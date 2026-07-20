from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from unienv_genesis.nodes.sensors.camera_node import CameraSensorNode
from unienv_genesis.nodes.sensors.keypoints.camera_port import GenesisCameraKeypointPort
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis_collection.objects.base_object import GraspCandidates
from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider


class ContactGraspNetGraspCandidateProvider(GraspCandidateProvider):
    """Camera-based grasp proposals from a PyTorch Contact-GraspNet port."""

    DEFAULT_CHECKPOINT_DIR = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "contact_graspnet_pytorch"
        / "checkpoints"
        / "contact_graspnet"
    )

    # The model checkpoint is *not* shipped in the package or committed to git.
    # When the default checkpoint directory is used and the checkpoint file is
    # missing, it is downloaded on demand from the canonical upstream
    # PyTorch-port GitHub repository.
    #
    # Downloads are never written into the (possibly read-only) installed
    # package directory. Instead they are materialized under a user-writable
    # cache directory (see :meth:`_default_download_cache_dir`), and
    # :meth:`_ensure_checkpoint_file` returns that cache directory for
    # loading; the configured ``checkpoint_dir`` is never mutated.
    #
    # The URL is pinned to an immutable upstream commit (not a floating
    # branch), and every downloaded/cached default checkpoint is validated
    # against the independently verified size and SHA-256 below.
    CHECKPOINT_UPSTREAM_COMMIT = "a3c92388a5134ac0ad6d200d6e51b6317656cc3f"
    DEFAULT_CHECKPOINT_URL = (
        "https://raw.githubusercontent.com/elchun/contact_graspnet_pytorch/"
        f"{CHECKPOINT_UPSTREAM_COMMIT}"
        "/checkpoints/contact_graspnet/checkpoints/model.pt"
    )
    EXPECTED_CHECKPOINT_SIZE = 26_628_146  # bytes
    EXPECTED_CHECKPOINT_SHA256 = (
        "39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762"
    )
    CHECKPOINT_CHUNK_SIZE = 1 << 20  # 1 MiB streaming chunks
    CHECKPOINT_URLOPEN_TIMEOUT = 60.0  # seconds; applied to every urlopen call
    MODEL_FILENAME = "model.pt"
    MODEL_SUBDIR = "checkpoints"

    def __init__(
        self,
        *,
        world: GenesisWorld,
        camera_node: CameraSensorNode,
        target_entity,
        checkpoint_dir: Optional[str] = None,
        forward_passes: int = 5,
        z_range: Tuple[float, float] = (0.2, 1.8),
        local_regions: bool = True,
        filter_grasps: bool = True,
        max_candidates: int = 64,
        contact_graspnet_gripper_depth: float = 0.1034,
        gripper_width_limit: Optional[float] = None,
    ) -> None:
        self.world = world
        self.camera_node = camera_node
        self.target_entity = target_entity
        # Immutable: whether this provider uses the *default* checkpoint
        # source is derived directly from this configured directory on every
        # ensure; it is never repointed to the download cache.
        self.checkpoint_dir = str(checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR)
        self.forward_passes = int(forward_passes)
        self.z_range = tuple(float(v) for v in z_range)
        self.local_regions = bool(local_regions)
        self.filter_grasps = bool(filter_grasps)
        self.max_candidates = int(max_candidates)
        self.contact_graspnet_gripper_depth = float(contact_graspnet_gripper_depth)
        self.gripper_width_limit = (
            None if gripper_width_limit is None else float(gripper_width_limit)
        )
        self._camera_port = GenesisCameraKeypointPort(camera_node, world)
        self._seg_index = None
        self._grasp_estimator = None
        self.last_debug = None

    def _require_unbatched(self) -> None:
        if self.world.batch_size is not None:
            raise NotImplementedError(
                "ContactGraspNetGraspCandidateProvider currently supports only "
                "unbatched environments."
            )

    def _target_segmap(self, raw_seg: torch.Tensor) -> torch.Tensor:
        if self._seg_index is None:
            self._camera_port.validate()
            self._seg_index = self._camera_port.build_segmentation_index()

        target_entity_idx = int(self.target_entity.idx)
        target_seg_ids = [
            seg_id
            for seg_id, entity_idx in self._seg_index.seg_to_entity_idx.items()
            if entity_idx == target_entity_idx
        ]
        if not target_seg_ids:
            raise RuntimeError(
                f"Camera '{self.camera_node.name}' segmentation does not contain "
                f"the target entity idx={target_entity_idx}. Ensure render_seg=True "
                "and the target is visible to the camera."
            )

        seg = raw_seg.detach()
        out = torch.zeros_like(seg, dtype=torch.int32)
        target_ids = torch.as_tensor(
            list(target_seg_ids), dtype=seg.dtype, device=seg.device
        )
        mask = torch.isin(seg, target_ids)
        out[mask] = 1
        return out

    @staticmethod
    def _debug_array(x) -> np.ndarray:
        """Coerce a torch/numpy/scalar to a NumPy array for npz serialization."""
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu").numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def save_debug(self, output_dir: str | Path) -> None:
        if self.last_debug is None:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = self.last_debug.get("arrays", {})
        meta = self.last_debug.get("meta", {})
        if arrays:
            np.savez(output_dir / "contact_graspnet_debug.npz", **arrays)
        if meta:
            (output_dir / "contact_graspnet_debug.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

    def _ensure_model(self):
        if self._grasp_estimator is not None:
            return self._grasp_estimator
        try:
            config_utils = importlib.import_module(
                "unienv_genesis_refpolicy.third_party.contact_graspnet_pytorch.config_utils"
            )
            estimator_mod = importlib.import_module(
                "unienv_genesis_refpolicy.third_party.contact_graspnet_pytorch.contact_grasp_estimator"
            )
            checkpoints_mod = importlib.import_module(
                "unienv_genesis_refpolicy.third_party.contact_graspnet_pytorch.checkpoints"
            )
        except Exception as exc:  # pragma: no cover - runtime path
            raise RuntimeError(
                "Failed to import vendored PyTorch Contact-GraspNet implementation."
            ) from exc

        resolved_checkpoint_dir = self._ensure_checkpoint_file()

        global_config = config_utils.load_config(
            str(resolved_checkpoint_dir),
            batch_size=self.forward_passes,
        )
        grasp_estimator = estimator_mod.GraspEstimator(global_config)
        checkpoint_io = checkpoints_mod.CheckpointIO(
            checkpoint_dir=str(resolved_checkpoint_dir / "checkpoints"),
            model=grasp_estimator.model,
        )
        checkpoint_io.load(self.MODEL_FILENAME)
        grasp_estimator.model.eval()
        self._grasp_estimator = grasp_estimator
        return self._grasp_estimator

    @staticmethod
    def _default_download_cache_dir() -> Path:
        """A user-writable directory for on-demand checkpoint downloads.

        Honors ``XDG_CACHE_HOME`` and falls back to ``~/.cache``. This is used
        so we never write into the installed package directory (which may be
        read-only).
        """
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            return Path(xdg) / "unienv_genesis_refpolicy" / "contact_graspnet"
        return Path.home() / ".cache" / "unienv_genesis_refpolicy" / "contact_graspnet"

    def _checkpoint_matches_expected(self, path: Path) -> bool:
        """Return True iff ``path`` matches the pinned size and SHA-256."""
        try:
            if path.stat().st_size != self.EXPECTED_CHECKPOINT_SIZE:
                return False
        except OSError:
            return False
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(self.CHECKPOINT_CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest() == self.EXPECTED_CHECKPOINT_SHA256

    def _download_checkpoint(self, target: Path) -> None:
        """Stream the pinned checkpoint to ``target`` atomically.

        The payload is streamed to a temporary file in the target directory
        while computing its size and SHA-256. The network read uses an
        explicit timeout (``CHECKPOINT_URLOPEN_TIMEOUT``) and aborts
        immediately if the streamed bytes exceed the expected size. The file
        is flushed, fsynced, validated against the exact size and SHA-256,
        and only then atomically ``os.replace``-d into place. The temporary
        file is always removed on failure, and any pre-existing target is
        left untouched until a fully validated replacement exists.
        """
        url = self.DEFAULT_CHECKPOINT_URL
        target.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[ContactGraspNet] Downloading checkpoint from '{url}' into "
            f"'{target}'..."
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".part", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            from urllib.request import urlopen

            hasher = hashlib.sha256()
            total = 0
            with os.fdopen(fd, "wb") as fh:
                with urlopen(url, timeout=self.CHECKPOINT_URLOPEN_TIMEOUT) as resp:
                    while True:
                        chunk = resp.read(self.CHECKPOINT_CHUNK_SIZE)
                        if not chunk:
                            break
                        fh.write(chunk)
                        hasher.update(chunk)
                        total += len(chunk)
                        if total > self.EXPECTED_CHECKPOINT_SIZE:
                            raise RuntimeError(
                                f"Contact-GraspNet checkpoint from '{url}' "
                                f"exceeds the expected "
                                f"{self.EXPECTED_CHECKPOINT_SIZE} bytes; "
                                f"aborting download."
                            )
                fh.flush()
                os.fsync(fh.fileno())

            if total != self.EXPECTED_CHECKPOINT_SIZE:
                raise RuntimeError(
                    f"Contact-GraspNet checkpoint size mismatch from '{url}': "
                    f"expected {self.EXPECTED_CHECKPOINT_SIZE} bytes, got "
                    f"{total} bytes."
                )
            digest = hasher.hexdigest()
            if digest != self.EXPECTED_CHECKPOINT_SHA256:
                raise RuntimeError(
                    f"Contact-GraspNet checkpoint SHA-256 mismatch from '{url}': "
                    f"expected {self.EXPECTED_CHECKPOINT_SHA256}, got {digest}."
                )
            os.replace(tmp_path, target)
        except BaseException as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            if isinstance(exc, Exception) and not isinstance(exc, RuntimeError):
                raise RuntimeError(
                    f"Failed to download Contact-GraspNet checkpoint from "
                    f"'{url}' to '{target}': {exc}"
                ) from exc
            raise

    def _ensure_checkpoint_file(self) -> Path:
        """Make sure the Contact-GraspNet checkpoint file is present locally.

        Returns the resolved checkpoint directory whose
        ``MODEL_SUBDIR/MODEL_FILENAME`` file is ready for loading. The
        configured ``self.checkpoint_dir`` is never mutated.

        Default vs custom behavior is derived directly from the immutable
        configured ``self.checkpoint_dir`` on every call.

        For a custom ``checkpoint_dir`` the behavior is explicit: an existing
        file is used as-is (no integrity check against the default pinned
        checkpoint, since users may supply their own weights), and a missing
        file raises :class:`FileNotFoundError` instead of silently
        downloading into a user-provided location.

        For the default checkpoint directory, the checkpoint is validated
        against the pinned size/SHA-256 on *every* call. An invalid file in
        the (possibly read-only) package directory raises a clear error. When
        the file is missing it is downloaded on demand from the immutable
        upstream URL into a user-writable cache directory (see
        :meth:`_default_download_cache_dir`) and that cache directory is
        returned; an invalid cached file is re-downloaded, with the atomic
        replace ensuring a failed re-download never destroys the previous
        file.
        """
        checkpoint_dir = Path(self.checkpoint_dir)
        is_default = (
            checkpoint_dir.resolve() == self.DEFAULT_CHECKPOINT_DIR.resolve()
        )

        if not is_default:
            target = checkpoint_dir / self.MODEL_SUBDIR / self.MODEL_FILENAME
            if target.exists():
                return checkpoint_dir
            raise FileNotFoundError(
                f"Contact-GraspNet checkpoint not found at '{target}'. A custom "
                f"checkpoint_dir was provided ('{self.checkpoint_dir}'); place the "
                f"checkpoint file at '{self.MODEL_SUBDIR}/{self.MODEL_FILENAME}' "
                f"inside it, or omit checkpoint_dir to download it automatically."
            )

        # Validate anything already present in the default package directory,
        # but never write into the (possibly read-only) installed package.
        package_dir = self.DEFAULT_CHECKPOINT_DIR
        package_target = package_dir / self.MODEL_SUBDIR / self.MODEL_FILENAME
        if package_target.exists():
            if self._checkpoint_matches_expected(package_target):
                return package_dir
            raise RuntimeError(
                f"Contact-GraspNet checkpoint at '{package_target}' does not "
                f"match the expected size/SHA-256 for pinned upstream commit "
                f"{self.CHECKPOINT_UPSTREAM_COMMIT}. Remove the corrupt file "
                f"or provide a custom checkpoint_dir."
            )

        # Materialize under a user-writable cache dir and return it.
        cache_dir = self._default_download_cache_dir()
        target = cache_dir / self.MODEL_SUBDIR / self.MODEL_FILENAME
        if target.exists():
            if self._checkpoint_matches_expected(target):
                return cache_dir
            print(
                f"[ContactGraspNet] Cached checkpoint at '{target}' failed "
                f"integrity validation; re-downloading."
            )

        self._download_checkpoint(target)
        print(f"[ContactGraspNet] Checkpoint ready at '{target}'.")
        return cache_dir

    def _grasps_cam_to_world_candidates(
        self,
        grasps_cam: torch.Tensor,
        extrinsics_cv: torch.Tensor,
    ) -> GraspCandidates:
        world_to_cam = torch.eye(4, dtype=torch.float32, device=self.world.device)
        world_to_cam[:3, :] = extrinsics_cv.to(dtype=torch.float32, device=self.world.device)
        cam_to_world = torch.linalg.inv(world_to_cam)
        grasps_cam = grasps_cam.to(dtype=torch.float32, device=self.world.device)
        grasps_world = torch.einsum("ij,njk->nik", cam_to_world, grasps_cam)

        opening = grasps_world[:, :3, 0]
        approach = grasps_world[:, :3, 2]
        opening = opening / (opening.norm(dim=-1, keepdim=True) + 1e-12)
        approach = approach / (approach.norm(dim=-1, keepdim=True) + 1e-12)
        position = grasps_world[:, :3, 3] + self.contact_graspnet_gripper_depth * approach

        dtype = torch.float32 if self.world.dtype is None else self.world.dtype
        return GraspCandidates(
            position=position.to(device=self.world.device, dtype=dtype),
            approach_direction=approach.to(device=self.world.device, dtype=dtype),
            opening_direction=opening.to(device=self.world.device, dtype=dtype),
        )

    def propose(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> GraspCandidates:
        del mask
        self._require_unbatched()
        obs = self.camera_node.get_observation()
        required = ("depth", "intrinsics_cv", "extrinsics_cv", "seg")
        missing = [key for key in required if key not in obs]
        if missing:
            raise RuntimeError(
                f"Camera '{self.camera_node.name}' is missing required observations "
                f"for Contact-GraspNet: {missing}. Enable render_depth and render_seg."
            )

        segmap = self._target_segmap(obs["seg"])
        if not bool((segmap == 1).any()):
            self.last_debug = {
                "meta": {"status": "empty_target_segmap"},
                "arrays": {"segmap": self._debug_array(segmap)},
            }
            return GraspCandidates.empty(device=self.world.device, dtype=self.world.dtype)

        estimator = self._ensure_model()
        rgb = obs["rgb"] if "rgb" in obs else None
        with torch.no_grad():
            pred_grasps_cam, scores, _contact_pts, openings = (
                estimator.predict_scene_grasps_from_depth_K_and_2d_seg(
                    None,
                    obs["depth"],
                    segmap,
                    obs["intrinsics_cv"],
                    z_range=list(self.z_range),
                    local_regions=self.local_regions,
                    filter_grasps=self.filter_grasps,
                    segmap_id=1,
                    rgb=rgb,
                    forward_passes=self.forward_passes,
                    seed=seed,
                )
            )

        key = 1 if 1 in pred_grasps_cam else (-1 if -1 in pred_grasps_cam else None)
        if key is None or pred_grasps_cam[key] is None or len(pred_grasps_cam[key]) == 0:
            self.last_debug = {
                "meta": {
                    "status": "no_raw_grasps",
                    "available_keys": sorted(int(k) for k in pred_grasps_cam.keys()),
                },
                "arrays": {
                    "segmap": self._debug_array(segmap),
                    "depth": self._debug_array(obs["depth"]),
                    "intrinsics_cv": self._debug_array(obs["intrinsics_cv"]),
                    "extrinsics_cv": self._debug_array(obs["extrinsics_cv"]),
                },
            }
            return GraspCandidates.empty(device=self.world.device, dtype=self.world.dtype)

        grasps_cam = pred_grasps_cam[key].to(torch.float32)
        scores = scores[key].to(torch.float32).reshape(-1)
        openings = openings[key].to(torch.float32).reshape(-1)
        raw_grasps_cam = grasps_cam.clone()
        raw_scores = scores.clone()
        raw_openings = openings.clone()

        if grasps_cam.numel() == 0:
            return GraspCandidates.empty(device=self.world.device, dtype=self.world.dtype)

        order = torch.argsort(scores, descending=True)
        grasps_cam = grasps_cam[order]
        scores = scores[order]
        openings = openings[order]
        if self.gripper_width_limit is not None:
            keep = openings <= (self.gripper_width_limit + 1e-6)
            grasps_cam = grasps_cam[keep]
            scores = scores[keep]
            openings = openings[keep]
        if self.max_candidates > 0:
            grasps_cam = grasps_cam[: self.max_candidates]
            scores = scores[: self.max_candidates]
            openings = openings[: self.max_candidates]
        if grasps_cam.numel() == 0:
            self.last_debug = {
                "meta": {
                    "status": "all_grasps_filtered",
                    "raw_candidate_count": int(raw_grasps_cam.shape[0]),
                    "width_limit": self.gripper_width_limit,
                },
                "arrays": {
                    "raw_grasp_poses_cam": self._debug_array(raw_grasps_cam),
                    "raw_scores": self._debug_array(raw_scores),
                    "raw_openings": self._debug_array(raw_openings),
                    "segmap": self._debug_array(segmap),
                    "depth": self._debug_array(obs["depth"]),
                    "intrinsics_cv": self._debug_array(obs["intrinsics_cv"]),
                    "extrinsics_cv": self._debug_array(obs["extrinsics_cv"]),
                },
            }
            return GraspCandidates.empty(device=self.world.device, dtype=self.world.dtype)
        candidates = self._grasps_cam_to_world_candidates(
            grasps_cam,
            obs["extrinsics_cv"],
        )
        self.last_debug = {
            "meta": {
                "status": "ok",
                "raw_candidate_count": int(raw_grasps_cam.shape[0]),
                "post_filter_candidate_count": int(grasps_cam.shape[0]),
                "width_limit": self.gripper_width_limit,
                "selected_segmap_key": int(key),
            },
            "arrays": {
                "raw_grasp_poses_cam": self._debug_array(raw_grasps_cam),
                "raw_scores": self._debug_array(raw_scores),
                "raw_openings": self._debug_array(raw_openings),
                "filtered_grasp_poses_cam": self._debug_array(grasps_cam),
                "filtered_scores": self._debug_array(scores),
                "filtered_openings": self._debug_array(openings),
                "candidate_positions_world": self._debug_array(candidates.position),
                "candidate_approach_world": self._debug_array(candidates.approach_direction),
                "candidate_opening_world": self._debug_array(candidates.opening_direction),
                "segmap": self._debug_array(segmap),
                "depth": self._debug_array(obs["depth"]),
                "intrinsics_cv": self._debug_array(obs["intrinsics_cv"]),
                "extrinsics_cv": self._debug_array(obs["extrinsics_cv"]),
            },
        }
        return candidates
