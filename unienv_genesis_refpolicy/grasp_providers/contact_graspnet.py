from __future__ import annotations

import importlib
import json
import os
import shutil
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
    # cache directory (see :meth:`_default_download_cache_dir`), and the
    # provider's ``checkpoint_dir`` is repointed there for loading.
    DEFAULT_CHECKPOINT_URL = (
        "https://raw.githubusercontent.com/elchun/contact_graspnet_pytorch/main/"
        "checkpoints/contact_graspnet/checkpoints/model.pt"
    )
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

        self._ensure_checkpoint_file()

        global_config = config_utils.load_config(
            self.checkpoint_dir,
            batch_size=self.forward_passes,
        )
        grasp_estimator = estimator_mod.GraspEstimator(global_config)
        checkpoint_io = checkpoints_mod.CheckpointIO(
            checkpoint_dir=str(Path(self.checkpoint_dir) / "checkpoints"),
            model=grasp_estimator.model,
        )
        checkpoint_io.load(self.MODEL_FILENAME)
        grasp_estimator.model.eval()
        self._grasp_estimator = grasp_estimator
        return self._grasp_estimator

    def _checkpoint_file_path(self) -> Path:
        return Path(self.checkpoint_dir) / self.MODEL_SUBDIR / self.MODEL_FILENAME

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

    def _ensure_checkpoint_file(self) -> None:
        """Make sure the Contact-GraspNet checkpoint file is present locally.

        If the file already exists at ``checkpoint_dir``, this is a no-op
        (preserving the existing behavior for both the default and custom
        ``checkpoint_dir`` paths).

        If the file is missing *and* the provider is using the default
        checkpoint directory, the checkpoint is downloaded on demand from the
        canonical upstream PyTorch-port GitHub URL into a user-writable cache
        directory (see :meth:`_default_download_cache_dir`), and
        ``checkpoint_dir`` is repointed there so subsequent loading reads the
        cached file. For a custom ``checkpoint_dir`` that lacks the file, a
        clear error is raised instead of silently downloading into an
        arbitrary user-provided location.
        """
        target = self._checkpoint_file_path()
        if target.exists():
            return

        using_default = (
            Path(self.checkpoint_dir).resolve()
            == self.DEFAULT_CHECKPOINT_DIR.resolve()
        )
        if not using_default:
            raise FileNotFoundError(
                f"Contact-GraspNet checkpoint not found at '{target}'. A custom "
                f"checkpoint_dir was provided ('{self.checkpoint_dir}'); place the "
                f"checkpoint file at '{self.MODEL_SUBDIR}/{self.MODEL_FILENAME}' "
                f"inside it, or omit checkpoint_dir to download it automatically."
            )

        # Don't write into the (possibly read-only) installed package
        # directory. Materialize under a user-writable cache dir and repoint
        # the loader there.
        cache_dir = self._default_download_cache_dir()
        self.checkpoint_dir = str(cache_dir)
        target = self._checkpoint_file_path()
        if target.exists():
            return

        url = self.DEFAULT_CHECKPOINT_URL

        target.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[ContactGraspNet] Downloading checkpoint from '{url}' into "
            f"'{target}'..."
        )
        try:
            from urllib.request import urlopen

            with urlopen(url) as resp, open(target, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Contact-GraspNet checkpoint from '{url}' "
                f"to '{target}': {exc}"
            ) from exc

        if not target.exists() or target.stat().st_size == 0:
            raise RuntimeError(
                f"Failed to download Contact-GraspNet checkpoint to '{target}'."
            )
        print(f"[ContactGraspNet] Checkpoint ready at '{target}'.")

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
