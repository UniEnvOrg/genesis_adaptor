"""Keypoint tracker sensor node for Genesis.

This sensor delegates image acquisition to an existing :class:`CameraSensorNode`
that provides segmentation and per-pixel 3D positions. At reset/reload time it:

1. Proposes keypoints from the camera image via a :class:`PointProposer`.
2. Maps each keypoint to a rigid entity using camera segmentation metadata.
3. Stores each keypoint's local offset in the entity frame.

After each environment step, keypoints are updated from entity poses only
(ground-truth tracking), and per-step 3D flow is produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
)
from xbarray.transformations.pointcloud.pytorch import (
    world_to_pixel_coordinate_and_depth,
)

from unienv_interface.space import DictSpace, BoxSpace
from unienv_interface.backends.pytorch import (
    PyTorchComputeBackend,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils import get_batch_aware_shape

from .base import SensorNode
from .camera_node import CameraSensorNode

__all__ = [
    "PointProposer",
    "UniformSegmentationPointProposer",
    "KeypointTrackerNode",
]


class PointProposer(ABC):
    """Base class for selecting keypoints from camera observations."""

    @abstractmethod
    def propose(
        self,
        *,
        camera_obs: Dict[str, torch.Tensor],
        seg_to_entity_idx: Mapping[int, int],
        entity_names: Mapping[int, str],
        max_points: int,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select keypoints from batched camera observations.

        The caller is responsible for pre-filtering ``camera_obs`` to only
        contain active entries.  That is, if only ``B'`` out of ``B`` total
        environments need proposals, the caller should slice all tensors in
        ``camera_obs`` down to ``B'`` before calling this method, so the
        batch dimension of every value in ``camera_obs`` is ``B'``.

        The ``mask`` parameter is **not** used for filtering — the caller
        handles that.  It is passed through so that proposers with internal
        per-environment state (e.g. history buffers) can update bookkeeping
        for the active entries.

        Args:
            camera_obs: Pre-filtered camera observation dictionary with
                batch size ``B'`` (only active environments).  Guaranteed to
                contain at least ``"seg"`` (``(B', H, W)`` int segmentation
                map) and ``"position"`` (``(B', H, W, 3)`` float32 world
                positions).  May also contain ``"rgb"``, ``"depth"``,
                ``"normal"``, ``"intrinsics_cv"``, ``"extrinsics_cv"``.
            seg_to_entity_idx: Maps segmentation IDs to entity indices.
            entity_names: Mapping from entity index to entity name string.
                Proposers can use this to filter which entities to track.
            max_points: Maximum number of keypoints to return per environment.
            mask: Optional ``(B',)`` bool tensor passed for proposers that
                maintain internal state.  All entries are ``True`` (the
                caller already filtered).  Stateless proposers may ignore it.
            generator: Optional torch RNG for reproducible randomization.

        Returns:
            ``(points_world, seg_ids, valid_mask)`` where ``points_world``
            is a ``(B', max_points, 3)`` float32 tensor, ``seg_ids`` is a
            ``(B', max_points)`` int64 tensor, and ``valid_mask`` is a
            ``(B', max_points)`` bool tensor indicating which slots hold
            proposed points.
        """


class UniformSegmentationPointProposer(PointProposer):
    """Simple point proposer using valid segmented pixels.

    Args:
        max_points_per_entity: Optional cap for points selected from each
            segmentation id.
        randomize: If ``True``, randomly sub-sample candidate pixels.
            If ``False``, keep deterministic raster-scan order.
        tracked_entity_names: If provided, only track entities whose names
            are in this set.
        tracked_entity_name_prefixes: If provided, also track entities
            whose names start with any of these prefixes.
    """

    def __init__(
        self,
        *,
        max_points_per_entity: Optional[int] = None,
        randomize: bool = False,
        tracked_entity_names: Optional[Sequence[str]] = None,
        tracked_entity_name_prefixes: Optional[Sequence[str]] = None,
    ):
        self.max_points_per_entity = max_points_per_entity
        self.randomize = randomize
        self.tracked_entity_names: Optional[Set[str]] = (
            set(tracked_entity_names) if tracked_entity_names is not None else None
        )
        self.tracked_entity_name_prefixes: Optional[Tuple[str, ...]] = (
            tuple(tracked_entity_name_prefixes)
            if tracked_entity_name_prefixes is not None
            else None
        )

    def _name_allowed(self, entity_name: str) -> bool:
        if self.tracked_entity_names is None and self.tracked_entity_name_prefixes is None:
            return True
        exact_ok = (
            self.tracked_entity_names is not None
            and entity_name in self.tracked_entity_names
        )
        prefix_ok = (
            self.tracked_entity_name_prefixes is not None
            and any(entity_name.startswith(p) for p in self.tracked_entity_name_prefixes)
        )
        return exact_ok or prefix_ok

    def _choose(
        self,
        pixels: torch.Tensor,
        n: int,
        *,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if pixels.shape[0] <= n:
            return pixels
        if self.randomize:
            perm = torch.randperm(pixels.shape[0], generator=generator, device=pixels.device)
            return pixels[perm[:n]]
        return pixels[:n]

    def _propose_single(
        self,
        segmentation: torch.Tensor,
        position: torch.Tensor,
        filtered_seg_to_entity_idx: Mapping[int, int],
        max_points: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propose keypoints for a single environment.

        Returns ``(points_world, seg_ids)`` with variable length ``N``.
        """
        valid_mask = torch.isfinite(position).all(dim=-1)  # (H, W)
        device = segmentation.device

        _empty = (
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
        )

        if max_points <= 0 or len(filtered_seg_to_entity_idx) == 0:
            return _empty

        allowed_seg_ids = torch.tensor(
            list(filtered_seg_to_entity_idx.keys()),
            dtype=segmentation.dtype,
            device=device,
        )
        candidate_mask = valid_mask & torch.isin(segmentation, allowed_seg_ids)
        candidate_pixels = torch.nonzero(candidate_mask, as_tuple=False)  # (K, 2)
        if candidate_pixels.numel() == 0:
            return _empty

        if self.max_points_per_entity is None:
            chosen_pixels = self._choose(candidate_pixels, max_points, generator=generator)
        else:
            seg_values = segmentation[candidate_pixels[:, 0], candidate_pixels[:, 1]]
            entity_values = torch.full_like(seg_values, -1, dtype=torch.int64)
            for seg_id, entity_idx in filtered_seg_to_entity_idx.items():
                entity_values[seg_values == int(seg_id)] = int(entity_idx)

            valid_entity = entity_values >= 0
            if not torch.any(valid_entity):
                return _empty
            candidate_pixels = candidate_pixels[valid_entity]
            entity_values = entity_values[valid_entity]

            selected: list[torch.Tensor] = []
            remaining = max_points
            for entity_idx in torch.unique(entity_values):
                if remaining <= 0:
                    break
                entity_pixels = candidate_pixels[entity_values == entity_idx]
                take = min(self.max_points_per_entity, entity_pixels.shape[0], remaining)
                if take <= 0:
                    continue
                selected.append(self._choose(entity_pixels, take, generator=generator))
                remaining -= take

            if len(selected) == 0:
                return _empty
            chosen_pixels = torch.cat(selected, dim=0)

        # Look up 3D positions and seg IDs at selected pixels.
        y = chosen_pixels[:, 0].long()
        x = chosen_pixels[:, 1].long()
        points_world = position[y, x].to(torch.float32)  # (N, 3)
        seg_ids = segmentation[y, x].to(torch.int64)      # (N,)
        return points_world, seg_ids

    def propose(
        self,
        *,
        camera_obs: Dict[str, torch.Tensor],
        seg_to_entity_idx: Mapping[int, int],
        entity_names: Mapping[int, str],
        max_points: int,
        mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        segmentation = camera_obs["seg"]       # (B, H, W)
        position = camera_obs["position"]      # (B, H, W, 3)
        B = segmentation.shape[0]
        device = segmentation.device

        filtered_seg_to_entity_idx = {
            seg_id: entity_idx
            for seg_id, entity_idx in seg_to_entity_idx.items()
            if self._name_allowed(entity_names.get(entity_idx, ""))
        }

        out_points = torch.zeros((B, max_points, 3), dtype=torch.float32, device=device)
        out_seg = torch.zeros((B, max_points), dtype=torch.int64, device=device)
        out_valid = torch.zeros((B, max_points), dtype=torch.bool, device=device)

        for b in range(B):
            pts, sids = self._propose_single(
                segmentation[b], position[b],
                filtered_seg_to_entity_idx, max_points, generator,
            )
            n = pts.shape[0]
            if n > 0:
                out_points[b, :n] = pts
                out_seg[b, :n] = sids
                out_valid[b, :n] = True

        return out_points, out_seg, out_valid


class KeypointTrackerNode(SensorNode[None, None, Dict[str, PyTorchArrayType]]):
    """Keypoint tracking sensor.

    Notes
    -----
    - Tracks keypoints on all visible ``RigidEntity`` objects, including
      multi-link articulated bodies.  Entity name filtering is handled by the
      :class:`PointProposer` passed at construction time.
    - The delegated camera must provide segmentation (``render_seg=True``)
      and either depth or position rendering.
    """

    reload_priorities = {0}
    reset_priorities = {0}

    # Must run after delegated camera after_reset/after_reload (-50).
    after_reload_priorities = {-60}
    after_reset_priorities = {-60}

    # Must run after delegated camera post step (0).
    post_environment_step_priorities = {-1}

    def __init__(
        self,
        world: GenesisWorld,
        name: str,
        camera_node: CameraSensorNode,
        *,
        max_points: int = 128,
        point_proposer: Optional[PointProposer] = None,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
    ):
        super().__init__(
            world=world,
            name=name,
            control_timestep=control_timestep,
            update_timestep=update_timestep,
        )
        if max_points <= 0:
            raise ValueError(f"max_points must be > 0, got {max_points}")
        if camera_node.world is not world:
            raise ValueError("camera_node must belong to the same world.")
        if not camera_node.render_seg:
            raise ValueError("camera_node must render segmentation (render_seg=True).")
        if not (camera_node.render_depth or camera_node.render_position):
            raise ValueError("camera_node must render depth or position.")

        self.world = world
        self.camera_node = camera_node
        self.max_points = int(max_points)
        self.point_proposer = point_proposer or UniformSegmentationPointProposer()

        self._n_envs = self.world.batch_size if self.world.batch_size is not None else 1
        self._device = self.world.device
        self._entity_by_idx: Dict[int, RigidEntity] = {}
        self._link_by_idx: Dict[int, RigidLink] = {}
        self._rng = torch.Generator(device=self._device)

        self._points_world = torch.zeros((self._n_envs, self.max_points, 3), dtype=torch.float32, device=self._device)
        self._prev_points_world = torch.zeros_like(self._points_world)
        self._flow_world = torch.zeros_like(self._points_world)
        self._local_offsets = torch.zeros_like(self._points_world)
        self._valid_mask = torch.zeros((self._n_envs, self.max_points), dtype=torch.bool, device=self._device)
        self._entity_idx = torch.full((self._n_envs, self.max_points), -1, dtype=torch.int32, device=self._device)
        self._link_idx = torch.full((self._n_envs, self.max_points), -1, dtype=torch.int32, device=self._device)
        self._seg_idxc = torch.full((self._n_envs, self.max_points), -1, dtype=torch.int32, device=self._device)
        self._points_2d = torch.zeros((self._n_envs, self.max_points, 2), dtype=torch.float32, device=self._device)
        self._visibility_mask = torch.zeros((self._n_envs, self.max_points), dtype=torch.bool, device=self._device)

        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        self.observation_space = self._get_observation_space()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_observation_space(self) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        B = self.world.batch_size
        spaces = {
            "point_positions": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=get_batch_aware_shape(B, (self.max_points, 3)),
                dtype=torch.float32,
                device=self._device,
            ),
            "points_2d": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=get_batch_aware_shape(B, (self.max_points, 2)),
                dtype=torch.float32,
                device=self._device,
            ),
            "point_movements": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=get_batch_aware_shape(B, (self.max_points, 3)),
                dtype=torch.float32,
                device=self._device,
            ),
            "point_segmentations": BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=2**31 - 1,
                shape=get_batch_aware_shape(B, (self.max_points,)),
                dtype=torch.int32,
                device=self._device,
            ),
            "valid_masks": BoxSpace(
                PyTorchComputeBackend,
                low=0,
                high=1,
                shape=get_batch_aware_shape(B, (self.max_points,)),
                dtype=torch.uint8,
                device=self._device,
            ),
            "visibility_masks": BoxSpace(
                PyTorchComputeBackend,
                low=0,
                high=1,
                shape=get_batch_aware_shape(B, (self.max_points,)),
                dtype=torch.uint8,
                device=self._device,
            ),
        }
        return DictSpace(PyTorchComputeBackend, spaces, device=self._device)

    def _zero_env_slots(self, env_idx: int) -> None:
        self._points_world[env_idx].zero_()
        self._prev_points_world[env_idx].zero_()
        self._flow_world[env_idx].zero_()
        self._local_offsets[env_idx].zero_()
        self._valid_mask[env_idx].zero_()
        self._entity_idx[env_idx].fill_(-1)
        self._link_idx[env_idx].fill_(-1)
        self._seg_idxc[env_idx].fill_(-1)
        self._points_2d[env_idx].zero_()
        self._visibility_mask[env_idx].zero_()

    def _camera_env_indices(self, n_camera_envs: int) -> torch.Tensor:
        """Map camera batch axis to world environment indices."""
        if self.world.batch_size is None:
            return torch.arange(n_camera_envs, dtype=torch.long, device=self._device)

        if n_camera_envs == self._n_envs:
            return torch.arange(self._n_envs, dtype=torch.long, device=self._device)

        sensor = self.camera_node.sensor
        rendered_envs_idx = None
        if hasattr(sensor, "_visualizer") and hasattr(sensor._visualizer, "_context"):
            rendered_envs_idx = getattr(sensor._visualizer._context, "rendered_envs_idx", None)
        if rendered_envs_idx is not None and len(rendered_envs_idx) == n_camera_envs:
            return torch.tensor(rendered_envs_idx, dtype=torch.long, device=self._device)

        if n_camera_envs == 1:
            env_idx = self.camera_node.option.env_idx if self.camera_node.option.env_idx is not None else 0
            return torch.tensor([int(env_idx)], dtype=torch.long, device=self._device)

        return torch.arange(min(n_camera_envs, self._n_envs), dtype=torch.long, device=self._device)

    def _get_camera_maps(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Fetch segmentation, position, and env indices from the camera.

        Returns:
            seg: ``(B, H, W)`` segmentation tensor.
            position: ``(B, H, W, 3)`` world-position tensor, or ``None`` if
                the camera provides neither position nor depth.
            env_indices: ``(B,)`` long tensor mapping camera batch to world envs.
        """
        obs = self.camera_node.get_observation()
        if obs is None:
            raise RuntimeError(
                f"Camera node '{self.camera_node.name}' has no observation yet; "
                "ensure this tracking node runs after the camera."
            )
        if "seg" not in obs:
            raise RuntimeError(f"Camera node '{self.camera_node.name}' must provide segmentation ('seg').")

        seg = obs["seg"]
        if "position" in obs:
            position = obs["position"]
        elif "depth" in obs:
            position = self.camera_node.depth_to_position_map(obs["depth"])
        else:
            # No 3D data available -- caller should handle gracefully.
            position = None

        if seg.ndim == 2:
            seg = seg.unsqueeze(0)
        elif seg.ndim != 3:
            raise ValueError(f"Unsupported segmentation shape: {tuple(seg.shape)}")

        if position is not None:
            if position.ndim == 3 and position.shape[-1] == 3:
                position = position.unsqueeze(0)
            elif position.ndim != 4 or position.shape[-1] != 3:
                raise ValueError(f"Unsupported position map shape: {tuple(position.shape)}")

            if seg.shape[0] != position.shape[0]:
                raise ValueError(
                    f"Segmentation and position batch sizes mismatch: {seg.shape[0]} vs {position.shape[0]}"
                )

        env_indices = self._camera_env_indices(seg.shape[0])
        if env_indices.shape[0] != seg.shape[0]:
            n = min(env_indices.shape[0], seg.shape[0])
            seg = seg[:n]
            if position is not None:
                position = position[:n]
            env_indices = env_indices[:n]

        return seg, position, env_indices

    @staticmethod
    def _seg_key_to_entity_and_link(seg_key: Union[int, Tuple[int, ...]]) -> Tuple[int, Optional[int]]:
        if isinstance(seg_key, tuple):
            entity_idx = int(seg_key[0])
            link_idx = int(seg_key[1]) if len(seg_key) >= 2 else None
            return entity_idx, link_idx
        return int(seg_key), None

    def _build_seg_to_entity_idx(self) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, str]]:
        """Build segmentation-ID to (entity_idx, link_idx) mapping.

        Returns:
            seg_to_entity_idx: Maps seg_idxc -> (entity_idx, link_idx).
            entity_names: Maps entity_idx -> entity name string.
        """
        seg_to_entity_idx: Dict[int, Tuple[int, int]] = {}
        entity_names: Dict[int, str] = {}
        self._entity_by_idx.clear()
        self._link_by_idx.clear()

        sensor = self.camera_node.sensor
        if not hasattr(sensor, "_visualizer"):
            return seg_to_entity_idx, entity_names

        seg_idx_dict = getattr(sensor._visualizer, "segmentation_idx_dict", None)
        if seg_idx_dict is None:
            return seg_to_entity_idx, entity_names

        rigid_solver = self.world.scene.rigid_solver
        rigid_entities = rigid_solver.entities
        rigid_links = rigid_solver.links
        n_entities = len(rigid_entities)
        n_links = len(rigid_links)

        for seg_idxc, seg_key in seg_idx_dict.items():
            entity_idx, link_idx = self._seg_key_to_entity_and_link(seg_key)
            if entity_idx < 0 or entity_idx >= n_entities:
                continue

            entity = rigid_entities[entity_idx]
            if not isinstance(entity, RigidEntity):
                continue

            resolved_link_idx = int(entity.base_link_idx) if link_idx is None else int(link_idx)
            if resolved_link_idx < 0 or resolved_link_idx >= n_links:
                continue

            link = rigid_links[resolved_link_idx]
            if not isinstance(link, RigidLink):
                continue

            seg_to_entity_idx[int(seg_idxc)] = (int(entity_idx), int(resolved_link_idx))
            entity_names[int(entity_idx)] = entity.name
            self._entity_by_idx[int(entity_idx)] = entity
            self._link_by_idx[int(resolved_link_idx)] = link

        return seg_to_entity_idx, entity_names

    def _populate_slots(
        self,
        *,
        env_idx: int,
        points_world: torch.Tensor,
        seg_ids: torch.Tensor,
        valid: torch.Tensor,
        seg_to_entity_and_link: Mapping[int, Tuple[int, int]],
    ) -> None:
        """Fill tracking slots for one environment from proposer output.

        Args:
            env_idx: Environment index.
            points_world: ``(max_points, 3)`` padded world positions.
            seg_ids: ``(max_points,)`` padded segmentation IDs.
            valid: ``(max_points,)`` bool mask from the proposer.
            seg_to_entity_and_link: Maps seg ID → (entity_idx, link_idx).
        """
        if not torch.any(valid):
            return

        pts = points_world[valid].float()
        sids = seg_ids[valid].long()
        n = pts.shape[0]

        entity_indices = torch.full((n,), -1, dtype=torch.int64, device=self._device)
        link_indices = torch.full((n,), -1, dtype=torch.int64, device=self._device)
        for i in range(n):
            assign = seg_to_entity_and_link.get(int(sids[i].item()), None)
            if assign is None:
                continue
            entity_indices[i] = int(assign[0])
            link_indices[i] = int(assign[1])

        resolved = (entity_indices >= 0) & (link_indices >= 0)
        if not torch.any(resolved):
            return

        sids = sids[resolved]
        pts = pts[resolved]
        entity_indices = entity_indices[resolved]
        link_indices = link_indices[resolved]
        n_valid = entity_indices.shape[0]

        local_offsets = torch.zeros((n_valid, 3), dtype=torch.float32, device=self._device)
        for link_idx in torch.unique(link_indices):
            link_idx_i = int(link_idx.item())
            link = self._link_by_idx.get(link_idx_i, None)
            if link is None:
                continue

            env_query = env_idx if self.world.batch_size is not None else None
            link_pos = link.get_pos(env_query).float()
            link_quat = link.get_quat(env_query).float()
            if link_pos.ndim == 2:
                link_pos = link_pos[0]
            if link_quat.ndim == 2:
                link_quat = link_quat[0]

            lm = link_indices == link_idx
            delta = pts[lm] - link_pos
            q_inv = quaternion_invert(link_quat.unsqueeze(0)).expand(delta.shape[0], -1)
            local_offsets[lm] = quaternion_apply(q_inv, delta)

        self._valid_mask[env_idx, :n_valid] = True
        self._visibility_mask[env_idx, :n_valid] = True
        self._entity_idx[env_idx, :n_valid] = entity_indices.to(torch.int32)
        self._link_idx[env_idx, :n_valid] = link_indices.to(torch.int32)
        self._seg_idxc[env_idx, :n_valid] = sids.to(torch.int32)
        self._local_offsets[env_idx, :n_valid] = local_offsets
        self._points_world[env_idx, :n_valid] = pts
        self._prev_points_world[env_idx, :n_valid] = pts
        self._flow_world[env_idx, :n_valid].zero_()

    def _init_tracks(self, mask: Optional[PyTorchArrayType] = None) -> None:
        seg, position, camera_env_indices = self._get_camera_maps()

        if mask is not None and self.world.batch_size is not None:
            mask_t = mask
        else:
            mask_t = None

        if mask_t is None:
            for env_idx in range(self._n_envs):
                self._zero_env_slots(env_idx)
        else:
            reset_envs = torch.where(mask_t)[0]
            for env_idx in reset_envs.tolist():
                self._zero_env_slots(int(env_idx))

        seg_to_entity_and_link, entity_names = self._build_seg_to_entity_idx()
        seg_to_entity_idx = {
            seg_idxc: entity_idx
            for seg_idxc, (entity_idx, _link_idx) in seg_to_entity_and_link.items()
        }

        # Build batched camera_obs with normalized seg/position from _get_camera_maps.
        raw_obs = self.camera_node.get_observation()
        camera_obs: Dict[str, torch.Tensor] = {"seg": seg, "position": position}
        for key in ("rgb", "depth", "normal", "intrinsics_cv", "extrinsics_cv"):
            if key in raw_obs:
                val = raw_obs[key]
                if val.ndim > 1 and val.shape[0] == seg.shape[0]:
                    camera_obs[key] = val
                else:
                    camera_obs[key] = val.unsqueeze(0).expand(seg.shape[0], *val.shape)

        # Build per-camera-batch active mask and pre-filter camera_obs.
        active_mask = (camera_env_indices >= 0) & (camera_env_indices < self._n_envs)
        if mask_t is not None:
            for batch_i in range(seg.shape[0]):
                ei = int(camera_env_indices[batch_i].item())
                if 0 <= ei < self._n_envs:
                    active_mask[batch_i] = active_mask[batch_i] & mask_t[ei]

        active_indices = torch.where(active_mask)[0]
        if active_indices.numel() == 0:
            self._refresh_observation()
            return

        # Slice camera_obs to only active entries (B' = active_indices.numel()).
        filtered_obs = {k: v[active_indices] for k, v in camera_obs.items()}

        points_world, seg_ids, proposer_valid = self.point_proposer.propose(
            camera_obs=filtered_obs,
            seg_to_entity_idx=seg_to_entity_idx,
            entity_names=entity_names,
            max_points=self.max_points,
            generator=self._rng,
        )

        # Map B' results back to env indices.
        for i, batch_i in enumerate(active_indices.tolist()):
            env_idx = int(camera_env_indices[batch_i].item())
            self._populate_slots(
                env_idx=env_idx,
                points_world=points_world[i],
                seg_ids=seg_ids[i],
                valid=proposer_valid[i],
                seg_to_entity_and_link=seg_to_entity_and_link,
            )

        self._refresh_observation()

    def _update_points_from_entity_pose(self) -> None:
        self._prev_points_world.copy_(self._points_world)

        if not torch.any(self._valid_mask):
            self._flow_world.zero_()
            self._refresh_observation()
            return

        for link_idx in torch.unique(self._link_idx[self._valid_mask]):
            link_idx_i = int(link_idx.item())
            link = self._link_by_idx.get(link_idx_i, None)
            if link is None:
                continue

            mask = self._valid_mask & (self._link_idx == link_idx)
            env_ids, point_ids = torch.nonzero(mask, as_tuple=True)
            if env_ids.numel() == 0:
                continue

            # get_pos()/get_quat() returns (D,) unbatched or (B, D) batched.
            link_pos = link.get_pos().float()
            link_quat = link.get_quat().float()

            if link_pos.ndim == 1:
                pos = link_pos.unsqueeze(0).expand(env_ids.numel(), -1)
                quat = link_quat.unsqueeze(0).expand(env_ids.numel(), -1)
            else:
                pos = link_pos[env_ids]
                quat = link_quat[env_ids]

            local = self._local_offsets[env_ids, point_ids]
            self._points_world[env_ids, point_ids] = pos + quaternion_apply(quat, local)

        self._flow_world.zero_()
        self._flow_world[self._valid_mask] = (
            self._points_world[self._valid_mask] - self._prev_points_world[self._valid_mask]
        )
        self._update_visibility()
        self._refresh_observation()

    def _update_visibility(self) -> None:
        """Project points to 2D and update visibility masks.

        Uses :func:`world_to_pixel_coordinate_and_depth` to project 3D points
        into pixel space.  A point is visible when its nearest-integer pixel
        falls inside the image and the segmentation at that pixel matches the
        point's cached segmentation ID.
        """
        self._points_2d.zero_()
        self._visibility_mask.zero_()
        if not torch.any(self._valid_mask):
            return

        cam_obs = self.camera_node.get_observation()
        if cam_obs is None or "seg" not in cam_obs:
            return

        seg = cam_obs["seg"]
        intrinsics = cam_obs["intrinsics_cv"]
        extrinsics = cam_obs["extrinsics_cv"]

        if seg.ndim == 2:
            seg = seg.unsqueeze(0)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if extrinsics.ndim == 2:
            extrinsics = extrinsics.unsqueeze(0)

        H, W = seg.shape[-2], seg.shape[-1]
        camera_env_indices = self._camera_env_indices(seg.shape[0])

        # Filter to valid env indices.
        valid_cam = (camera_env_indices >= 0) & (camera_env_indices < self._n_envs)
        cam_idx = torch.arange(seg.shape[0], device=self._device)[valid_cam]
        env_idx = camera_env_indices[valid_cam]
        if env_idx.numel() == 0:
            return

        # Gather points and project all envs at once: (B, max_points, 3).
        pts = self._points_world[env_idx]
        pixel_xy, depth = world_to_pixel_coordinate_and_depth(
            pts, intrinsics[cam_idx], extrinsics[cam_idx],
        )  # pixel_xy: (B, max_points, 2), depth: (B, max_points)
        self._points_2d[env_idx] = pixel_xy

        # Visibility: valid, in front of camera, within image bounds.
        valid = self._valid_mask[env_idx]  # (B, max_points)
        u, v = pixel_xy[..., 0], pixel_xy[..., 1]
        in_bounds = valid & (depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # Seg lookup via gather on flattened spatial dims.
        u_px = u.clamp(0, W - 1).round().long()
        v_px = v.clamp(0, H - 1).round().long()
        seg_batch = seg[cam_idx]  # (B, H, W)
        pixel_flat = v_px * W + u_px  # (B, max_points)
        seg_at_pixel = seg_batch.reshape(seg_batch.shape[0], -1).gather(1, pixel_flat)

        expected_seg = self._seg_idxc[env_idx]  # (B, max_points)
        self._visibility_mask[env_idx] = in_bounds & (seg_at_pixel == expected_seg)

    def _refresh_observation(self) -> None:
        obs: Dict[str, PyTorchArrayType] = {
            "point_positions": self._points_world,
            "points_2d": self._points_2d,
            "point_movements": self._flow_world,
            "point_segmentations": self._seg_idxc,
            "valid_masks": self._valid_mask.to(torch.uint8),
            "visibility_masks": self._visibility_mask.to(torch.uint8),
        }
        if self.world.batch_size is None:
            obs = {k: v[0] for k, v in obs.items()}
        self._last_observation = obs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs) -> None:
        del mask, kwargs
        if priority in self.reload_priorities and seed is not None:
            self._rng.manual_seed(seed)

    def reset(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs) -> None:
        del mask, kwargs
        if priority in self.reset_priorities and seed is not None:
            self._rng.manual_seed(seed)

    def after_reload(self, *, priority: int = 0, mask: Optional[PyTorchArrayType] = None) -> None:
        if priority in self.after_reload_priorities:
            self._init_tracks(mask=mask)

    def after_reset(self, *, priority: int = 0, mask: Optional[PyTorchArrayType] = None) -> None:
        if priority in self.after_reset_priorities:
            self._init_tracks(mask=mask)

    def post_environment_step(self, dt, *, priority: int = 0) -> None:
        del dt
        if priority in self.post_environment_step_priorities:
            self._update_points_from_entity_pose()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> Dict[str, PyTorchArrayType]:
        return self._last_observation
