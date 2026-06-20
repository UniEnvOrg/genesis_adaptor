"""Data types for keypoint tracking.

This module contains the pure data structures used by the keypoint tracker:
``SegmentationTarget``, ``SegmentationIndex``, ``KeypointTrackState``, and
``CameraFrame``.  No Genesis-specific logic or camera access lives here —
those concerns are handled by :mod:`camera_port` and
:mod:`keypoint_tracker_node`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from genesis.engine.entities.rigid_entity import RigidLink

__all__ = [
    "SegmentationTarget",
    "SegmentationIndex",
    "KeypointTrackState",
    "CameraFrame",
]


@dataclass(frozen=True)
class SegmentationTarget:
    """Internal representation of a segmentation ID's target.

    Maps a Genesis segmentation ID to the entity, link, and (optionally) geom
    it represents. This provides an explicit internal model instead of parsing
    the raw segmentation metadata implicitly.

    Attributes:
        seg_id: The Genesis segmentation ID (as it appears in the seg map).
        entity_idx: Index of the rigid entity.
        link_idx: Index of the rigid link (always valid for link/geom modes).
        geom_idx: Index of the geom within the link. ``-1`` for link mode,
            or a valid geom index for geom mode.
    """
    seg_id: int
    entity_idx: int
    link_idx: int
    geom_idx: int = -1

    @property
    def is_geom_mode(self) -> bool:
        """Return True if this target represents a geom-level segmentation."""
        return self.geom_idx >= 0


@dataclass(frozen=True)
class SegmentationIndex:
    """Immutable index mapping Genesis segmentation IDs to rigid entities and links.

    Built once at reload time (via
    :meth:`GenesisCameraKeypointPort.build_segmentation_index`) and reused
    across resets, since scene structure does not change between resets.
    This separates the static scene/camera metadata from the mutable
    per-environment track state.
    """

    seg_to_entity_link: Mapping[int, Tuple[int, int]]
    seg_to_entity_idx: Mapping[int, int]
    entity_names: Mapping[int, str]
    seg_target_by_id: Mapping[int, SegmentationTarget]
    link_by_idx: Mapping[int, RigidLink]


@dataclass
class KeypointTrackState:
    """Mutable tensor buffers for tracked keypoints across batched environments.

    Consolidates the parallel per-environment tensors that were previously
    held as individual attributes on :class:`KeypointTrackerNode`.  All
    tensors share the same ``(n_envs, max_points, ...)`` leading shape and
    must stay shape/dtype/lifecycle-consistent.
    """

    points_world: torch.Tensor
    prev_points_world: torch.Tensor
    flow_world: torch.Tensor
    local_offsets: torch.Tensor
    valid_mask: torch.Tensor
    entity_idx: torch.Tensor
    link_idx: torch.Tensor
    seg_ids: torch.Tensor
    geom_idx: torch.Tensor
    points_2d: torch.Tensor
    visibility_mask: torch.Tensor
    valid_mask_u8: torch.Tensor
    visibility_mask_u8: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        n_envs: int,
        max_points: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "KeypointTrackState":
        """Allocate zero-initialised track state buffers."""
        return cls(
            points_world=torch.zeros((n_envs, max_points, 3), dtype=dtype, device=device),
            prev_points_world=torch.zeros((n_envs, max_points, 3), dtype=dtype, device=device),
            flow_world=torch.zeros((n_envs, max_points, 3), dtype=dtype, device=device),
            local_offsets=torch.zeros((n_envs, max_points, 3), dtype=dtype, device=device),
            valid_mask=torch.zeros((n_envs, max_points), dtype=torch.bool, device=device),
            entity_idx=torch.full((n_envs, max_points), -1, dtype=torch.int32, device=device),
            link_idx=torch.full((n_envs, max_points), -1, dtype=torch.int32, device=device),
            seg_ids=torch.full((n_envs, max_points), -1, dtype=torch.int32, device=device),
            geom_idx=torch.full((n_envs, max_points), -1, dtype=torch.int32, device=device),
            points_2d=torch.zeros((n_envs, max_points, 2), dtype=dtype, device=device),
            visibility_mask=torch.zeros((n_envs, max_points), dtype=torch.bool, device=device),
            valid_mask_u8=torch.zeros((n_envs, max_points), dtype=torch.uint8, device=device),
            visibility_mask_u8=torch.zeros((n_envs, max_points), dtype=torch.uint8, device=device),
        )

    def zero_env(self, env_idx: int) -> None:
        """Zero all buffers for a single environment index."""
        self.points_world[env_idx].zero_()
        self.prev_points_world[env_idx].zero_()
        self.flow_world[env_idx].zero_()
        self.local_offsets[env_idx].zero_()
        self.valid_mask[env_idx].zero_()
        self.entity_idx[env_idx].fill_(-1)
        self.link_idx[env_idx].fill_(-1)
        self.seg_ids[env_idx].fill_(-1)
        self.geom_idx[env_idx].fill_(-1)
        self.points_2d[env_idx].zero_()
        self.visibility_mask[env_idx].zero_()
        self.valid_mask_u8[env_idx].zero_()
        self.visibility_mask_u8[env_idx].zero_()

    def as_observation(self, *, unbatched: bool) -> Dict[str, torch.Tensor]:
        """Build the observation dictionary from the current state.

        .. note::
            The returned tensors are **borrowed views** of the internal
            buffers.  Mutating them will corrupt the tracker's state.
            Callers that need a defensive copy should ``clone()`` the
            tensors they intend to modify.
        """
        self.valid_mask_u8.copy_(self.valid_mask)
        self.visibility_mask_u8.copy_(self.visibility_mask)
        obs: Dict[str, torch.Tensor] = {
            "point_positions": self.points_world,
            "points_2d": self.points_2d,
            "point_movements": self.flow_world,
            "point_segmentations": self.seg_ids,
            "valid_masks": self.valid_mask_u8,
            "visibility_masks": self.visibility_mask_u8,
        }
        if unbatched:
            obs = {k: v[0] for k, v in obs.items()}
        return obs


@dataclass(frozen=True)
class CameraFrame:
    """Normalized camera observation with guaranteed batch dimension.

    All tensor fields have a leading batch dimension ``(B, ...)``.  Unbatched
    observations are automatically unsqueezed by the port that constructs this
    frame.  ``position`` is always populated when either ``position`` or
    ``depth`` is available from the camera (depth is converted via
    ``depth_to_position_map``).
    """

    seg: torch.Tensor                  # (B, H, W)
    intrinsics_cv: torch.Tensor        # (B, 3, 3)
    extrinsics_cv: torch.Tensor        # (B, 3, 4) or (B, 4, 4)
    position: Optional[torch.Tensor] = None   # (B, H, W, 3)
    depth: Optional[torch.Tensor] = None      # (B, H, W)
    rgb: Optional[torch.Tensor] = None        # (B, H, W, 3)
    normal: Optional[torch.Tensor] = None     # (B, H, W, 3)

    @property
    def batch_size(self) -> int:
        """Number of environments in the batch dimension."""
        return self.seg.shape[0]

    @property
    def height(self) -> int:
        """Image height."""
        return self.seg.shape[-2]

    @property
    def width(self) -> int:
        """Image width."""
        return self.seg.shape[-1]

    def as_proposer_dict(self) -> Dict[str, torch.Tensor]:
        """Return as a dict suitable for :meth:`PointProposer.propose`.

        Only non-``None`` optional fields are included.  ``seg``,
        ``intrinsics_cv``, and ``extrinsics_cv`` are always present.
        """
        d: Dict[str, torch.Tensor] = {
            "seg": self.seg,
            "intrinsics_cv": self.intrinsics_cv,
            "extrinsics_cv": self.extrinsics_cv,
        }
        if self.position is not None:
            d["position"] = self.position
        if self.depth is not None:
            d["depth"] = self.depth
        if self.rgb is not None:
            d["rgb"] = self.rgb
        if self.normal is not None:
            d["normal"] = self.normal
        return d
