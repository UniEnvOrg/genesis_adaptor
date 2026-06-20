"""Keypoint tracker sensor node for Genesis.

This sensor delegates image acquisition to an existing :class:`CameraSensorNode`
that provides segmentation and per-pixel 3D positions. At reset/reload time it:

1. Proposes keypoints from the camera image via a :class:`PointProposer`.
2. Maps each keypoint to a rigid entity using camera segmentation metadata.
3. Stores each keypoint's local offset in the entity frame.

After each environment step, keypoints are updated from entity poses only
(ground-truth tracking), and per-step 3D flow is produced.

**Rendering assumption**: The tracker requires that *all* environments are
rendered by the delegated camera on every frame, in order.  The camera's
batch dimension is treated as the environment index directly — no remapping
of rendered-env indices is performed.

The node is intentionally thin — it delegates camera/Genesis access to
:class:`GenesisCameraKeypointPort`, track state to
:class:`KeypointTrackState`, and visibility math to :mod:`visibility`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
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

from ..base import SensorNode
from ..camera_node import CameraSensorNode
from .types import KeypointTrackState, SegmentationIndex
from .proposers import PointProposer, UniformSegmentationPointProposer
from .camera_port import GenesisCameraKeypointPort
from .pose_port import GenesisRigidPosePort
from .visibility import update_visibility

__all__ = ["KeypointTrackerNode"]


class KeypointTrackerNode(SensorNode[None, None, Dict[str, PyTorchArrayType]]):
    """Keypoint tracking sensor.

    Notes
    -----
    - Tracks keypoints on all visible ``RigidEntity`` objects, including
      multi-link articulated bodies.  Entity name filtering is handled by the
      :class:`PointProposer` passed at construction time.
    - The delegated camera must provide segmentation (``render_seg=True``)
      and either depth or position rendering.
    - **All environments must be rendered** by the delegated camera on every
      frame, in order.  The camera batch dimension is treated as the
      environment index directly (no rendered-env remapping).
    - Visibility is determined by a **strict z-buffer depth test**: a tracked
      point is visible when its projected pixel is in-bounds and the rendered
      depth at that pixel matches the point's projected depth within
      ``depth_tolerance``.  Segmentation is used only at initialisation (point
      proposal and entity/link mapping), not on the per-step hot path.
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
        depth_tolerance: float = 0.05,
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

        # The port quarantines all Genesis-specific camera access.
        self._port = GenesisCameraKeypointPort(camera_node, world)
        self._port.validate()
        self._segmentation_level = self._port.validate_segmentation_level()

        # Keep camera_node for backward compatibility.
        self.camera_node = camera_node
        self.max_points = int(max_points)
        self.point_proposer = point_proposer or UniformSegmentationPointProposer()
        self.depth_tolerance = float(depth_tolerance)

        self._n_envs = self.world.batch_size if self.world.batch_size is not None else 1
        self._rng = torch.Generator(device=self.world.device)

        self._state = KeypointTrackState.allocate(
            n_envs=self._n_envs,
            max_points=self.max_points,
            dtype=self.world.dtype,
            device=self.world.device,
        )
        self._seg_index: Optional[SegmentationIndex] = None
        self._pose_port: Optional[GenesisRigidPosePort] = None

        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        self.observation_space = self._get_observation_space()

    # ------------------------------------------------------------------
    # Observation space
    # ------------------------------------------------------------------

    def _get_observation_space(self) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        B = self.world.batch_size
        spaces = {
            "point_positions": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=get_batch_aware_shape(B, (self.max_points, 3)),
                dtype=self.world.dtype,
                device=self.world.device,
            ),
            "points_2d": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=get_batch_aware_shape(B, (self.max_points, 2)),
                dtype=self.world.dtype,
                device=self.world.device,
            ),
            "point_movements": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=get_batch_aware_shape(B, (self.max_points, 3)),
                dtype=self.world.dtype,
                device=self.world.device,
            ),
            "point_segmentations": BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=2**31 - 1,
                shape=get_batch_aware_shape(B, (self.max_points,)),
                dtype=torch.int32,
                device=self.world.device,
            ),
            "valid_masks": BoxSpace(
                PyTorchComputeBackend,
                low=0,
                high=1,
                shape=get_batch_aware_shape(B, (self.max_points,)),
                dtype=torch.uint8,
                device=self.world.device,
            ),
            "visibility_masks": BoxSpace(
                PyTorchComputeBackend,
                low=0,
                high=1,
                shape=get_batch_aware_shape(B, (self.max_points,)),
                dtype=torch.uint8,
                device=self.world.device,
            ),
        }
        return DictSpace(PyTorchComputeBackend, spaces, device=self.world.device)

    # ------------------------------------------------------------------
    # Track initialisation
    # ------------------------------------------------------------------

    def _populate_slots(
        self,
        *,
        env_idx: int,
        points_world: torch.Tensor,
        seg_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Fill tracking slots for one environment from proposer output.

        Args:
            env_idx: Environment index.
            points_world: ``(max_points, 3)`` padded world positions.
            seg_ids: ``(max_points,)`` padded segmentation IDs.
            valid: ``(max_points,)`` bool mask from the proposer.
        """
        if not torch.any(valid):
            return

        state = self._state
        seg_index = self._seg_index

        pts = points_world[valid].to(dtype=self.world.dtype)
        sids = seg_ids[valid].long()
        n = pts.shape[0]

        entity_indices = torch.full((n,), -1, dtype=torch.int64, device=self.world.device)
        link_indices = torch.full((n,), -1, dtype=torch.int64, device=self.world.device)
        geom_indices = torch.full((n,), -1, dtype=torch.int64, device=self.world.device)
        for i in range(n):
            seg_id_i = int(sids[i].item())
            assign = seg_index.seg_to_entity_link.get(seg_id_i, None)
            if assign is None:
                continue
            entity_indices[i] = int(assign[0])
            link_indices[i] = int(assign[1])
            # Look up geom_idx from the target if available
            target = seg_index.seg_target_by_id.get(seg_id_i, None)
            if target is not None:
                geom_indices[i] = target.geom_idx

        resolved = (entity_indices >= 0) & (link_indices >= 0)
        if not torch.any(resolved):
            return

        sids = sids[resolved]
        pts = pts[resolved]
        entity_indices = entity_indices[resolved]
        link_indices = link_indices[resolved]
        geom_indices = geom_indices[resolved]
        n_valid = entity_indices.shape[0]

        local_offsets = torch.zeros((n_valid, 3), dtype=self.world.dtype, device=self.world.device)
        for link_idx in torch.unique(link_indices):
            link_idx_i = int(link_idx.item())
            link = seg_index.link_by_idx.get(link_idx_i, None)
            if link is None:
                continue

            env_query = env_idx if self.world.batch_size is not None else None
            link_pos = link.get_pos(env_query).to(dtype=self.world.dtype)
            link_quat = link.get_quat(env_query).to(dtype=self.world.dtype)
            if link_pos.ndim == 2:
                link_pos = link_pos[0]
            if link_quat.ndim == 2:
                link_quat = link_quat[0]

            lm = link_indices == link_idx
            delta = pts[lm] - link_pos
            q_inv = quaternion_invert(link_quat.unsqueeze(0)).expand(delta.shape[0], -1)
            local_offsets[lm] = quaternion_apply(q_inv, delta)

        state.valid_mask[env_idx, :n_valid] = True
        state.entity_idx[env_idx, :n_valid] = entity_indices.to(torch.int32)
        state.link_idx[env_idx, :n_valid] = link_indices.to(torch.int32)
        state.seg_ids[env_idx, :n_valid] = sids.to(torch.int32)
        state.geom_idx[env_idx, :n_valid] = geom_indices.to(torch.int32)
        state.local_offsets[env_idx, :n_valid] = local_offsets
        state.points_world[env_idx, :n_valid] = pts
        state.prev_points_world[env_idx, :n_valid] = pts
        state.flow_world[env_idx, :n_valid].zero_()

    def _init_tracks(self, mask: Optional[PyTorchArrayType] = None) -> None:
        """Initialise (or re-initialise) keypoint tracks from camera segmentation.

        Uses the :class:`SegmentationIndex` built during ``after_reload`` to
        map segmentation IDs to entities/links.  This method only re-proposes
        keypoints and populates track slots — it does not rebuild the
        segmentation index.

        Args:
            mask: Optional ``(B,)`` bool tensor indicating which environments
                should be (re)initialised.  When ``None``, all environments
                are initialised.
        """
        if self._seg_index is None:
            raise RuntimeError(
                f"KeypointTrackerNode '{self.name}': segmentation index not "
                f"built.  Ensure after_reload has been called before after_reset."
            )
        seg_index = self._seg_index

        frame = self._port.current_frame()
        if frame.position is None:
            raise RuntimeError(
                f"KeypointTrackerNode '{self.name}': camera node "
                f"'{self.camera_node.name}' did not provide a position map "
                f"(neither 'position' nor 'depth' in its observation). "
                f"Track initialisation requires per-pixel 3D positions. "
                f"Enable render_position or render_depth on the camera."
            )
        B = frame.batch_size

        # Determine which envs to (re)initialise.
        if mask is not None and self.world.batch_size is not None:
            reset_envs = torch.where(mask)[0]
        else:
            reset_envs = torch.arange(B, dtype=torch.long, device=self.world.device)

        for env_idx in reset_envs.tolist():
            self._state.zero_env(int(env_idx))

        if reset_envs.numel() == 0:
            self._update_visibility()
            self._last_observation = self._state.as_observation(
                unbatched=self.world.batch_size is None,
            )
            return

        # Build proposer dict and slice to only the envs being (re)initialised.
        proposer_obs = frame.as_proposer_dict()
        filtered_obs = {k: v[reset_envs] for k, v in proposer_obs.items()}

        points_world, seg_ids, proposer_valid = self.point_proposer.propose(
            camera_obs=filtered_obs,
            seg_to_entity_idx=seg_index.seg_to_entity_idx,
            entity_names=seg_index.entity_names,
            max_points=self.max_points,
            generator=self._rng,
        )

        # Write proposer results back to the corresponding env slots.
        # Batch index i in proposer output corresponds to reset_envs[i].
        for i, env_idx in enumerate(reset_envs.tolist()):
            self._populate_slots(
                env_idx=int(env_idx),
                points_world=points_world[i],
                seg_ids=seg_ids[i],
                valid=proposer_valid[i],
            )

        # Compute points_2d and proper visibility_masks before publishing
        # the first observation, so consumers see valid projections.
        self._update_visibility()
        self._last_observation = self._state.as_observation(
            unbatched=self.world.batch_size is None,
        )

    # ------------------------------------------------------------------
    # Per-step tracking
    # ------------------------------------------------------------------

    def _update_points_from_entity_pose(self) -> None:
        """Recompute world positions from entity poses and update flow.

        Fully vectorised: queries all tracked link poses in one fixed-size
        CPU loop (no GPU sync), then gathers the appropriate pose for each
        point and applies the local offset in a single batched quaternion
        operation.  No per-link Python loop, no ``.item()`` calls.
        """
        state = self._state
        pose_port = self._pose_port

        state.prev_points_world.copy_(state.points_world)

        # No early-exit on torch.any(valid_mask) — that would force a GPU
        # sync every step.  The vectorised ops below handle the all-invalid
        # case naturally (torch.where keeps old values).

        # Query all tracked link poses: (B, n_tracked, 3) and (B, n_tracked, 4).
        link_pos, link_quat = pose_port.get_tracked_link_poses(self.world.dtype)

        # Map state.link_idx (B, P) → local tracked-link index via LUT.
        B, P = state.link_idx.shape
        safe_link_idx = state.link_idx.clamp_min(0).long()
        local_link_idx = pose_port.link_local_idx_lut[safe_link_idx]  # (B, P)

        # Gather pose for each point: (B, P, 3) and (B, P, 4).
        env_idx = torch.arange(B, device=state.points_world.device)[:, None].expand(B, P)
        safe_local = local_link_idx.clamp_min(0)
        pos_per_point = link_pos[env_idx, safe_local]     # (B, P, 3)
        quat_per_point = link_quat[env_idx, safe_local]    # (B, P, 4)

        # Apply local offset: world = pos + quat * local_offset.
        local = state.local_offsets  # (B, P, 3)
        new_points = pos_per_point + quaternion_apply(
            quat_per_point.reshape(-1, 4), local.reshape(-1, 3),
        ).reshape(B, P, 3)

        # Only update valid points with a resolved local link index.
        valid = state.valid_mask & (local_link_idx >= 0)
        state.points_world.copy_(
            torch.where(valid[..., None], new_points, state.points_world)
        )

        # Compute flow for valid points.
        state.flow_world.copy_(
            torch.where(
                state.valid_mask[..., None],
                state.points_world - state.prev_points_world,
                torch.zeros_like(state.flow_world),
            )
        )

        self._update_visibility()
        self._last_observation = state.as_observation(
            unbatched=self.world.batch_size is None,
        )

    def _update_visibility(self) -> None:
        """Project points, run z-buffer test, update visibility mask."""
        frame = self._port.current_frame()
        update_visibility(self._state, frame, self.depth_tolerance)

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
            # Re-validate segmentation level (scene may have been rebuilt
            # with different vis_options during reload).
            self._segmentation_level = self._port.validate_segmentation_level()

            # Build the static segmentation index once per reload.
            self._seg_index = self._port.build_segmentation_index()
            # Build the pose port from the index's tracked links.
            self._pose_port = GenesisRigidPosePort(
                self._seg_index.link_by_idx, self.world,
            )
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
        if self._last_observation is None:
            raise RuntimeError(
                f"KeypointTrackerNode '{self.name}' has no observation yet. "
                f"Ensure the node has been initialised via after_reload/after_reset."
            )
        return self._last_observation
