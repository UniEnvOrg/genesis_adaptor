"""Rigid pose port for keypoint tracking.

This module provides batched access to rigid link poses (position +
quaternion) for all tracked links in a single call, eliminating the
per-link Python loop and GPU↔CPU synchronization that the previous
``_update_points_from_entity_pose`` required.

The :class:`GenesisRigidPosePort` precomputes a lookup table (LUT) at
construction time that maps Genesis link indices to compact local indices
(0..N-1).  At each step, ``get_tracked_link_poses()`` queries all tracked
links in a fixed-size Python loop over a **CPU** list (no GPU sync) and
stacks the results into ``(B, n_tracked, 3)`` / ``(B, n_tracked, 4)``
tensors, ready for a single vectorized gather + quaternion apply.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import torch
from genesis.engine.entities.rigid_entity import RigidLink

from unienv_genesis.world.genesis_world import GenesisWorld

__all__ = [
    "RigidPosePort",
    "GenesisRigidPosePort",
]


class RigidPosePort:
    """Interface for batched rigid link pose access.

    Implementations provide:

    - ``n_tracked_links``: the number of tracked links.
    - ``link_local_idx_lut``: a tensor mapping Genesis link indices to
      compact local indices (0..N-1), with ``-1`` for untracked links.
    - ``get_tracked_link_poses()``: returns batched ``(B, N, 3)`` and
      ``(B, N, 4)`` tensors of link positions and quaternions.

    The LUT is on the world's device so it can be indexed directly by
    GPU tensors (e.g. ``state.link_idx``).
    """

    @property
    def n_tracked_links(self) -> int:
        """Number of tracked links."""
        ...

    @property
    def link_local_idx_lut(self) -> torch.Tensor:
        """Tensor of shape ``(max_link_idx + 1,)`` mapping link_idx → local index.

        Untracked indices are ``-1``.  The tensor is on the world's device.
        """
        ...

    def get_tracked_link_poses(
        self, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return batched link poses for all tracked links.

        Args:
            dtype: Desired floating-point dtype for the returned tensors.

        Returns:
            ``(link_pos, link_quat)`` where ``link_pos`` is
            ``(B, n_tracked, 3)`` and ``link_quat`` is
            ``(B, n_tracked, 4)`` (w, x, y, z).  For an unbatched world,
            ``B == 1``.
        """
        ...


class GenesisRigidPosePort(RigidPosePort):
    """Concrete pose port backed by Genesis ``RigidLink`` objects.

    All per-link ``get_pos()`` / ``get_quat()`` access is confined to
    :meth:`get_tracked_link_poses`.  The tracked-link list is precomputed
    at construction time as a CPU tuple, so the per-step query loop never
    touches the GPU (no ``torch.unique``, no ``.item()`` synchronization).
    """

    def __init__(
        self,
        link_by_idx: Mapping[int, RigidLink],
        world: GenesisWorld,
    ):
        self._link_by_idx = link_by_idx
        self._world = world
        self._n_envs = world.batch_size if world.batch_size is not None else 1

        # Precompute tracked link IDs as a CPU tuple (sorted for determinism).
        self._tracked_link_ids: Tuple[int, ...] = tuple(sorted(link_by_idx.keys()))
        self._n_tracked = len(self._tracked_link_ids)

        # Build LUT: Genesis link_idx → local index (0..N-1), -1 for untracked.
        # Placed on the world's device so it can be indexed by GPU tensors.
        if self._n_tracked > 0:
            max_link_idx = max(self._tracked_link_ids)
            lut = torch.full(
                (max_link_idx + 1,), -1, dtype=torch.int64, device=world.device
            )
            for local_idx, link_idx in enumerate(self._tracked_link_ids):
                lut[link_idx] = local_idx
            self._lut = lut
        else:
            self._lut = torch.empty(0, dtype=torch.int64, device=world.device)

    @property
    def n_tracked_links(self) -> int:
        return self._n_tracked

    @property
    def link_local_idx_lut(self) -> torch.Tensor:
        return self._lut

    def get_tracked_link_poses(
        self, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query all tracked links' poses in one fixed-size CPU loop.

        The loop iterates over ``self._tracked_link_ids`` (a CPU tuple),
        so no GPU↔CPU synchronization occurs.  Each ``link.get_pos()``
        call returns a GPU tensor directly; the results are stacked and
        transposed into ``(B, n_tracked, 3/4)``.
        """
        n = self._n_tracked
        device = self._world.device
        B = self._n_envs

        if n == 0:
            return (
                torch.zeros((B, 0, 3), dtype=dtype, device=device),
                torch.zeros((B, 0, 4), dtype=dtype, device=device),
            )

        poses_pos: list[torch.Tensor] = []
        poses_quat: list[torch.Tensor] = []
        for link_idx in self._tracked_link_ids:
            link = self._link_by_idx[link_idx]
            pos = link.get_pos().to(dtype=dtype)    # (3,) or (B, 3)
            quat = link.get_quat().to(dtype=dtype)  # (4,) or (B, 4)
            if pos.ndim == 1:
                pos = pos.unsqueeze(0).expand(B, -1)
                quat = quat.unsqueeze(0).expand(B, -1)
            poses_pos.append(pos)   # (B, 3)
            poses_quat.append(quat)  # (B, 4)

        # Stack: (n, B, 3) → transpose → (B, n, 3)
        link_pos = torch.stack(poses_pos).transpose(0, 1).contiguous()
        link_quat = torch.stack(poses_quat).transpose(0, 1).contiguous()
        return link_pos, link_quat
