"""Utility for treating a collection of Genesis entities as a single rigid group.

Genesis has no native multi-entity grouping API.  ``EntityGroup`` fills that gap
by recording each member's **local pose** relative to a common anchor point, then
recomputing world poses whenever the anchor is moved or rotated.

Typical usage
-------------
Build the group during ``reload`` when you add entities to the scene::

    from unienv_genesis.utils.entity_group import EntityGroup

    group = EntityGroup(pos=(0, 0, 0.8), quat=(1, 0, 0, 0))

    tabletop = scene.add_entity(gs.morphs.Box(size=(1, 0.6, 0.02), fixed=True))
    group.add_member(tabletop, local_pos=(0, 0, -0.01))

    leg = scene.add_entity(gs.morphs.Cylinder(radius=0.025, height=0.76, fixed=True))
    group.add_member(leg, local_pos=(0.45, 0.27, -0.39))

Then during ``reset`` move the whole group::

    group.set_pose(new_pos, new_quat)

Coordinate convention
---------------------
``pos`` and ``quat`` describe the **anchor** frame.  A member with
``local_pos=(0, 0, 0)`` and ``local_quat=(1, 0, 0, 0)`` sits exactly at the
anchor.  Local positions are expressed in the **unrotated** anchor frame; they
are rotated by ``quat`` when computing world positions.

Quaternion format: ``(w, x, y, z)`` throughout.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import torch
from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_multiply
)

# ---------------------------------------------------------------------------
# Tensor helper
# ---------------------------------------------------------------------------

def _as_tensor(v, dtype=torch.float32) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.to(dtype=dtype)
    return torch.tensor(v, dtype=dtype)


# ---------------------------------------------------------------------------
# EntityGroup
# ---------------------------------------------------------------------------

class EntityGroup:
    """A logical group of Genesis entities sharing a common anchor pose.

    The group stores each member's pose **relative to the anchor**.  Calling
    :meth:`set_pose` (or :meth:`set_pos` / :meth:`set_quat`) recomputes and
    applies world poses to every member entity.

    Batched environments
    --------------------
    All pose setters accept both **unbatched** tensors ``(3,)``/``(4,)`` and
    **batched** tensors ``(n_envs, 3)``/``(n_envs, 4)``.  Member local poses
    are always stored as ``(3,)``/``(4,)`` and are broadcast to the anchor's
    batch dimension at application time.

    When ``envs_idx`` is supplied to update a *subset* of environments and the
    anchor is batched, the pre-selected rows (``w_pos[envs_idx]``) are
    forwarded to ``entity.set_pos``/``set_quat``.

    Args:
        pos: Initial world position of the anchor ``(x, y, z)`` or
            ``(n_envs, 3)`` for batched initialisation.
        quat: Initial orientation of the anchor ``(w, x, y, z)`` or
            ``(n_envs, 4)``.  Defaults to identity.
    """

    def __init__(
        self,
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self._pos: torch.Tensor = _as_tensor(pos)
        self._quat: torch.Tensor = _as_tensor(quat)
        # list of (entity, local_pos (3,), local_quat (4,))
        self._members: list[tuple] = []

    # ------------------------------------------------------------------
    # Building the group
    # ------------------------------------------------------------------

    def add_member(
        self,
        entity,
        local_pos: Sequence[float] = (0.0, 0.0, 0.0),
        local_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Register an entity as a member of this group.

        The entity's **current world pose** is NOT used; the caller must supply
        ``local_pos`` and ``local_quat`` explicitly in the anchor frame.

        Args:
            entity: A Genesis ``RigidEntity``.
            local_pos: Position relative to the anchor ``(x, y, z)``.
            local_quat: Orientation relative to the anchor ``(w, x, y, z)``.
        """
        self._members.append((
            entity,
            _as_tensor(local_pos),
            _as_tensor(local_quat),
        ))

    @property
    def entities(self):
        """List of member entities in insertion order."""
        return [m[0] for m in self._members]

    # ------------------------------------------------------------------
    # World-pose helpers (for use before entities are added to the scene)
    # ------------------------------------------------------------------

    def world_pose(
        self,
        local_pos: Union[Sequence[float], torch.Tensor] = (0.0, 0.0, 0.0),
        local_quat: Union[Sequence[float], torch.Tensor] = (1.0, 0.0, 0.0, 0.0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the world pose of a member at the given local offset.

        Useful for computing the correct morph ``pos``/``quat`` to pass to
        Genesis *before* adding the entity to the scene, so no post-build
        fixup is needed.

        Args:
            local_pos: Position relative to the anchor ``(x, y, z)``.
            local_quat: Orientation relative to the anchor ``(w, x, y, z)``.

        Returns:
            ``(world_pos, world_quat)`` as float32 tensors.
        """
        return self._world_pose(_as_tensor(local_pos), _as_tensor(local_quat))

    # ------------------------------------------------------------------
    # Pose manipulation
    # ------------------------------------------------------------------

    def _world_pose(
        self,
        local_pos: torch.Tensor,
        local_quat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute world pose from anchor pose and member local pose.

        Handles both unbatched anchors ``(3,)``/``(4,)`` and batched anchors
        ``(n_envs, 3)``/``(n_envs, 4)``.  The member's local tensors are always
        ``(3,)``/``(4,)`` and are expanded along the batch dimension as needed.
        """
        if self._quat.dim() > 1:
            # Batched anchor — broadcast local tensors to (n_envs, *).
            n_envs = self._quat.shape[0]
            lp = local_pos.unsqueeze(0).expand(n_envs, -1)
            lq = local_quat.unsqueeze(0).expand(n_envs, -1)
            world_pos = self._pos + quaternion_apply(self._quat, lp)
            world_quat = quaternion_multiply(self._quat, lq)
        else:
            world_pos = self._pos + quaternion_apply(self._quat, local_pos)
            world_quat = quaternion_multiply(self._quat, local_quat)
        return world_pos, world_quat

    def _apply_all(self, envs_idx=None) -> None:
        """Push current anchor pose to all member entities.

        When the anchor is batched ``(n_envs, 3)`` and ``envs_idx`` selects a
        subset of environments, only the rows for those indices are forwarded to
        each entity.
        """
        for entity, local_pos, local_quat in self._members:
            w_pos, w_quat = self._world_pose(local_pos, local_quat)
            if envs_idx is not None:
                if w_pos.dim() > 1:
                    # Batched poses: pass only the rows for envs_idx.
                    entity.set_pos(w_pos[envs_idx], envs_idx=envs_idx)
                    entity.set_quat(w_quat[envs_idx], envs_idx=envs_idx)
                else:
                    entity.set_pos(w_pos, envs_idx=envs_idx)
                    entity.set_quat(w_quat, envs_idx=envs_idx)
            else:
                entity.set_pos(w_pos)
                entity.set_quat(w_quat)

    def set_pose(
        self,
        pos: Union[Sequence[float], torch.Tensor],
        quat: Union[Sequence[float], torch.Tensor],
        envs_idx=None,
    ) -> None:
        """Move the group anchor to ``(pos, quat)`` and update all members.

        Args:
            pos: New anchor world position ``(x, y, z)`` or ``(n_envs, 3)``.
            quat: New anchor orientation ``(w, x, y, z)`` or ``(n_envs, 4)``.
            envs_idx: Optional environment indices for batched resets.
        """
        self._pos = _as_tensor(pos)
        self._quat = _as_tensor(quat)
        self._apply_all(envs_idx=envs_idx)

    def set_pos(
        self,
        pos: Union[Sequence[float], torch.Tensor],
        envs_idx=None,
    ) -> None:
        """Translate the group anchor (orientation unchanged).

        Args:
            pos: New anchor world position ``(x, y, z)``.
            envs_idx: Optional environment indices for batched resets.
        """
        self._pos = _as_tensor(pos)
        self._apply_all(envs_idx=envs_idx)

    def set_quat(
        self,
        quat: Union[Sequence[float], torch.Tensor],
        envs_idx=None,
    ) -> None:
        """Rotate the group anchor (position unchanged).

        Args:
            quat: New anchor orientation ``(w, x, y, z)``.
            envs_idx: Optional environment indices for batched resets.
        """
        self._quat = _as_tensor(quat)
        self._apply_all(envs_idx=envs_idx)

    # ------------------------------------------------------------------
    # Current anchor pose
    # ------------------------------------------------------------------

    @property
    def pos(self) -> torch.Tensor:
        """Current anchor world position ``(3,)``."""
        return self._pos

    @property
    def quat(self) -> torch.Tensor:
        """Current anchor orientation ``(4,)`` in ``(w, x, y, z)``."""
        return self._quat
