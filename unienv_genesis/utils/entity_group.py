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

def _as_tensor(v, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Coerce *v* to a :class:`torch.Tensor`.

    - If *v* is already a tensor:
        - with an explicit *dtype*, cast to it;
        - otherwise, if *v* is floating-point, **preserve its dtype** (so
          float64 worlds are not silently downcast to float32);
        - otherwise (non-floating input tensor), cast to *dtype* or
          ``torch.float32`` as a safe float fallback.
    - If *v* is a plain Python sequence / scalar, build a tensor with the
      explicit *dtype* (or ``torch.float32`` when *dtype* is ``None``).
    """
    if isinstance(v, torch.Tensor):
        if dtype is not None:
            return v.to(dtype=dtype)
        if v.dtype.is_floating_point:
            return v
        return v.to(dtype=torch.float32)
    return torch.tensor(v, dtype=dtype if dtype is not None else torch.float32)


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
        dtype: Optional floating-point dtype for internally stored pose
            tensors.  When ``None``, the dtype is inferred from the inputs
            (preserving a floating tensor's dtype, or falling back to
            ``torch.float32`` for plain Python sequences).  Pass
            ``world.dtype`` to keep group poses consistent with the world.
    """

    def __init__(
        self,
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._dtype = dtype
        self._pos: torch.Tensor = _as_tensor(pos, dtype=dtype)
        self._quat: torch.Tensor = _as_tensor(quat, dtype=dtype)
        # Enforce a single canonical anchor device/dtype from the start
        # (the position component defines it when inputs disagree).
        self._set_anchor()
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
            _as_tensor(local_pos, dtype=self._dtype),
            _as_tensor(local_quat, dtype=self._dtype),
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
            ``(world_pos, world_quat)`` as floating-point tensors.  The
            dtype matches the group's ``dtype`` (if set), the dtype of any
            floating tensor input, or ``torch.float32`` otherwise.
        """
        return self._world_pose(
            _as_tensor(local_pos, dtype=self._dtype),
            _as_tensor(local_quat, dtype=self._dtype),
        )

    # ------------------------------------------------------------------
    # Pose manipulation
    # ------------------------------------------------------------------

    def _set_anchor(
        self,
        pos: Union[Sequence[float], torch.Tensor, None] = None,
        quat: Union[Sequence[float], torch.Tensor, None] = None,
    ) -> None:
        """Update anchor components under one canonical device/dtype.

        The anchor (``_pos`` + ``_quat``) must always live on a single
        device/dtype pair.  A floating **tensor** input makes its
        device/dtype canonical and the *other* anchor component is migrated
        alongside (``pos`` takes precedence if both are tensors); non-tensor
        inputs adopt the current canonical device/dtype.  This guarantees
        that e.g. ``set_pos(cuda_tensor)`` followed by ``set_quat(...)`` on
        a CPU-initialised group can never leave the anchor internally
        mixed-device.  ``Tensor.to`` is a no-op when already aligned.
        """
        pos_t = _as_tensor(pos, dtype=self._dtype) if pos is not None else None
        quat_t = _as_tensor(quat, dtype=self._dtype) if quat is not None else None

        device, dtype = self._pos.device, self._pos.dtype
        for raw, t in ((pos, pos_t), (quat, quat_t)):
            if t is not None and isinstance(raw, torch.Tensor):
                device, dtype = t.device, t.dtype
                break

        self._pos = (pos_t if pos_t is not None else self._pos).to(device=device, dtype=dtype)
        self._quat = (quat_t if quat_t is not None else self._quat).to(device=device, dtype=dtype)

    def _world_pose(
        self,
        local_pos: torch.Tensor,
        local_quat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute world pose from anchor pose and member local pose.

        Handles both unbatched anchors ``(3,)``/``(4,)`` and batched anchors
        ``(n_envs, 3)``/``(n_envs, 4)``.  The member's local tensors are always
        ``(3,)``/``(4,)`` and are expanded along the batch dimension as needed.

        Member local tensors are stored device-neutrally (they are typically
        created from plain Python sequences at build time); they are aligned to
        the anchor's device/dtype here so CUDA-anchored groups stay
        device-consistent.  ``Tensor.to`` is a no-op when already aligned.
        """
        local_pos = local_pos.to(device=self._pos.device, dtype=self._pos.dtype)
        local_quat = local_quat.to(device=self._quat.device, dtype=self._quat.dtype)
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
        self._set_anchor(pos=pos, quat=quat)
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
        self._set_anchor(pos=pos)
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
        self._set_anchor(quat=quat)
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
