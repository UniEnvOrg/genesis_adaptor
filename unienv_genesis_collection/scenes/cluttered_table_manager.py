"""ClutteredTableManager — randomly places distractor objects on a table surface.

Follows the :class:`CachedTableBuilder` pattern:

* During ``reload()`` (priority 25): loads every object from the pool into the
  scene at the parking position ``(0, 0, -1000)``.
* During ``reset()`` (priority 25): selects a random subset, computes
  collision-free placements via rejection sampling, and positions them on the
  table surface.  Unused objects remain parked.

Priority 25 is within the ``[0, 50)`` objects / props range, so it runs
**after** the table (priority 100) and robot (priority 50) have been set up.

Supports batched environments: each environment gets an independent random
placement.  When a ``mask`` is supplied to ``reset()``, only the flagged
environments are re-randomised; the others keep their current layout.

Per-environment ``avoid_spaces`` can be passed through ``reset()`` as a tuple
``(centres, radii)`` of tensors shaped ``(n_envs, n_avoid, 2)`` and
``(n_envs, n_avoid)`` respectively.  These are concatenated with the static
avoid spaces from ``__init__``.

Orientation is delegated to each :class:`ManipulatableObject` via its
:meth:`~ManipulatableObject.set_pos_3d` method, so objects can define their own
"standing" vs "random" orientation policy.

Example::

    from unienv_genesis_collection.objects import BoxObject, CylinderObject

    pool = [
        BoxObject(size=(0.04, 0.04, 0.04), name="cube", color=(1, 0, 0, 1)),
        CylinderObject(radius=0.02, height=0.08, name="can", color=(0, 1, 0, 1)),
        SphereObject(radius=0.03, name="ball", color=(0, 0, 1, 1)),
    ]
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=pool,
        n_objects_range=(3, 6),
        table_size=(1.0, 0.6),
        table_pos=(0.0, 0.0, 0.65),
        avoid_spaces=[((0.0, 0.0, 0.65), 0.1)],  # keep centre clear
    )
"""

from __future__ import annotations

import math
import logging
from typing import Optional, List, Tuple, TYPE_CHECKING

import torch
from unienv_interface.space import BoxSpace
from unienv_interface.world.node import WorldNode

from unienv_genesis_collection.objects.base_object import (
    ManipulatableObject,
    bbox_size,
    bbox_center,
)

if TYPE_CHECKING:
    from unienv_genesis.world.genesis_world import GenesisWorld

__all__ = ["ClutteredTableManager"]

LOGGER = logging.getLogger(__name__)

_PARK_POS = (0.0, 0.0, -1000.0)
_PARK_QUAT = (1.0, 0.0, 0.0, 0.0)


class ClutteredTableManager(WorldNode):
    """Randomly places distractor :class:`ManipulatableObject` instances on a
    table surface without overlap.

    Args:
        world: :class:`GenesisWorld` instance.
        name: Node name.
        object_pool: Descriptors to choose from.
        n_objects_range: ``(min, max)`` number of objects per reset.
        table_size: Placement region ``(width_x, depth_y)`` in metres.
        table_pos: Table surface centre ``(x, y, z)``.
        table_yaw: Rotation of the table around the Z axis in radians.
        avoid_spaces: Regions where objects must NOT be placed.  Each entry
            is ``((cx, cy, cz), radius)`` — a sphere exclusion zone checked
            during rejection sampling alongside inter-object overlap.
        placement_margin: Margin from table edges for object centres (m).
        max_placement_attempts: Rejection-sampling budget per object.
        random_quaternion: Whether to request random orientations from each
            object's :meth:`~ManipulatableObject.set_pos_3d`.

    Raises:
        ValueError: If the upper bound of *n_objects_range* exceeds the
            pool size.
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    reload_priorities = {25}
    after_reload_priorities = {25}
    reset_priorities = {25}

    def __init__(
        self,
        world: "GenesisWorld",
        name: str = "clutter",
        object_pool: Optional[List[ManipulatableObject]] = None,
        n_objects_range: Tuple[int, int] = (3, 8),
        table_size: Tuple[float, float] = (1.0, 0.6),
        table_pos: Tuple[float, float, float] = (0.0, 0.0, 0.65),
        table_yaw: float = 0.0,
        avoid_spaces: Optional[List[Tuple[Tuple[float, float, float], float]]] = None,
        placement_margin: float = 0.05,
        max_placement_attempts: int = 100,
        random_quaternion: bool = True,
    ):
        self.world = world
        self._name = name
        self.object_pool = list(object_pool) if object_pool else []
        self.n_objects_range = n_objects_range
        self.table_size = table_size
        self.table_pos = table_pos
        self.table_yaw = table_yaw
        self._cos_yaw = math.cos(table_yaw)
        self._sin_yaw = math.sin(table_yaw)
        self.avoid_spaces = list(avoid_spaces) if avoid_spaces else []
        self.placement_margin = placement_margin
        self.max_placement_attempts = max_placement_attempts
        self.random_quaternion = random_quaternion

        if self.object_pool and n_objects_range[1] > len(self.object_pool):
            raise ValueError(
                f"n_objects_range upper bound ({n_objects_range[1]}) exceeds "
                f"object_pool size ({len(self.object_pool)})"
            )

        # RNG state — reseeded when an explicit seed is provided.
        # Use the world's device so that torch.rand/randint work on GPU backends.
        self.rng: torch.Generator = torch.Generator(device=world.device)
        self.rng.seed()

        # Pre-tensorise static avoid spaces (XY only), inverse-rotated into
        # the table's local frame so placement sampling stays axis-aligned.
        if self.avoid_spaces:
            world_xy = torch.tensor(
                [[c[0], c[1]] for c, _r in self.avoid_spaces],
                dtype=torch.float32,
            )  # (n_static, 2)
            # Inverse-rotate world → local table frame.
            offset = world_xy - torch.tensor(
                [table_pos[0], table_pos[1]], dtype=torch.float32,
            )
            c_y, s_y = self._cos_yaw, self._sin_yaw
            self._static_avoid_xy = torch.stack([
                offset[:, 0] * c_y + offset[:, 1] * s_y,
                -offset[:, 0] * s_y + offset[:, 1] * c_y,
            ], dim=-1)  # (n_static, 2)
            self._static_avoid_r = torch.tensor(
                [r for _c, r in self.avoid_spaces],
                dtype=torch.float32,
            )  # (n_static,)
        else:
            self._static_avoid_xy = torch.empty(0, 2)
            self._static_avoid_r = torch.empty(0)

        # Populated during reload().
        self._loaded_entities: List = []
        self._loaded_objects: List[ManipulatableObject] = []
        self._bboxes: List[BoxSpace] = []

        # Pre-computed bbox tensors (set after reload).
        self._dims_t: Optional[torch.Tensor] = None         # (n_pool, 3)
        self._centers_t: Optional[torch.Tensor] = None       # (n_pool, 3)
        self._footprint_r_t: Optional[torch.Tensor] = None   # (n_pool,)

        # Per-environment active-object mask (set after first reset).
        # Unbatched: (n_pool,) bool.  Batched: (n_pool, n_envs) bool.
        self._active_mask_t: Optional[torch.Tensor] = None

        # Pending avoid spaces queued via set_next_reset_avoid_spaces().
        self._pending_avoid_spaces: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def name(self) -> str:
        return self._name

    def _reseed(self, seed: int) -> None:
        """Reseed the torch RNG from *seed*."""
        self.rng.manual_seed(seed)

    def set_next_reset_avoid_spaces(
        self,
        centres: torch.Tensor,
        radii: torch.Tensor
    ) -> None:
        """Queue per-env avoid spaces for the next ``reset()`` call.

        These are merged with any ``avoid_spaces`` passed via ``reset()`` kwargs and the static avoid spaces from ``__init__``.  The queued value is automatically consumed (cleared) after the next reset.

        Args:
            centres, shaped `([n_envs], n_zones, 2 or 3)`
            radii, shaped `([n_envs], n_zones)`.
        The `n_envs` depends on the number of environments in the world, it's not optional if the world is batched
        """
        self._pending_avoid_spaces = (
            centres,
            radii
        )

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority != 25:
            return
        if seed is not None:
            self._reseed(seed)

        batch_fixed_verts = self.world.batch_size is not None

        self._loaded_entities = []
        self._loaded_objects = []
        self._bboxes = []
        self._active_mask_t = None

        for i, obj_desc in enumerate(self.object_pool):
            entity = obj_desc.load_into_scene(
                world=self.world,
                pos=_PARK_POS,
                quat=_PARK_QUAT,
                name_override=f"{self._name}_{obj_desc.name}_{i}",
                batch_fixed_verts=batch_fixed_verts,
            )
            self._loaded_entities.append(entity)
            self._loaded_objects.append(obj_desc)
            self._bboxes.append(obj_desc.get_local_bounding_box(entity))

        # Pre-compute bbox tensors for vectorised placement.
        if self._bboxes:
            self._dims_t = torch.stack(
                [bbox_size(b) for b in self._bboxes]
            )  # (n_pool, 3)
            self._centers_t = torch.stack(
                [bbox_center(b) for b in self._bboxes]
            )  # (n_pool, 3)
            self._footprint_r_t = (
                torch.linalg.norm(self._dims_t[:, :2], dim=1) / 2.0
            )  # (n_pool,)

    def after_reload(self, *, priority=0, mask=None, **kwargs):
        if priority != 25:
            return
        # After a fresh reload every environment needs placement.
        self.reset(priority=25, seed=None, mask=None, **kwargs)

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority != 25:
            return
        if not self._loaded_entities:
            return

        if seed is not None:
            self._reseed(seed)

        n_envs = self.world.batch_size
        n_pool = len(self.object_pool)

        if n_pool == 0:
            return

        # --- Merge pending + kwargs avoid spaces -----------------------
        extra = kwargs.get("avoid_spaces", None)
        pending = self._pending_avoid_spaces
        self._pending_avoid_spaces = None  # consume

        if pending is not None:
            if extra is not None:
                extra = (
                    torch.cat([extra[0], pending[0]], dim=-2),
                    torch.cat([extra[1], pending[1]], dim=-2),
                )
            else:
                extra = pending

        # --- Build avoid-space tensors --------------------------------
        avoid_xy, avoid_r = self._prepare_avoid_spaces(n_envs, mask, extra)

        if n_envs is None:
            # Unbatched — single environment (mask ignored).
            positions, active = self._compute_placement_vectorized(
                1, avoid_xy, avoid_r,
            )
            self._active_mask_t = active.squeeze(1)  # (n_pool,)
            self._apply_placements(positions.squeeze(1), envs_idx=None)
        else:
            # Batched — respect the mask.
            if mask is not None:
                mask_t = (
                    mask if isinstance(mask, torch.Tensor)
                    else torch.as_tensor(mask, dtype=torch.bool)
                )
                reset_idx = torch.where(mask_t)[0]
            else:
                reset_idx = torch.arange(n_envs)

            n_reset = reset_idx.numel()
            if n_reset == 0:
                return

            # Initialise active mask on first call.
            if self._active_mask_t is None:
                self._active_mask_t = torch.zeros(
                    n_pool, n_envs, dtype=torch.bool,
                )

            positions, active = self._compute_placement_vectorized(
                n_reset, avoid_xy, avoid_r,
            )

            # Update active mask for reset environments only.
            self._active_mask_t[:, reset_idx] = active

            envs_idx = reset_idx if mask is not None else None
            self._apply_placements(positions, envs_idx=envs_idx)

    def get_observation(self):
        return None

    # ------------------------------------------------------------------
    # Avoid-space preparation
    # ------------------------------------------------------------------

    def _prepare_avoid_spaces(
        self,
        n_envs: Optional[int],
        mask: Optional[torch.Tensor],
        extra: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge static and per-env avoid spaces into broadcast-ready tensors.

        Returns:
            avoid_xy: ``(n_reset, n_total_avoid, 2)``
            avoid_r:  ``(n_reset, n_total_avoid)``
        """
        if n_envs is None:
            n_reset = 1
            reset_idx = None
            extra = (item[None] for item in extra) if extra is not None else None
        elif mask is not None:
            mask_t = (
                mask if isinstance(mask, torch.Tensor)
                else torch.as_tensor(mask, dtype=torch.bool)
            )
            reset_idx = torch.where(mask_t)[0]
            n_reset = reset_idx.numel()
        else:
            n_reset = n_envs
            reset_idx = None

        parts_xy: List[torch.Tensor] = []
        parts_r: List[torch.Tensor] = []

        # Static avoid spaces — broadcast to (1, n_static, 2).
        n_static = self._static_avoid_xy.shape[0]
        if n_static > 0:
            parts_xy.append(
                self._static_avoid_xy.unsqueeze(0).expand(n_reset, -1, -1)
            )
            parts_r.append(
                self._static_avoid_r.unsqueeze(0).expand(n_reset, -1)
            )

        # Per-env avoid spaces from kwargs (expected in world frame).
        if extra is not None:
            ext_xy, ext_r = extra  # (n_envs_full, n_extra, 2|3), (n_envs_full, n_extra)
            # Slice XY only if 3D centres were provided.
            if ext_xy.dim() == 3 and ext_xy.shape[-1] > 2:
                ext_xy = ext_xy[..., :2]
            # Index by reset_idx if partial reset.
            if reset_idx is not None:
                ext_xy = ext_xy[reset_idx]
                ext_r = ext_r[reset_idx]
            # Inverse-rotate world → local table frame.
            tp = torch.tensor(
                [self.table_pos[0], self.table_pos[1]], dtype=torch.float32,
            )
            offset = ext_xy.float() - tp
            c_y, s_y = self._cos_yaw, self._sin_yaw
            local_xy = torch.stack([
                offset[..., 0] * c_y + offset[..., 1] * s_y,
                -offset[..., 0] * s_y + offset[..., 1] * c_y,
            ], dim=-1)
            parts_xy.append(local_xy)
            parts_r.append(ext_r.float())

        if parts_xy:
            return torch.cat(parts_xy, dim=1), torch.cat(parts_r, dim=1)
        return (
            torch.empty(n_reset, 0, 2),
            torch.empty(n_reset, 0),
        )

    # ------------------------------------------------------------------
    # Vectorised placement computation
    # ------------------------------------------------------------------

    def _compute_placement_vectorized(
        self,
        n_envs: int,
        avoid_xy: torch.Tensor,
        avoid_r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute collision-free placements for *n_envs* environments in
        parallel using PyTorch.

        The algorithm walks through object *slots* 0 … max_n−1 sequentially
        (each slot depends on previously placed objects), but processes **all
        environments simultaneously** within each slot via batched rejection
        sampling.

        Args:
            n_envs: Number of environments to compute for.
            avoid_xy: ``(n_envs, n_avoid, 2)`` exclusion-zone centres.
            avoid_r:  ``(n_envs, n_avoid)`` exclusion-zone radii.

        Returns:
            positions:   ``(n_pool, n_envs, 3)`` — world positions (parked for
                         inactive objects).
            active_mask: ``(n_pool, n_envs)``    — *True* where the object is
                         placed on the table in that environment.
        """
        n_pool = len(self.object_pool)

        # Start every object at the parking position.
        positions = torch.zeros(n_pool, n_envs, 3)
        positions[:, :, 2] = _PARK_POS[2]
        active_mask = torch.zeros(n_pool, n_envs, dtype=torch.bool)

        if n_pool == 0:
            return positions, active_mask

        # --- Random object counts and permutations --------------------
        n_objs = torch.randint(
            self.n_objects_range[0],
            self.n_objects_range[1] + 1,
            (n_envs,),
            generator=self.rng,
        )
        max_n = int(n_objs.max().item())

        if max_n == 0:
            return positions, active_mask

        # Random permutations via argsort of uniform noise.
        noise = torch.rand(n_envs, n_pool, generator=self.rng)
        perms = noise.argsort(dim=1)  # (n_envs, n_pool)

        # --- Table geometry -------------------------------------------
        table_half_x = self.table_size[0] / 2.0 - self.placement_margin
        table_half_y = self.table_size[1] / 2.0 - self.placement_margin
        table_half = torch.tensor(
            [table_half_x, table_half_y], dtype=torch.float32,
        )
        table_pos = torch.tensor(self.table_pos, dtype=torch.float32)

        n_avoid = avoid_xy.shape[1]

        # --- Footprint tracking ---------------------------------------
        placed_xy = torch.zeros(max_n, n_envs, 2)
        placed_r = torch.zeros(max_n, n_envs)
        placed_valid = torch.zeros(max_n, n_envs, dtype=torch.bool)
        n_placed = 0

        for k in range(max_n):
            slot_active = k < n_objs  # (n_envs,)
            if not slot_active.any():
                break

            obj_idx = perms[:, k]  # (n_envs,)

            # Per-env bbox look-up.
            obj_fp_r = self._footprint_r_t[obj_idx]   # (n_envs,)
            obj_half = self._dims_t[obj_idx, :2] / 2.0  # (n_envs, 2)
            obj_dims = self._dims_t[obj_idx]             # (n_envs, 3)
            obj_ctrs = self._centers_t[obj_idx]          # (n_envs, 3)

            # Valid placement range.
            xy_range = table_half.unsqueeze(0) - obj_half  # (n_envs, 2)
            valid = (
                slot_active
                & (xy_range[:, 0] > 0)
                & (xy_range[:, 1] > 0)
            )

            if not valid.any():
                n_placed += 1
                continue

            # --- Rejection sampling across all envs simultaneously ----
            placed = torch.zeros(n_envs, dtype=torch.bool)
            best_xy = torch.zeros(n_envs, 2)

            for _att in range(self.max_placement_attempts):
                need = valid & ~placed  # (n_envs,)
                if not need.any():
                    break

                n_need = int(need.sum().item())
                raw = torch.rand(n_need, 2, generator=self.rng) * 2 - 1
                cand = torch.zeros(n_envs, 2)
                cand[need] = raw * xy_range[need]

                # Check overlap with already-placed footprints.
                overlap = torch.zeros(n_envs, dtype=torch.bool)
                if n_placed > 0:
                    dx = placed_xy[:n_placed, :, 0] - cand[:, 0].unsqueeze(0)
                    dy = placed_xy[:n_placed, :, 1] - cand[:, 1].unsqueeze(0)
                    dist = torch.sqrt(dx * dx + dy * dy)
                    min_d = obj_fp_r.unsqueeze(0) + placed_r[:n_placed]
                    overlap = (
                        (dist < min_d) & placed_valid[:n_placed]
                    ).any(dim=0)

                # Check avoid-space exclusion zones (per-env aware).
                # Both cand and avoid_xy are in the table's local frame.
                if n_avoid > 0:
                    cand_expanded = cand.unsqueeze(1)  # (n_envs, 1, 2)
                    adist = torch.sqrt(
                        ((avoid_xy - cand_expanded) ** 2).sum(dim=-1)
                    )  # (n_envs, n_avoid)
                    amin = obj_fp_r.unsqueeze(1) + avoid_r  # (n_envs, n_avoid)
                    overlap = overlap | (adist < amin).any(dim=1)

                newly_placed = need & ~overlap
                placed = placed | newly_placed
                best_xy[newly_placed] = cand[newly_placed]

            # Record footprint for this slot.
            placed_xy[n_placed] = best_xy
            placed_r[n_placed] = obj_fp_r
            placed_valid[n_placed] = placed & valid
            n_placed += 1

            success = placed & valid
            if not success.any():
                continue

            # --- World positions (forward-rotate local → world) ---------
            c_y, s_y = self._cos_yaw, self._sin_yaw
            wx = table_pos[0] + best_xy[:, 0] * c_y - best_xy[:, 1] * s_y
            wy = table_pos[1] + best_xy[:, 0] * s_y + best_xy[:, 1] * c_y
            wz = table_pos[2] + obj_dims[:, 2] / 2.0 - obj_ctrs[:, 2]

            # --- Write via advanced indexing (no Python env loop) ------
            s_env = torch.where(success)[0]
            s_pool = obj_idx[success]
            positions[s_pool, s_env, 0] = wx[success]
            positions[s_pool, s_env, 1] = wy[success]
            positions[s_pool, s_env, 2] = wz[success]
            active_mask[s_pool, s_env] = True

        return positions, active_mask

    # ------------------------------------------------------------------
    # Apply placements to Genesis entities
    # ------------------------------------------------------------------

    def _apply_placements(
        self,
        positions: torch.Tensor,
        envs_idx: Optional[torch.Tensor],
    ) -> None:
        """Set entity positions and orientations via each object's
        :meth:`~ManipulatableObject.set_pos_3d`.

        Args:
            positions: ``(n_pool, n_envs, 3)`` or ``(n_pool, 3)``
                (unbatched).
            envs_idx:  ``None`` for full writes (all envs or unbatched).
                ``(n_reset,)`` LongTensor for partial (masked) updates —
                passed through to
                :meth:`~ManipulatableObject.set_pos_3d` as *envs_idx*.
        """
        for i, (obj, entity) in enumerate(
            zip(self._loaded_objects, self._loaded_entities),
        ):
            obj.set_pos_3d(
                entity,
                positions[i],
                random_quaternion=self.random_quaternion,
                rng=self.rng,
                envs_idx=envs_idx,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_objects(self) -> List[ManipulatableObject]:
        """Descriptors for the active objects in env 0."""
        if self._active_mask_t is None:
            return []
        mask = (
            self._active_mask_t
            if self._active_mask_t.ndim == 1
            else self._active_mask_t[:, 0]
        )
        return [self._loaded_objects[i] for i in torch.where(mask)[0].tolist()]

    @property
    def active_entities(self) -> List:
        """Genesis entity references for the active objects in env 0."""
        if self._active_mask_t is None:
            return []
        mask = (
            self._active_mask_t
            if self._active_mask_t.ndim == 1
            else self._active_mask_t[:, 0]
        )
        return [self._loaded_entities[i] for i in torch.where(mask)[0].tolist()]
