"""ClutteredTableManager — randomly places distractor objects on a table surface.

Follows the :class:`CachedTableBuilder` pattern:

* During ``reload()`` (priority 25): loads every object from the pool into the
  scene at a **unique** parking position far below the table (z = park_z,
  default -1000) and well-separated in XY so multi-mesh collision hulls never
  overlap before the Genesis scene is built / settled.
* During ``reset()`` (priority 25): selects a random subset, computes
  collision-free placements via rejection sampling, and positions them on the
  table surface.  Unused objects remain parked at their unique per-object
  parking spots (fixed ``_PARK_QUAT`` orientation).

Unique parking (vs the historical single shared ``(0, 0, -1000)`` point) is
motivated by the GSO / YCB catalogs' multi-mesh (V-HACD / CoACD) collision
hulls: piling the whole inactive pool on one point inflates the broad-phase
candidate-pair count past Genesis's default budget and halts the settle step.
The parking grid is laid out as a square-ish 2-D lattice with uniform spacing
``2 * max_footprint_radius + park_clearance``, which guarantees the centre-to-
centre separation between any two parked objects meets or exceeds the sum of
their (per-object) footprint radii (the spacing equals ``2 * max_radius`` when
``park_clearance`` is ``0``, so two max-radius objects end up exactly touching;
a positive ``park_clearance`` makes the separation strict).  Off-centre object origins are compensated
for: the grid targets are bbox **centres**, and each object's loaded
``entity-origin`` position is ``target_bbox_centre − local_bbox_centre`` so the
resulting AABB centres land exactly on the lattice.

The geometry cache (per-descriptor bounding boxes, dims, centres, footprint
radii, max radius, and the deterministic park ``entity-origin`` positions) is
computed once on the **first** reload — before any ``load_into_scene`` call —
from each descriptor's no-arg ``get_local_bounding_box()`` and
``get_footprint_radius()``.  Subsequent reloads reuse the cache *iff* the
pool's lightweight identity signature (a tuple of descriptor object ids, in
order — see ``_pool_signature``) is unchanged, in which case neither geometry
query is re-invoked.  Replacing a descriptor, reordering the pool, or
adding/removing an entry changes the signature and rebuilds the cache; in-
place geometry mutation of an existing descriptor instance is not detected.

Priority 25 is within the ``[0, 50)`` objects / props range, so it runs
**after** the table (priority 100) and robot (priority 50) have been set up.

Supports batched environments: each environment gets an independent random
placement.  When a ``mask`` is supplied to ``reset()``, only the flagged
environments are re-randomised; the others keep their current layout.

Per-environment ``avoid_spaces`` can be passed through ``reset()`` as a tuple
``(centres, radii)`` of tensors shaped ``(n_envs, n_avoid, 2)`` and
``(n_envs, n_avoid)`` respectively.  These are concatenated with the static
avoid spaces from ``__init__``.

Inactive (parked) objects always receive the identity ``_PARK_QUAT`` and no
RNG is consumed for their orientation.  Active (on-table) objects delegate
orientation to each :class:`ManipulatableObject` via its
:meth:`~ManipulatableObject.set_pos_3d` / ``_standing_quaternion`` behaviour,
so objects can define their own "standing" vs "random" orientation policy.

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

# Parking altitude sentinel (far below the table).  The shared parking
# *origin* point is no longer used; inactive objects are now spread across a
# 2-D grid whose target bbox-centre altitude is ``_PARK_POS[2]``.
_PARK_POS = (0.0, 0.0, -1000.0)
_PARK_QUAT = (1.0, 0.0, 0.0, 0.0)

# Default extra centre-to-centre gap (m) added on top of the
# ``2 * max_footprint_radius`` minimum-pair-distance park-grid spacing.
_DEFAULT_PARK_CLEARANCE = 0.0


class ClutteredTableManager(WorldNode):
    """Randomly places distractor :class:`ManipulatableObject` instances on a
    table surface without overlap.

    Inactive (unused) objects are parked at unique, well-separated positions
    derived from the pool's per-object bounding boxes (see the module
    docstring for the parking-grid rationale).

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
        park_clearance: Extra centre-to-centre gap (m) added on top of the
            ``2 * max_footprint_radius`` minimum pairwise park-grid
            spacing.  Must be non-negative.  ``0.0`` makes the spacing
            exactly ``2 * max_footprint_radius`` — i.e. every parked pair
            is just touching by their circumscribed footprints, which is
            already enough for sweep-and-prune broad-phase filtering.
            Increasing it adds a safety margin against numerical
            tolerance.  The park altitude is fixed at ``_PARK_POS[2]``
            (``-1000``).

    Raises:
        ValueError: If the upper bound of *n_objects_range* exceeds the
            pool size, or if *park_clearance* is negative.
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
        park_clearance: float = _DEFAULT_PARK_CLEARANCE,
        seed: Optional[int] = None,
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
        self.park_clearance = float(park_clearance)

        if not math.isfinite(self.park_clearance) or self.park_clearance < 0.0:
            raise ValueError(
                f"park_clearance must be a non-negative finite float, "
                f"got {self.park_clearance}"
            )

        if self.object_pool and n_objects_range[1] > len(self.object_pool):
            raise ValueError(
                f"n_objects_range upper bound ({n_objects_range[1]}) exceeds "
                f"object_pool size ({len(self.object_pool)})"
            )

        # RNG state — reseeded when an explicit seed is provided.
        # Use the world's device so that torch.rand/randint work on GPU backends.
        self.rng: torch.Generator = torch.Generator(device=world.device)
        if seed is None:
            self.rng.seed()
        else:
            self.rng.manual_seed(seed)

        # Pre-tensorise static avoid spaces (XY only), inverse-rotated into
        # the table's local frame so placement sampling stays axis-aligned.
        # All tensors live on the world's device so that GPU backends never
        # have to cross-hostcopies during reset().
        if self.avoid_spaces:
            world_xy = torch.tensor(
                [[c[0], c[1]] for c, _r in self.avoid_spaces],
                dtype=world.dtype,
                device=world.device,
            )  # (n_static, 2)
            # Inverse-rotate world → local table frame.
            offset = world_xy - torch.tensor(
                [table_pos[0], table_pos[1]],
                dtype=world.dtype,
                device=world.device,
            )
            c_y, s_y = self._cos_yaw, self._sin_yaw
            self._static_avoid_xy = torch.stack([
                offset[:, 0] * c_y + offset[:, 1] * s_y,
                -offset[:, 0] * s_y + offset[:, 1] * c_y,
            ], dim=-1)  # (n_static, 2)
            self._static_avoid_r = torch.tensor(
                [r for _c, r in self.avoid_spaces],
                dtype=world.dtype,
                device=world.device,
            )  # (n_static,)
        else:
            self._static_avoid_xy = torch.empty(
                0, 2, dtype=world.dtype, device=world.device,
            )
            self._static_avoid_r = torch.empty(
                0, dtype=world.dtype, device=world.device,
            )

        # Populated during reload().
        self._loaded_entities: List = []
        self._loaded_objects: List[ManipulatableObject] = []
        self._bboxes: List[BoxSpace] = []

        # Pre-computed bbox tensors (cached on the first reload and reused on
        # subsequent reloads of the same manager instance).  The cache is
        # invalidated (and rebuilt) when the object pool's lightweight identity
        # signature — a tuple of descriptor object ids in pool order — changes
        # between reloads (see ``_pool_signature``).  Computing the signature
        # invokes no geometry queries, so an unchanged pool is detected cheaply
        # and the cached tensors are reused verbatim.  Replacing, reordering,
        # adding, or removing a descriptor rebuilds; in-place geometry mutation
        # of an existing descriptor instance is *not* detected.
        self._dims_t: Optional[torch.Tensor] = None         # (n_pool, 3)
        self._centers_t: Optional[torch.Tensor] = None     # (n_pool, 3)
        self._footprint_r_t: Optional[torch.Tensor] = None  # (n_pool,)
        self._max_footprint_radius: float = 0.0            # max over pool
        # Deterministic per-object *entity-origin* parking positions (n_pool, 3),
        # computed once from the bounding-box cache so each inactive object
        # occupies a unique lattice cell whose bbox centre lands at park_z.
        self._park_positions_t: Optional[torch.Tensor] = None
        # Lightweight identity signature (tuple of descriptor object ids, in pool
        # order) of the descriptor set the current cache was built from; ``None``
        # until the first successful build.  Reading it never queries geometry.
        self._pool_signature_cached: Optional[tuple] = None

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

    def _pool_signature(self) -> tuple:
        """Compute a lightweight stable identity signature for the current
        object pool.

        The signature captures pool membership **and** order through the
        Python-level *object identity* of each descriptor: it is simply
        ``tuple(id(obj) for obj in self.object_pool)``.  Replacing a
        descriptor, reordering the pool, or adding/removing an entry all yield
        a different signature and therefore force a geometry-cache rebuild on
        the next :meth:`_ensure_geometry_cache` call.  Two distinct but
        geometrically-identical descriptor instances are *not* considered
        equal — the second reload of a replaced pool rebuilds the cache.

        Crucially, computing this signature does **not** invoke any geometry
        query (``get_local_bounding_box`` / ``get_footprint_radius``); it only
        reads ``id()``.  This means the signature can be checked on every
        reload at near-zero cost, and — for an unchanged pool — the cached
        geometry tensors are reused without re-querying the descriptors.

        In-place mutation of a descriptor's geometry (e.g. resizing a box in
        place without replacing the instance) is intentionally *not* detected:
        the cache is only rebuilt when the descriptor set or its ordering
        changes.

        Returns:
            A hashable, comparable tuple of ``id()`` ints (one per
            descriptor, in pool order).  Returns ``()`` for an empty pool.
        """
        return tuple(id(obj_desc) for obj_desc in self.object_pool)

    def _ensure_geometry_cache(self) -> None:
        """Populate the bounding-box / footprint / parking cache once.

        On the first ``reload()`` (before any ``load_into_scene`` call) walk
        the pool, cache each descriptor's no-arg
        :meth:`~ManipulatableObject.get_local_bounding_box` and
        :meth:`~ManipulatableObject.get_footprint_radius`, derive the stacked
        ``dims`` / ``centers`` / ``footprint_r`` tensors, the scalar
        ``max_footprint_radius``, and the deterministic per-object
        ``entity-origin`` parking positions for a 2-D lattice with uniform
        spacing ``2 * max_footprint_radius + park_clearance``.

        On subsequent reloads the cache is reused *iff* the pool's lightweight
        identity signature (a tuple of descriptor object ids, in pool order —
        see :meth:`_pool_signature`) is unchanged.  Computing the signature
        invokes no geometry queries, so the reuse check is essentially free and
        an unchanged pool never re-queries ``get_local_bounding_box`` /
        ``get_footprint_radius``.  If a descriptor was replaced, the pool
        reordered, or an entry added/removed, the cache is invalidated and
        rebuilt so the park lattice and footprint radii stay consistent with the
        descriptors that are actually about to be loaded.  In-place geometry
        mutation of an existing descriptor instance is not detected.
        """
        current_sig = self._pool_signature()
        if (
            self._park_positions_t is not None
            and self._pool_signature_cached == current_sig
        ):
            # Geometry cache still valid for the current pool; reuse.
            return

        n_pool = len(self.object_pool)
        device = self.world.device
        dtype = self.world.dtype

        self._bboxes = [
            obj_desc.get_local_bounding_box()
            for obj_desc in self.object_pool
        ]
        radii = [
            obj_desc.get_footprint_radius()
            for obj_desc in self.object_pool
        ]

        if n_pool == 0:
            self._dims_t = torch.zeros(0, 3, dtype=dtype, device=device)
            self._centers_t = torch.zeros(0, 3, dtype=dtype, device=device)
            self._footprint_r_t = torch.zeros(0, dtype=dtype, device=device)
            self._max_footprint_radius = 0.0
            self._park_positions_t = torch.zeros(0, 3, dtype=dtype, device=device)
            self._pool_signature_cached = current_sig
            return

        self._dims_t = torch.stack(
            [bbox_size(b) for b in self._bboxes]
        ).to(dtype=dtype, device=device)  # (n_pool, 3)
        self._centers_t = torch.stack(
            [bbox_center(b) for b in self._bboxes]
        ).to(dtype=dtype, device=device)  # (n_pool, 3)
        self._footprint_r_t = torch.tensor(
            radii, dtype=dtype, device=device,
        )  # (n_pool,)
        self._max_footprint_radius = float(self._footprint_r_t.max().item())

        # Square-ish 2-D lattice: cell (i) at column ``(i % n_cols)`` and row
        # ``(i // n_cols)`` multiplied by a uniform spacing that meets or
        # exceeds the sum of any pair's footprint radii (bounded by the max);
        # the separation is strict when ``park_clearance`` is positive.
        spacing = 2.0 * self._max_footprint_radius + self.park_clearance
        n_cols = max(1, int(math.ceil(math.sqrt(n_pool))))
        idx = torch.arange(n_pool, dtype=dtype, device=device)
        cols = (idx % n_cols) * spacing
        rows = torch.div(idx, n_cols, rounding_mode="floor") * spacing

        # Parking *bbox-centre* targets in the world XY plane; z is fixed at
        # the park altitude (park_z, default -1000) so the parked AABB
        # centres share a single altitude far below the table.
        park_centers = torch.stack(
            [cols, rows, torch.full_like(cols, float(_PARK_POS[2]))],
            dim=-1,
        )  # (n_pool, 3)

        # Compensate each object's local bbox centre so its AABB centre lands
        # exactly on the lattice target — accounts for off-centre object
        # origins commonly found in GSO/YCB meshes.
        self._park_positions_t = (park_centers - self._centers_t)  # (n_pool, 3)

        self._pool_signature_cached = current_sig

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority != 25:
            return
        if seed is not None:
            self._reseed(seed)

        batch_fixed_verts = self.world.batch_size is not None

        # Build the geometry cache (unique park positions included) before any
        # entity enters the scene so every object is loaded directly at its
        # unique, well-separated parking pose — no shared pile before settle.
        self._ensure_geometry_cache()

        self._loaded_entities = []
        self._loaded_objects = []
        self._active_mask_t = None

        for i, obj_desc in enumerate(self.object_pool):
            park_pos = self._park_positions_t[i]
            entity = obj_desc.load_into_scene(
                world=self.world,
                pos=(
                    float(park_pos[0].item()),
                    float(park_pos[1].item()),
                    float(park_pos[2].item()),
                ),
                quat=_PARK_QUAT,
                name_override=f"{self._name}_{obj_desc.name}_{i}",
                batch_fixed_verts=batch_fixed_verts,
            )
            self._loaded_entities.append(entity)
            self._loaded_objects.append(obj_desc)

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
            self._apply_placements(
                positions.squeeze(1), active.squeeze(1), envs_idx=None,
            )
        else:
            # Batched — respect the mask.
            if mask is not None:
                mask_t = (
                    mask if isinstance(mask, torch.Tensor)
                    else torch.as_tensor(
                        mask, dtype=torch.bool, device=self.world.device,
                    )
                )
                # Ensure the mask lives on the world's device so that the
                # derived index tensor (and subsequent indexing) stays
                # device-local on GPU backends.
                if mask_t.device != self.world.device:
                    mask_t = mask_t.to(self.world.device)
                reset_idx = torch.where(mask_t)[0]
            else:
                reset_idx = torch.arange(
                    n_envs, device=self.world.device,
                )

            n_reset = reset_idx.numel()
            if n_reset == 0:
                return

            # Initialise active mask on first call on the world's device so
            # later indexing/assignment stays device-local on GPU backends.
            if self._active_mask_t is None:
                self._active_mask_t = torch.zeros(
                    n_pool, n_envs, dtype=torch.bool, device=self.world.device,
                )

            positions, active = self._compute_placement_vectorized(
                n_reset, avoid_xy, avoid_r,
            )

            # Update active mask for reset environments only.
            self._active_mask_t[:, reset_idx] = active

            envs_idx = reset_idx if mask is not None else None
            self._apply_placements(positions, active, envs_idx=envs_idx)

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
                else torch.as_tensor(
                    mask, dtype=torch.bool, device=self.world.device,
                )
            )
            if mask_t.device != self.world.device:
                mask_t = mask_t.to(self.world.device)
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
            # Move dynamic avoid spaces onto the world's device so that the
            # local index slicing and downstream math stay device-local.
            ext_xy = ext_xy.to(self.world.device)
            ext_r = ext_r.to(self.world.device)
            # Slice XY only if 3D centres were provided.
            if ext_xy.dim() == 3 and ext_xy.shape[-1] > 2:
                ext_xy = ext_xy[..., :2]
            # Index by reset_idx if partial reset.
            if reset_idx is not None:
                ext_xy = ext_xy[reset_idx]
                ext_r = ext_r[reset_idx]
            # Inverse-rotate world → local table frame.
            tp = torch.tensor(
                [self.table_pos[0], self.table_pos[1]],
                dtype=ext_xy.dtype, device=ext_xy.device,
            )
            offset = ext_xy.to(dtype=tp.dtype) - tp
            c_y, s_y = self._cos_yaw, self._sin_yaw
            local_xy = torch.stack([
                offset[..., 0] * c_y + offset[..., 1] * s_y,
                -offset[..., 0] * s_y + offset[..., 1] * c_y,
            ], dim=-1)
            parts_xy.append(local_xy)
            parts_r.append(ext_r.to(dtype=ext_xy.dtype))

        if parts_xy:
            return torch.cat(parts_xy, dim=1), torch.cat(parts_r, dim=1)
        return (
            torch.empty(n_reset, 0, 2, dtype=self.world.dtype, device=self.world.device),
            torch.empty(n_reset, 0, dtype=self.world.dtype, device=self.world.device),
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
            positions:   ``(n_pool, n_envs, 3)`` — world positions (the unique
                         per-object parking entity-origin position for
                         inactive objects; on-table entity-origin position for
                         active ones).
            active_mask: ``(n_pool, n_envs)``    — *True* where the object is
                         placed on the table in that environment.
        """
        n_pool = len(self.object_pool)

        # Start every object at its unique, well-separated parking position
        # (broadcast across the reset scope).  Inactive objects never land on
        # the shared ``(0,0,-1000)`` pile — they keep this per-object slot.
        positions = self._park_positions_t.unsqueeze(1).expand(
            n_pool, n_envs, 3,
        ).contiguous()
        active_mask = torch.zeros(n_pool, n_envs, dtype=torch.bool, device=self.world.device)

        if n_pool == 0:
            return positions, active_mask

        # --- Random object counts and permutations --------------------
        n_objs = torch.randint(
            self.n_objects_range[0],
            self.n_objects_range[1] + 1,
            (n_envs,),
            generator=self.rng,
            device=self.world.device,
        )
        max_n = int(n_objs.max().item())

        if max_n == 0:
            return positions, active_mask

        # Random permutations via argsort of uniform noise.
        noise = torch.rand(n_envs, n_pool, generator=self.rng, dtype=self.world.dtype, device=self.world.device)
        perms = noise.argsort(dim=1)  # (n_envs, n_pool)

        # --- Table geometry -------------------------------------------
        table_half_x = self.table_size[0] / 2.0 - self.placement_margin
        table_half_y = self.table_size[1] / 2.0 - self.placement_margin
        table_half = torch.tensor(
            [table_half_x, table_half_y], dtype=self.world.dtype, device=self.world.device,
        )
        table_pos = torch.tensor(self.table_pos, dtype=self.world.dtype, device=self.world.device)

        n_avoid = avoid_xy.shape[1]

        # --- Footprint tracking ---------------------------------------
        placed_xy = torch.zeros(max_n, n_envs, 2, dtype=self.world.dtype, device=self.world.device)
        placed_r = torch.zeros(max_n, n_envs, dtype=self.world.dtype, device=self.world.device)
        placed_valid = torch.zeros(max_n, n_envs, dtype=torch.bool, device=self.world.device)
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
            placed = torch.zeros(n_envs, dtype=torch.bool, device=self.world.device)
            best_xy = torch.zeros(n_envs, 2, dtype=self.world.dtype, device=self.world.device)

            for _att in range(self.max_placement_attempts):
                need = valid & ~placed  # (n_envs,)
                if not need.any():
                    break

                n_need = int(need.sum().item())
                raw = torch.rand(n_need, 2, generator=self.rng, dtype=self.world.dtype, device=self.world.device) * 2 - 1
                cand = torch.zeros(n_envs, 2, dtype=self.world.dtype, device=self.world.device)
                cand[need] = raw * xy_range[need]

                # Check overlap with already-placed footprints.
                overlap = torch.zeros(n_envs, dtype=torch.bool, device=self.world.device)
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
        active_mask: torch.Tensor,
        envs_idx: Optional[torch.Tensor],
    ) -> None:
        """Set entity positions and orientations in an active-mask-aware way.

        Active objects use each descriptor's ``set_pos_3d`` orientation
        behaviour (random yaw by default; subclasses may customise via
        ``_standing_quaternion``); inactive objects are pinned to their unique
        cached parking position with the fixed ``_PARK_QUAT`` orientation and
        consume **no** RNG, regardless of ``random_quaternion``.

        Args:
            positions: ``(n_pool, n_envs, 3)`` (batched / masked) or
                ``(n_pool, 3)`` (unbatched).
            active_mask: ``(n_pool, n_envs)`` or ``(n_pool,)`` bool tensor
                matching ``positions`` — *True* where the object goes on the
                table, *False* where it stays parked.
            envs_idx: ``None`` for full writes (all envs or unbatched).
                ``(n_reset,)`` LongTensor for partial (masked) updates —
                ``positions`` are then ``(n_pool, n_reset, 3)`` and only the
                listed environments are overwritten; the rest keep their
                current pose.
        """
        if positions.ndim == 2:
            # Unbatched — single environment.
            for i, (obj, entity) in enumerate(
                zip(self._loaded_objects, self._loaded_entities),
            ):
                pos_i = positions[i]
                if bool(active_mask[i]):
                    # Active: honour the descriptor's orientation policy via
                    # the public set_pos_3d (which delegates to
                    # ``_standing_quaternion`` internally).  No private call.
                    obj.set_pos_3d(
                        entity,
                        pos_i,
                        random_quaternion=self.random_quaternion,
                        rng=self.rng,
                        envs_idx=None,
                    )
                else:
                    # Parked: pinned directly to the cached per-object park
                    # position with the fixed ``_PARK_QUAT``.  No RNG.
                    park_pos = self._park_positions_t[i].to(
                        dtype=pos_i.dtype, device=pos_i.device,
                    )
                    entity.set_pos(park_pos)
                    entity.set_quat(
                        torch.tensor(
                            _PARK_QUAT, dtype=pos_i.dtype, device=pos_i.device,
                        )
                    )
            return

        # Batched (full reset or masked reset sub-scope).
        # ``positions`` is ``(n_pool, n_reset, 3)`` where ``n_reset`` indexes
        # the reset scope; ``envs_idx`` (when not ``None``) maps the reset
        # scope to the **global** environment axis.
        n_reset = positions.shape[1]
        device = positions.device
        dtype = positions.dtype
        park_quat = torch.tensor(_PARK_QUAT, dtype=dtype, device=device)

        for i, (obj, entity) in enumerate(
            zip(self._loaded_objects, self._loaded_entities),
        ):
            pos_i = positions[i]        # (n_reset, 3)
            act_i = active_mask[i]      # (n_reset,) bool
            active_idx = torch.where(act_i)[0]      # local reset-scope idxs
            inactive_idx = torch.where(~act_i)[0]   # local reset-scope idxs

            # Map the local reset-scope indices to global env indices.  When
            # ``envs_idx`` is ``None`` the reset scope *is* the full env axis,
            # so the local and global indices coincide.
            if envs_idx is None:
                global_active_idx = active_idx
                global_inactive_idx = inactive_idx
            else:
                global_active_idx = envs_idx[active_idx]
                global_inactive_idx = envs_idx[inactive_idx]

            # --- Active subset: route through the descriptor's public
            # set_pos_3d so each object's "standing" / "random" orientation
            # policy is honoured.  RNG is drawn only for the active envs
            # (inside set_pos_3d → _standing_quaternion), so masked resets
            # with few active objects stay cheap and reproducible.
            # set_pos_3d accepts a boolean mask selector only, so the active
            # global indices are expanded to a (batch_size,) bool mask.
            if active_idx.numel() > 0:
                active_positions = pos_i[active_idx]  # (n_active, 3)
                global_active_mask = torch.zeros(
                    self.world.batch_size, dtype=torch.bool, device=device,
                )
                global_active_mask[global_active_idx] = True
                obj.set_pos_3d(
                    entity,
                    active_positions,
                    random_quaternion=self.random_quaternion,
                    rng=self.rng,
                    envs_idx=global_active_mask,
                )

            # --- Inactive subset: pinned directly to the cached per-object
            # park position with the fixed ``_PARK_QUAT``.  No RNG is drawn
            # for inactive objects, regardless of ``random_quaternion``.
            if inactive_idx.numel() > 0:
                n_inactive = inactive_idx.numel()
                park_pos = self._park_positions_t[i].to(
                    dtype=dtype, device=device,
                )  # (3,)
                inactive_positions = park_pos.unsqueeze(0).expand(
                    n_inactive, 3,
                )
                inactive_quats = park_quat.unsqueeze(0).expand(
                    n_inactive, 4,
                )
                cur_pos = entity.get_pos().clone()
                cur_quat = entity.get_quat().clone()
                cur_pos[global_inactive_idx] = inactive_positions
                cur_quat[global_inactive_idx] = inactive_quats
                entity.set_pos(cur_pos)
                entity.set_quat(cur_quat)

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
