"""FloatingVisualPropsManager — static, visual-only floating GSO/YCB props.

A sibling of :class:`ClutteredTableManager` that places a small random
subset of :class:`MeshObject`-compatible descriptors (typically
:class:`GSOObject` / :class:`YCBObject`) **floating around/above** the
current table.  Unlike the physical on-table clutter manager, these
props are:

* **Static** — ``fixed=True`` (no gravity / dynamics / contact).
* **Non-colliding** — ``collision=False`` so they never participate in
  the broad/narrow phase and never perturb the settled robot/table/clutter.
* **Visual-only** — rendered in RGB and depth, but never touched by the
  physics solver.
* Loaded as **rigid visual mesh entities** (``gs.morphs.Mesh``), *not*
  via the YCB URDF/GSO MJCF loaders.  This is mandatory because the
  Genesis ``BatchRenderer`` reads the rigid-solver visual geometry; the
  URDF/MJCF loaders bake textures into material overrides that the
  batched rasterizer does not pick up.  The original visual OBJ mesh is
  loaded as a bare mesh; when the descriptor exposes a
  ``visual_texture_path`` (e.g. GSO ``texture.png`` or YCB
  ``texture_map.png``) the manager builds a textured
  :class:`genesis.surfaces.Plastic` surface (``diffuse_texture`` =
  :class:`genesis.textures.ImageTexture`) so the asset's intrinsic
  texture survives the loss of its native MJCF/URDF material binding.
  Descriptors without a ``visual_texture_path`` (e.g. a generic
  :class:`MeshObject`) are added with ``surface=None`` and fall back to
  the Genesis default surface.
* Sampled with **full SO(3)** random orientations (Shoemake's method)
  when ``random_quaternion=True``; inactive objects are parked out of
  view at unique, well-separated lattice cells with a fixed identity
  quaternion (no RNG consumed for parked objects).

Lifecycle priorities are 20 — *after* the table (priority 100) and
robot (priority 50) are placed, *after* the physical clutter
(priority 25), and *before* the camera/sensors (priority 0/-50).
Reload/after_reload/reset priorities execute **highest-first**, so
priority 20 runs after the priority-25 physical clutter manager and
still before the priority-0 camera so the renderer picks up the
per-env pose at scene build time.  Sampling a table-aware subclass
refreshes ``self.table_size`` from the
:class:`RandomizedTableBuilder`'s re-sampled width/depth on every
``after_reload`` so pool/active placement tracks the actual table
footprint.

The bounding-box geometry cache (local AABB dims, centres, footprint
radii, max radius, and the deterministic parking lattice) is built
once on the first ``reload()`` and reused on subsequent reloads when
the pool's lightweight identity signature (a tuple of descriptor
``id()``s) is unchanged.

Per `ClutteredTableManager`, parking is a square-ish 2-D lattice with
spacing ``2 * max_footprint_radius + park_clearance``, as the
multi-mesh collision hulls of the GSO/YCB catalogs would otherwise
overlap at the default shared parking point.  Even though floating
props do not collide, keeping them spatially separated while parked
keeps the render culling broad-phase cheap and visually consistent.

Example::

    from unienv_genesis_collection.scenes.floating_visual_props_manager import (
        FloatingVisualPropsManager,
    )

    mgr = FloatingVisualPropsManager(
        world,
        name="float_props",
        object_pool=pool,
        n_objects_range=(0, 3),
        table_size=(1.0, 0.6),
        table_pos=(0.0, 0.0, 0.65),
        z_offset_range=(0.15, 0.65),
        xy_margin=0.05,
        min_separation=0.10,
        random_quaternion=True,
    )
"""

from __future__ import annotations

import math
import logging
from typing import Optional, List, Tuple, Sequence, TYPE_CHECKING

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

__all__ = ["FloatingVisualPropsManager"]

LOGGER = logging.getLogger(__name__)

# Parking altitude sentinel (far below the table).  Like the physical
# clutter manager, inactive objects are laid out on a 2-D lattice whose
# target bbox-centre altitude is ``_PARK_POS[2]``.
_PARK_POS = (0.0, 0.0, -1000.0)
_PARK_QUAT = (1.0, 0.0, 0.0, 0.0)

_DEFAULT_PARK_CLEARANCE = 0.0


def _shoemake_quaternions(
    rng: torch.Generator,
    n: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Return *n* uniformly-random unit quaternions ``(n, 4)`` ``(w, x, y, z)``.

    Shoemake's surrogate method: spherical-component parametrisation
    yields a uniform distribution over SO(3) for independent uniform
    inputs.  The output is a unit-norm tensor of shape ``(n, 4)``.
    """
    if n <= 0:
        return torch.zeros(0, 4, dtype=dtype, device=device)
    u1 = torch.rand(n, generator=rng, dtype=dtype, device=device)
    u2 = torch.rand(n, generator=rng, dtype=dtype, device=device)
    u3 = torch.rand(n, generator=rng, dtype=dtype, device=device)
    sqrt_1mu1 = torch.sqrt(torch.clamp(1.0 - u1, min=0.0))
    sqrt_u1 = torch.sqrt(torch.clamp(u1, min=0.0))
    a1 = 2.0 * math.pi * u2
    a2 = 2.0 * math.pi * u3
    sin1 = torch.sin(a1)
    cos1 = torch.cos(a1)
    sin2 = torch.sin(a2)
    cos2 = torch.cos(a2)
    return torch.stack(
        [sqrt_1mu1 * cos1, sqrt_1mu1 * sin1, sqrt_u1 * cos2, sqrt_u1 * sin2],
        dim=-1,
    )


class FloatingVisualPropsManager(WorldNode):
    """Static, visual-only floating GSO/YCB props around/above the table.

    Inactive (unused) props are parked at unique, well-separated lattice
    positions derived from the pool's per-object bounding boxes (see
    :class:`ClutteredTableManager` for the parking-grid rationale).

    Args:
        world: :class:`GenesisWorld` instance.
        name: Node name.
        object_pool: Descriptors (GSO/YCB or any :class:`MeshObject`) to
            load as visual entities.
        n_objects_range: ``(min, max)`` number of *active* floating props
            per environment per reset (default ``(0, 3)``).
        table_size: Placement region ``(width_x, depth_y)`` in metres.
        table_pos: Table surface centre ``(x, y, z)``.
        table_yaw: Rotation of the table around the Z axis in radians.
        z_offset_range: ``(min, max)`` metres *above the table surface*
            the active prop bbox centre is placed.  Default ``(0.15,
            0.65)``.
        xy_margin: Margin (m) kept between prop centres and the table
            edge along X / Y.  Default ``0.05``.
        min_separation: Minimum 3D centre-to-centre separation (m)
            between any two active floating props *within the same
            environment*.  Default ``0.10``.
        max_placement_attempts: Per-slot rejection-sampling budget.
        random_quaternion: Whether active props receive a random SO(3)
            orientation; if ``False`` the identity quaternion is used
            (still no dynamics — fixed=True).
        park_clearance: Extra centre-to-centre gap (m) added on top of
            the ``2 * max_footprint_radius`` minimum pairwise park-grid
            spacing.  Must be non-negative.

    Raises:
        ValueError: If the upper bound of *n_objects_range* exceeds the
            pool size, or if *park_clearance* is negative or non-finite.
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    # After table (100) / robot (50); after physical clutter (25) and
    # before camera/sensors (0 / -50).  Higher priority executes first,
    # so priority 20 runs after the priority-25 physical clutter and
    # still before the priority-0 camera — the renderer therefore picks
    # up the per-env pose at scene build time.
    reload_priorities = {20}
    after_reload_priorities = {20}
    reset_priorities = {20}

    def __init__(
        self,
        world: "GenesisWorld",
        name: str = "float_props",
        object_pool: Optional[List[ManipulatableObject]] = None,
        n_objects_range: Tuple[int, int] = (0, 3),
        table_size: Tuple[float, float] = (1.0, 0.6),
        table_pos: Tuple[float, float, float] = (0.0, 0.0, 0.65),
        table_yaw: float = 0.0,
        z_offset_range: Tuple[float, float] = (0.15, 0.65),
        xy_margin: float = 0.05,
        min_separation: float = 0.10,
        max_placement_attempts: int = 100,
        random_quaternion: bool = True,
        park_clearance: float = _DEFAULT_PARK_CLEARANCE,
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
        self.z_offset_range = z_offset_range
        self.xy_margin = float(xy_margin)
        self.min_separation = float(min_separation)
        self.max_placement_attempts = int(max_placement_attempts)
        self.random_quaternion = random_quaternion
        self.park_clearance = float(park_clearance)

        if not math.isfinite(self.park_clearance) or self.park_clearance < 0.0:
            raise ValueError(
                f"park_clearance must be a non-negative finite float, "
                f"got {self.park_clearance}"
            )
        zmin, zmax = self.z_offset_range
        if not (math.isfinite(zmin) and math.isfinite(zmax)):
            raise ValueError(
                "z_offset_range must contain finite floats"
            )
        if not (zmin <= zmax):
            raise ValueError(
                "z_offset_range must satisfy min <= max"
            )
        if not math.isfinite(self.xy_margin) or self.xy_margin < 0.0:
            raise ValueError(
                "xy_margin must be a non-negative finite float (m)"
            )
        if not math.isfinite(self.min_separation) or self.min_separation < 0.0:
            raise ValueError(
                "min_separation must be a non-negative finite float (m)"
            )

        if self.object_pool and n_objects_range[1] > len(self.object_pool):
            raise ValueError(
                f"n_objects_range upper bound ({n_objects_range[1]}) exceeds "
                f"object_pool size ({len(self.object_pool)})"
            )

        # RNG state — reseeded when an explicit seed is provided.
        self.rng: torch.Generator = torch.Generator(device=world.device)
        self.rng.seed()

        # Populated during reload().
        self._loaded_entities: List = []
        self._loaded_objects: List[ManipulatableObject] = []
        self._bboxes: List[BoxSpace] = []

        # Pre-computed bbox tensors and the parking lattice (cached on
        # the first reload; reused on subsequent reloads of the same
        # pool identity signature — see :meth:`_pool_signature`).
        self._dims_t: Optional[torch.Tensor] = None          # (n_pool, 3)
        self._centers_t: Optional[torch.Tensor] = None       # (n_pool, 3)
        self._footprint_r_t: Optional[torch.Tensor] = None   # (n_pool,)
        self._bound_r_t: Optional[torch.Tensor] = None       # (n_pool,)  3D
        self._max_footprint_radius: float = 0.0
        self._max_bound_radius: float = 0.0
        self._park_positions_t: Optional[torch.Tensor] = None  # (n_pool, 3) entity-origin
        self._park_quats_t: Optional[torch.Tensor] = None  # (n_pool, 4)
        self._pool_signature_cached: Optional[tuple] = None

        # Per-environment active-object mask, set after the first reset.
        # Unbatched: (n_pool,) bool.  Batched: (n_pool, n_envs) bool.
        self._active_mask_t: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def active_objects(self) -> List[ManipulatableObject]:
        """Descriptors for the active floating props in env 0."""
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
        """Genesis entity references for the active props in env 0."""
        if self._active_mask_t is None:
            return []
        mask = (
            self._active_mask_t
            if self._active_mask_t.ndim == 1
            else self._active_mask_t[:, 0]
        )
        return [self._loaded_entities[i] for i in torch.where(mask)[0].tolist()]

    @property
    def active_mask(self) -> Optional[torch.Tensor]:
        """The current active-object mask (``None`` until the first reset).

        Shape ``(n_pool,)`` (unbatched) or ``(n_pool, n_envs)`` (batched).
        """
        return self._active_mask_t

    # ------------------------------------------------------------------
    # RNG
    # ------------------------------------------------------------------

    def _reseed(self, seed: int) -> None:
        self.rng.manual_seed(seed)
    # ------------------------------------------------------------------
    # Geometry cache
    # ------------------------------------------------------------------

    def _pool_signature(self) -> tuple:
        """Identity signature (tuple of descriptor ``id()`` integers)."""
        return tuple(id(obj_desc) for obj_desc in self.object_pool)

    def _ensure_geometry_cache(self) -> None:
        """Populate the bbox / footprint / 3D bounding radius / parking cache.

        On the first ``reload()`` (before any ``load_into_scene`` call)
        walk the pool, query each descriptor's no-arg
        :meth:`~ManipulatableObject.get_local_bounding_box` and
        :meth:`~ManipulatableObject.get_footprint_radius`, derive the
        stacked ``dims`` / ``centers`` / ``footprint_r`` tensors and the
        per-object 3D circumscribed ``bound_r`` (= half the local AABB
        diagonal; used for 3D sphere overlap rejection among active
        props), plus the deterministic per-object entity-origin parking
        locations.
        """
        current_sig = self._pool_signature()
        if (
            self._park_positions_t is not None
            and self._pool_signature_cached == current_sig
        ):
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
            self._bound_r_t = torch.zeros(0, dtype=dtype, device=device)
            self._max_footprint_radius = 0.0
            self._max_bound_radius = 0.0
            self._park_positions_t = torch.zeros(0, 3, dtype=dtype, device=device)
            self._park_quats_t = torch.tensor(
                [_PARK_QUAT], dtype=dtype, device=device,
            )[:0]  # (0, 4)
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
        # 3D circumscribed radius per local AABB (half-diagonal).  Used as
        # a conservative orientation-independent upper bound on the
        # prop's spatial extent for 3D sphere overlap rejection between
        # active floating props.
        self._bound_r_t = torch.linalg.norm(self._dims_t, dim=-1) / 2.0
        self._max_footprint_radius = float(self._footprint_r_t.max().item())
        self._max_bound_radius = float(self._bound_r_t.max().item())

        # Square-ish 2-D lattice spaced by 2 * max_footprint_radius +
        # park_clearance (matching the physical-clutter spacing).  Even
        # though floating props do not collide, the per-object separation
        # is required so parked entities do not visually stack at the
        # off-camera parking altitude and so the broad-phase / culling
        # machinery stays bounded.
        spacing = 2.0 * self._max_footprint_radius + self.park_clearance
        n_cols = max(1, int(math.ceil(math.sqrt(n_pool))))
        idx = torch.arange(n_pool, dtype=dtype, device=device)
        cols = (idx % n_cols) * spacing
        rows = torch.div(idx, n_cols, rounding_mode="floor") * spacing
        park_centers = torch.stack(
            [cols, rows, torch.full_like(cols, float(_PARK_POS[2]))],
            dim=-1,
        )  # (n_pool, 3)
        # Compensate each object's local bbox centre so the AABB centre
        # lands exactly on the lattice target (handles off-centre origins
        # which are common in GSO/YCB meshes).
        self._park_positions_t = park_centers - self._centers_t  # (n_pool, 3)
        self._park_quats_t = torch.tensor(
            [_PARK_QUAT] * n_pool, dtype=dtype, device=device,
        )  # (n_pool, 4)

        self._pool_signature_cached = current_sig

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority != 20:
            return
        if seed is not None:
            self._reseed(seed)

        batch_fixed_verts = self.world.batch_size is not None

        self._ensure_geometry_cache()

        self._loaded_entities = []
        self._loaded_objects = []
        self._active_mask_t = None

        for i, obj_desc in enumerate(self.object_pool):
            park_pos = self._park_positions_t[i]
            # Ensure Hub assets are present *before* the morph reads the
            # file path.  Use the small public helper rather than the
            # private ``_ensure_assets`` underscore API.
            if hasattr(obj_desc, "ensure_assets"):
                obj_desc.ensure_assets()
            entity = self._load_visual_entity(
                obj_desc,
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
        if priority != 20:
            return
        # After a fresh reload every environment needs placement.
        self.reset(priority=20, seed=None, mask=None, **kwargs)

    # ------------------------------------------------------------------
    # Entity loading — raw visual mesh, optional textured surface
    # ------------------------------------------------------------------

    def _load_visual_entity(
        self,
        obj_desc: ManipulatableObject,
        *,
        world: "GenesisWorld" = None,
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        name_override: Optional[str] = None,
        batch_fixed_verts: bool = False,
    ):
        """Load a rigid **visual mesh** entity from ``obj_desc.mesh_file``.

        Uses ``gs.morphs.Mesh(file=..., scale=..., fixed=True,
        collision=False, visualization=True, batch_fixed_verts=...)``
        directly.  The surface is chosen so the polygon's intrinsic
        texture survives the loss of its native URDF/MJCF material
        binding:

        * If ``obj_desc.visual_texture_path`` is set **and** the file
          exists on disk, a textured :class:`genesis.surfaces.Plastic`
          surface is built from a :class:`genesis.textures.ImageTexture`
          bound as the ``diffuse_texture``.  This is the GSO/YCB path —
          the texture the MJCF/URDF would normally bind (GSO
          ``texture.png`` / YCB ``texture_map.png``) is re-attached to
          the bare visual mesh so the :class:`FloatingVisualPropsManager`
          can render the asset correctly without going through the
          URDF/MJCF loaders (whose texture overrides the Genesis
          ``BatchRenderer`` cannot read).
        * Otherwise ``surface`` is left ``None`` so the Genesis default
          surface is used — a generic :class:`MeshObject` with no
          explicit texture falls back to the default appearance.

        ``decimate=False`` keeps the visual mesh's full surface detail
        (we are not using the mesh for physics collision).

        Args:
            obj_desc: A :class:`MeshObject`-compatible descriptor
                (``GSOObject`` / ``YCBObject`` / :class:`MeshObject`).
            world: World to add the entity into; defaults to ``self.world``.
            pos, quat: Initial entity pose.
            name_override: Override the default ``self.name``.
            batch_fixed_verts: Enable per-env position overrides for
                fixed entities in batched worlds.

        Returns:
            The added Genesis entity.
        """
        import genesis as gs

        if world is None:
            world = self.world

        # ``obj_desc.scale`` is normalised to a per-axis tuple on the
        # MeshObject base class (see mesh_object.MeshObject.__init__).
        # ``gs.morphs.Mesh`` accepts both scalar and (sx, sy, sz) tuples.
        scale = getattr(obj_desc, "scale", 1.0)

        morph = gs.morphs.Mesh(
            file=str(obj_desc.mesh_file),
            scale=scale,
            pos=pos,
            quat=quat,
            fixed=True,
            collision=False,
            visualization=True,
            batch_fixed_verts=batch_fixed_verts,
            decimate=False,
        )

        surface = self._build_visual_surface(obj_desc)
        return world.scene.add_entity(
            morph,
            surface=surface,
            name=name_override or obj_desc.name,
        )

    @staticmethod
    def _build_visual_surface(obj_desc: ManipulatableObject):
        """Return a textured Plastic surface for textured descriptors.

        Looks up ``obj_desc.visual_texture_path`` (set by
        :class:`GSOObject` / :class:`YCBObject` to their asset-directory
        texture file).  When the path is set and points at a real file,
        a :class:`genesis.surfaces.Plastic` surface with a
        ``diffuse_texture`` :class:`genesis.textures.ImageTexture` is
        returned so the asset's intrinsic texture is preserved while
        still being a rigid-compatible surface the BatchRenderer can
        read.  Otherwise ``None`` is returned and the Genesis default
        surface is used.
        """
        import genesis as gs

        tex_path = getattr(obj_desc, "visual_texture_path", None)
        if tex_path is None:
            return None
        try:
            from pathlib import Path

            if not Path(tex_path).is_file():
                return None
        except (OSError, ValueError):
            return None
        return gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(image_path=tex_path),
        )

    # ------------------------------------------------------------------
    # Reset / placement computation
    # ------------------------------------------------------------------

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority != 20:
            return
        if not self._loaded_entities:
            return

        if seed is not None:
            self._reseed(seed)

        n_envs = self.world.batch_size
        n_pool = len(self.object_pool)

        if n_pool == 0:
            return

        if n_envs is None:
            # Unbatched — single environment (mask ignored).
            positions, quats, active = self._compute_placement_vectorized(1)
            self._active_mask_t = active.squeeze(1)  # (n_pool,)
            self._apply_placements(
                positions.squeeze(1), quats.squeeze(1),
                active.squeeze(1), envs_idx=None,
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

            if self._active_mask_t is None:
                self._active_mask_t = torch.zeros(
                    n_pool, n_envs, dtype=torch.bool, device=self.world.device,
                )

            positions, quats, active = self._compute_placement_vectorized(n_reset)

            self._active_mask_t[:, reset_idx] = active

            envs_idx = reset_idx if mask is not None else None
            self._apply_placements(positions, quats, active, envs_idx=envs_idx)

    def get_observation(self):
        return None

    # ------------------------------------------------------------------
    # Vectorised placement computation
    # ------------------------------------------------------------------

    def _compute_placement_vectorized(
        self,
        n_envs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 3D placements for the active floating props in *n_envs*
        environments in parallel.

        Active props are sampled within the randomized table-footprint XY
        bounds (margin-shrunk, **table-local frame**) at world z =
        ``table_pos[2] + z_offset``, with a uniformly random SO(3)
        orientation when ``random_quaternion=True``.  A 3D *sphere*
        rejection test (using the per-prop bounding radius plus
        ``min_separation``) keeps active props apart within each
        environment; it runs in the table-local frame (a Z-axis yaw
        preserves pairwise distances) and the local→world rotation +
        translation is applied to the bbox-centre target before the
        bbox-centre compensation so both terms share the world frame.
        Inactive props keep their per-object parking pose.

        Args:
            n_envs: Number of environments to compute for.

        Returns:
            positions:   ``(n_pool, n_envs, 3)`` world positions.
            quats:       ``(n_pool, n_envs, 4)`` world quaternions.
            active_mask: ``(n_pool, n_envs)`` — *True* where the prop is
                         floating around the table in that environment.
        """
        from xbarray.transformations.rotation_conversions.pytorch import (
            quaternion_apply,
        )

        n_pool = len(self.object_pool)
        device = self.world.device
        dtype = self.world.dtype

        # Start every prop at its unique parked entity-origin pose,
        # broadcast across the reset scope.
        positions = self._park_positions_t.unsqueeze(1).expand(
            n_pool, n_envs, 3,
        ).contiguous()
        quats = self._park_quats_t.unsqueeze(1).expand(
            n_pool, n_envs, 4,
        ).contiguous()
        active_mask = torch.zeros(
            n_pool, n_envs, dtype=torch.bool, device=device,
        )

        if n_pool == 0:
            return positions, quats, active_mask

        # --- Random active counts and permutations ---------------------
        n_objs = torch.randint(
            self.n_objects_range[0],
            self.n_objects_range[1] + 1,
            (n_envs,),
            generator=self.rng,
            device=device,
        )
        max_n = int(n_objs.max().item())

        if max_n == 0:
            return positions, quats, active_mask

        # Random permutations via argsort of uniform noise.
        noise = torch.rand(
            n_envs, n_pool, generator=self.rng, dtype=dtype, device=device,
        )
        perms = noise.argsort(dim=1)  # (n_envs, n_pool)

        # --- Table geometry.  Sampling + rejection stay in the
        #     table-local frame (yaw about Z preserves the pairwise
        #     distances the 3D overlap test relies on); the local→world
        #     rotation + translation is applied to the bbox-centre target
        #     when composing the entity-origin position (see below). -----
        table_half_x = self.table_size[0] / 2.0 - self.xy_margin
        table_half_y = self.table_size[1] / 2.0 - self.xy_margin
        table_half = torch.tensor(
            [table_half_x, table_half_y], dtype=dtype, device=device,
        )
        table_pos = torch.tensor(self.table_pos, dtype=dtype, device=device)
        zmin, zmax = self.z_offset_range

        # --- Footprint / bound tracking for 3D sphere overlap ----------
        placed_xyz = torch.zeros(max_n, n_envs, 3, dtype=dtype, device=device)
        placed_r = torch.zeros(max_n, n_envs, dtype=dtype, device=device)
        placed_valid = torch.zeros(
            max_n, n_envs, dtype=torch.bool, device=device,
        )
        n_placed = 0

        for k in range(max_n):
            slot_active = k < n_objs  # (n_envs,)
            if not slot_active.any():
                break

            obj_idx = perms[:, k]  # (n_envs,)

            obj_radius = self._bound_r_t[obj_idx]            # (n_envs,)
            obj_centers = self._centers_t[obj_idx]           # (n_envs, 3)

            # Active prop XY targets are sampled within the
            # margin-shrunk table footprint; we ignore each object's
            # individual half-extent along XY (we are placing around/above
            # the table, not on it, so edge clearance is not geometrically
            # required — the z_offset_range already lifts props clear).
            valid = slot_active & (table_half[0] > 0) & (table_half[1] > 0)
            if not valid.any():
                n_placed += 1
                continue

            placed = torch.zeros(n_envs, dtype=torch.bool, device=device)
            best_xyz = torch.zeros(n_envs, 3, dtype=dtype, device=device)

            for _att in range(self.max_placement_attempts):
                need = valid & ~placed
                if not need.any():
                    break

                n_need = int(need.sum().item())
                # Uniform in [-1, 1] then scaled to the table half-extent.
                raw_xy = (
                    torch.rand(n_need, 2, generator=self.rng, dtype=dtype, device=device) * 2 - 1
                )
                raw_z = (
                    torch.rand(n_need, generator=self.rng, dtype=dtype, device=device)
                    * (zmax - zmin) + zmin
                )
                cand_xy = torch.zeros(n_envs, 2, dtype=dtype, device=device)
                cand_z = torch.zeros(n_envs, dtype=dtype, device=device)
                # Scale per-env by the table half-extent broadcast over XY.
                cand_xy[need] = raw_xy * table_half.unsqueeze(0).expand(n_need, 2)
                cand_z[need] = raw_z

                # ``cand_xy`` is sampled **table-local** (axes aligned
                # with the rotated table footprint); ``cand_z`` is a height
                # above the table surface.  We compose the candidate as
                # ``(local_x, local_y, table_z + z_offset)`` so the 3D
                # sphere-overlap rejection below stays in a single
                # consistent frame — a Z-axis yaw plus an XY translation
                # preserve pairwise distances and therefore leave the
                # overlap test invariant.  The local→world transform is
                # applied at the very end (see ``per_env_target_world``)
                # exactly as :class:`ClutteredTableManager` does, so the
                # bbox-centre compensation stays world-frame consistent.
                cand_xyz = torch.stack(
                    [
                        cand_xy[:, 0], cand_xy[:, 1],
                        table_pos[2] + cand_z,
                    ],
                    dim=-1,
                )  # (n_envs, 3) table-local XY / world Z

                # 3D sphere overlap against already-placed active props.
                overlap = torch.zeros(n_envs, dtype=torch.bool, device=device)
                if n_placed > 0:
                    diff = placed_xyz[:n_placed] - cand_xyz.unsqueeze(0)
                    dist = torch.linalg.norm(diff, dim=-1)  # (n_placed, n_envs)
                    min_d = (
                        obj_radius.unsqueeze(0)
                        + placed_r[:n_placed]
                        + self.min_separation
                    )
                    overlap = (
                        (dist < min_d) & placed_valid[:n_placed]
                    ).any(dim=0)

                newly_placed = need & ~overlap
                placed = placed | newly_placed
                best_xyz[newly_placed] = cand_xyz[newly_placed]

            placed_xyz[n_placed] = best_xyz
            placed_r[n_placed] = obj_radius
            placed_valid[n_placed] = placed & valid
            n_placed += 1

            success = placed & valid
            if not success.any():
                continue

            # Convert the *bbox-centre* target back into an
            # *entity-origin* target by compensating the local bbox
            # centre rotated into the world frame by this slot's random
            # quaternion.  We draw one quaternion per (env, slot) so the
            # compensation is orientation-consistent.  Batched draws on
            # the world's RNG stay device-local on GPU backends.
            n_quat = int(success.sum().item())
            if self.random_quaternion:
                slot_quats_local = _shoemake_quaternions(
                    self.rng, n_quat, dtype=dtype, device=device,
                )  # (n_quat, 4)  (w, x, y, z)
            else:
                slot_quats_local = torch.tensor(
                    [_PARK_QUAT] * n_quat, dtype=dtype, device=device,
                )

            # Build local copies of best_xyz / success indexing.
            filled_envs = torch.where(success)[0]              # (n_quat,)
            filled_centers = obj_centers[success]               # (n_quat, 3)  local
            per_env_target = best_xyz[success]                  # (n_quat, 3)  table-local
            per_env_pool = obj_idx[success]                     # (n_quat,)

            # Forward-rotate the **table-local** bbox-centre target into
            # the world frame — yaw about Z then translate by
            # ``table_pos`` — matching :class:`ClutteredTableManager`'s
            # ``wx/wy/wz`` compose.  ``best_xyz`` stored table-local XY
            # plus a world-Z altitude; the Z component is invariant
            # under the Z-axis yaw, so only XY need rotating.  This must
            # happen *before* the bbox-centre compensation so both
            # terms of ``per_env_target_world − self_local_rot`` live in
            # the same (world) frame.
            c_y, s_y = self._cos_yaw, self._sin_yaw
            local_x = per_env_target[:, 0]
            local_y = per_env_target[:, 1]
            world_x = table_pos[0] + local_x * c_y - local_y * s_y
            world_y = table_pos[1] + local_x * s_y + local_y * c_y
            world_z = per_env_target[:, 2]
            per_env_target_world = torch.stack(
                [world_x, world_y, world_z], dim=-1,
            )  # (n_quat, 3) world-frame bbox-centre target

            # Compose the *entity-origin* target: the sampled target is a
            # *bbox-centre* target; the entity's origin is offset from the
            # bbox centre by the (rotation-frame) local bbox centre.  We
            # rotate the local bbox centre into the world frame by this
            # slot's quaternion and subtract it so the world-frame AABB
            # centre lands exactly at the sampled target.
            self_local_rot = quaternion_apply(
                slot_quats_local, filled_centers,
            )  # (n_quat, 3)  world-rotated local bbox centre
            entity_origin_target = per_env_target_world - self_local_rot

            # Scatter-write.
            positions[per_env_pool, filled_envs, 0] = entity_origin_target[:, 0]
            positions[per_env_pool, filled_envs, 1] = entity_origin_target[:, 1]
            positions[per_env_pool, filled_envs, 2] = entity_origin_target[:, 2]
            quats[per_env_pool, filled_envs] = slot_quats_local
            active_mask[per_env_pool, filled_envs] = True

        return positions, quats, active_mask

    # ------------------------------------------------------------------
    # Apply placements to Genesis entities
    # ------------------------------------------------------------------

    def _apply_placements(
        self,
        positions: torch.Tensor,
        quats: torch.Tensor,
        active_mask: torch.Tensor,
        envs_idx: Optional[torch.Tensor],
    ) -> None:
        """Set entity positions and orientations in an active-mask-aware way.

        Active objects receive the per-env world-frame pose computed by
        :meth:`_compute_placement_vectorized`.  Inactive objects are
        pinned to their unique cached parked pose (fixed ``_PARK_QUAT``)
        with no RNG consumption, regardless of ``random_quaternion``.

        Args:
            positions: ``(n_pool, n_envs, 3)`` (batched / masked) or
                ``(n_pool, 3)`` (unbatched).
            quats: ``(n_pool, n_envs, 4)`` or ``(n_pool, 4)``.
            active_mask: ``(n_pool, n_envs)`` or ``(n_pool,)`` bool
                tensor matching ``positions`` — *True* where the prop is
                active, *False* where it stays parked.
            envs_idx: ``None`` for full writes (all envs or unbatched).
                ``(n_reset,)`` LongTensor for partial (masked) updates —
                ``positions`` / ``quats`` are then ``(n_pool, n_reset,
                *)`` and only the listed environments are overwritten.
        """
        if positions.ndim == 2:
            # Unbatched — single environment.
            park_q = torch.tensor(
                _PARK_QUAT, dtype=quats.dtype, device=quats.device,
            )
            for i, entity in enumerate(self._loaded_entities):
                if bool(active_mask[i]):
                    entity.set_pos(positions[i].to(entity.get_pos().dtype))
                    entity.set_quat(quats[i].to(entity.get_quat().dtype))
                else:
                    park_pos = self._park_positions_t[i].to(
                        dtype=positions.dtype, device=positions.device,
                    )
                    entity.set_pos(park_pos)
                    entity.set_quat(park_q)
            return

        # Batched (full reset or masked reset sub-scope).
        n_reset = positions.shape[1]
        device = positions.device
        dtype = positions.dtype
        park_quat = torch.tensor(_PARK_QUAT, dtype=dtype, device=device)

        for i, entity in enumerate(self._loaded_entities):
            pos_i = positions[i]        # (n_reset, 3)
            quat_i = quats[i]           # (n_reset, 4)
            act_i = active_mask[i]      # (n_reset,) bool
            active_idx = torch.where(act_i)[0]
            inactive_idx = torch.where(~act_i)[0]

            if envs_idx is None:
                global_active_idx = active_idx
                global_inactive_idx = inactive_idx
            else:
                global_active_idx = envs_idx[active_idx]
                global_inactive_idx = envs_idx[inactive_idx]

            if active_idx.numel() > 0 or inactive_idx.numel() > 0:
                cur_pos = entity.get_pos().clone()
                cur_quat = entity.get_quat().clone()
                if active_idx.numel() > 0:
                    cur_pos[global_active_idx] = pos_i[active_idx].to(cur_pos.dtype)
                    cur_quat[global_active_idx] = quat_i[active_idx].to(cur_quat.dtype)
                if inactive_idx.numel() > 0:
                    n_inactive = inactive_idx.numel()
                    park_pos = self._park_positions_t[i].to(
                        dtype=dtype, device=device,
                    )
                    cur_pos[global_inactive_idx] = park_pos.unsqueeze(0).expand(
                        n_inactive, 3,
                    ).to(cur_pos.dtype)
                    cur_quat[global_inactive_idx] = park_quat.unsqueeze(0).expand(
                        n_inactive, 4,
                    ).to(cur_quat.dtype)
                entity.set_pos(cur_pos)
                entity.set_quat(cur_quat)
