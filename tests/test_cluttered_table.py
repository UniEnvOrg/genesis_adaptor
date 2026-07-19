"""Tests for ClutteredTableManager with primitive shapes.

Covers:
  - __init__ validation: n_objects_range upper bound vs pool size
  - Unbatched: objects placed on table, inactive parked, correct active counts
  - Seed reproducibility (same seed -> same layout)
  - avoid_spaces exclusion zone respected
  - Batched: per-environment independent placement
  - Batched masked reset: only flagged envs re-randomised, others preserved
  - Per-env avoid_spaces via reset() kwargs
"""

import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.cached_table_builder import CachedTableBuilder
from unienv_genesis_collection.scenes.cluttered_table_manager import (
    ClutteredTableManager,
    _PARK_POS,
    _PARK_QUAT,
)
from unienv_genesis_collection.objects import (
    BoxObject,
    SphereObject,
    CylinderObject,
)
from unienv_genesis_collection.objects.base_object import make_bbox
from unienv_interface.world import WorldEnv

_TABLE_POS = (0.0, 0.0, 0.65)
_TABLE_SIZE = (0.8, 0.5)
_PARK_Z = _PARK_POS[2]

# Shared object pool for tests
def _make_pool():
    return [
        BoxObject(size=(0.04, 0.04, 0.04), name="cube_r", color=(1, 0, 0, 1)),
        BoxObject(size=(0.05, 0.03, 0.06), name="brick_g", color=(0, 1, 0, 1)),
        SphereObject(radius=0.025, name="ball_b", color=(0, 0, 1, 1)),
        CylinderObject(radius=0.02, height=0.08, name="can_y", color=(1, 1, 0, 1)),
        BoxObject(size=(0.03, 0.03, 0.03), name="cube_w", color=(1, 1, 1, 1)),
        SphereObject(radius=0.02, name="ball_m", color=(1, 0, 1, 1)),
    ]


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def test_pool_size_validation():
    """n_objects_range upper bound must not exceed pool size."""
    world = GenesisWorld()
    small_pool = [BoxObject(size=(0.04, 0.04, 0.04), name="cube")]

    try:
        ClutteredTableManager(
            world,
            object_pool=small_pool,
            n_objects_range=(1, 5),  # 5 > len(pool)==1
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "5" in str(e) and "1" in str(e)
        print(f"  PASS: validation caught bad range — {e}")

    # Equal is fine
    mgr = ClutteredTableManager(
        world, object_pool=small_pool, n_objects_range=(1, 1),
    )
    assert mgr is not None
    print("  PASS: n_objects_range[1] == pool size accepted")


# ------------------------------------------------------------------
# Unbatched
# ------------------------------------------------------------------

def test_unbatched_placement():
    """Unbatched: objects placed on table surface, inactive parked."""
    world = GenesisWorld()
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=pool,
        n_objects_range=(3, 5),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)

    n_pool = len(pool)
    n_active = len(clutter.active_objects)
    n_active_entities = len(clutter.active_entities)

    assert 3 <= n_active <= 5, f"Expected 3-5 active objects, got {n_active}"
    assert n_active == n_active_entities

    # Active mask shape
    assert clutter._active_mask_t is not None
    assert clutter._active_mask_t.shape == (n_pool,)
    assert clutter._active_mask_t.sum().item() == n_active

    print(f"  PASS: {n_active} active objects out of {n_pool}")

    # Active objects should be near the table surface (z > 0.5)
    # Inactive objects should be parked (z == -1000)
    for i, entity in enumerate(clutter._loaded_entities):
        pos = entity.get_pos()
        if clutter._active_mask_t[i]:
            assert pos[2].item() > 0.5, \
                f"Active object {i} at z={pos[2]:.1f}, expected near table"
            # Check XY within table bounds (with margin)
            dx = abs(pos[0].item() - _TABLE_POS[0])
            dy = abs(pos[1].item() - _TABLE_POS[1])
            assert dx <= _TABLE_SIZE[0] / 2.0, \
                f"Object {i} x={pos[0]:.3f} outside table"
            assert dy <= _TABLE_SIZE[1] / 2.0, \
                f"Object {i} y={pos[1]:.3f} outside table"
        else:
            assert pos[2].item() == _PARK_Z, \
                f"Inactive object {i} at z={pos[2]:.1f}, expected {_PARK_Z}"

    print("  PASS: active objects on table, inactive parked")

    env.close()


def test_seed_reproducibility():
    """Same seed produces identical placement."""
    world = GenesisWorld()
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=pool,
        n_objects_range=(3, 5),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)

    # Collect positions with seed=42
    env.reset(seed=42)
    positions_a = [e.get_pos().clone() for e in clutter._loaded_entities]
    mask_a = clutter._active_mask_t.clone()

    # Reset with same seed
    env.reset(seed=42)
    positions_b = [e.get_pos().clone() for e in clutter._loaded_entities]
    mask_b = clutter._active_mask_t.clone()

    assert torch.equal(mask_a, mask_b), "Same seed gave different active masks"
    for i, (pa, pb) in enumerate(zip(positions_a, positions_b)):
        assert torch.allclose(pa, pb), \
            f"Object {i}: pos differs between same-seed resets"

    print("  PASS: same seed -> identical placement")

    env.close()


def test_avoid_spaces():
    """Objects avoid the centre exclusion zone."""
    world = GenesisWorld()
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)

    avoid_radius = 0.15
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=pool,
        n_objects_range=(4, 6),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
        avoid_spaces=[((_TABLE_POS[0], _TABLE_POS[1], _TABLE_POS[2]), avoid_radius)],
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)

    for seed in range(10):
        env.reset(seed=seed, reload=(seed == 0))
        for i, entity in enumerate(clutter._loaded_entities):
            if not clutter._active_mask_t[i]:
                continue
            pos = entity.get_pos()
            dist_xy = torch.sqrt(
                (pos[0] - _TABLE_POS[0]) ** 2 + (pos[1] - _TABLE_POS[1]) ** 2
            ).item()
            fp_r = clutter._footprint_r_t[i].item()
            # The object centre + its footprint should not overlap the avoid zone
            # Allow a small tolerance for floating point
            assert dist_xy + 0.001 >= avoid_radius - fp_r, (
                f"Seed {seed}, object {i}: dist_xy={dist_xy:.4f}, "
                f"fp_r={fp_r:.4f}, avoid_r={avoid_radius}"
            )

    print("  PASS: avoid_spaces exclusion zone respected across 10 seeds")

    env.close()



# ------------------------------------------------------------------
# Batched
# ------------------------------------------------------------------

def test_batched_placement():
    """Batched: per-environment independent placement."""
    N_ENVS = 8
    world = GenesisWorld(batch_size=N_ENVS)
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=pool,
        n_objects_range=(2, 4),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)

    n_pool = len(pool)

    # Active mask shape
    assert clutter._active_mask_t.shape == (n_pool, N_ENVS)

    # Per-env active counts should be in [2, 4]
    for env_j in range(N_ENVS):
        n_act = clutter._active_mask_t[:, env_j].sum().item()
        assert 2 <= n_act <= 4, \
            f"Env {env_j}: {n_act} active objects, expected 2-4"

    print(f"  PASS: all {N_ENVS} envs have 2-4 active objects")

    # Environments should (very likely) have different layouts
    masks = [
        clutter._active_mask_t[:, j].tolist() for j in range(N_ENVS)
    ]
    unique = len(set(tuple(m) for m in masks))
    assert unique > 1, "All envs have identical active masks — no randomisation"
    print(f"  PASS: {unique} distinct layouts across {N_ENVS} environments")

    # Active objects on table, inactive parked — check per env
    for i, entity in enumerate(clutter._loaded_entities):
        pos = entity.get_pos()  # (N_ENVS, 3)
        assert pos.shape == (N_ENVS, 3)
        for env_j in range(N_ENVS):
            if clutter._active_mask_t[i, env_j]:
                assert pos[env_j, 2].item() > 0.5, \
                    f"Active object {i} env {env_j} at z={pos[env_j, 2]:.1f}"
            else:
                assert pos[env_j, 2].item() == _PARK_Z, \
                    f"Inactive object {i} env {env_j} at z={pos[env_j, 2]:.1f}"

    print("  PASS: batched positions correct (active on table, inactive parked)")

    env.close()


def test_batched_masked_reset():
    """Masked reset: only flagged envs change, others preserved."""
    N_ENVS = 8
    world = GenesisWorld(batch_size=N_ENVS)
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=pool,
        n_objects_range=(2, 4),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)

    # Capture state of all envs
    positions_before = [e.get_pos().clone() for e in clutter._loaded_entities]
    quats_before = [e.get_quat().clone() for e in clutter._loaded_entities]
    mask_before = clutter._active_mask_t.clone()

    # Masked reset: only envs 0, 3, 7
    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[0, 3, 7]] = True
    env.reset(seed=99, mask=reset_mask)

    preserved_envs = [1, 2, 4, 5, 6]

    # Preserved envs should have identical positions and quats
    for i, entity in enumerate(clutter._loaded_entities):
        pos_after = entity.get_pos()
        quat_after = entity.get_quat()
        for env_j in preserved_envs:
            assert torch.allclose(pos_after[env_j], positions_before[i][env_j], atol=1e-5), \
                f"Object {i} env {env_j}: position changed after masked reset"
            assert torch.allclose(quat_after[env_j], quats_before[i][env_j], atol=1e-5), \
                f"Object {i} env {env_j}: quaternion changed after masked reset"

    # Preserved envs should have identical active masks
    for env_j in preserved_envs:
        assert torch.equal(
            clutter._active_mask_t[:, env_j],
            mask_before[:, env_j],
        ), f"Env {env_j}: active mask changed after masked reset"

    print("  PASS: preserved envs unchanged after masked reset")

    # Reset envs should (very likely) differ from before
    changed = False
    for env_j in [0, 3, 7]:
        for i in range(len(pool)):
            if not torch.allclose(
                clutter._loaded_entities[i].get_pos()[env_j],
                positions_before[i][env_j],
                atol=1e-5,
            ):
                changed = True
                break
        if changed:
            break
    assert changed, "Reset envs did not change — masked reset may not be working"

    print("  PASS: reset envs changed after masked reset")

    env.close()


def test_batched_per_env_avoid_spaces():
    """Per-env avoid_spaces passed through reset() kwargs."""
    N_ENVS = 8
    world = GenesisWorld(batch_size=N_ENVS)
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=pool,
        n_objects_range=(2, 4),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)

    # Each env has a different avoid centre along the X axis
    avoid_centres = torch.zeros(N_ENVS, 1, 2)
    for j in range(N_ENVS):
        avoid_centres[j, 0, 0] = _TABLE_POS[0] + (j - N_ENVS / 2) * 0.05
        avoid_centres[j, 0, 1] = _TABLE_POS[1]
    avoid_radii = torch.full((N_ENVS, 1), 0.10)

    env.reset(
        seed=10,
        pernode_kwargs={"clutter": {"avoid_spaces": (avoid_centres, avoid_radii)}},
    )

    # Verify active objects respect per-env avoid zones
    for env_j in range(N_ENVS):
        ac = avoid_centres[env_j, 0]  # (2,)
        ar = avoid_radii[env_j, 0].item()
        for i, entity in enumerate(clutter._loaded_entities):
            if not clutter._active_mask_t[i, env_j]:
                continue
            pos = entity.get_pos()[env_j]  # (3,)
            dist = torch.sqrt(
                (pos[0] - ac[0]) ** 2 + (pos[1] - ac[1]) ** 2
            ).item()
            fp_r = clutter._footprint_r_t[i].item()
            assert dist + 0.001 >= ar - fp_r, (
                f"Env {env_j}, object {i}: dist={dist:.4f}, "
                f"fp_r={fp_r:.4f}, avoid_r={ar}"
            )

    print("  PASS: per-env avoid_spaces respected")

    env.close()


# ------------------------------------------------------------------
# Unique upstream parking — geometry cache, lattice, compensation,
# inactive / masked behaviour, fixed parked quaternion.
# ------------------------------------------------------------------

class _ShiftedBoxObject(BoxObject):
    """BoxObject variant whose local AABB is offset from the entity origin.

    Used to exercise off-centre bbox-origin compensation: the manager's
    parking entity-origin position must equal ``lattice_center − bbox_center``
    so the resulting AABB centre lands exactly on the lattice cell.
    """

    def __init__(self, size, center_offset, **kwargs):
        super().__init__(size=size, **kwargs)
        self._co = tuple(float(c) for c in center_offset)

    def get_local_bounding_box(self):
        from unienv_genesis_collection.objects.base_object import make_bbox
        import torch
        half = torch.tensor(self.size) / 2.0
        co = torch.tensor(self._co)
        return make_bbox(co - half, co + half)


def _expected_max_radius(pool):
    return max(o.get_footprint_radius() for o in pool)


def test_park_clearance_validation():
    """park_clearance must be a non-negative finite float."""
    world = GenesisWorld()
    pool = _make_pool()
    try:
        ClutteredTableManager(world, object_pool=pool, park_clearance=-0.01)
        assert False, "negative park_clearance should raise"
    except ValueError as e:
        assert "park_clearance" in str(e)
        print(f"  PASS: negative clearance rejected — {e}")

    try:
        ClutteredTableManager(world, object_pool=pool, park_clearance=float("nan"))
        assert False, "NaN park_clearance should raise"
    except ValueError as e:
        print(f"  PASS: NaN clearance rejected — {e}")

    mgr = ClutteredTableManager(
        world, object_pool=pool, n_objects_range=(1, len(pool)), park_clearance=0.1,
    )
    assert mgr.park_clearance == 0.1
    print("  PASS: valid park_clearance accepted")


def test_geometry_cache_radius_and_max():
    """Per-descriptor footprint radius cache and max over pool."""
    world = GenesisWorld()
    pool = _make_pool()
    mgr = ClutteredTableManager(
        world, object_pool=pool, n_objects_range=(1, len(pool)), park_clearance=0.0,
    )
    mgr._ensure_geometry_cache()

    expected_r = [o.get_footprint_radius() for o in pool]
    got_r = mgr._footprint_r_t.tolist()
    for exp, got in zip(expected_r, got_r):
        assert abs(got - exp) < 1e-6, (exp, got)
    assert abs(mgr._max_footprint_radius - _expected_max_radius(pool)) < 1e-6
    print("  PASS: footprint radius cache matches descriptors and max is correct")


def test_park_grid_spacing():
    """All pairwise lattice centre distances are >= 2*max_radius (+ clearance)."""
    world = GenesisWorld()
    pool = _make_pool()
    clearance = 0.05
    mgr = ClutteredTableManager(
        world, object_pool=pool, n_objects_range=(1, len(pool)), park_clearance=clearance,
    )
    mgr._ensure_geometry_cache()

    n = len(pool)
    centers = mgr._centers_t
    park_origin = mgr._park_positions_t
    # Reconstruct the parked AABB centres = entity_origin + center.
    parked_centers = park_origin + centers
    # XY spacing between distinct centres must be >= 2*max + clearance.
    min_sep = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(torch.linalg.norm(parked_centers[i, :2] - parked_centers[j, :2]).item())
            min_sep = min(min_sep, d)
    expected_min = 2.0 * mgr._max_footprint_radius + clearance
    assert min_sep >= expected_min - 1e-6, (min_sep, expected_min)
    # Bounding equality at the row/col grid step:
    # the smallest grid step equals exactly `spacing` (clearance aside).
    # z invariant:
    assert all(abs(float(parked_centers[i, 2].item()) - _PARK_Z) < 1e-6
               for i in range(n))
    print(f"  PASS: min park centre separation {min_sep:.4f} >= {expected_min:.4f}")


def test_park_offcenter_compensation():
    """Off-centre bbox origin: entity-origin park pos == lattice_center − bbox_center."""
    from unienv_genesis_collection.objects.base_object import bbox_center
    world = GenesisWorld()
    # Single object whose AABB centre is not at the entity origin.
    co = (0.1, -0.07, 0.02)
    obj = _ShiftedBoxObject(size=(0.04, 0.06, 0.08), center_offset=co, name="shifted")
    pool = [obj]
    mgr = ClutteredTableManager(
        world, object_pool=pool, n_objects_range=(0, 1), park_clearance=0.0,
    )
    mgr._ensure_geometry_cache()

    # Single object → cell 0 ⇒ lattice centre (0, 0, park_z).
    park_origin = mgr._park_positions_t[0]
    assert abs(float(park_origin[0].item()) - (0.0 - co[0])) < 1e-4
    assert abs(float(park_origin[1].item()) - (0.0 - co[1])) < 1e-4
    # float32 around -1000 has ~2e-5 quantisation; tolerate a coarser bound.
    assert abs(float(park_origin[2].item()) - (_PARK_Z - co[2])) < 1e-3
    # And the AABB centre ends on (0,0,_PARK_Z).
    center = bbox_center(pool[0].get_local_bounding_box())
    parked_center = park_origin + center
    assert abs(float(parked_center[2].item()) - _PARK_Z) < 1e-3
    print("  PASS: off-centre bbox-origin compensation lands AABB centre on lattice")


def test_geometry_cache_reuse():
    """Geometry cache reuse: an unchanged pool triggers *no* geometry queries
    (``get_local_bounding_box`` / ``get_footprint_radius``) on a second
    ``_ensure_geometry_cache`` call or ``reload``; replacing or reordering the
    pool invalidates the cache and rebuilds."""
    world = GenesisWorld()
    pool = _make_pool()

    # Per-descriptor call counters for the two geometry queries the manager
    # invokes directly via ``_ensure_geometry_cache``.  Wrapping the bound
    # methods on each instance lets us observe every call, including the base
    # ``get_footprint_radius``'s internal ``get_local_bounding_box`` call.
    counts = {"bbox": [], "radius": []}

    def _wrap(obj):
        idx = len(counts["bbox"])
        counts["bbox"].append(0)
        counts["radius"].append(0)
        orig_bbox = obj.get_local_bounding_box
        orig_radius = obj.get_footprint_radius

        def bbox_w(*a, **kw):
            counts["bbox"][idx] += 1
            return orig_bbox(*a, **kw)

        def radius_w(*a, **kw):
            counts["radius"][idx] += 1
            return orig_radius(*a, **kw)

        obj.get_local_bounding_box = bbox_w
        obj.get_footprint_radius = radius_w

    for obj in pool:
        _wrap(obj)

    mgr = ClutteredTableManager(
        world, object_pool=pool, n_objects_range=(1, len(pool)),
    )

    # --- First build: every descriptor is queried at least once. -----------
    mgr._ensure_geometry_cache()
    first_bbox = list(counts["bbox"])
    first_radius = list(counts["radius"])
    assert sum(first_bbox) >= len(pool), first_bbox
    assert sum(first_radius) >= len(pool), first_radius
    dims_ref = mgr._dims_t
    park_ref = mgr._park_positions_t

    # --- Second ensure on the identical pool: cache reused, no new calls. -
    mgr._ensure_geometry_cache()
    assert counts["bbox"] == first_bbox, counts["bbox"]
    assert counts["radius"] == first_radius, counts["radius"]
    assert mgr._dims_t is dims_ref
    assert mgr._park_positions_t is park_ref
    print("  PASS: unchanged pool reuses cache; 0 geometry queries on 2nd ensure")

    # --- Reordering the pool invalidates the signature → rebuild. ---------
    mgr.object_pool = [pool[2], pool[0], pool[1]] + pool[3:]
    mgr._ensure_geometry_cache()
    assert sum(counts["bbox"]) > sum(first_bbox), counts["bbox"]
    assert sum(counts["radius"]) > sum(first_radius), counts["radius"]
    print("  PASS: reordered pool invalidates cache and rebuilds")

    reorder_bbox = list(counts["bbox"])
    reorder_radius = list(counts["radius"])

    # --- Replacing a descriptor with a fresh instance (new id) → rebuild. -
    replacement = BoxObject(
        size=(0.04, 0.04, 0.04), name="cube_r", color=(1, 0, 0, 1),
    )
    _wrap(replacement)  # count its geometry queries too
    mgr.object_pool[3] = replacement
    mgr._ensure_geometry_cache()
    assert sum(counts["bbox"]) > sum(reorder_bbox), counts["bbox"]
    assert sum(counts["radius"]) > sum(reorder_radius), counts["radius"]
    print("  PASS: replaced descriptor invalidates cache and rebuilds")

    replaced_bbox = list(counts["bbox"])
    replaced_radius = list(counts["radius"])

    # --- Second reload of the (now stable) replaced pool: cache reused. ---
    mgr._ensure_geometry_cache()
    assert counts["bbox"] == replaced_bbox, counts["bbox"]
    assert counts["radius"] == replaced_radius, counts["radius"]
    print("  PASS: stable replaced pool reuses cache; 0 geometry queries")


def test_unique_initial_load_poses():
    """After reload, every pool object is loaded at a unique park XY position,
    all parked far below the table."""
    world = GenesisWorld()
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world, name="clutter", object_pool=pool,
        n_objects_range=(0, 0),  # no on-table placement for this check
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)

    # All inactive (range (0,0)) ⇒ every object parked.
    xs = []
    ys = []
    for entity in clutter._loaded_entities:
        pos = entity.get_pos()
        xs.append(float(pos[0].item())); ys.append(float(pos[1].item()))
        assert abs(float(pos[2].item()) - _PARK_Z) < 1e-6
    # Primitives are centred ⇒ entity origin == park lattice origin ⇒ unique XY.
    assert len(set(zip(xs, ys))) == len(pool), (xs, ys)
    # Parked quaternion is identity.
    park_q = torch.tensor(_PARK_QUAT, dtype=clutter._loaded_entities[0].get_quat().dtype)
    for entity in clutter._loaded_entities:
        assert torch.allclose(entity.get_quat(), park_q)
    print("  PASS: every pool object loaded at a unique, parked pose with fixed quat")
    env.close()


def test_inactive_unique_poses_after_reset_unbatched():
    """After a reset that activates only a subset, inactive objects keep their
    unique park positions and fixed orientation."""
    world = GenesisWorld()
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world, name="clutter", object_pool=pool, n_objects_range=(2, 2),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=1, reload=True)

    park_xy = set()
    for i, entity in enumerate(clutter._loaded_entities):
        pos = entity.get_pos()
        if not bool(clutter._active_mask_t[i]):
            park_xy.add((round(float(pos[0].item()), 6), round(float(pos[1].item()), 6)))
            assert abs(float(pos[2].item()) - _PARK_Z) < 1e-6
            assert torch.allclose(entity.get_quat(),
                                  torch.tensor(_PARK_QUAT, dtype=entity.get_quat().dtype))
    # Every inactive object occupies a distinct parked XY slot.
    assert len(park_xy) == (len(pool) - int(clutter._active_mask_t.sum().item()))
    print("  PASS: inactive unbatched objects parked at unique poses with fixed quat")
    env.close()


def test_batched_inactive_unique_poses():
    """In batched reset, within every environment the inactive objects have
    mutually-distinct parked XY positions."""
    N_ENVS = 6
    world = GenesisWorld(batch_size=N_ENVS)
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world, name="clutter", object_pool=pool, n_objects_range=(2, 3),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)

    for env_j in range(N_ENVS):
        xy = set()
        for i, entity in enumerate(clutter._loaded_entities):
            if not bool(clutter._active_mask_t[i, env_j]):
                p = entity.get_pos()[env_j]
                xy.add((round(float(p[0].item()), 6), round(float(p[1].item()), 6)))
                assert abs(float(p[2].item()) - _PARK_Z) < 1e-6
        assert len(xy) > 1, f"env {env_j}: inactive parked XY not unique ({xy})"
    print("  PASS: inactive batched objects have unique parked XY per env")
    env.close()


def test_masked_reset_inactive_unique_and_preserved():
    """Masked reset: reset envs get unique parked inactive poses; preserved envs
    keep their prior state exactly."""
    N_ENVS = 8
    world = GenesisWorld(batch_size=N_ENVS)
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world, name="clutter", object_pool=pool, n_objects_range=(2, 4),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)

    before = [e.get_pos().clone() for e in clutter._loaded_entities]
    before_q = [e.get_quat().clone() for e in clutter._loaded_entities]
    mask_before = clutter._active_mask_t.clone()

    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[0, 4, 5]] = True
    env.reset(seed=42, mask=reset_mask)

    preserved = [j for j in range(N_ENVS) if not bool(reset_mask[j])]
    for i, entity in enumerate(clutter._loaded_entities):
        pa = entity.get_pos(); qa = entity.get_quat()
        for j in preserved:
            assert torch.allclose(pa[j], before[i][j], atol=1e-6)
            assert torch.allclose(qa[j], before_q[i][j], atol=1e-6)
    # Preserved envs keep their active mask.
    for j in preserved:
        assert torch.equal(clutter._active_mask_t[:, j], mask_before[:, j])

    # Reset envs: inactive objects still occupy unique parked XY positions.
    for j in [0, 4, 5]:
        xy = set()
        for i, entity in enumerate(clutter._loaded_entities):
            if not bool(clutter._active_mask_t[i, j]):
                p = entity.get_pos()[j]
                xy.add((round(float(p[0].item()), 6), round(float(p[1].item()), 6)))
        assert len(xy) > 1, (j, xy)
    print("  PASS: masked reset preserves others, reset envs have unique park XY")
    env.close()


def test_inactive_no_rng_consumed():
    """An all-inactive reset (n_objects_range=(0,0)) with random_quaternion=True
    must keep every parked object at the fixed _PARK_QUAT orientation — the
    random-yaw RNG path is never invoked for inactive objects."""
    world = GenesisWorld()
    pool = _make_pool()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world, name="clutter", object_pool=pool, n_objects_range=(0, 0),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS, random_quaternion=True,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=3, reload=True)

    park_q = torch.tensor(_PARK_QUAT, dtype=clutter._loaded_entities[0].get_quat().dtype)
    for entity in clutter._loaded_entities:
        assert torch.allclose(entity.get_quat(), park_q), \
            f"random_quaternion leaked into inactive park pose: {entity.get_quat()}"
    print("  PASS: all-inactive reset kept fixed _PARK_QUAT (no random-yaw RNG)")
    env.close()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== test_pool_size_validation ===")
    test_pool_size_validation()
    print()

    print("=== test_unbatched_placement ===")
    test_unbatched_placement()
    print()

    print("=== test_seed_reproducibility ===")
    test_seed_reproducibility()
    print()

    print("=== test_avoid_spaces ===")
    test_avoid_spaces()
    print()

    print("=== test_batched_placement ===")
    test_batched_placement()
    print()

    print("=== test_batched_masked_reset ===")
    test_batched_masked_reset()
    print()

    print("=== test_batched_per_env_avoid_spaces ===")
    test_batched_per_env_avoid_spaces()
    print()

    print("=== test_park_clearance_validation ===")
    test_park_clearance_validation()
    print()
    print("=== test_geometry_cache_radius_and_max ===")
    test_geometry_cache_radius_and_max()
    print()
    print("=== test_park_grid_spacing ===")
    test_park_grid_spacing()
    print()
    print("=== test_park_offcenter_compensation ===")
    test_park_offcenter_compensation()
    print()
    print("=== test_geometry_cache_reuse ===")
    test_geometry_cache_reuse()
    print()
    print("=== test_unique_initial_load_poses ===")
    test_unique_initial_load_poses()
    print()
    print("=== test_inactive_unique_poses_after_reset_unbatched ===")
    test_inactive_unique_poses_after_reset_unbatched()
    print()
    print("=== test_batched_inactive_unique_poses ===")
    test_batched_inactive_unique_poses()
    print()
    print("=== test_masked_reset_inactive_unique_and_preserved ===")
    test_masked_reset_inactive_unique_and_preserved()
    print()
    print("=== test_inactive_no_rng_consumed ===")
    test_inactive_no_rng_consumed()
    print()

    print("All tests passed!")
