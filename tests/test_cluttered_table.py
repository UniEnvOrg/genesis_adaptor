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

gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.cached_table_builder import CachedTableBuilder
from unienv_genesis_collection.scenes.cluttered_table_manager import (
    ClutteredTableManager,
    _PARK_POS,
)
from unienv_genesis_collection.objects import (
    BoxObject,
    SphereObject,
    CylinderObject,
)
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

    print("All tests passed!")
