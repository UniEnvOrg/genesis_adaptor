"""Tests for CachedTableBuilder.

Covers:
  - Correct number of variants built during reload()
  - size_range and height_range constraints respected
  - Active variant positioned at target; inactive variants parked
  - Seed reproducibility (same seed → same variant)
  - Persistent RNG state when no seed is given
  - Batched worlds: per-environment independent selection and correct positions
"""

import torch
import genesis as gs
from genesis.constants import backend as gs_backend

gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.cached_table_builder import CachedTableBuilder, _PARK_POS
from unienv_interface.world import WorldEnv

_TABLE_POS = (0.0, 0.0, 0.65)
_TABLE_YAW = 0.0
_SIZE_RANGE = ((0.8, 1.2), (0.5, 0.7))
_HEIGHT_RANGE = (0.55, 0.75)
_N_VARIANTS = 8

_TARGET_POS_T = torch.tensor(_TABLE_POS, dtype=torch.float32)
_PARK_POS_T = torch.tensor(_PARK_POS, dtype=torch.float32)


def test_cached_table_builder():
    """Unbatched CachedTableBuilder: construction, positions, seeding, RNG persistence."""
    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(
        world,
        name="table",
        n_variants=_N_VARIANTS,
        pos=_TABLE_POS,
        yaw=_TABLE_YAW,
        size_range=_SIZE_RANGE,
        height_range=_HEIGHT_RANGE,
    )
    env = WorldEnv(world, [plane, table], render_mode=None)

    # ------------------------------------------------------------------
    # Initial reload+reset
    # ------------------------------------------------------------------
    env.reset(seed=0, reload=True)

    # Correct number of variants created
    assert len(table._variants) == _N_VARIANTS, \
        f"Expected {_N_VARIANTS} variants, got {len(table._variants)}"
    assert len(table._variant_params) == _N_VARIANTS

    # size_range and height_range honoured
    for p in table._variant_params:
        w, d, _ = p["size"]
        h = p["height"]
        assert _SIZE_RANGE[0][0] <= w <= _SIZE_RANGE[0][1], f"width {w:.4f} out of range"
        assert _SIZE_RANGE[1][0] <= d <= _SIZE_RANGE[1][1], f"depth {d:.4f} out of range"
        assert _HEIGHT_RANGE[0] <= h <= _HEIGHT_RANGE[1], f"height {h:.4f} out of range"

    print("  PASS: variants built with correct count and parameters")

    # ------------------------------------------------------------------
    # Active at target position; others parked
    # ------------------------------------------------------------------
    chosen = table._chosen_indices[0]
    active_pos = table._variants[chosen].pos
    assert torch.allclose(active_pos, _TARGET_POS_T), \
        f"Active variant {chosen} pos {active_pos} != target {_TARGET_POS_T}"

    for i, group in enumerate(table._variants):
        if i == chosen:
            continue
        parked_pos = group.pos
        assert torch.allclose(parked_pos, _PARK_POS_T), \
            f"Variant {i} pos {parked_pos} != park {_PARK_POS_T}"

    print("  PASS: active variant at target, others parked")

    # ------------------------------------------------------------------
    # Seed reproducibility: same reset seed → same variant
    # ------------------------------------------------------------------
    env.reset(seed=42)
    first = table._chosen_indices[0]
    env.reset(seed=42)
    second = table._chosen_indices[0]
    assert first == second, \
        f"Same reset seed 42 gave different variants: {first} vs {second}"

    print("  PASS: same reset seed yields same variant")

    # ------------------------------------------------------------------
    # Different reset seeds → (very likely) different variants across many trials
    # ------------------------------------------------------------------
    choices = set()
    for s in range(30):
        env.reset(seed=s)
        choices.add(table._chosen_indices[0])
    assert len(choices) > 1, "30 distinct seeds all chose the same variant — RNG is broken"

    print(f"  PASS: {len(choices)} distinct variants chosen across 30 seeds")

    # ------------------------------------------------------------------
    # Persistent RNG: no seed → state carries over, variants keep changing
    # ------------------------------------------------------------------
    env.reset(seed=7, reload=True)  # anchor the RNG state
    unseeded_choices = []
    for _ in range(15):
        env.reset()  # no seed — RNG state continues
        unseeded_choices.append(table._chosen_indices[0])
    assert len(set(unseeded_choices)) > 1, \
        "RNG state did not advance across unseeded resets"

    print(f"  PASS: persistent RNG — {len(set(unseeded_choices))} distinct variants across 15 unseeded resets")

    env.close()
    print("test_cached_table_builder PASSED\n")


def test_cached_table_builder_batched():
    """Batched CachedTableBuilder: per-environment selection and correct pose tensors."""
    N_ENVS = 12
    world = GenesisWorld(batch_size=N_ENVS)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(
        world,
        name="table",
        n_variants=_N_VARIANTS,
        pos=_TABLE_POS,
        yaw=_TABLE_YAW,
    )
    env = WorldEnv(world, [plane, table], render_mode=None)

    env.reset(seed=5, reload=True)

    # One chosen index per environment
    assert len(table._chosen_indices) == N_ENVS, \
        f"Expected {N_ENVS} chosen indices, got {len(table._chosen_indices)}"

    print("  PASS: one chosen index per environment")

    # For every variant, its (N_ENVS, 3) pos tensor should be correct:
    #   target pos for envs that chose it, park pos for all others
    for i, group in enumerate(table._variants):
        group_pos = group.pos  # (N_ENVS, 3)
        assert group_pos.shape == (N_ENVS, 3), \
            f"Variant {i} pos shape {group_pos.shape} != ({N_ENVS}, 3)"
        for env_j in range(N_ENVS):
            expected = _TARGET_POS_T if table._chosen_indices[env_j] == i else _PARK_POS_T
            assert torch.allclose(group_pos[env_j], expected), \
                f"Variant {i}, env {env_j}: pos {group_pos[env_j]} != expected {expected} " \
                f"(chosen={table._chosen_indices[env_j]})"

    print("  PASS: each variant's position tensor correct for all environments")

    # Different environments should (with very high probability) pick different variants
    assert len(set(table._chosen_indices)) > 1, \
        "All environments chose the same variant — per-env selection not working"

    print(f"  PASS: {len(set(table._chosen_indices))} distinct variants across {N_ENVS} environments")

    # ------------------------------------------------------------------
    # Seed reproducibility in batched mode
    # ------------------------------------------------------------------
    env.reset(seed=99)
    first_indices = list(table._chosen_indices)
    env.reset(seed=99)
    second_indices = list(table._chosen_indices)
    assert first_indices == second_indices, \
        f"Same seed gave different per-env selections:\n  {first_indices}\n  {second_indices}"

    print("  PASS: same reset seed yields identical per-environment selections")

    env.close()
    print("test_cached_table_builder_batched PASSED\n")


if __name__ == "__main__":
    print("=== test_cached_table_builder ===")
    test_cached_table_builder()
    print("=== test_cached_table_builder_batched ===")
    test_cached_table_builder_batched()
    print("All tests passed!")
