"""CUDA device-safety tests for CachedTableBuilder (Phase 2 runtime lane).

These tests require CUDA and initialize Genesis with the GPU backend.  They
mirror ``tests/test_cached_table_builder.py`` but assert that every runtime
tensor produced by the builder (sampled sizes/heights, ``size``/``height``
properties, group anchor poses, random draws through the device-matched
generator) lives on ``world.device``, for both unbatched and batched/masked
reset flows.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required for CachedTableBuilder device tests", allow_module_level=True)

import genesis as gs
from genesis.constants import backend as gs_backend

# Reinitialize genesis with GPU if it was already initialized (e.g. CPU).
if gs._initialized:
    gs.destroy()
gs.init(backend=gs_backend.gpu)

from unienv_interface.world import WorldEnv

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.cached_table_builder import (
    CachedTableBuilder,
    _PARK_POS,
)

_TABLE_POS = (0.0, 0.0, 0.65)
_N_VARIANTS = 8
_SIZE_RANGE = ((0.8, 1.2), (0.5, 0.7))
_HEIGHT_RANGE = (0.55, 0.75)


def _make_table(world):
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(
        world,
        name="table",
        n_variants=_N_VARIANTS,
        pos=_TABLE_POS,
        yaw=0.0,
        size_range=_SIZE_RANGE,
        height_range=_HEIGHT_RANGE,
    )
    env = WorldEnv(world, [plane, table], render_mode=None)
    return env, table


def test_cached_table_builder_cuda_unbatched():
    """Unbatched CUDA world: random draw and all exposed tensors on device."""
    world = GenesisWorld()
    assert world.device.type == "cuda"
    env, table = _make_table(world)

    env.reset(seed=0, reload=True)

    # Sampled parameter tensors live on the world device.
    assert table._sampled_sizes_t.device == world.device
    assert table._sampled_heights_t.device == world.device

    # size / height properties follow the active variant, on device.
    chosen = int(table._chosen_indices[0])
    assert table.size.device == world.device
    assert table.height.device == world.device
    expected_size = torch.tensor(
        table._variant_params[chosen]["size"], dtype=world.dtype, device=world.device,
    )
    assert torch.allclose(table.size, expected_size)

    # Group anchor poses are device-local; active at target, others parked.
    target_pos = torch.tensor(_TABLE_POS, dtype=world.dtype, device=world.device)
    park_pos = torch.tensor(_PARK_POS, dtype=world.dtype, device=world.device)
    for i, group in enumerate(table._variants):
        assert group.pos.device == world.device
        assert group.quat.device == world.device
        expected = target_pos if i == chosen else park_pos
        assert torch.allclose(group.pos, expected), (
            f"Variant {i}: pos {group.pos} != expected {expected}"
        )

    # A few unseeded resets keep working on the CUDA generator.
    for _ in range(5):
        env.reset()
        assert 0 <= int(table._chosen_indices[0]) < _N_VARIANTS

    env.close()
    print("test_cached_table_builder_cuda_unbatched PASSED")


def test_cached_table_builder_cuda_batched_masked_reset():
    """Batched CUDA world: masked reset stays device-safe and preserves unmasked envs."""
    N_ENVS = 8
    world = GenesisWorld(batch_size=N_ENVS)
    assert world.device.type == "cuda"
    env, table = _make_table(world)

    env.reset(seed=5, reload=True)

    assert table.size.shape == (N_ENVS, 3)
    assert table.size.device == world.device
    assert table.height.device == world.device
    for group in table._variants:
        assert group.pos.device == world.device
        assert group.pos.shape == (N_ENVS, 3)

    chosen_before = table._chosen_indices.tolist()
    poses_before = [group.pos.clone() for group in table._variants]

    # Masked reset on a CUDA bool mask: only envs 1, 4 may change.
    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool, device=world.device)
    reset_mask[[1, 4]] = True
    env.reset(mask=reset_mask)

    preserved = [0, 2, 3, 5, 6, 7]
    chosen_after = table._chosen_indices.tolist()
    for env_j in preserved:
        assert chosen_after[env_j] == chosen_before[env_j]
    for env_j in (1, 4):
        assert 0 <= chosen_after[env_j] < _N_VARIANTS

    target_pos = torch.tensor(_TABLE_POS, dtype=world.dtype, device=world.device)
    park_pos = torch.tensor(_PARK_POS, dtype=world.dtype, device=world.device)
    for i, group in enumerate(table._variants):
        for env_j in preserved:
            assert torch.allclose(group.pos[env_j], poses_before[i][env_j]), (
                f"Variant {i}, env {env_j}: pose changed after masked reset"
            )
        for env_j in (1, 4):
            expected = target_pos if chosen_after[env_j] == i else park_pos
            assert torch.allclose(group.pos[env_j], expected)

    # size stays consistent with the updated selection, on device.
    expected_sizes = torch.tensor(
        [table._variant_params[i]["size"] for i in chosen_after],
        dtype=world.dtype, device=world.device,
    )
    assert table.size.device == world.device
    assert torch.allclose(table.size, expected_sizes)

    # The strict mask contract rejects integer indices and wrong-device
    # masks on a CUDA world.
    with pytest.raises(AssertionError):
        env.reset(mask=torch.tensor([0, 2], device=world.device))
    with pytest.raises(AssertionError):
        table.reset(
            priority=100,
            mask=torch.zeros(N_ENVS, dtype=torch.bool, device="cpu"),  # wrong device
        )

    assert table._chosen_indices.shape == (N_ENVS,)

    env.close()
    print("test_cached_table_builder_cuda_batched_masked_reset PASSED")


if __name__ == "__main__":
    test_cached_table_builder_cuda_unbatched()
    test_cached_table_builder_cuda_batched_masked_reset()
    print("All tests passed!")
