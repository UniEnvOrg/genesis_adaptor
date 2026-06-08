"""Focused API tests for TableBuilder defaults and fixed tensor parameters."""

import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_interface.world import WorldEnv


def test_table_builder_defaults_unbatched():
    """Unbatched TableBuilder exposes default_* fields and fixed tensor size/height."""
    table_size = (1.1, 0.7, 0.03)
    table_height = 0.68
    leg_color = (0.11, 0.22, 0.33, 1.0)

    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")
    table = TableBuilder(
        world,
        name="table",
        size=table_size,
        height=table_height,
        leg_color=leg_color,
    )
    env = WorldEnv(world, [plane, table], render_mode=None)

    env.reset(seed=0, reload=True)

    assert table.default_size == table_size
    assert table.default_height == table_height
    assert table.default_texture_path is None
    assert table.default_leg_color == leg_color

    assert not hasattr(table, "texture_path")
    assert not hasattr(table, "leg_color")

    assert isinstance(table.size, torch.Tensor)
    assert isinstance(table.height, torch.Tensor)
    assert table.size.shape == (3,)
    assert table.height.ndim == 0
    assert torch.allclose(table.size, torch.tensor(table_size, dtype=world.dtype))
    assert torch.isclose(table.height, torch.tensor(table_height, dtype=world.dtype))
    assert table._group is not None

    env.close()


def test_table_builder_defaults_batched():
    """Batched TableBuilder exposes fixed repeated tensor size/height."""
    n_envs = 6
    table_size = (0.9, 0.55, 0.02)
    table_height = 0.61

    world = GenesisWorld(batch_size=n_envs)
    plane = PlaneBuilder(world, name="floor")
    table = TableBuilder(
        world,
        name="table",
        size=table_size,
        height=table_height,
    )
    env = WorldEnv(world, [plane, table], render_mode=None)

    env.reset(seed=3, reload=True)

    assert table.size.shape == (n_envs, 3)
    assert table.height.shape == (n_envs,)
    assert torch.allclose(
        table.size,
        torch.tensor(table_size, dtype=world.dtype).unsqueeze(0).repeat(n_envs, 1),
    )
    assert torch.allclose(
        table.height,
        torch.full((n_envs,), table_height, dtype=world.dtype),
    )

    env.close()


if __name__ == "__main__":
    test_table_builder_defaults_unbatched()
    test_table_builder_defaults_batched()
    print("All tests passed!")
