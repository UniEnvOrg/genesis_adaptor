"""Tests for TactileSensorNode (unbatched, CPU) on a box+plane scene: both sensor types, bands and patches."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode

from unienv_genesis import (
    GenesisWorld,
    TactileSensorNode,
    TactileSensorNodeOptions,
    TaxelPatch,
)
from unienv_genesis.options.attach_option import AttachFromName

BOX_LINK = "box_baselink"
BOX_SIZE = 0.1


class BoxEntityNode(WorldNode):
    """Plane + box resting on it, so contact sensors have something to read."""

    reload_priorities = {20}
    reset_priorities = {20}
    pre_environment_step_priorities = {20}
    post_environment_step_priorities = {20}

    def __init__(self, world: GenesisWorld, name: str = "box_env"):
        self.world = world
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reload(self, *, priority=0, **kw):
        if priority == 20:
            self.world.scene.add_entity(gs.morphs.Plane(), name="plane")
            self.world.scene.add_entity(
                gs.morphs.Box(size=(BOX_SIZE, BOX_SIZE, BOX_SIZE), pos=(0, 0, BOX_SIZE / 2)),
                name="box",
            )

    def reset(self, *, priority=0, **kw):
        pass

    def pre_environment_step(self, dt, *, priority=0):
        pass

    def post_environment_step(self, dt, *, priority=0):
        pass

    def get_observation(self):
        return None


def _make_env(world: GenesisWorld, node: TactileSensorNode, steps: int = 40) -> WorldEnv:
    """Build the scene and settle it for ``steps``; probes are placed in reload, so geometry checks can pass 0."""
    env = WorldEnv(world, [BoxEntityNode(world), node], render_mode=None)
    env.reset(seed=0, reload=True)
    for _ in range(steps):
        env.step(None)
    return env


N_AROUND, N_ALONG = 3, 4
N_TAXELS = N_AROUND * N_ALONG


def _make_options(**kw) -> TactileSensorNodeOptions:
    kw.setdefault("attaches", [AttachFromName("box", BOX_LINK)])
    kw.setdefault("face_axis", "-z")
    kw.setdefault("n_around", N_AROUND)
    kw.setdefault("n_along", N_ALONG)
    return TactileSensorNodeOptions(**kw)


def test_contact_depth_shapes_and_signal():
    world = GenesisWorld()
    node = TactileSensorNode(world, "hand_tactile", _make_options(sensor_type="contact_depth"))
    env = _make_env(world, node)
    obs = node.get_observation()

    assert set(obs) == {BOX_LINK}
    assert obs[BOX_LINK].shape == (N_TAXELS,)
    assert obs[BOX_LINK].dtype == world.dtype
    assert torch.all(obs[BOX_LINK] >= 0.0)
    assert float(obs[BOX_LINK].max()) > 0.0  # pressed into the plane

    space = node.observation_space.spaces[BOX_LINK]
    assert obs[BOX_LINK].shape == space.shape
    env.close()


def test_kinematic_shapes_and_signal():
    world = GenesisWorld()
    node = TactileSensorNode(world, "hand_tactile", _make_options(sensor_type="kinematic"))
    env = _make_env(world, node)
    obs = node.get_observation()

    assert set(obs[BOX_LINK]) == {"force", "torque"}
    assert obs[BOX_LINK]["force"].shape == (N_TAXELS, 3)
    assert obs[BOX_LINK]["torque"].shape == (N_TAXELS, 3)
    assert float(torch.linalg.norm(obs[BOX_LINK]["force"], dim=-1).max()) > 0.0

    space = node.observation_space.spaces[BOX_LINK]
    assert obs[BOX_LINK]["force"].shape == space.spaces["force"].shape
    env.close()


def test_probes_lie_on_the_mesh_surface():
    """Regression: probes used to sit on the AABB face, 9.5 mm off the contoured palm."""
    import trimesh
    from unienv_genesis.nodes.sensors.tactile_node import _build_link_mesh

    world = GenesisWorld()
    node = TactileSensorNode(world, "hand_tactile", _make_options())
    env = _make_env(world, node, steps=0)

    pos = node.probe_positions[BOX_LINK]
    assert pos.shape == (N_TAXELS, 3)
    link = world.scene.get_entity("box").get_link(BOX_LINK)
    dist = trimesh.proximity.closest_point(_build_link_mesh(link), pos)[1]
    assert float(dist.max()) < 1e-4, f"probe {dist.max()*1000:.2f} mm off the surface"
    env.close()


def test_probe_normals_are_unit_and_outward():
    world = GenesisWorld()
    node = TactileSensorNode(world, "hand_tactile", _make_options())
    env = _make_env(world, node, steps=0)

    pos, nrm = node.probe_positions[BOX_LINK], node.probe_normals[BOX_LINK]
    assert nrm.shape == (N_TAXELS, 3)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)
    # outward = away from the box centre, which sits at the origin of the link frame
    assert np.all(np.einsum("ij,ij->i", nrm, pos) > 0.0)
    env.close()


def test_rejects_unknown_sensor_type():
    world = GenesisWorld()
    with pytest.raises(ValueError):
        TactileSensorNode(
            world, "t",
            TactileSensorNodeOptions(
                attaches=[AttachFromName("box", BOX_LINK)],
                sensor_type="bogus",
            ),
        )


def test_rejects_duplicate_link_names():
    world = GenesisWorld()
    with pytest.raises(ValueError):
        TactileSensorNode(
            world, "t",
            TactileSensorNodeOptions(
                attaches=[
                    AttachFromName("box", BOX_LINK),
                    AttachFromName("box", BOX_LINK),
                ],
            ),
        )


@pytest.mark.parametrize("overrides, expected", [(None, -1.0), ({BOX_LINK: "+z"}, 1.0)])
def test_face_axis_override_moves_the_band(overrides, expected):
    """The override picks the face; without one the band stays on ``face_axis`` ("-z")."""
    # n_around=1 puts every probe dead centre on the chosen face, so the
    # normals read the face axis directly instead of wrapping onto a corner.
    world = GenesisWorld()
    node = TactileSensorNode(
        world, "t", _make_options(n_around=1, face_axis_overrides=overrides)
    )
    env = _make_env(world, node, steps=0)
    normals = node.probe_normals[BOX_LINK]
    assert np.allclose(normals[:, 2], expected, atol=1e-6), normals
    env.close()


def test_rejects_face_axis_override_for_unattached_link():
    world = GenesisWorld()
    with pytest.raises(ValueError, match="not attached"):
        TactileSensorNode(world, "t", _make_options(face_axis_overrides={"nope": "+z"}))


PATCH_SHAPE = (2, 3)
PATCH_CELLS = [0, 2, 4, 5]  # cells 1 and 3 have no taxel


def _place_on_bottom(link):
    """Four probes on the box's bottom face, which rests on the plane."""
    xy = np.array([[-0.02, -0.02], [0.02, -0.02], [-0.02, 0.02], [0.02, 0.02]])
    pos = np.column_stack([xy, np.full(len(xy), -BOX_SIZE / 2)])
    return pos, np.tile([0.0, 0.0, -1.0], (len(xy), 1))


def _make_patch(**kw) -> TaxelPatch:
    kw.setdefault("name", "pad")
    kw.setdefault("attach", AttachFromName("box", BOX_LINK))
    kw.setdefault("shape", PATCH_SHAPE)
    kw.setdefault("cells", PATCH_CELLS)
    kw.setdefault("place", _place_on_bottom)
    return TaxelPatch(**kw)


def test_patch_reads_back_as_its_grid():
    world = GenesisWorld()
    node = TactileSensorNode(world, "t", TactileSensorNodeOptions(patches=[_make_patch()]))
    env = _make_env(world, node)
    reading = node.get_observation()["pad"]

    assert reading.shape == PATCH_SHAPE == node.observation_space.spaces["pad"].shape
    flat = reading.reshape(-1)
    assert torch.all(flat[PATCH_CELLS] > 0.0)  # every placed taxel is pressed into the plane
    assert torch.all(flat[[1, 3]] == 0.0)  # cells without a taxel stay empty
    assert np.array_equal(node.probe_cells["pad"], PATCH_CELLS)
    env.close()


def test_patch_kinematic_grid_shape():
    world = GenesisWorld()
    node = TactileSensorNode(
        world, "t", TactileSensorNodeOptions(patches=[_make_patch()], sensor_type="kinematic")
    )
    env = _make_env(world, node)
    force = node.get_observation()["pad"]["force"]
    assert force.shape == (*PATCH_SHAPE, 3) == node.observation_space.spaces["pad"].spaces["force"].shape
    assert float(torch.linalg.norm(force.reshape(-1, 3)[PATCH_CELLS], dim=-1).min()) > 0.0
    env.close()


def test_patch_and_band_share_a_link():
    world = GenesisWorld()
    node = TactileSensorNode(world, "t", _make_options(patches=[_make_patch()]))
    env = _make_env(world, node)
    obs = node.get_observation()
    assert obs[BOX_LINK].shape == (N_TAXELS,)
    assert obs["pad"].shape == PATCH_SHAPE
    env.close()


@pytest.mark.parametrize("cells", [[0, 0, 4, 5], [0, 2, 4, 6], [-1, 2, 4, 5]])
def test_rejects_bad_patch_cells(cells):
    world = GenesisWorld()
    with pytest.raises(ValueError, match="cells"):
        TactileSensorNode(world, "t", TactileSensorNodeOptions(patches=[_make_patch(cells=cells)]))


def test_rejects_patch_named_like_a_band_link():
    world = GenesisWorld()
    with pytest.raises(ValueError, match="duplicate"):
        TactileSensorNode(world, "t", _make_options(patches=[_make_patch(name=BOX_LINK)]))


def test_rejects_patch_placing_the_wrong_number_of_probes():
    world = GenesisWorld()
    node = TactileSensorNode(
        world, "t", TactileSensorNodeOptions(patches=[_make_patch(cells=[0, 2, 4])])
    )
    with pytest.raises(ValueError, match="placed 4 probes, expected 3"):
        _make_env(world, node, steps=0)
