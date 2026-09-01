"""Tests for TactileSensorNode (unbatched, CPU) on a box+plane scene: both sensor types."""

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


def _options(**kw) -> TactileSensorNodeOptions:
    kw.setdefault("attaches", [AttachFromName("box", BOX_LINK)])
    kw.setdefault("face_axis", "-z")
    kw.setdefault("n_around", N_AROUND)
    kw.setdefault("n_along", N_ALONG)
    return TactileSensorNodeOptions(**kw)


def test_contact_depth_shapes_and_signal():
    world = GenesisWorld()
    node = TactileSensorNode(world, "hand_tactile", _options(sensor_type="contact_depth"))
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
    node = TactileSensorNode(world, "hand_tactile", _options(sensor_type="kinematic"))
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
    from unienv_genesis.nodes.sensors.tactile_node import _link_mesh

    world = GenesisWorld()
    node = TactileSensorNode(world, "hand_tactile", _options())
    env = _make_env(world, node, steps=0)

    pos = node.probe_positions[BOX_LINK]
    assert pos.shape == (N_TAXELS, 3)
    link = world.scene.get_entity("box").get_link(BOX_LINK)
    dist = trimesh.proximity.closest_point(_link_mesh(link), pos)[1]
    assert float(dist.max()) < 1e-4, f"probe {dist.max()*1000:.2f} mm off the surface"
    env.close()


def test_probe_normals_are_unit_and_outward():
    world = GenesisWorld()
    node = TactileSensorNode(world, "hand_tactile", _options())
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
