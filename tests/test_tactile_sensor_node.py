"""Tests for TactileSensorNode (unbatched, CPU) on a box+plane scene: both sensor types."""

from __future__ import annotations

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


def _make_env(world: GenesisWorld, node: TactileSensorNode) -> WorldEnv:
    env = WorldEnv(world, [BoxEntityNode(world), node], render_mode=None)
    env.reset(seed=0, reload=True)
    for _ in range(40):
        env.step(None)
    return env


def test_contact_depth_shapes_and_signal():
    world = GenesisWorld()
    node = TactileSensorNode(
        world, "hand_tactile",
        TactileSensorNodeOptions(
            attaches=[AttachFromName("box", BOX_LINK)],
            sensor_type="contact_depth",
            face_axis="-z",
        ),
    )
    env = _make_env(world, node)
    obs = node.get_observation()

    assert set(obs) == {BOX_LINK}
    assert obs[BOX_LINK].shape == (1,)
    assert obs[BOX_LINK].dtype == world.dtype
    assert torch.all(obs[BOX_LINK] >= 0.0)
    assert float(obs[BOX_LINK].max()) > 0.0  # pressed into the plane

    space = node.observation_space.spaces[BOX_LINK]
    assert obs[BOX_LINK].shape == space.shape
    env.close()


def test_kinematic_shapes_and_signal():
    world = GenesisWorld()
    node = TactileSensorNode(
        world, "hand_tactile",
        TactileSensorNodeOptions(
            attaches=[AttachFromName("box", BOX_LINK)],
            sensor_type="kinematic",
            face_axis="-z",
        ),
    )
    env = _make_env(world, node)
    obs = node.get_observation()

    assert set(obs[BOX_LINK]) == {"force", "torque"}
    assert obs[BOX_LINK]["force"].shape == (3,)
    assert obs[BOX_LINK]["torque"].shape == (3,)
    assert float(torch.linalg.norm(obs[BOX_LINK]["force"])) > 0.0

    space = node.observation_space.spaces[BOX_LINK]
    assert obs[BOX_LINK]["force"].shape == space.spaces["force"].shape
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
