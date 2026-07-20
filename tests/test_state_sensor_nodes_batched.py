"""Batched (GPU) tests for the Tier-1 state sensor nodes.

These tests require CUDA and initialize Genesis with the GPU backend.  They
mirror ``tests/test_state_sensor_nodes.py`` but assert the batched leading
dimension.  No renderer is needed for state sensors, so the batched world is
created without a ``BatchRenderer`` (which would otherwise require a camera).
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required for batched Genesis tests", allow_module_level=True)

import genesis as gs
from genesis.constants import backend as gs_backend

# Reinitialize genesis with GPU if it was already initialized (e.g. CPU).
if gs._initialized:
    gs.destroy()
gs.init(backend=gs_backend.gpu)

from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors import (
    JointStateSensorNode,
    JointStateSensorNodeOptions,
    EndEffectorStateSensorNode,
    EndEffectorStateSensorNodeOptions,
    IMUSensorNode,
    IMUSensorNodeOptions,
    ContactForceSensorNode,
    ContactForceSensorNodeOptions,
    ContactSensorNode,
    ContactSensorNodeOptions,
)
from unienv_genesis.options.attach_option import AttachFromName

N_ENVS = 2
BOX_LINK = "box_baselink"
N_MANUAL_STEPS = 40


class BoxEntityNode(WorldNode):
    """Adds a plane + free-floating box for batched scenes."""

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
                gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05)),
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


def _make_batched_world() -> GenesisWorld:
    # State sensors do not need rendering; a batched world without a renderer
    # avoids the BatchRenderer's "add at least one camera" requirement.
    return GenesisWorld(batch_size=N_ENVS)


def _make_env(world: GenesisWorld, sensor_nodes) -> WorldEnv:
    box = BoxEntityNode(world)
    env = WorldEnv(world, [box, *sensor_nodes], render_mode=None)
    env.reset(seed=0, reload=True)
    return env


def _step_env(env: WorldEnv, n: int = N_MANUAL_STEPS) -> None:
    for _ in range(n):
        env.step(None)


def _assert_space_matches(node) -> None:
    obs = node.get_observation()
    for key, space in node.observation_space.spaces.items():
        assert key in obs, f"Key {key!r} in obs space but not in obs"
        assert obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={obs[key].shape}, space={space.shape}"
        )


# ---------------------------------------------------------------------------
# JointStateSensorNode
# ---------------------------------------------------------------------------


def test_batched_joint_state_shapes():
    world = _make_batched_world()
    node = JointStateSensorNode(
        world=world,
        name="joint_state",
        option=JointStateSensorNodeOptions(
            attach=AttachFromName("box"),
            read_root_pose=True,
        ),
    )
    env = _make_env(world, [node])
    obs = node.get_observation()

    n_dofs = world.scene.get_entity("box").n_dofs
    assert obs["joint_position"].shape == (N_ENVS, n_dofs)
    assert obs["joint_velocity"].shape == (N_ENVS, n_dofs)
    assert obs["joint_torque"].shape == (N_ENVS, n_dofs)
    assert obs["root_pos"].shape == (N_ENVS, 3)
    assert obs["root_quat"].shape == (N_ENVS, 4)
    _assert_space_matches(node)
    env.close()


# ---------------------------------------------------------------------------
# EndEffectorStateSensorNode
# ---------------------------------------------------------------------------


def test_batched_eef_state_shapes():
    world = _make_batched_world()
    node = EndEffectorStateSensorNode(
        world=world,
        name="eef",
        option=EndEffectorStateSensorNodeOptions(
            attach=AttachFromName("box", BOX_LINK, offset_pos=(0.0, 0.0, 0.1)),
        ),
    )
    env = _make_env(world, [node])
    obs = node.get_observation()

    assert obs["eef_position"].shape == (N_ENVS, 3)
    assert obs["eef_quaternion"].shape == (N_ENVS, 4)
    _assert_space_matches(node)
    env.close()


# ---------------------------------------------------------------------------
# IMUSensorNode
# ---------------------------------------------------------------------------


def test_batched_imu_shapes():
    world = _make_batched_world()
    node = IMUSensorNode(
        world=world,
        name="imu",
        option=IMUSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _step_env(env)

    obs = node.get_observation()
    assert obs["lin_acc"].shape == (N_ENVS, 3)
    assert obs["ang_vel"].shape == (N_ENVS, 3)
    assert obs["mag"].shape == (N_ENVS, 3)
    # Gravity should appear in the z-component for every env.
    assert torch.all(obs["lin_acc"][:, 2] > 5.0)
    _assert_space_matches(node)
    env.close()


# ---------------------------------------------------------------------------
# ContactForceSensorNode
# ---------------------------------------------------------------------------


def test_batched_contact_force_shapes():
    world = _make_batched_world()
    node = ContactForceSensorNode(
        world=world,
        name="contact_force",
        option=ContactForceSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _step_env(env)

    obs = node.get_observation()
    assert obs["contact_force"].shape == (N_ENVS, 3)
    # Box resting on plane -> upward (positive z) normal force in every env.
    assert torch.all(obs["contact_force"][:, 2] > 0.0)
    _assert_space_matches(node)
    env.close()


# ---------------------------------------------------------------------------
# ContactSensorNode
# ---------------------------------------------------------------------------


def test_batched_contact_shapes():
    world = _make_batched_world()
    node = ContactSensorNode(
        world=world,
        name="contact",
        option=ContactSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _step_env(env)

    obs = node.get_observation()
    assert obs["in_contact"].shape == (N_ENVS,)
    assert obs["in_contact"].dtype == torch.bool
    assert torch.all(obs["in_contact"])
    _assert_space_matches(node)
    env.close()


# ---------------------------------------------------------------------------
# No-advance sensor-cache refresh after reset (batched)
# ---------------------------------------------------------------------------


def test_batched_first_reset_reads_empty_caches():
    """Batched reset/reload finalization never steps the physics: right after
    the first (reload) reset every batched sensor cache is still empty."""
    world = GenesisWorld(batch_size=N_ENVS)
    imu = IMUSensorNode(
        world=world,
        name="imu",
        option=IMUSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    cf = ContactForceSensorNode(
        world=world,
        name="contact_force",
        option=ContactForceSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    ct = ContactSensorNode(
        world=world,
        name="contact",
        option=ContactSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [imu, cf, ct])

    # No manual stepping — the world must not have advanced the simulation.
    imu_obs = imu.get_observation()
    cf_obs = cf.get_observation()
    ct_obs = ct.get_observation()

    assert imu_obs["lin_acc"].shape == (N_ENVS, 3)
    assert torch.all(imu_obs["lin_acc"] == 0.0)

    assert cf_obs["contact_force"].shape == (N_ENVS, 3)
    assert torch.all(cf_obs["contact_force"] == 0.0)

    assert ct_obs["in_contact"].shape == (N_ENVS,)
    assert not torch.any(ct_obs["in_contact"])
    env.close()


def test_batched_cache_refresh_populates_imu_on_subsequent_reset():
    """After a plain non-reload reset, the no-advance cache refresh populates
    the batched IMU with gravity for every env — with zero physics steps —
    while contact-based sensors stay empty until the next ``env.step()``."""
    world = GenesisWorld(batch_size=N_ENVS)
    imu = IMUSensorNode(
        world=world,
        name="imu",
        option=IMUSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    cf = ContactForceSensorNode(
        world=world,
        name="contact_force",
        option=ContactForceSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [imu, cf])

    _step_env(env, n=3)
    env.reset(seed=1)

    imu_obs = imu.get_observation()
    cf_obs = cf.get_observation()
    assert torch.all(imu_obs["lin_acc"][:, 2] > 5.0)
    # No automatic physics stepping: contacts stay cleared until env.step().
    assert torch.all(cf_obs["contact_force"] == 0.0)
    env.close()
