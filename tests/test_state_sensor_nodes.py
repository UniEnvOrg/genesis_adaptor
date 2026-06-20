"""Tests for the Tier-1 state sensor nodes (unbatched, CPU).

Covers JointStateSensorNode, EndEffectorStateSensorNode, IMUSensorNode,
ContactForceSensorNode and ContactSensorNode.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

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

BOX_LINK = "box_baselink"
N_SETTLE_STEPS = 40


class BoxEntityNode(WorldNode):
    """Adds a plane + free-floating box so the scene is non-empty.

    The box rests (initially) on the plane so contact-based sensors have
    something to detect after a few simulation steps.
    """

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


def _settle(env: WorldEnv, n: int = N_SETTLE_STEPS) -> None:
    """Step the env ``n`` times so Genesis populates the sensor cache."""
    for _ in range(n):
        env.step(None)


def _make_env(world: GenesisWorld, sensor_nodes) -> WorldEnv:
    """Build a WorldEnv from the box helper + the given sensor nodes."""
    box = BoxEntityNode(world)
    env = WorldEnv(world, [box, *sensor_nodes], render_mode=None)
    env.reset(seed=0, reload=True)
    return env


# ---------------------------------------------------------------------------
# JointStateSensorNode
# ---------------------------------------------------------------------------


def test_joint_state_shapes():
    world = GenesisWorld()
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
    assert obs["joint_position"].shape == (n_dofs,)
    assert obs["joint_velocity"].shape == (n_dofs,)
    assert obs["joint_torque"].shape == (n_dofs,)
    assert obs["root_pos"].shape == (3,)
    assert obs["root_quat"].shape == (4,)
    assert obs["joint_position"].dtype == world.dtype
    env.close()


def test_joint_state_observation_space_matches():
    world = GenesisWorld()
    node = JointStateSensorNode(
        world=world,
        name="joint_state",
        option=JointStateSensorNodeOptions(
            attach=AttachFromName("box"),
            read_motor_torque=True,
            read_root_pose=True,
        ),
    )
    env = _make_env(world, [node])
    obs = node.get_observation()

    for key, space in node.observation_space.spaces.items():
        assert key in obs, f"Key {key!r} in obs space but not in obs"
        assert obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={obs[key].shape}, space={space.shape}"
        )
    env.close()


# ---------------------------------------------------------------------------
# EndEffectorStateSensorNode
# ---------------------------------------------------------------------------


def test_eef_state_shapes():
    world = GenesisWorld()
    node = EndEffectorStateSensorNode(
        world=world,
        name="eef",
        option=EndEffectorStateSensorNodeOptions(
            attach=AttachFromName("box", BOX_LINK),
        ),
    )
    env = _make_env(world, [node])
    obs = node.get_observation()

    assert obs["eef_position"].shape == (3,)
    assert obs["eef_quaternion"].shape == (4,)
    assert obs["eef_position"].dtype == world.dtype
    env.close()


def test_eef_state_offset():
    """With offset_pos=(0,0,0.1) the eef_position should be offset from
    the no-offset case by ~0.1 along the link's local z-axis."""
    # No offset
    world_a = GenesisWorld()
    node_a = EndEffectorStateSensorNode(
        world=world_a,
        name="eef_a",
        option=EndEffectorStateSensorNodeOptions(
            attach=AttachFromName("box", BOX_LINK),
        ),
    )
    env_a = _make_env(world_a, [node_a])
    pos_a = node_a.get_observation()["eef_position"].clone()
    env_a.close()

    # With offset
    world_b = GenesisWorld()
    node_b = EndEffectorStateSensorNode(
        world=world_b,
        name="eef_b",
        option=EndEffectorStateSensorNodeOptions(
            attach=AttachFromName("box", BOX_LINK, offset_pos=(0.0, 0.0, 0.1)),
        ),
    )
    env_b = _make_env(world_b, [node_b])
    pos_b = node_b.get_observation()["eef_position"].clone()
    env_b.close()

    diff = pos_b - pos_a
    # The box starts with identity orientation, so the local z offset maps to
    # the world z axis.
    assert torch.allclose(diff, torch.tensor([0.0, 0.0, 0.1], dtype=diff.dtype), atol=1e-4), (
        f"Expected offset (0,0,0.1), got {diff.tolist()}"
    )


def test_eef_state_observation_space_matches():
    world = GenesisWorld()
    node = EndEffectorStateSensorNode(
        world=world,
        name="eef",
        option=EndEffectorStateSensorNodeOptions(
            attach=AttachFromName("box", BOX_LINK, offset_pos=(0.0, 0.0, 0.1)),
        ),
    )
    env = _make_env(world, [node])
    obs = node.get_observation()

    for key, space in node.observation_space.spaces.items():
        assert key in obs, f"Key {key!r} in obs space but not in obs"
        assert obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={obs[key].shape}, space={space.shape}"
        )
    env.close()


def test_eef_state_base_frame_offset():
    """When ``base_frame`` carries an offset, the EEF position is expressed
    relative to the *virtual* base frame (the physical link origin shifted by
    the offset), not the physical link origin."""
    world = GenesisWorld()
    node_no_base = EndEffectorStateSensorNode(
        world=world,
        name="eef_no_base",
        option=EndEffectorStateSensorNodeOptions(
            attach=AttachFromName("box", BOX_LINK),
        ),
    )
    node_offset_base = EndEffectorStateSensorNode(
        world=world,
        name="eef_offset_base",
        option=EndEffectorStateSensorNodeOptions(
            attach=AttachFromName("box", BOX_LINK),
            base_frame=AttachFromName("box", BOX_LINK, offset_pos=(0.0, 0.0, 0.05)),
        ),
    )
    env = _make_env(world, [node_no_base, node_offset_base])

    pos_no_base = node_no_base.get_observation()["eef_position"].clone()
    pos_offset_base = node_offset_base.get_observation()["eef_position"].clone()
    env.close()

    # The physical base link and the EEF link are the same link (box_baselink),
    # at the same world position with identity orientation at reset.  The
    # virtual base frame is shifted +0.05 along the link's local z-axis (which
    # is the world z-axis at rest).  So the EEF position expressed in the
    # virtual base frame is -0.05 along z, regardless of where the link
    # actually is in the world.
    assert torch.allclose(
        pos_offset_base,
        torch.tensor([0.0, 0.0, -0.05], dtype=pos_offset_base.dtype),
        atol=1e-4,
    ), f"Expected eef_pos_base (0,0,-0.05), got {pos_offset_base.tolist()}"

    # Sanity: the no-base EEF position is the world-frame link position
    # (non-zero z), and the offset-base position differs from it by the
    # virtual base offset.
    assert pos_no_base[2].item() != 0.0


# ---------------------------------------------------------------------------
# IMUSensorNode
# ---------------------------------------------------------------------------


def test_imu_shapes():
    world = GenesisWorld()
    node = IMUSensorNode(
        world=world,
        name="imu",
        option=IMUSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _settle(env)

    obs = node.get_observation()
    assert obs["lin_acc"].shape == (3,)
    assert obs["ang_vel"].shape == (3,)
    assert obs["mag"].shape == (3,)
    assert obs["lin_acc"].dtype == world.dtype
    # After settling, gravity should show up in the z-component of lin_acc.
    assert obs["lin_acc"][2] > 5.0
    env.close()


def test_imu_observation_space_matches():
    world = GenesisWorld()
    node = IMUSensorNode(
        world=world,
        name="imu",
        option=IMUSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _settle(env)

    obs = node.get_observation()
    for key, space in node.observation_space.spaces.items():
        assert key in obs, f"Key {key!r} in obs space but not in obs"
        assert obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={obs[key].shape}, space={space.shape}"
        )
    env.close()


# ---------------------------------------------------------------------------
# ContactForceSensorNode
# ---------------------------------------------------------------------------


def test_contact_force_shapes():
    world = GenesisWorld()
    node = ContactForceSensorNode(
        world=world,
        name="contact_force",
        option=ContactForceSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _settle(env)

    obs = node.get_observation()
    assert obs["contact_force"].shape == (3,)
    assert obs["contact_force"].dtype == world.dtype
    # Box resting on plane -> upward (positive z) normal force in the link frame.
    assert obs["contact_force"][2] > 0.0
    env.close()


def test_contact_force_observation_space_matches():
    world = GenesisWorld()
    node = ContactForceSensorNode(
        world=world,
        name="contact_force",
        option=ContactForceSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _settle(env)

    obs = node.get_observation()
    for key, space in node.observation_space.spaces.items():
        assert key in obs, f"Key {key!r} in obs space but not in obs"
        assert obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={obs[key].shape}, space={space.shape}"
        )
    env.close()


# ---------------------------------------------------------------------------
# ContactSensorNode
# ---------------------------------------------------------------------------


def test_contact_shapes():
    world = GenesisWorld()
    node = ContactSensorNode(
        world=world,
        name="contact",
        option=ContactSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _settle(env)

    obs = node.get_observation()
    assert obs["in_contact"].shape == ()
    assert obs["in_contact"].dtype == torch.bool
    assert bool(obs["in_contact"]) is True
    env.close()


def test_contact_observation_space_matches():
    world = GenesisWorld()
    node = ContactSensorNode(
        world=world,
        name="contact",
        option=ContactSensorNodeOptions(attach=AttachFromName("box", BOX_LINK)),
    )
    env = _make_env(world, [node])
    _settle(env)

    obs = node.get_observation()
    for key, space in node.observation_space.spaces.items():
        assert key in obs, f"Key {key!r} in obs space but not in obs"
        assert obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={obs[key].shape}, space={space.shape}"
        )
    env.close()


# ---------------------------------------------------------------------------
# n_settle_steps: meaningful observations immediately after reset
# ---------------------------------------------------------------------------


def test_settle_steps_populates_imu_and_contacts_after_reset():
    """With ``n_settle_steps`` configured, every sensor returns meaningful
    data on the very first ``env.reset()`` — no manual settling needed."""
    world = GenesisWorld(n_settle_steps=5)
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

    # No _settle() call — observations must be meaningful right after reset.
    imu_obs = imu.get_observation()
    cf_obs = cf.get_observation()
    ct_obs = ct.get_observation()

    # IMU accelerometer should reflect gravity.
    assert imu_obs["lin_acc"].shape == (3,)
    assert imu_obs["lin_acc"][2] > 5.0, f"IMU lin_acc[2]={imu_obs['lin_acc'][2]}"

    # Box resting on plane → positive z normal force in the link frame.
    assert cf_obs["contact_force"].shape == (3,)
    assert cf_obs["contact_force"][2] > 0.0, f"contact_force[2]={cf_obs['contact_force'][2]}"

    # Box touching plane → in_contact is True.
    assert ct_obs["in_contact"].shape == ()
    assert bool(ct_obs["in_contact"]) is True
    env.close()


def test_settle_steps_populates_on_subsequent_reset():
    """After a non-reload reset, sensors should still be populated."""
    world = GenesisWorld(n_settle_steps=5)
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

    # Step a few times then do a plain (non-reload) reset.
    _settle(env, n=3)
    env.reset(seed=1)

    imu_obs = imu.get_observation()
    cf_obs = cf.get_observation()
    assert imu_obs["lin_acc"][2] > 5.0
    assert cf_obs["contact_force"][2] > 0.0
    env.close()
