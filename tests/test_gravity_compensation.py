"""Tests for the gravity-compensation feedforward of joint-position controllers.

Covers:
  - ``GravityFeedforward.torque()`` equals ``-sum_l J_com,l^T m_l g`` from
    Genesis' per-link Jacobians, unbatched and per env in a batched scene.
  - Holding a pose: the PD sag disappears with compensation, while the motor
    still carries the gravity load (control force ~= g(q)).
  - ``act_bias[0]`` is restored on the next step after a Kp change (which
    resets it).
  - Guards: velocity control and batched worlds without batched DOF info
    reject ``gravity_compensation=True``.
"""

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.robots.franka_emika_panda import (
    FrankaEmikaPanda,
    HOME_JOINT_POSITIONS,
)
from unienv_genesis_collection.robots.tianji_marvin_wuji import TianjiMarvinWuji
from unienv_interface.world import WorldEnv
from xbarray.transformations.rotation_conversions.pytorch import (
    matrix_to_euler_angles,
    quaternion_to_matrix,
)

ARM = list(range(7))
POSE_OFFSET = [0.3, -0.4, 0.2, 0.3, -0.5, 0.4, 0.1]


def _make(robot_cls, world=None, **kwargs):
    world = GenesisWorld() if world is None else world
    robot = robot_cls(world, **kwargs)
    env = WorldEnv(world, [robot], render_mode=None)
    env.reset(seed=0, reload=True)
    return world, robot, env


def _batched_world(n_envs=2):
    return GenesisWorld(
        batch_size=n_envs,
        scene_init_kwargs=dict(rigid_options=gs.options.RigidOptions(batch_dofs_info=True)),
    )


def _jacobian_gravity_torque(entity, dofs):
    """Reference ``-sum_l J_com,l^T m_l g`` (``(..., len(dofs))``) from per-link Jacobians."""
    gravity = entity.solver.get_gravity()
    com = entity.get_links_pos(ref=gs.link_ref_frame.link_COM)
    link_pos = entity.get_links_pos(relative=False)
    link_rot = quaternion_to_matrix(entity.get_links_quat(relative=False))
    mass = entity.get_links_mass()
    tau = 0.0
    for link in entity.links:
        i = link.idx_local
        if float(mass[..., i].max()) <= 0:
            continue
        # link-local COM; identical across envs without COM randomization
        local = (link_rot[..., i, :, :].transpose(-1, -2) @ (com[..., i, :] - link_pos[..., i, :])[..., None])[..., 0]
        jac = entity.get_jacobian(link, local_point=local.reshape(-1, 3)[0])
        tau = tau - (jac[..., :3, :] * (mass[..., i, None] * gravity)[..., :, None]).sum(-2)
    return tau[..., dofs]


def _franka_pose(world):
    pose = HOME_JOINT_POSITIONS.to(dtype=world.dtype, device=gs.device).clone()
    pose[:7] += torch.tensor(POSE_OFFSET, dtype=world.dtype, device=gs.device)
    return pose


def test_torque_matches_jacobian():
    world, robot, env = _make(FrankaEmikaPanda, name="franka", controller_type="joint_position",
                              gravity_compensation=True)
    robot.entity.set_dofs_position(_franka_pose(world))
    ff = robot.controller._gravity_feedforward
    tau = ff.torque()
    ref = _jacobian_gravity_torque(robot.entity, ff.dofs_idx_local)
    assert tau.shape == ref.shape == (robot.entity.n_dofs,)
    assert tau[ARM].abs().max() > 1.0, "test pose should load the arm joints"
    torch.testing.assert_close(tau, ref, atol=1e-4, rtol=1e-4)


def test_batched_torque_is_per_env():
    world, robot, env = _make(FrankaEmikaPanda, world=_batched_world(), name="franka",
                              controller_type="joint_position", gravity_compensation=True)
    home = HOME_JOINT_POSITIONS.to(dtype=world.dtype, device=gs.device)
    poses = torch.stack([home, _franka_pose(world)])
    robot.entity.set_dofs_position(poses)
    ff = robot.controller._gravity_feedforward
    tau = ff.torque()
    assert tau.shape == (2, robot.entity.n_dofs)
    assert (tau[0] - tau[1]).abs().max() > 1.0
    torch.testing.assert_close(tau, _jacobian_gravity_torque(robot.entity, ff.dofs_idx_local), atol=1e-4, rtol=1e-4)

    # apply_action writes each env's own torque into act_bias[0]
    env.step(poses)
    bias0, _, _ = robot.entity.get_dofs_act_bias()
    torch.testing.assert_close(bias0, tau)


def _franka_hold(gravity_compensation):
    world, robot, env = _make(FrankaEmikaPanda, name="franka", controller_type="joint_position",
                              gravity_compensation=gravity_compensation)
    home = HOME_JOINT_POSITIONS.to(dtype=world.dtype, device=gs.device)
    for _ in range(int(round(1.0 / world.scene.dt))):
        env.step(home)
    error = float((robot.entity.get_dofs_position() - home)[ARM].abs().max())
    return error, robot


def test_franka_hold_removes_sag():
    plain, _ = _franka_hold(False)
    compensated, robot = _franka_hold(True)
    assert compensated < 0.1 * plain, f"arm error {plain:.2e} -> {compensated:.2e} rad"
    # the motor carries the gravity load: at rest the control force is the feedforward
    force = robot.entity.get_dofs_control_force(ARM)
    ghat = robot.controller._gravity_feedforward.torque()[ARM]
    torch.testing.assert_close(force, ghat, atol=1e-2, rtol=1e-2)
    assert ghat.abs().max() > 1.0


def _tianji_palm_drift(gravity_compensation):
    world, robot, env = _make(TianjiMarvinWuji, gravity_compensation=gravity_compensation)
    obs = robot.controller.get_observation()
    action = {}
    for side in ("left", "right"):
        arm = obs[f"{side}_arm"]
        euler = matrix_to_euler_angles(quaternion_to_matrix(arm["eef_quaternion"]), "XYZ")
        action[f"{side}_arm"] = torch.cat([arm["eef_position"], euler])
        action[f"{side}_hand"] = torch.zeros(20, device=arm["eef_position"].device)
    start = obs["right_arm"]["eef_position"].clone()
    for _ in range(int(round(2.0 / world.scene.dt))):
        env.step(action)
    end = robot.controller.get_observation()["right_arm"]["eef_position"]
    return 1000 * float((end - start).norm())


def test_tianji_eef_hold_removes_drift():
    plain = _tianji_palm_drift(False)
    compensated = _tianji_palm_drift(True)
    assert compensated < 1.0, f"right palm drift {plain:.2f} -> {compensated:.2f} mm"


def test_feedforward_restored_after_kp_change():
    world, robot, env = _make(FrankaEmikaPanda, name="franka", controller_type="joint_position",
                              gravity_compensation=True)
    home = HOME_JOINT_POSITIONS.to(dtype=world.dtype, device=gs.device)
    env.step(home)
    robot.controller.Kp = 500.0
    bias0, _, _ = robot.entity.get_dofs_act_bias()
    assert torch.all(bias0 == 0), "set_dofs_kp is expected to reset act_bias[0]"

    expected = robot.controller._gravity_feedforward.torque()
    env.step(home)
    bias0, bias1, _ = robot.entity.get_dofs_act_bias()
    torch.testing.assert_close(bias0, expected)
    torch.testing.assert_close(bias1, torch.full_like(bias1, -500.0))


def test_velocity_control_rejects_gravity_compensation():
    # the robot node builds its controller on construction
    with pytest.raises(ValueError, match="velocity control"):
        FrankaEmikaPanda(GenesisWorld(), name="franka", controller_type="joint_velocity", gravity_compensation=True)


def test_batched_world_requires_batched_dofs_info():
    with pytest.raises(ValueError, match="batch_dofs_info"):
        FrankaEmikaPanda(GenesisWorld(batch_size=2), name="franka", controller_type="joint_position",
                         gravity_compensation=True)
