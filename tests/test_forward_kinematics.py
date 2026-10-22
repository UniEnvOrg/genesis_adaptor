"""Forward kinematics on the current Genesis API (``forward_kinematics_query``).

Genesis removed ``RigidEntity.forward_kinematics`` (genesis-world#3233), which
the robot node's FK primitive and the EEF controllers'
``compute_eef_pose_from_qpos`` / ``get_action_dict`` called.
"""

import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.utils import forward_kinematics
from unienv_genesis_collection.robots.franka_emika_panda import FrankaEmikaPanda
from unienv_interface.world import WorldEnv


def _franka(batch_size=None):
    world = GenesisWorld(batch_size=batch_size) if batch_size is not None else GenesisWorld()
    robot = FrankaEmikaPanda(world, name="franka")
    env = WorldEnv(world, [robot], render_mode=None)
    env.reset(seed=0, reload=True)
    return robot, env


def test_forward_kinematics_matches_live_and_set_poses():
    robot, env = _franka()
    try:
        entity = robot.entity
        arm = robot.controller["arm"]
        link = arm.target_link

        # At the live configuration: the live link pose.
        pos, quat = forward_kinematics(entity, entity.get_qpos(), links_idx_local=[link.idx_local])
        assert torch.allclose(pos[0], link.get_pos(), atol=1e-5)
        assert torch.allclose(quat[0], link.get_quat(), atol=1e-5)

        # At another arm configuration: the pose the link takes once set, the live state untouched.
        q_live = entity.get_qpos().clone()
        q_arm = entity.get_dofs_position(dofs_idx_local=arm.joint_indexes) + 0.2
        pos, quat = robot.forward_kinematics(q_arm, qs_idx_local=arm.joint_indexes, links_idx_local=[link.idx_local])
        assert torch.allclose(entity.get_qpos(), q_live)
        entity.set_dofs_position(q_arm, dofs_idx_local=arm.joint_indexes)
        assert torch.allclose(pos[0], link.get_pos(), atol=1e-5)
        assert torch.allclose(quat[0], link.get_quat(), atol=1e-5)
    finally:
        env.close()


def test_eef_pose_from_qpos_and_action_dict_batched():
    robot, env = _franka(batch_size=2)
    try:
        arm = robot.controller["arm"]
        q = arm.entity.get_dofs_position(dofs_idx_local=arm.joint_indexes)
        assert q.shape == (2, 7)
        pos, quat = arm.compute_eef_pose_from_qpos(q)
        live_pos, live_quat = arm.get_eef_pose_base()
        assert pos.shape == (2, 3) and quat.shape == (2, 4)
        assert torch.allclose(pos, live_pos, atol=1e-5)
        assert torch.allclose(quat.abs(), live_quat.abs(), atol=1e-5)

        # get_action_dict reports the FK pose of the (clipped) joint target.
        action = arm.get_action_world(live_pos, live_quat)
        action_dict = arm.get_action_dict(action)
        assert all(torch.isfinite(v).all() for v in action_dict.values() if isinstance(v, torch.Tensor))
    finally:
        env.close()
