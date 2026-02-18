"""Tianji Marvin dual-arm robot with Robotiq 2F-85 grippers.

The Marvin robot is a 14-DOF dual-arm manipulator with a fixed torso base.
Each arm has 7 revolute joints and a Robotiq 2F-85 gripper attached at the
flange. The MJCF was assembled from the tianji_sim marvin_description and the
mujoco_menagerie robotiq_2f85_v4 model.

MJCF sources:
  Arms:     tianji_sim/marvin_description (converted from URDF)
  Grippers: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85_v4
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis_collection.asset_path import get_asset_path

# Default home position: 26 DOF, all zeros (arms neutral, grippers open)
HOME_JOINT_POSITIONS = torch.zeros(26)

# End-effector link names (last actuated arm link before gripper)
LEFT_EE_LINK_NAME = "left_link7"
RIGHT_EE_LINK_NAME = "right_link7"

# Joint names for each arm and gripper (resolved to indexes at controller init)
LEFT_ARM_JOINT_NAMES = [f"left_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"right_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_DRIVER_NAMES = ["lg_left_driver_joint", "lg_right_driver_joint"]
RIGHT_GRIPPER_DRIVER_NAMES = ["rg_left_driver_joint", "rg_right_driver_joint"]

# Default combined sub-controller configuration
_DEFAULT_SUB_CONTROLLERS = {
    "left_arm": {
        "controller_type": "eef_pose",
        "joint_names": LEFT_ARM_JOINT_NAMES,
        "target_link": LEFT_EE_LINK_NAME,
    },
    "right_arm": {
        "controller_type": "eef_pose",
        "joint_names": RIGHT_ARM_JOINT_NAMES,
        "target_link": RIGHT_EE_LINK_NAME,
    },
    "left_gripper": {
        "controller_type": "mimic_joint_position",
        "joint_names": LEFT_GRIPPER_DRIVER_NAMES,
        # Gripper convention: -1 = fully open (driver=0), +1 = fully closed (driver=0.9)
        # finger_pos = action * multiplier + shift
        "multiplier": 0.45,
        "shift": 0.45,
        "control_low": -1,
        "control_high": 1,
    },
    "right_gripper": {
        "controller_type": "mimic_joint_position",
        "joint_names": RIGHT_GRIPPER_DRIVER_NAMES,
        "multiplier": 0.45,
        "shift": 0.45,
        "control_low": -1,
        "control_high": 1,
    },
}


class TianjiMarvinWithRobotiq2F85(RobotDescriptionNode):
    """Tianji Marvin 14-DOF dual-arm robot with Robotiq 2F-85 grippers.

    Each arm has 7 revolute joints plus a Robotiq 2F-85 gripper (6 joints
    per gripper, 1 actuated DOF via driver-joint mimic control).

    Default controller is ``"combined"``:
      - ``"left_arm"``      sub-controller: ``eef_pose``
      - ``"right_arm"``     sub-controller: ``eef_pose``
      - ``"left_gripper"``  sub-controller: ``mimic_joint_position``
      - ``"right_gripper"`` sub-controller: ``mimic_joint_position``

    Gripper action convention::

        -1  ->  fully open  (driver joints at 0 rad)
        +1  ->  fully closed (driver joints at 0.9 rad)

    Args:
        world: GenesisWorld instance
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller. For the default
            ``"combined"`` controller, pass ``sub_controllers=...`` to override.

    Example::

        from unienv_genesis import GenesisWorld
        from unienv_genesis_collection.robots.tianji_marvin_robotiq_2f85 import (
            TianjiMarvinWithRobotiq2F85,
        )

        world = GenesisWorld()
        robot = TianjiMarvinWithRobotiq2F85(world, name="tianji")

        # Action dict expected by the default combined controller:
        # {
        #   "left_arm":      <eef_pose action tensor>,
        #   "right_arm":     <eef_pose action tensor>,
        #   "left_gripper":  tensor([-1.0]),   # open
        #   "right_gripper": tensor([-1.0]),   # open
        # }
    """

    keyframes = {
        "reset": RobotKeyframe(joint_position=HOME_JOINT_POSITIONS),
    }

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "tianji",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        if controller_type == "eef_pose" and "target_link" not in controller_kwargs:
            controller_kwargs["target_link"] = LEFT_EE_LINK_NAME

        super().__init__(
            world=world,
            description_file=str(get_asset_path(
                "robots", "tianji_marvin",
                "tianji_marvin_CCS_with_robotiq_2f85.xml",
            )),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
