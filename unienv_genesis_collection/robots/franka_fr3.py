"""Franka Research FR3 robot descriptor with Franka Hand gripper.

MJCF sources (Apache 2.0):
  Arm:     https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis_collection.asset_path import get_asset_path

# Default home joint positions for FR3 arm (7 DOF) + Franka Hand (2 DOF, fully open)
HOME_JOINT_POSITIONS = torch.tensor([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# End-effector link name for IK control
EE_LINK_NAME = "fr3_link7"

class FrankaFR3(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm.

    The arm MJCF comes from the mujoco_menagerie ``franka_fr3`` model.

    Args:
        world: GenesisWorld instance
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"eef_pose"``.
        **controller_kwargs: Forwarded to the controller. 

    Example::

        from unienv_genesis import GenesisWorld
        from unienv_genesis_collection.robots.franka_fr3 import FrankaFR3

        world = GenesisWorld()
        robot = FrankaFR3(world, name="franka")
    """

    keyframes = {
        "reset": RobotKeyframe(joint_position=HOME_JOINT_POSITIONS),
    }

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "franka",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "eef_pose",
        **controller_kwargs,
    ):
        if controller_type == "eef_pose" and "target_link" not in controller_kwargs:
            controller_kwargs["target_link"] = EE_LINK_NAME

        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3", "fr3.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
