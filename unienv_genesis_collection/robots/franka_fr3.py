"""Franka Research FR3 robot descriptor."""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis_collection.asset_path import get_asset_path

# Default home joint positions for FR3 (from MJCF keyframe)
FR3_HOME_JOINT_POSITIONS = torch.tensor([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Default end-effector link name for FR3 (used in IK control)
FR3_EE_LINK_NAME = "fr3_link7"

class FrankaFR3(RobotDescriptionNode):
    """Franka Research FR3 7-DOF manipulator.

    The FR3 is the successor to the Panda robot with improved performance,
    safety features, and reliability. It has 7 revolute joints with
    torque sensors in each joint.

    Args:
        world: GenesisWorld instance
        name: Name for the robot entity
        controller_type: Type of controller to use (default: "joint_position_and_velocity")
        keyframes: Dictionary of keyframes for reset positions. If not provided,
            uses home position: [0, 0, 0, -1.57, 0, 1.57, -0.785]
        **controller_kwargs: Additional arguments for the controller (e.g., Kp, Kd)

    Example:
        >>> from unienv_genesis import GenesisWorld
        >>> from unienv_genesis_collection.robots.franka_fr3 import FrankaFR3
        >>>
        >>> world = GenesisWorld()
        >>> franka = FrankaFR3(world, name="franka")
        >>> 
        >>> # Or with custom controller settings
        >>> franka = FrankaFR3(
        ...     world,
        ...     name="franka",
        ...     controller_type="joint_position",
        ...     Kp=100.0
        ... )
    """

    keyframes = {
        "reset": RobotKeyframe(joint_position=FR3_HOME_JOINT_POSITIONS),
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
        if controller_type == 'eef_pose' and 'target_link' not in controller_kwargs:
            controller_kwargs['target_link'] = FR3_EE_LINK_NAME

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
