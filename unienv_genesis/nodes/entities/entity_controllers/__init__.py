from typing import Literal
from .base import EntityController, DiscreteEntityController
from .joint_position_velocity_control import JointPositionAndVelocityController
from .joint_position import JointPositionController
from .joint_velocity import JointVelocityController
from .end_effector_pose import EndEffectorPoseController

CONTROL_MODE_T = Literal[
    'joint_position_and_velocity',
    'joint_position',
    'joint_velocity',
    'eef_pose',
]
CONTROL_MODE_MAP = {
    'joint_position_and_velocity': JointPositionAndVelocityController,
    "joint_position": JointPositionController,
    "joint_velocity": JointVelocityController,
    "eef_pose": EndEffectorPoseController,
}

__all__ = [
    "EntityController",
    "DiscreteEntityController",
    "JointPositionAndVelocityController",
    "JointPositionController",
    "JointVelocityController",
    "EndEffectorPoseController",
    "CONTROL_MODE_T",
    "CONTROL_MODE_MAP",
]