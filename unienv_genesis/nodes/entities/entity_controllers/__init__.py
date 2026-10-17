from typing import Literal
from .base import EntityController, DiscreteEntityController, UnknownControllerOptionWarning, resolve_joint_indexes
from .joint_position_velocity_control import JointPositionAndVelocityController
from .joint_position import JointPositionController
from .joint_velocity import JointVelocityController
from .end_effector_pose import EndEffectorPoseController
from .combined_controller import CombinedController
from .mimic_controllers import (
    MimicJointPositionAndVelocityController,
    MimicJointPositionController,
    MimicJointVelocityController,
)

CONTROL_MODE_T = Literal[
    'joint_position_and_velocity',
    'joint_position',
    'joint_velocity',
    'eef_pose',
    'combined',
    'mimic_joint_position_and_velocity',
    'mimic_joint_position',
    'mimic_joint_velocity',
]
CONTROL_MODE_MAP = {
    'joint_position_and_velocity': JointPositionAndVelocityController,
    "joint_position": JointPositionController,
    "joint_velocity": JointVelocityController,
    "eef_pose": EndEffectorPoseController,
    "combined": CombinedController,
    "mimic_joint_position_and_velocity": MimicJointPositionAndVelocityController,
    "mimic_joint_position": MimicJointPositionController,
    "mimic_joint_velocity": MimicJointVelocityController,
}

__all__ = [
    "resolve_joint_indexes",
    "EntityController",
    "UnknownControllerOptionWarning",
    "DiscreteEntityController",
    "JointPositionAndVelocityController",
    "JointPositionController",
    "JointVelocityController",
    "EndEffectorPoseController",
    "CombinedController",
    "MimicJointPositionAndVelocityController",
    "MimicJointPositionController",
    "MimicJointVelocityController",
    "CONTROL_MODE_T",
    "CONTROL_MODE_MAP",
]