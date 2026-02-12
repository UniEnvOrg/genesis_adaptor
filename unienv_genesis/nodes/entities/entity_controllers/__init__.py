from typing import Literal
from .base import EntityControllerNode, DiscreteEntityControllerNode
from .joint_position_velocity_control import JointPositionAndVelocityControllerNode
from .joint_position import JointPositionControllerNode
from .joint_velocity import JointVelocityControllerNode

CONTROL_MODE_T = Literal[
    'joint_position_and_velocity',
    'joint_position',
    'joint_velocity',
]
CONTROL_MODE_MAP = {
    'joint_position_and_velocity': JointPositionAndVelocityControllerNode,
    "joint_position": JointPositionControllerNode,
    "joint_velocity": JointVelocityControllerNode,
}