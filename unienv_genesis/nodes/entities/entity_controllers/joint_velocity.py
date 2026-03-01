from typing import Literal, Optional, Dict, Any, List, Tuple, Union, Generic, TypeVar, Sequence
import abc
import genesis as gs
from genesis.engine import entities as gs_entities
import torch
from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_genesis.world.genesis_world import GenesisWorld
from .base import DiscreteEntityController, EntityType, ActType
from .joint_position_velocity_control import JointPositionAndVelocityController

class JointVelocityController(
    JointPositionAndVelocityController
):
    controller_type = "joint_velocity"

    def __init__(
        self,
        world : GenesisWorld,
        entity : gs_entities.RigidEntity,
        Kp : Optional[Union[float, PyTorchArrayType]] = None,
        Kd : Optional[Union[float, PyTorchArrayType]] = None,
        joint_indexes : Optional[Sequence[int]] = None,
        min_joint_vel : Optional[Union[float, PyTorchArrayType]] = None,
        max_joint_vel : Optional[Union[float, PyTorchArrayType]] = None,
        joint_names : Optional[Sequence[str]] = None,
        is_grasping_fn = None,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes, min_joint_vel, max_joint_vel, joint_names=joint_names, is_grasping_fn=is_grasping_fn)
        self.action_space = self.action_space['target_joint_velocity']

    def apply_action(self, action):
        self.entity.control_dofs_velocity(
            action,
            dofs_idx_local=self.joint_indexes
        )
