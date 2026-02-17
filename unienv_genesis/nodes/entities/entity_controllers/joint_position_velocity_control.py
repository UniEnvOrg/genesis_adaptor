from typing import Literal, Optional, Dict, Any, List, Tuple, Union, Generic, TypeVar, Sequence
import abc
import genesis as gs
from genesis.engine import entities as gs_entities
import torch
from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import (
    broadcast_to_batch,
    get_batch_aware_shape,
    is_batched,
)
from .base import DiscreteEntityController, EntityType, ActType

class JointPositionAndVelocityController(
    DiscreteEntityController[
        PyTorchArrayType
    ]
):
    controller_type = "joint_position_and_velocity"

    def __init__(
        self,
        world : GenesisWorld,
        entity : gs_entities.RigidEntity,
        Kp : Optional[Union[float, PyTorchArrayType]] = None,
        Kd : Optional[Union[float, PyTorchArrayType]] = None,
        joint_indexes : Optional[Sequence[int]] = None,
        min_joint_vel : Optional[Union[float, PyTorchArrayType]] = None,
        max_joint_vel : Optional[Union[float, PyTorchArrayType]] = None,
    ) -> None:
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        batch_size = world.batch_size

        # Get limits from entity - returns (low, high) tuple of shape (batch_size, n_dofs) or (n_dofs,)
        joint_limit_low, joint_limit_high = entity.get_dofs_limit(dofs_idx_local=joint_indexes)

        # Build velocity limit tensors if provided
        vel_low = (
            broadcast_to_batch(min_joint_vel, batch_size, n_dofs, device=joint_limit_low.device)
            if min_joint_vel is not None else None
        )
        vel_high = (
            broadcast_to_batch(max_joint_vel, batch_size, n_dofs, device=joint_limit_low.device)
            if max_joint_vel is not None else None
        )

        action_space = DictSpace(
            PyTorchComputeBackend,
            {
                "target_joint_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=joint_limit_low,
                    high=joint_limit_high,
                    dtype=joint_limit_low.dtype,
                    device=joint_limit_low.device,
                ),
                "target_joint_velocity": BoxSpace(
                    PyTorchComputeBackend,
                    low=vel_low if vel_low is not None else -torch.inf,
                    high=vel_high if vel_high is not None else torch.inf,
                    shape=None if vel_low is not None else get_batch_aware_shape(batch_size, n_dofs),
                    dtype=joint_limit_low.dtype,
                    device=joint_limit_low.device,
                ),
            },
            device=joint_limit_low.device,
        )
        super().__init__(world, entity, joint_indexes, action_space)
        self.Kp = Kp
        self.Kd = Kd

    @property
    def Kp(self) -> Optional[Union[float, PyTorchArrayType]]:
        return self._Kp

    @Kp.setter
    def Kp(self, value: Optional[Union[float, PyTorchArrayType]]) -> None:
        self._Kp = value
        if value is not None:
            self._Kp_expanded = broadcast_to_batch(
                value, self.world.batch_size, self.entity.n_dofs, device=self.world.device
            )
            self.entity.set_dofs_kp(
                self._Kp_expanded, dofs_idx_local=self.joint_indexes
            )
        else:
            self._Kp_expanded = None

    @property
    def Kd(self) -> Optional[Union[float, PyTorchArrayType]]:
        return self._Kd

    @Kd.setter
    def Kd(self, value: Optional[Union[float, PyTorchArrayType]]) -> None:
        self._Kd = value
        if value is not None:
            self._Kd_expanded = broadcast_to_batch(
                value, self.world.batch_size, self.entity.n_dofs, device=self.world.device
            )
            self.entity.set_dofs_kv(
                self._Kd_expanded, dofs_idx_local=self.joint_indexes
            )
        else:
            self._Kd_expanded = None

    def reload(self, mask = None):
        if mask is not None:
            env_idx = torch.nonzero(mask)[0]
        else:
            env_idx = None
        # Set the controller for the entity
        if self._Kp_expanded is not None:
            self.entity.set_dofs_kp(self._Kp_expanded, dofs_idx_local=self.joint_indexes, envs_idx=env_idx)
        if self._Kd_expanded is not None:
            self.entity.set_dofs_kv(self._Kd_expanded, dofs_idx_local=self.joint_indexes, envs_idx=env_idx)

    def apply_action(self, action):
        self.entity.control_dofs_position_velocity(
            action['target_joint_position'],
            action['target_joint_velocity'],
            dofs_idx_local=self.joint_indexes
        )
