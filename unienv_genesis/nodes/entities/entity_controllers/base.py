from typing import Literal, Optional, Dict, Any, List, Tuple, Union, Generic, TypeVar, Sequence
import abc
import genesis as gs
from genesis.engine import entities as gs_entities
from genesis.engine.entities.base_entity import Entity
import torch
from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_genesis.world.genesis_world import GenesisWorld
from ..base import EntityNode, EntityType, ContextType, ObsType, ActType

class EntityController(
    abc.ABC,
    Generic[EntityType, ObsType, ActType]
):
    """Base controller for Genesis entities."""
    controller_type : str

    def __init__(
        self, 
        world : GenesisWorld,
        entity: EntityType,
        observation_space : Optional[Space[ObsType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]],
        action_space : Space[ActType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType],
    ):
        self.world = world
        self.entity = entity
        self.observation_space = observation_space
        self.action_space = action_space

    def get_observation(self) -> Optional[ObsType]:
        return None

    @abc.abstractmethod
    def apply_action(self, action : ActType) -> None:
        raise NotImplementedError
    
    def reload(self, mask : Optional[PyTorchArrayType] = None) -> None:
        pass

    def reset(self, mask : Optional[PyTorchArrayType] = None) -> None:
        pass

    def after_reload(self, mask : Optional[PyTorchArrayType] = None) -> None:
        pass

    def after_reset(self, mask : Optional[PyTorchArrayType] = None) -> None:
        pass

class DiscreteEntityController(
    abc.ABC,
    EntityController[
        gs_entities.RigidEntity,
        Dict[str, Any],
        ActType
    ],
    Generic[ActType]
):
    @staticmethod
    def _broadcast_to_dofs(
        scalar_or_array: Union[float, PyTorchArrayType],
        batch_size: int,
        n_dofs: int,
        device: Optional[PyTorchDeviceType] = None,
    ) -> PyTorchArrayType:
        if isinstance(scalar_or_array, float):
            array = torch.full((batch_size, n_dofs), scalar_or_array, device=device)
        else:
            if scalar_or_array.ndim == 1:
                array = scalar_or_array.unsqueeze(0).expand(batch_size, -1)
            else:
                array = scalar_or_array
        return array

    def __init__(
        self,
        world : GenesisWorld,
        entity : gs_entities.RigidEntity,
        joint_indexes : Optional[Sequence[int]] = None,
        action_space : Optional[Space[ActType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]] = None,
    ) -> None:
        super().__init__(world, entity, action_space)
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        n_envs = world.batch_size
        self.joint_position_low, self.joint_position_high = self._broadcast_to_dofs(entity.get_dofs_limit(dofs_idx_local=joint_indexes), n_envs, n_dofs)
        self.joint_force_low, self.joint_force_high = self._broadcast_to_dofs(entity.get_dofs_force_range(dofs_idx_local=joint_indexes), n_envs, n_dofs)
        self.joint_indexes = joint_indexes

        self.observation_space = DictSpace(
            PyTorchComputeBackend,
            {
                "joint_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=self.joint_position_low[..., 0],
                    high=self.joint_position_high[..., 1],
                    dtype=self.joint_position_low.dtype,
                    device=self.joint_position_low.device,
                ),
                "joint_velocity": BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=(n_envs, n_dofs),
                    dtype=self.joint_position_low.dtype,
                    device=self.joint_position_low.device,
                ),
                "joint_torque": BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=(n_envs, n_dofs),
                    dtype=self.joint_force_low.dtype,
                    device=self.joint_force_low.device,
                ),
                "joint_motor_torque": BoxSpace(
                    PyTorchComputeBackend,
                    low=self.joint_force_low[..., 0],
                    high=self.joint_force_high[..., 1],
                    dtype=self.joint_force_low.dtype,
                    device=self.joint_force_low.device,
                ),
                "root_pos": BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=(n_envs, 3),
                    dtype=self.joint_force_low.dtype,
                    device=self.joint_force_low.device,
                ),
                "root_quat": BoxSpace(
                    PyTorchComputeBackend,
                    low=-1,
                    high=1,
                    shape=(n_envs, 4),
                    dtype=self.joint_force_low.dtype,
                    device=self.joint_force_low.device,
                ),
            },
            device=self.joint_position_low.device,
        )

    def get_observation(self):
        qpos = self.entity.get_dofs_position(dofs_idx_local=self.joint_indexes)
        qvel = self.entity.get_dofs_velocity(dofs_idx_local=self.joint_indexes)
        joint_torque = self.entity.get_dofs_force(dofs_idx_local=self.joint_indexes)
        joint_motor_torque = self.entity.get_dofs_control_force(dofs_idx_local=self.joint_indexes)
        root_pos = self.entity.get_pos()
        root_quat = self.entity.get_quat()
        return {
            'joint_position': qpos,
            'joint_velocity': qvel,
            'joint_torque': joint_torque,
            'joint_motor_torque': joint_motor_torque,
            'root_pos': root_pos,
            'root_quat': root_quat
        }

    @abc.abstractmethod
    def apply_action(self, action):
        raise NotImplementedError