from typing import Literal, Optional, Dict, Any, List, Tuple, Union, Generic, TypeVar, Sequence
import abc
import genesis as gs
from genesis.engine import entities as gs_entities
from genesis.engine.entities.base_entity import Entity
import torch
from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import (
    get_batch_aware_shape,
    is_batched,
)
from ..base import EntityNode, EntityType, ContextType, ObsType, ActType


def resolve_joint_indexes(
    entity: gs_entities.RigidEntity,
    joint_indexes: Optional[Sequence[int]] = None,
    joint_names: Optional[Sequence[str]] = None,
) -> Optional[Sequence[int]]:
    """Resolve joint selection to numeric DOF indexes.

    Users may specify joints either by index (``joint_indexes``) or by name
    (``joint_names``).  At most one may be provided.  If neither is given the
    function returns ``None``, which conventionally means *all joints*.

    Args:
        entity: The rigid entity whose joints are queried.
        joint_indexes: Numeric DOF indexes (passed through unchanged).
        joint_names: Joint names to look up on the entity.

    Returns:
        A list of integer DOF indexes, or ``None``.

    Raises:
        ValueError: If both ``joint_indexes`` and ``joint_names`` are given,
            or if any name in ``joint_names`` is not found on the entity.
    """
    if joint_names is not None and joint_indexes is not None:
        raise ValueError("Specify joint_indexes or joint_names, not both")
    if joint_names is not None:
        name_to_idx = {j.name: j.dofs_idx_local for j in entity.joints}
        missing = [n for n in joint_names if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"Joint names not found on entity: {missing}. "
                f"Available joints: {list(name_to_idx.keys())}"
            )
        # dofs_idx_local returns a list per joint; flatten into a single list.
        joint_indexes = [idx for n in joint_names for idx in name_to_idx[n]]
    return joint_indexes


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
        **kwargs,
    ):
        self.world = world
        self.entity = entity
        self.observation_space = observation_space
        self.action_space = action_space
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_observation(self) -> Optional[ObsType]:
        return None

    @abc.abstractmethod
    def apply_action(self, action : ActType) -> None:
        raise NotImplementedError
    
    def reload(self) -> None:
        pass

    def reset(self, mask : Optional[PyTorchArrayType] = None) -> None:
        pass

    def after_reload(self) -> None:
        pass

    def after_reset(self, mask : Optional[PyTorchArrayType] = None) -> None:
        pass

class DiscreteEntityController(
    EntityController[
        gs_entities.RigidEntity,
        Dict[str, Any],
        ActType
    ],
    Generic[ActType]
):
    def __init__(
        self,
        world : GenesisWorld,
        entity : gs_entities.RigidEntity,
        joint_indexes : Optional[Sequence[int]] = None,
        action_space : Optional[Space[ActType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]] = None,
        joint_names : Optional[Sequence[str]] = None,
        is_grasping_fn = None,
        **kwargs,
    ) -> None:
        joint_indexes = resolve_joint_indexes(entity, joint_indexes, joint_names)
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        batch_size = world.batch_size
        
        # Get limits from entity - returns (low, high) tuple of shape (batch_size, n_dofs) or (n_dofs,)
        joint_position_low, joint_position_high = entity.get_dofs_limit(dofs_idx_local=joint_indexes)
        joint_force_low, joint_force_high = entity.get_dofs_force_range(dofs_idx_local=joint_indexes)
        
        # Build observation space before calling super().__init__
        obs_space = DictSpace(
            PyTorchComputeBackend,
            {
                "joint_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=joint_position_low,
                    high=joint_position_high,
                    dtype=joint_position_low.dtype,
                    device=joint_position_low.device,
                ),
                "joint_velocity": BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=get_batch_aware_shape(batch_size, n_dofs),
                    dtype=joint_position_low.dtype,
                    device=joint_position_low.device,
                ),
                "joint_torque": BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=get_batch_aware_shape(batch_size, n_dofs),
                    dtype=joint_force_low.dtype,
                    device=joint_force_low.device,
                ),
                "joint_motor_torque": BoxSpace(
                    PyTorchComputeBackend,
                    low=joint_force_low,
                    high=joint_force_high,
                    dtype=joint_force_low.dtype,
                    device=joint_force_low.device,
                ),
                "root_pos": BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=get_batch_aware_shape(batch_size, 3),
                    dtype=joint_force_low.dtype,
                    device=joint_force_low.device,
                ),
                "root_quat": BoxSpace(
                    PyTorchComputeBackend,
                    low=-1,
                    high=1,
                    shape=get_batch_aware_shape(batch_size, 4),
                    dtype=joint_force_low.dtype,
                    device=joint_force_low.device,
                ),
            },
            device=joint_position_low.device,
        )
        
        super().__init__(world, entity, obs_space, action_space, **kwargs)

        # Store the broadcasted limits as attributes
        self.joint_position_low = joint_position_low
        self.joint_position_high = joint_position_high
        self.joint_force_low = joint_force_low
        self.joint_force_high = joint_force_high
        self.joint_indexes = joint_indexes

        self._is_grasping_fn = is_grasping_fn

    def is_grasping(self, target_entity):
        """Check if this controller's entity is grasping the target.

        Returns ``None`` if no ``is_grasping`` function was configured.
        The callable signature is ``(robot_entity, target_entity) -> bool | Tensor[bool]``.
        """
        if self._is_grasping_fn is None:
            return None
        return self._is_grasping_fn(self.entity, target_entity)

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