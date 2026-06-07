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
from .base import DiscreteEntityController, EntityType, ActType, resolve_joint_indexes

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
        max_joint_vel : Optional[Union[float, PyTorchArrayType]] = None,
        max_joint_pos_vel : Optional[Union[float, PyTorchArrayType]] = None,
        joint_names : Optional[Sequence[str]] = None,
        is_grasping_fn = None,
        position_control_low : Optional[Union[float, PyTorchArrayType]] = None,
        position_control_high : Optional[Union[float, PyTorchArrayType]] = None,
        **kwargs,
    ) -> None:
        joint_indexes = resolve_joint_indexes(entity, joint_indexes, joint_names)
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        batch_size = world.batch_size

        # Normalize tensor-valued config for device safety before any
        # broadcast_to_batch / BoxSpace construction.
        dev = world.device
        Kp = Kp.to(device=dev, dtype=torch.float32) if isinstance(Kp, torch.Tensor) else Kp
        Kd = Kd.to(device=dev, dtype=torch.float32) if isinstance(Kd, torch.Tensor) else Kd
        max_joint_vel = max_joint_vel.to(device=dev, dtype=torch.float32) if isinstance(max_joint_vel, torch.Tensor) else max_joint_vel
        max_joint_pos_vel = max_joint_pos_vel.to(device=dev, dtype=torch.float32) if isinstance(max_joint_pos_vel, torch.Tensor) else max_joint_pos_vel
        position_control_low = position_control_low.to(device=dev, dtype=torch.float32) if isinstance(position_control_low, torch.Tensor) else position_control_low
        position_control_high = position_control_high.to(device=dev, dtype=torch.float32) if isinstance(position_control_high, torch.Tensor) else position_control_high

        # Get limits from entity - returns (low, high) tuple of shape (batch_size, n_dofs) or (n_dofs,)
        joint_limit_low, joint_limit_high = entity.get_dofs_limit(dofs_idx_local=joint_indexes)

        # Build velocity limit tensor if provided (symmetric around zero)
        vel_limit = (
            broadcast_to_batch(max_joint_vel, batch_size, n_dofs, device=joint_limit_low.device)
            if max_joint_vel is not None else None
        )

        # Position action-space bounds: use explicit overrides if provided,
        # otherwise fall back to the entity's joint position limits.
        pos_action_low = (
            broadcast_to_batch(position_control_low, batch_size, n_dofs, device=joint_limit_low.device)
            if position_control_low is not None else joint_limit_low
        )
        pos_action_high = (
            broadcast_to_batch(position_control_high, batch_size, n_dofs, device=joint_limit_low.device)
            if position_control_high is not None else joint_limit_high
        )

        action_space = DictSpace(
            PyTorchComputeBackend,
            {
                "target_joint_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=pos_action_low,
                    high=pos_action_high,
                    dtype=joint_limit_low.dtype,
                    device=joint_limit_low.device,
                ),
                "target_joint_velocity": BoxSpace(
                    PyTorchComputeBackend,
                    low=-vel_limit if vel_limit is not None else -torch.inf,
                    high=vel_limit if vel_limit is not None else torch.inf,
                    shape=None if vel_limit is not None else get_batch_aware_shape(batch_size, n_dofs),
                    dtype=joint_limit_low.dtype,
                    device=joint_limit_low.device,
                ),
            },
            device=joint_limit_low.device,
        )
        super().__init__(
            world, entity, joint_indexes, action_space,
            is_grasping_fn=is_grasping_fn,
            max_joint_vel=max_joint_vel,
            max_joint_pos_vel=max_joint_pos_vel,
            **kwargs,
        )
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

    def reload(self):
        super().reload()
        # Set the controller for the entity
        if self._Kp_expanded is not None:
            self.entity.set_dofs_kp(self._Kp_expanded, dofs_idx_local=self.joint_indexes)
        if self._Kd_expanded is not None:
            self.entity.set_dofs_kv(self._Kd_expanded, dofs_idx_local=self.joint_indexes)

    def apply_action(self, action, dt: Optional[float] = None):
        target_qpos = self._clip_target_qpos(action['target_joint_position'], dt)
        target_qvel = self._clip_target_qvel(action['target_joint_velocity'])
        self.entity.control_dofs_position_velocity(
            target_qpos,
            target_qvel,
            dofs_idx_local=self.joint_indexes
        )

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return structured action dict for combined position+velocity control.

        ``action`` is expected to be a dict with keys ``target_joint_position``
        and ``target_joint_velocity``.  The returned dict includes both
        (after rate-limit / velocity clipping), plus optional FK-based EEF
        pose derived from the clipped position target.
        """
        target_qpos = self._clip_target_qpos(action['target_joint_position'], dt)
        target_qvel = self._clip_target_qvel(action['target_joint_velocity'])
        result = self._get_action_dict_from_qpos(target_qpos)
        result['target_joint_velocity'] = target_qvel
        return result
