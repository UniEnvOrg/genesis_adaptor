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

class JointPositionController(
    JointPositionAndVelocityController
):
    controller_type = "joint_position"

    def __init__(
        self,
        world : GenesisWorld,
        entity : gs_entities.RigidEntity,
        Kp : Optional[Union[float, PyTorchArrayType]] = None,
        Kd : Optional[Union[float, PyTorchArrayType]] = None,
        joint_indexes : Optional[Sequence[int]] = None,
        joint_names : Optional[Sequence[str]] = None,
        is_grasping_fn = None,
        position_control_low : Optional[Union[float, PyTorchArrayType]] = None,
        position_control_high : Optional[Union[float, PyTorchArrayType]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            world, entity, Kp, Kd, joint_indexes,
            joint_names=joint_names,
            is_grasping_fn=is_grasping_fn,
            position_control_low=position_control_low,
            position_control_high=position_control_high,
            **kwargs,
        )
        self.action_space = self.action_space['target_joint_position']

    def apply_action(self, action, dt: Optional[float] = None):
        target_qpos = self._clip_target_qpos(action, dt)
        self.entity.control_dofs_position(
            target_qpos,
            dofs_idx_local=self.joint_indexes
        )

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return structured action dict for joint position control.

        ``action`` is a tensor of target joint positions.  The returned dict
        includes ``target_joint_position`` (after rate-limit clipping) and,
        when an end-effector is configured, the FK-based
        ``target_eef_position`` / ``target_eef_quaternion``.
        """
        target_qpos = self._clip_target_qpos(action, dt)
        return self._get_action_dict_from_qpos(target_qpos)
