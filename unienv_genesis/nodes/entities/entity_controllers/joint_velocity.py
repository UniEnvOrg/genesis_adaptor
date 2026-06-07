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
        max_joint_vel : Optional[Union[float, PyTorchArrayType]] = None,
        joint_names : Optional[Sequence[str]] = None,
        is_grasping_fn = None,
        **kwargs,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes, max_joint_vel, joint_names=joint_names, is_grasping_fn=is_grasping_fn, **kwargs)
        self.action_space = self.action_space['target_joint_velocity']

    def apply_action(self, action, dt: Optional[float] = None):
        target_qvel = self._clip_target_qvel(action)
        self.entity.control_dofs_velocity(
            target_qvel,
            dofs_idx_local=self.joint_indexes
        )

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return structured action dict for joint velocity control.

        ``action`` is a tensor of target joint velocities.  Only
        ``target_joint_velocity`` is returned (after velocity clipping).
        """
        target_qvel = self._clip_target_qvel(action)
        return {"target_joint_velocity": target_qvel}
