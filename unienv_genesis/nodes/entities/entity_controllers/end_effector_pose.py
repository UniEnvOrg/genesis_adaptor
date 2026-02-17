from typing import Literal, Optional, Dict, Any, List, Tuple, Union, Generic, TypeVar, Sequence
import abc
import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
import torch

from xbarray.transformations.rotation_conversions.pytorch import (
    euler_angles_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_invert,
)
from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import (
    index_batch_dimension,
    get_batch_aware_shape,
    is_batched,
)
from .base import DiscreteEntityController, EntityType, ActType
from .joint_position_velocity_control import JointPositionAndVelocityController
from typing import TYPE_CHECKING

class AbsoluteRotationConventionUtils:
    AVAILABLE_REPRESENTATIONS = ["euler", "quat", "rot6d"]
    AVAILABLE_REPRESENTATIONS_T = Literal["euler", "quat", "rot6d"]

    @staticmethod
    def to_quaternion(representation: AVAILABLE_REPRESENTATIONS_T, rotation: PyTorchArrayType) -> PyTorchArrayType:
        if representation == "euler":
            rot_matrix = euler_angles_to_matrix(rotation, convention="XYZ")
            quat = matrix_to_quaternion(rot_matrix)
            return quat
        elif representation == "quat":
            return rotation
        elif representation == "rot6d":
            rot_matrix = rotation_6d_to_matrix(rotation)
            quat = matrix_to_quaternion(rot_matrix)
            return quat
        else:
            raise ValueError(f"Unsupported rotation representation: {representation}")
    
    @staticmethod
    def get_action_space(
        representation: AVAILABLE_REPRESENTATIONS_T, 
        batch_size: Optional[int],
        device: Optional[PyTorchDeviceType]
    ) -> BoxSpace[PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        if representation == "euler":
            rot_dim = 3
        elif representation == "quat":
            rot_dim = 4
        elif representation == "rot6d":
            rot_dim = 6
        else:
            raise ValueError(f"Unsupported rotation representation: {representation}")
        
        # Position (3) + Rotation (rot_dim)
        action_dim = 3 + rot_dim
        return BoxSpace(
            PyTorchComputeBackend,
            low=-torch.inf,
            high=torch.inf,
            shape=get_batch_aware_shape(batch_size, action_dim),
            dtype=torch.float32,
            device=device,
        )
    
    @staticmethod
    def normalize_action(representation: AVAILABLE_REPRESENTATIONS_T, action: PyTorchArrayType) -> PyTorchArrayType:
        if representation == 'euler':
            action = (action + torch.pi) % (2 * torch.pi) - torch.pi # wrap to [-pi, pi]
            action = action / torch.pi # normalize to [-1, 1]
        return action
    
    @staticmethod
    def unnormalize_action(representation: AVAILABLE_REPRESENTATIONS_T, action: PyTorchArrayType) -> PyTorchArrayType:
        if representation == 'euler':
            action = action * torch.pi # unnormalize from [-1, 1] to [-pi, pi]
        elif representation == 'quat':
            action = action / torch.clip(torch.norm(action, dim=-1, keepdim=True), min=1e-6) # normalize to unit quaternion
        return action

class EndEffectorPoseController(
    JointPositionAndVelocityController
):
    controller_type = "eef_pose"

    def __init__(
        self,
        world : GenesisWorld,
        entity : RigidEntity,
        target_link : Union[str, RigidLink],
        base_frame : Optional[Union[str, RigidLink, RigidEntity]] = None,
        Kp : Optional[Union[float, PyTorchArrayType]] = None,
        Kd : Optional[Union[float, PyTorchArrayType]] = None,
        joint_indexes : Optional[Sequence[int]] = None,
        rotation_representation : str = "euler", # "euler", "quat", "rot6d"
        normalize_rotation_action : bool = False,
        **ik_kwargs,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes)
        self.base_frame = base_frame if isinstance(base_frame, (RigidLink, RigidEntity)) else entity.get_link(base_frame) if base_frame is not None else None
        self.target_link = target_link if isinstance(target_link, RigidLink) else entity.get_link(target_link)
        self.rotation_representation = rotation_representation
        self.normalize_rotation_action = normalize_rotation_action
        self.ik_kwargs = ik_kwargs
        
        batch_size = world.batch_size
        
        # Action space: 3D position + rotation (depends on representation)
        self.action_space = AbsoluteRotationConventionUtils.get_action_space(
            rotation_representation, batch_size, world.device
        )
        self.observation_space['eef_position'] = BoxSpace(
            PyTorchComputeBackend,
            low=-torch.inf,
            high=torch.inf,
            shape=get_batch_aware_shape(batch_size, 3),
            dtype=torch.float32,
            device=world.device,
        )
        self.observation_space['eef_quaternion'] = BoxSpace(
            PyTorchComputeBackend,
            low=-1,
            high=1,
            shape=get_batch_aware_shape(batch_size, 4),
            dtype=torch.float32,
            device=world.device,
        )
        self.observation_space['last_ik_solved_position'] = BoxSpace(
            PyTorchComputeBackend,
            low=-torch.inf,
            high=torch.inf,
            shape=get_batch_aware_shape(batch_size, 3),
            dtype=torch.float32,
            device=world.device,
        )
        self.observation_space['last_ik_solved_quaternion'] = BoxSpace(
            PyTorchComputeBackend,
            low=-torch.inf,
            high=torch.inf,
            shape=get_batch_aware_shape(batch_size, 4),
            dtype=torch.float32,
            device=world.device,
        )
        self.observation_space['last_ik_solved_error'] = BoxSpace(
            PyTorchComputeBackend,
            low=-torch.inf,
            high=torch.inf,
            shape=get_batch_aware_shape(batch_size, 6),
            dtype=torch.float32,
            device=world.device,
        )
        self._last_ik_result = {}
    
    def reset(self, mask: Optional[PyTorchArrayType] = None) -> None:
        """Reset the controller's IK result cache."""
        self._last_ik_result = {}
    
    def get_base_transform(self) -> Optional[Tuple[PyTorchArrayType, PyTorchArrayType]]:
        if self.base_frame is None:
            return None
        elif isinstance(self.base_frame, (RigidEntity, RigidLink)):
            base_position, base_quaternion = self.base_frame.get_pos(), self.base_frame.get_quat()
        return base_position, base_quaternion

    def convert_base_to_absolute(self, target_position_base : Optional[PyTorchArrayType], target_rotation_base : Optional[PyTorchArrayType]) -> Tuple[Optional[PyTorchArrayType], Optional[PyTorchArrayType]]:
        base_transform = self.get_base_transform()
        if base_transform is None:
            return target_position_base, target_rotation_base
        base_position, base_quaternion = base_transform
        if target_position_base is not None:
            target_position_absolute = quaternion_apply(base_quaternion, target_position_base) + base_position
        else:
            target_position_absolute = None
        if target_rotation_base is not None:
            target_rotation_absolute = quaternion_apply(base_quaternion, target_rotation_base)
        else:
            target_rotation_absolute = None
        return target_position_absolute, target_rotation_absolute

    def convert_absolute_to_base(self, target_position_absolute : Optional[PyTorchArrayType], target_rotation_absolute : Optional[PyTorchArrayType]) -> Tuple[Optional[PyTorchArrayType], Optional[PyTorchArrayType]]:
        base_transform = self.get_base_transform()
        if base_transform is None:
            return target_position_absolute, target_rotation_absolute
        base_position, base_quaternion = base_transform
        if target_position_absolute is not None:
            target_position_base = quaternion_apply(quaternion_invert(base_quaternion), target_position_absolute - base_position)
        else:
            target_position_base = None
        if target_rotation_absolute is not None:
            target_rotation_base = quaternion_apply(quaternion_invert(base_quaternion), target_rotation_absolute)
        else:
            target_rotation_base = None
        return target_position_base, target_rotation_base

    def solve_ik_absolute(self, target_position : Optional[PyTorchArrayType], target_quaternion : Optional[PyTorchArrayType]) -> Tuple[PyTorchArrayType, PyTorchArrayType, PyTorchArrayType, PyTorchArrayType]:
        """
        Solve IK for the target end-effector position and rotation
        Args:
            target_position: (batch_size, 3) tensor of target end-effector positions
            target_quaternion: (batch_size, 4) tensor of target end-effector rotations in quaternion format
        Returns:
            target_joint_positions: (batch_size, n_dof) tensor of target joint positions
            achieved_position: (batch_size, 3) tensor of achieved end-effector positions
            achieved_quaternion: (batch_size, 4) tensor of achieved end-effector rotations in quaternion format
            achieved_error: (batch_size, 6) tensor of end-effector pose error (position error + rotation error), where position error = target_position - achieved_position, rotation error = quat_to_rotvec(achieved_quaternion^{-1} * target_quaternion)
        """
        # This is a placeholder implementation. You should replace this with your actual IK solver.
        full_target_joint_positions, achieved_error = self.entity.inverse_kinematics(
            self.target_link,
            target_position,
            target_quaternion,
            return_error=True,
            dofs_idx_local=self.joint_indexes,
            **self.ik_kwargs,
        )

        if self.joint_indexes is not None:
            target_joint_positions = index_batch_dimension(
                full_target_joint_positions,
                self.joint_indexes,
                batch_size=self.world.batch_size,
            )
        else:
            target_joint_positions = full_target_joint_positions
        
        
        achieved_position_alllinks, achieved_quaternion_alllinks = self.entity.forward_kinematics(
            target_joint_positions
        )
        
        link_idx_local = self.target_link.idx - self.entity.link_start
        achieved_position = index_batch_dimension(
            achieved_position_alllinks,
            link_idx_local,
            batch_size=self.world.batch_size,
        )
        achieved_quaternion = index_batch_dimension(
            achieved_quaternion_alllinks,
            link_idx_local,
            batch_size=self.world.batch_size,
        )
        return target_joint_positions, achieved_position, achieved_quaternion, achieved_error

    def solve_ik_base(self, target_position : Optional[PyTorchArrayType], target_quaternion : Optional[PyTorchArrayType]) -> Tuple[PyTorchArrayType, PyTorchArrayType, PyTorchArrayType, PyTorchArrayType]:
        """
        Solve IK for the target end-effector position and rotation in the end-effector's base frame
        Args:
            target_position: (batch_size, 3) tensor of target end-effector positions in the end-effector's base frame
            target_quaternion: (batch_size, 4) tensor of target end-effector rotations in quaternion format in the end-effector's base frame
        Returns:
            target_joint_positions: (batch_size, n_dof) tensor of target joint positions
            achieved_position: (batch_size, 3) tensor of achieved end-effector positions in the end-effector's base frame
            achieved_quaternion: (batch_size, 4) tensor of achieved end-effector rotations in quaternion format in the end-effector's base frame
            achieved_error: (batch_size, 6) tensor of end-effector pose error (position error + rotation error), where position error = target_position - achieved_position, rotation error = quat_to_rotvec(achieved_quaternion^{-1} * target_quaternion)
        """
        target_position_absolute, target_quaternion_absolute = self.convert_base_to_absolute(target_position, target_quaternion)
        target_joint_positions, achieved_position_absolute, achieved_quaternion_absolute, achieved_error_absolute = self.solve_ik_absolute(target_position_absolute, target_quaternion_absolute)
        achieved_position_base, achieved_quaternion_base = self.convert_absolute_to_base(achieved_position_absolute, achieved_quaternion_absolute)
        
        base_transform = self.get_base_transform()
        if base_transform is None:
            achieved_error = achieved_error_absolute
        else:
            _, base_quaternion = base_transform
            achieved_position_error = quaternion_apply(quaternion_invert(base_quaternion), achieved_error_absolute[..., :3])
            achieved_rotation_error = achieved_error_absolute[..., 3:]
            achieved_error = torch.cat([achieved_position_error, achieved_rotation_error], dim=-1)
        return target_joint_positions, achieved_position_base, achieved_quaternion_base, achieved_error

    def get_eef_pose_absolute(self) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        achieved_position_absolute, achieved_quaternion_absolute = self.target_link.get_pos(), self.target_link.get_quat()
        return achieved_position_absolute, achieved_quaternion_absolute

    def get_eef_pose_base(self) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        achieved_position_absolute, achieved_quaternion_absolute = self.get_eef_pose_absolute()
        achieved_position_base, achieved_quaternion_base = self.convert_absolute_to_base(achieved_position_absolute, achieved_quaternion_absolute)
        return achieved_position_base, achieved_quaternion_base

    def get_observation(self):
        obs = super().get_observation()
        eef_pos_base, eef_quat_base = self.get_eef_pose_base()
        obs['eef_position'] = eef_pos_base
        obs['eef_quaternion'] = eef_quat_base
        obs['last_ik_solved_position'] = self._last_ik_result.get('achieved_position', eef_pos_base)
        obs['last_ik_solved_quaternion'] = self._last_ik_result.get('achieved_quaternion', eef_quat_base)
        if 'achieved_error' in self._last_ik_result:
            obs['last_ik_solved_error'] = self._last_ik_result['achieved_error']
        else:
            obs['last_ik_solved_error'] = torch.zeros(get_batch_aware_shape(self.world.batch_size, 6), device=self.world.device)
        return obs

    def apply_action(self, action):
        target_pos = action[..., :3]
        target_rot = action[..., 3:]
        if self.normalize_rotation_action:
            target_rot = AbsoluteRotationConventionUtils.unnormalize_action(self.rotation_representation, target_rot)
        target_rot_quat = AbsoluteRotationConventionUtils.to_quaternion(self.rotation_representation, target_rot)
        target_joint_positions, achieved_position, achieved_quaternion, achieved_error = self.solve_ik_base(
            target_pos, target_rot_quat
        ) # (batch_size, n_dof), (batch_size, 3), (batch_size, 4), (batch_size, 6)
        
        self.entity.control_dofs_position(
            target_joint_positions,
            dofs_idx_local=self.joint_indexes
        )
        self._last_ik_result = {
            'target_joint_positions': target_joint_positions,
            'achieved_position': achieved_position,
            'achieved_quaternion': achieved_quaternion,
            'achieved_error': achieved_error,
        }
