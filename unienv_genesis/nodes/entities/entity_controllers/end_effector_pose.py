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
    quaternion_to_matrix,
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
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
from .joint_position import JointPositionController
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
        device: Optional[PyTorchDeviceType],
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

    @staticmethod
    def from_quaternion(representation: AVAILABLE_REPRESENTATIONS_T, quaternion: PyTorchArrayType) -> PyTorchArrayType:
        """Convert a quaternion ``(w, x, y, z)`` to the given rotation representation."""
        if representation == "euler":
            return matrix_to_euler_angles(quaternion_to_matrix(quaternion), "XYZ")
        elif representation == "quat":
            return quaternion
        elif representation == "rot6d":
            return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))
        else:
            raise ValueError(f"Unsupported rotation representation: {representation}")

class EndEffectorPoseController(
    JointPositionController
):
    """End-effector pose controller using inverse kinematics.

    Converts Cartesian end-effector targets (position + rotation) into joint
    positions via the Genesis damped-least-squares IK solver, then delegates to
    the underlying :class:`JointPositionController` for PD tracking.

    .. note:: **Virtual EEF point convention**

       When ``eef_offset_pos`` is provided, all poses reported by
       :meth:`get_eef_pose_absolute` / :meth:`get_eef_pose_base` and all
       targets accepted by :meth:`solve_ik_absolute` /
       :meth:`get_action_world` refer to a *virtual* EEF point, **not** the
       physical ``target_link`` origin.

       The offset should place this virtual point at the **gripper fingertip**
       — i.e. the farthest extent along the approach direction, centred in
       the opening direction.  Grasp candidate generators (e.g.
       ``base_object.list_box_gripper_grasping_candidates_local``) assume the
       EEF position is at the fingertip so that the fingers wrap around the
       object correctly.
    """

    controller_type = "eef_pose"

    def __init__(
        self,
        world : GenesisWorld,
        entity : RigidEntity,
        target_link : str,
        base_frame : Optional[str] = None,
        Kp : Optional[Union[float, PyTorchArrayType]] = None,
        Kd : Optional[Union[float, PyTorchArrayType]] = None,
        joint_indexes : Optional[Sequence[int]] = None,
        max_joint_vel : Optional[Union[float, PyTorchArrayType]] = None,
        max_joint_pos_vel : Optional[Union[float, PyTorchArrayType]] = None,
        rotation_representation : str = "euler", # "euler", "quat", "rot6d"
        normalize_rotation_action : bool = False,
        joint_names : Optional[Sequence[str]] = None,
        eef_offset_pos : Optional[Union[Sequence[float], PyTorchArrayType]] = None,
        eef_offset_quat : Optional[Union[Sequence[float], PyTorchArrayType]] = None,
        is_grasping_fn = None,
        ik_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        # EEF metadata (target_link, base_frame, eef_offset_*) is forwarded to
        # the DiscreteEntityController base via **kwargs so that the base class
        # owns link resolution, offset transforms, and EEF observation keys.
        super().__init__(
            world, entity, Kp, Kd, joint_indexes,
            joint_names=joint_names,
            is_grasping_fn=is_grasping_fn,
            max_joint_vel=max_joint_vel,
            max_joint_pos_vel=max_joint_pos_vel,
            target_link=target_link,
            base_frame=base_frame,
            eef_offset_pos=eef_offset_pos,
            eef_offset_quat=eef_offset_quat,
            **kwargs,
        )
        self.rotation_representation = rotation_representation
        self.normalize_rotation_action = normalize_rotation_action
        self.ik_kwargs = ik_kwargs or {}

        batch_size = world.batch_size

        # Action space: 3D position + rotation (depends on representation),
        # unconstrained (no translational position bounds).
        self.action_space = AbsoluteRotationConventionUtils.get_action_space(
            rotation_representation, batch_size, world.device,
        )
        # EEF observation keys (eef_position, eef_quaternion) are already added
        # by the DiscreteEntityController base when target_link is provided.
        self.observation_space['last_ik_solved_error'] = BoxSpace(
            PyTorchComputeBackend,
            low=-torch.inf,
            high=torch.inf,
            shape=get_batch_aware_shape(batch_size, 6),
            dtype=torch.float32,
            device=world.device,
        )
        self._last_ik_result = {}

    def reload(self) -> None:
        # Link re-resolution is handled by DiscreteEntityController.reload().
        super().reload()

    def reset(self, mask: Optional[PyTorchArrayType] = None) -> None:
        """Reset the controller's IK result cache."""
        self._last_ik_result = {}
    
    def solve_ik_absolute(self, target_position : Optional[PyTorchArrayType], target_quaternion : Optional[PyTorchArrayType]) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Solve IK for the target end-effector position and rotation.

        Seeds the solver with the current joint positions to ensure solution
        continuity across frames.  See the class docstring for details on why
        this matters.

        Args:
            target_position: ``(3,)`` or ``(batch_size, 3)`` target EEF positions.
            target_quaternion: ``(4,)`` or ``(batch_size, 4)`` target EEF orientations (w, x, y, z).

        Returns:
            target_joint_positions: ``(n_dof,)`` or ``(batch_size, n_dof)`` solved joint positions.
            achieved_error: ``(6,)`` or ``(batch_size, 6)`` pose error
                ``[pos_err_x, pos_err_y, pos_err_z, rot_err_x, rot_err_y, rot_err_z]``.
        
        .. note:: **IK solver seeding and solution continuity**

        Genesis's IK solver (``entity.inverse_kinematics``) uses an iterative DLS method.  
        When the solver cannot reach the target within tolerance with the initial joint configuration, 
        it enters a **random-restart loop** (``max_samples`` attempts, default 50),
        each time drawing a new random initial joint configuration.  
        The best result across all attempts is returned.

        For unreachable targets this causes **frame-to-frame oscillation**: each
        call discovers a *different* local minimum because the random seeds
        differ, so the PD controller chases a flickering joint target.

        To avoid this, this controller defaults ``max_samples`` to **1** (no random restarts).  
        This guarantees that the returned solution is
        always the closest reachable configuration to the current state,
        producing smooth, continuous behaviour even when the target is out of
        reach.  Both defaults can be overridden via ``ik_kwargs``.
        """
        # Build IK kwargs: seed with current joint positions and default to
        # max_samples=1 (no random restarts) unless the caller overrides.
        ik_call_kwargs = {"max_samples": 1}
        ik_call_kwargs.update(self.ik_kwargs)

        # Transform virtual EEF target to physical target_link frame.
        link_position, link_quaternion = self._virtual_to_link(
            target_position, target_quaternion,
        )

        full_target_joint_positions, achieved_error = self.entity.inverse_kinematics(
            self.target_link,
            link_position,
            link_quaternion,
            return_error=True,
            dofs_idx_local=self.joint_indexes,
            **ik_call_kwargs,
        )

        if self.joint_indexes is not None:
            target_joint_positions = index_batch_dimension(
                full_target_joint_positions,
                self.joint_indexes,
                batch_size=self.world.batch_size,
            )
        else:
            target_joint_positions = full_target_joint_positions
        
        
        return target_joint_positions, achieved_error

    def solve_ik_base(self, target_position : Optional[PyTorchArrayType], target_quaternion : Optional[PyTorchArrayType]) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """
        Solve IK for the target end-effector position and rotation in the end-effector's base frame
        Args:
            target_position: (batch_size, 3) tensor of target end-effector positions in the end-effector's base frame
            target_quaternion: (batch_size, 4) tensor of target end-effector rotations in quaternion format in the end-effector's base frame
        Returns:
            target_joint_positions: (batch_size, n_dof) tensor of target joint positions
            achieved_error: (batch_size, 6) tensor of end-effector pose error (position error + rotation error), where position error = target_position - achieved_position, rotation error = quat_to_rotvec(achieved_quaternion^{-1} * target_quaternion)
        """
        target_position_absolute, target_quaternion_absolute = self.convert_base_to_absolute(target_position, target_quaternion)
        target_joint_positions, achieved_error_absolute = self.solve_ik_absolute(target_position_absolute, target_quaternion_absolute)
        
        base_transform = self.get_base_transform()
        if base_transform is None:
            achieved_error = achieved_error_absolute
        else:
            _, base_quaternion = base_transform
            achieved_position_error = quaternion_apply(quaternion_invert(base_quaternion), achieved_error_absolute[..., :3])
            achieved_rotation_error = achieved_error_absolute[..., 3:]
            achieved_error = torch.cat([achieved_position_error, achieved_rotation_error], dim=-1)
        return target_joint_positions, achieved_error

    def get_observation(self):
        obs = super().get_observation()
        # eef_position and eef_quaternion are provided by the base class.
        # This controller only adds last_ik_solved_error.
        if 'achieved_error' in self._last_ik_result:
            obs['last_ik_solved_error'] = self._last_ik_result['achieved_error']
        else:
            obs['last_ik_solved_error'] = torch.zeros(get_batch_aware_shape(self.world.batch_size, 6), device=self.world.device)
        return obs

    def get_action_world(
        self,
        pos: PyTorchArrayType,
        quat: PyTorchArrayType,
    ) -> PyTorchArrayType:
        """Convert a world-frame target pose to this controller's action format.

        This is the inverse of the transformation applied inside
        :meth:`apply_action`: it converts an absolute (world-frame) position
        and quaternion into the ``[pos_base, rot_repr]`` action tensor that the
        controller expects, accounting for the base frame and the configured
        rotation representation.

        Args:
            pos: ``(..., 3)`` target position in world frame.
            quat: ``(..., 4)`` target quaternion ``(w, x, y, z)`` in world
                frame.

        Returns:
            action: ``(..., 3 + rot_dim)`` action tensor suitable for
                :meth:`apply_action` or ``set_next_action``.
        """
        pos_base, quat_base = self.convert_absolute_to_base(pos, quat)
        rot = AbsoluteRotationConventionUtils.from_quaternion(
            self.rotation_representation, quat_base,
        )
        if self.normalize_rotation_action:
            rot = AbsoluteRotationConventionUtils.normalize_action(
                self.rotation_representation, rot,
            )
        return torch.cat([pos_base, rot], dim=-1)

    def apply_action(self, action, dt: Optional[float] = None):
        target_pos = action[..., :3]
        target_rot = action[..., 3:]
        if self.normalize_rotation_action:
            target_rot = AbsoluteRotationConventionUtils.unnormalize_action(self.rotation_representation, target_rot)
        target_rot_quat = AbsoluteRotationConventionUtils.to_quaternion(self.rotation_representation, target_rot)
        target_joint_positions, achieved_error = self.solve_ik_base(
            target_pos, target_rot_quat
        ) # (batch_size, n_dof), (batch_size, 3), (batch_size, 4), (batch_size, 6)
        
        # Delegate rate-limit clipping (max_joint_pos_vel) to the parent
        # JointPositionController.apply_action, which handles dt-aware
        # clipping internally.
        super().apply_action(target_joint_positions, dt=dt)
        self._last_ik_result = {
            'target_joint_positions': target_joint_positions,
            'achieved_error': achieved_error,
        }

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return structured action dict for EEF pose control.

        Parses the raw action tensor into position + rotation, solves IK to
        obtain the target joint positions, then delegates to
        :meth:`JointPositionController.get_action_dict` which handles
        rate-limit clipping and builds the structured dict (including
        FK-based EEF pose of the clipped qpos).
        """
        target_pos = action[..., :3]
        target_rot = action[..., 3:]
        if self.normalize_rotation_action:
            target_rot = AbsoluteRotationConventionUtils.unnormalize_action(
                self.rotation_representation, target_rot,
            )
        target_quat = AbsoluteRotationConventionUtils.to_quaternion(
            self.rotation_representation, target_rot,
        )

        # Solve IK to obtain target joint positions
        target_joint_positions, _ = self.solve_ik_base(target_pos, target_quat)

        # Delegate to JointPositionController.get_action_dict which handles
        # rate-limit clipping and builds the structured dict (including
        # FK-based EEF pose of the clipped qpos).
        return super().get_action_dict(target_joint_positions, dt=dt)

    def get_action_dict_space(self) -> DictSpace:
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        The EEF pose controller's :meth:`get_action_dict` delegates to
        :meth:`JointPositionController.get_action_dict`, so the space
        mirrors that schema: ``target_joint_position`` plus (when an
        end-effector is configured) ``target_eef_position`` /
        ``target_eef_quaternion``.
        """
        return super().get_action_dict_space()
