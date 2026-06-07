from typing import Literal, Optional, Dict, Any, List, Tuple, Union, Generic, TypeVar, Sequence
import abc
import genesis as gs
from genesis.engine import entities as gs_entities
from genesis.engine.entities.rigid_entity import RigidLink
from genesis.engine.entities.base_entity import Entity
import torch

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
)

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

    def get_action_dict(self, action: Any) -> Dict[str, Any]:
        """Return a structured dict describing the commanded target.

        This is an introspection helper — it does **not** apply the action.
        Subclasses should override this to provide semantically meaningful
        keys (e.g. ``target_joint_position``, ``target_eef_position``).

        The base implementation wraps the raw action under the key
        ``"action"`` so that callers always receive a dict.
        """
        return {"action": action}

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
        # Optional end-effector metadata
        target_link : Optional[str] = None,
        base_frame : Optional[str] = None,
        eef_offset_pos : Optional[Union[Sequence[float], PyTorchArrayType]] = None,
        eef_offset_quat : Optional[Union[Sequence[float], PyTorchArrayType]] = None,
        **kwargs,
    ) -> None:
        joint_indexes = resolve_joint_indexes(entity, joint_indexes, joint_names)
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        batch_size = world.batch_size
        
        # Get limits from entity - returns (low, high) tuple of shape (batch_size, n_dofs) or (n_dofs,)
        joint_position_low, joint_position_high = entity.get_dofs_limit(dofs_idx_local=joint_indexes)
        joint_force_low, joint_force_high = entity.get_dofs_force_range(dofs_idx_local=joint_indexes)
        
        # Store EEF metadata before building observation space
        self._target_link_name : Optional[str] = target_link
        self._base_frame_name : Optional[str] = base_frame

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

        # Conditionally add EEF observation keys when an end-effector is configured
        self._has_eef = target_link is not None
        if self._has_eef:
            obs_space['eef_position'] = BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=get_batch_aware_shape(batch_size, 3),
                dtype=torch.float32,
                device=world.device,
            )
            obs_space['eef_quaternion'] = BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=1,
                shape=get_batch_aware_shape(batch_size, 4),
                dtype=torch.float32,
                device=world.device,
            )
        
        super().__init__(world, entity, obs_space, action_space, **kwargs)

        # Store the broadcasted limits as attributes
        self.joint_position_low = joint_position_low
        self.joint_position_high = joint_position_high
        self.joint_force_low = joint_force_low
        self.joint_force_high = joint_force_high
        self.joint_indexes = joint_indexes

        self._is_grasping_fn = is_grasping_fn

        # EEF offset: translates the virtual EEF point away from the physical
        # target_link origin.  Stored in the target_link's local frame.
        if self._has_eef and eef_offset_pos is not None:
            self._eef_offset_pos = torch.as_tensor(
                eef_offset_pos, dtype=torch.float32, device=world.device,
            )
            if eef_offset_quat is not None:
                self._eef_offset_quat = torch.as_tensor(
                    eef_offset_quat, dtype=torch.float32, device=world.device,
                )
            else:
                self._eef_offset_quat = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=world.device,
                )
        else:
            self._eef_offset_pos = None
            self._eef_offset_quat = None

        # Resolve link references from names (needs entity to be built)
        if self._has_eef:
            self._resolve_ee_links()
        else:
            self.target_link = None
            self.base_frame = None

    def is_grasping(self, target_entity):
        """Check if this controller's entity is grasping the target.

        Returns ``None`` if no ``is_grasping`` function was configured.
        The callable signature is ``(robot_entity, target_entity) -> bool | Tensor[bool]``.
        """
        if self._is_grasping_fn is None:
            return None
        return self._is_grasping_fn(self.entity, target_entity)

    # ------------------------------------------------------------------
    # End-effector metadata helpers
    # ------------------------------------------------------------------

    def _resolve_ee_links(self) -> None:
        """Re-resolve target_link and base_frame from the current entity by name."""
        if not self._has_eef:
            return
        self.target_link: RigidLink = self.entity.get_link(self._target_link_name)
        self.base_frame: Optional[RigidLink] = (
            self.entity.get_link(self._base_frame_name)
            if self._base_frame_name is not None
            else None
        )

    def reload(self) -> None:
        """Re-resolve link references after entity rebuild."""
        if self._has_eef:
            self._resolve_ee_links()

    def _link_to_virtual(
        self,
        link_pos: PyTorchArrayType,
        link_quat: PyTorchArrayType,
    ) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Forward offset: physical target_link pose -> virtual EEF pose."""
        if self._eef_offset_pos is None:
            return link_pos, link_quat
        virt_pos = link_pos + quaternion_apply(link_quat, self._eef_offset_pos)
        virt_quat = quaternion_multiply(link_quat, self._eef_offset_quat)
        return virt_pos, virt_quat

    def _virtual_to_link(
        self,
        virt_pos: PyTorchArrayType,
        virt_quat: PyTorchArrayType,
    ) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Inverse offset: virtual EEF pose -> physical target_link pose."""
        if self._eef_offset_pos is None:
            return virt_pos, virt_quat
        link_quat = quaternion_multiply(virt_quat, quaternion_invert(self._eef_offset_quat))
        link_pos = virt_pos - quaternion_apply(link_quat, self._eef_offset_pos)
        return link_pos, link_quat

    def get_base_transform(self) -> Optional[Tuple[PyTorchArrayType, PyTorchArrayType]]:
        """Return ``(base_position, base_quaternion)`` for the configured base frame.

        Returns ``None`` when no ``base_frame`` is configured.
        """
        if not self._has_eef or self.base_frame is None:
            return None
        base_position = self.base_frame.get_pos()
        base_quaternion = self.base_frame.get_quat()
        return base_position, base_quaternion

    def convert_base_to_absolute(
        self,
        target_position_base: Optional[PyTorchArrayType],
        target_rotation_base: Optional[PyTorchArrayType],
    ) -> Tuple[Optional[PyTorchArrayType], Optional[PyTorchArrayType]]:
        """Transform a pose from the configured base frame to world frame."""
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

    def convert_absolute_to_base(
        self,
        target_position_absolute: Optional[PyTorchArrayType],
        target_rotation_absolute: Optional[PyTorchArrayType],
    ) -> Tuple[Optional[PyTorchArrayType], Optional[PyTorchArrayType]]:
        """Transform a pose from world frame to the configured base frame."""
        base_transform = self.get_base_transform()
        if base_transform is None:
            return target_position_absolute, target_rotation_absolute
        base_position, base_quaternion = base_transform
        if target_position_absolute is not None:
            target_position_base = quaternion_apply(
                quaternion_invert(base_quaternion),
                target_position_absolute - base_position,
            )
        else:
            target_position_base = None
        if target_rotation_absolute is not None:
            target_rotation_base = quaternion_apply(
                quaternion_invert(base_quaternion),
                target_rotation_absolute,
            )
        else:
            target_rotation_base = None
        return target_position_base, target_rotation_base

    def get_eef_pose_absolute(self) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Return the virtual EEF pose in world frame.

        Only valid when an end-effector has been configured (``target_link`` was
        provided).  Returns ``(position, quaternion)`` each of shape
        ``(batch_size, 3/4)`` or ``(3/4,)``.
        """
        link_pos = self.target_link.get_pos()
        link_quat = self.target_link.get_quat()
        return self._link_to_virtual(link_pos, link_quat)

    def get_eef_pose_base(self) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Return the virtual EEF pose in the configured base frame (or world if no base)."""
        achieved_position_absolute, achieved_quaternion_absolute = self.get_eef_pose_absolute()
        achieved_position_base, achieved_quaternion_base = self.convert_absolute_to_base(
            achieved_position_absolute, achieved_quaternion_absolute,
        )
        return achieved_position_base, achieved_quaternion_base

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def _target_qpos_to_eef(
        self,
        target_qpos: PyTorchArrayType,
    ) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Compute the virtual EEF pose (in base/world frame) for given joint positions.

        Uses :meth:`entity.forward_kinematics` for the configured ``target_link``
        while leaving un-controlled joints at their current values.  The virtual
        EEF offset is applied and the result is converted to the configured base
        frame (or world frame if no ``base_frame`` is set).

        Args:
            target_qpos: Target joint positions for the controlled DOFs —
                shape ``(n_dofs,)`` or ``(batch_size, n_dofs)``.

        Returns:
            ``(eef_position, eef_quaternion)`` in the base/world frame.
        """
        link_idx = [self.target_link.idx_local]
        links_pos, links_quat = self.entity.forward_kinematics(
            target_qpos,
            qs_idx_local=self.joint_indexes,
            links_idx_local=link_idx,
        )
        # FK returns (n_links, 3/4) or (n_envs, n_links, 3/4); squeeze link dim.
        link_pos = links_pos.squeeze(-2)
        link_quat = links_quat.squeeze(-2)
        virt_pos, virt_quat = self._link_to_virtual(link_pos, link_quat)
        base_pos, base_quat = self.convert_absolute_to_base(virt_pos, virt_quat)
        return base_pos, base_quat

    def _get_action_dict_from_qpos(
        self,
        target_qpos: PyTorchArrayType,
    ) -> Dict[str, PyTorchArrayType]:
        """Build a standard action dict from target joint positions.

        Always includes ``target_joint_position``.  When an end-effector is
        configured, also includes ``target_eef_position`` and
        ``target_eef_quaternion`` obtained via FK + virtual offset + base-frame
        conversion.
        """
        result: Dict[str, PyTorchArrayType] = {
            "target_joint_position": target_qpos,
        }
        if self._has_eef:
            eef_pos, eef_quat = self._target_qpos_to_eef(target_qpos)
            result["target_eef_position"] = eef_pos
            result["target_eef_quaternion"] = eef_quat
        return result

    def get_action_dict(self, action: Any) -> Dict[str, Any]:
        """Return a structured dict describing the commanded target.

        ``DiscreteEntityController`` does not know the action semantics on its
        own; concrete subclasses (position, velocity, …) override this.
        The default delegates to the generic base implementation.
        """
        return super().get_action_dict(action)

    def get_observation(self):
        qpos = self.entity.get_dofs_position(dofs_idx_local=self.joint_indexes)
        qvel = self.entity.get_dofs_velocity(dofs_idx_local=self.joint_indexes)
        joint_torque = self.entity.get_dofs_force(dofs_idx_local=self.joint_indexes)
        joint_motor_torque = self.entity.get_dofs_control_force(dofs_idx_local=self.joint_indexes)
        root_pos = self.entity.get_pos()
        root_quat = self.entity.get_quat()
        obs = {
            'joint_position': qpos,
            'joint_velocity': qvel,
            'joint_torque': joint_torque,
            'joint_motor_torque': joint_motor_torque,
            'root_pos': root_pos,
            'root_quat': root_quat,
        }
        if self._has_eef:
            eef_pos_base, eef_quat_base = self.get_eef_pose_base()
            obs['eef_position'] = eef_pos_base
            obs['eef_quaternion'] = eef_quat_base
        return obs

    @abc.abstractmethod
    def apply_action(self, action):
        raise NotImplementedError