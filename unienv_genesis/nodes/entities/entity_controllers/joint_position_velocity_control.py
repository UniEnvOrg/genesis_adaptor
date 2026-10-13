from typing import Literal, Optional, Dict, Any, List, Tuple, Union, Generic, TypeVar, Sequence
import abc
import genesis as gs
import numpy as np
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
from .gravity_compensation import GravityFeedforward

class JointPositionAndVelocityController(
    DiscreteEntityController[
        PyTorchArrayType
    ]
):
    """Joint position + velocity impedance controller.

    Genesis applies ``tau = Kp (q* - q) - Kd (q_dot - q_dot*)`` every substep.
    With ``gravity_compensation`` set, the controller also writes the gravity
    torque ``g(q)`` of the controlled joints (see :class:`GravityFeedforward`),
    scaled by that factor, into Genesis' actuator feedforward ``act_bias[0]``
    on every :meth:`apply_action`; it is held across the substeps of that step.
    ``True`` / ``1.0`` removes the PD steady-state sag under gravity while the
    motors still carry the load; a factor off 1 models a real robot that
    under- or over-compensates (``False`` / ``0.0`` = off).  Held / grasped
    objects are not compensated.  In a batched world
    this needs ``RigidOptions(batch_dofs_info=True)``, since ``g(q)`` differs per
    env.
    """
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
        min_joint_pos_delta_per_step : Optional[Union[float, PyTorchArrayType]] = None,
        joint_names : Optional[Sequence[str]] = None,
        is_grasping_fn = None,
        position_control_low : Optional[Union[float, PyTorchArrayType]] = None,
        position_control_high : Optional[Union[float, PyTorchArrayType]] = None,
        gravity_compensation : Union[bool, float] = False,
        **kwargs,
    ) -> None:
        joint_indexes = resolve_joint_indexes(entity, joint_indexes, joint_names)
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        batch_size = world.batch_size

        # Normalize tensor-valued config for device safety before any
        # broadcast_to_batch / BoxSpace construction.
        dev = world.device
        def _coerce_gain(v):
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                return v.to(device=dev, dtype=torch.float32)
            if isinstance(v, (list, tuple, np.ndarray)):
                return torch.as_tensor(v, dtype=torch.float32, device=dev)
            return float(v)
        Kp = _coerce_gain(Kp)
        Kd = _coerce_gain(Kd)
        max_joint_vel = max_joint_vel.to(device=dev, dtype=torch.float32) if isinstance(max_joint_vel, torch.Tensor) else max_joint_vel
        max_joint_pos_vel = max_joint_pos_vel.to(device=dev, dtype=torch.float32) if isinstance(max_joint_pos_vel, torch.Tensor) else max_joint_pos_vel
        min_joint_pos_delta_per_step = min_joint_pos_delta_per_step.to(device=dev, dtype=torch.float32) if isinstance(min_joint_pos_delta_per_step, torch.Tensor) else min_joint_pos_delta_per_step
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
            min_joint_pos_delta_per_step=min_joint_pos_delta_per_step,
            **kwargs,
        )
        self.Kp = Kp
        self.Kd = Kd
        self._gravity_feedforward = None
        if float(gravity_compensation):
            if world.is_batched and not world.dofs_info_batched:
                raise ValueError(
                    "gravity_compensation in a batched world writes a per-env "
                    "torque, which requires the scene to store batched DOF info.  "
                    "Construct the world with scene_init_kwargs=dict(rigid_options="
                    "gs.options.RigidOptions(batch_dofs_info=True), ...)."
                )
            self._gravity_feedforward = GravityFeedforward(
                entity, self.joint_indexes, scale=float(gravity_compensation),
            )

    def _expand_gains(
        self, value: Union[float, PyTorchArrayType], gain_name: str,
    ) -> PyTorchArrayType:
        """Broadcast a PD gain to the shape Genesis accepts for DOF info.

        Genesis stores DOF gains per env only when the scene was created with
        ``RigidOptions(batch_dofs_info=True)``; otherwise the gain table is a
        single ``(n_dofs,)`` row shared by all envs.  Accordingly this returns
        ``(batch_size, n_dofs)`` when per-env DOF info is enabled and
        ``(n_dofs,)`` otherwise — even in batched worlds.

        Raises:
            ValueError: If *value* is a per-env ``(batch_size, n_dofs)`` tensor
                but the scene does not store batched DOF info.
        """
        n_controlled_dofs = (
            len(self.joint_indexes) if self.joint_indexes is not None
            else self.entity.n_dofs
        )
        dofs_info_batched = self.world.dofs_info_batched
        if (
            not dofs_info_batched
            and isinstance(value, torch.Tensor)
            and value.ndim == 2
        ):
            raise ValueError(
                f"Per-env {gain_name} of shape {tuple(value.shape)} requires the "
                f"scene to store batched DOF info, but it does not.  Construct "
                f"the world with scene_init_kwargs=dict(rigid_options="
                f"gs.options.RigidOptions(batch_dofs_info=True), ...) or pass a "
                f"scalar / (n_dofs,) {gain_name} shared by all envs."
            )
        gains_batch_size = self.world.batch_size if dofs_info_batched else None
        return broadcast_to_batch(
            value, gains_batch_size, n_controlled_dofs, device=self.world.device
        )

    @property
    def Kp(self) -> Optional[Union[float, PyTorchArrayType]]:
        return self._Kp

    @Kp.setter
    def Kp(self, value: Optional[Union[float, PyTorchArrayType]]) -> None:
        if value is not None:
            expanded = self._expand_gains(value, "Kp")
            self.entity.set_dofs_kp(expanded, dofs_idx_local=self.joint_indexes)
            self._Kp_expanded = expanded
        else:
            self._Kp_expanded = None
        self._Kp = value

    @property
    def Kd(self) -> Optional[Union[float, PyTorchArrayType]]:
        return self._Kd

    @Kd.setter
    def Kd(self, value: Optional[Union[float, PyTorchArrayType]]) -> None:
        if value is not None:
            expanded = self._expand_gains(value, "Kd")
            self.entity.set_dofs_kv(expanded, dofs_idx_local=self.joint_indexes)
            self._Kd_expanded = expanded
        else:
            self._Kd_expanded = None
        self._Kd = value

    def reload(self):
        super().reload()
        # Set the controller for the entity
        if self._Kp_expanded is not None:
            self.entity.set_dofs_kp(self._Kp_expanded, dofs_idx_local=self.joint_indexes)
        if self._Kd_expanded is not None:
            self.entity.set_dofs_kv(self._Kd_expanded, dofs_idx_local=self.joint_indexes)
        if self._gravity_feedforward is not None:
            self._gravity_feedforward.rebuild(self.entity)

    @property
    def gravity_compensation(self) -> float:
        """Gravity-compensation factor (``0.0`` when off)."""
        if self._gravity_feedforward is None:
            return 0.0
        return self._gravity_feedforward.scale

    def _apply_gravity_compensation(self) -> None:
        if self._gravity_feedforward is not None:
            self._gravity_feedforward.apply()

    def apply_action(self, action, dt: Optional[float] = None):
        target_qpos = self._clip_target_qpos(action['target_joint_position'], dt)
        target_qvel = self._clip_target_qvel(action['target_joint_velocity'])
        self.entity.control_dofs_position_velocity(
            target_qpos,
            target_qvel,
            dofs_idx_local=self.joint_indexes
        )
        self._apply_gravity_compensation()

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return structured action dict for combined position+velocity control.

        ``action`` is expected to be a dict with keys ``target_joint_position``
        and ``target_joint_velocity``.  The returned dict includes both
        (after rate-limit / velocity clipping), plus optional FK-based EEF
        pose derived from the clipped position target.
        """
        target_qpos = self._clip_target_qpos(action['target_joint_position'], dt)
        target_qvel = self._clip_target_qvel(action['target_joint_velocity'])
        result: Dict[str, PyTorchArrayType] = {
            "target_joint_position": target_qpos,
        }
        if self._has_eef:
            eef_pos, eef_quat = self._target_qpos_to_eef(target_qpos)
            result["target_eef_position"] = eef_pos
            result["target_eef_quaternion"] = eef_quat
        result["target_joint_velocity"] = target_qvel
        return result

    def get_action_dict_space(self) -> DictSpace:
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        Includes ``target_joint_position``, ``target_joint_velocity``, and
        (when an end-effector is configured) ``target_eef_position`` and
        ``target_eef_quaternion``.
        """
        batch_size = self.world.batch_size
        device = self.world.device

        spaces: Dict[str, Space] = {
            "target_joint_position": BoxSpace(
                PyTorchComputeBackend,
                low=self.joint_position_low,
                high=self.joint_position_high,
                dtype=self.joint_position_low.dtype,
                device=device,
            ),
        }
        if self._has_eef:
            spaces["target_eef_position"] = BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=get_batch_aware_shape(batch_size, 3),
                dtype=torch.float32,
                device=device,
            )
            spaces["target_eef_quaternion"] = BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=1,
                shape=get_batch_aware_shape(batch_size, 4),
                dtype=torch.float32,
                device=device,
            )
        spaces["target_joint_velocity"] = self.action_space["target_joint_velocity"]
        return DictSpace(PyTorchComputeBackend, spaces, device=device)
