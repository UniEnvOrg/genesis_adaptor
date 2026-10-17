from typing import Optional, Union, Sequence, Dict, Any
import torch
from genesis.engine import entities as gs_entities
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
from unienv_genesis.utils.space_metadata import dof_names
from unienv_interface.space.space_utils.metadata_utils import action_metadata
from .joint_position_velocity_control import JointPositionAndVelocityController
from .joint_position import JointPositionController
from .joint_velocity import JointVelocityController


def _prepare_1d(
    value: Union[float, PyTorchArrayType],
    n_dofs: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """Return a (n_dofs,) float32 tensor. Scalar input is broadcast."""
    t = torch.as_tensor(value, dtype=torch.float32, device=device)
    if t.dim() == 0:
        return t.expand(n_dofs)
    if t.dim() == 1 and t.shape[0] == n_dofs:
        return t
    raise ValueError(f"{name} must be a scalar or 1-d tensor of length {n_dofs}, got {tuple(t.shape)}")


def _apply_mimic(
    action: torch.Tensor,      # (..., 1)
    multiplier: torch.Tensor,  # (n_dofs,)
    shift: torch.Tensor,       # (n_dofs,)
) -> torch.Tensor:             # (..., n_dofs)
    """action * multiplier + shift, broadcasting (…, 1) × (n_dofs,) → (…, n_dofs)."""
    return action * multiplier + shift


def _is_finite_bound(value) -> bool:
    if value is None:
        return False
    return bool(torch.isfinite(torch.as_tensor(value, dtype=torch.float32)).all())


def _mimic_metadata(
    controller,
    control: str,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
    control_low,
    control_high,
) -> Dict[str, Any]:
    """Action metadata of a mimic controller's scalar control signal.

    The signal drives every controlled joint as ``joint = signal *
    multiplier + shift``; ``mimic_joints`` / ``multiplier`` / ``shift``
    record that mapping. A controller configured with the gripper description
    (``gripper_open_action`` / ``gripper_close_action``, see
    :attr:`EntityController.extra_attributes`) is a gripper command
    (``control="gripper"``); with finite control bounds it is a normalized
    signal, and ``open_value`` / ``closed_value`` give the fully open / closed
    commands. ``control`` is ``"joint_position"`` or ``"joint_velocity"``.
    """
    extra: Dict[str, Any] = {
        "mimic_joints": dof_names(controller.entity, controller.joint_indexes),
        # float32 values, written with float32 precision (-0.02, not -0.0199999995)
        "multiplier": [float(f"{v:.7g}") for v in multiplier.tolist()],
        "shift": [float(f"{v:.7g}") for v in shift.tolist()],
    }
    open_value = getattr(controller, "gripper_open_action", None)
    closed_value = getattr(controller, "gripper_close_action", None)
    if open_value is None or closed_value is None:
        return action_metadata(control=control, mode="absolute", names=["mimic"], **extra)
    normalized = _is_finite_bound(control_low) and _is_finite_bound(control_high)
    return action_metadata(
        control="gripper",
        mode="absolute" if control == "joint_position" else "velocity",
        names=["gripper"],
        normalized=True if normalized else None,
        units="normalized" if normalized else None,
        open_value=float(open_value),
        closed_value=float(closed_value),
        **extra,
    )


class MimicJointPositionAndVelocityController(JointPositionAndVelocityController):
    """Joint position+velocity controller driven by a single scalar control signal.

    The command sent to the entity is::

        actual_position = clip(action['target_joint_position'] * position_multiplier + position_shift,
                               joint_position_low, joint_position_high)
        actual_velocity = clip(action['target_joint_velocity'] * velocity_multiplier + velocity_shift,
                               -max_joint_vel, max_joint_vel)

    where both action tensors have shape ``(n_envs, 1)`` (or ``(1,)`` unbatched).

    Args:
        position_multiplier: Scalar or ``(n_dofs,)`` per-joint gain for position.
        position_shift: Scalar or ``(n_dofs,)`` per-joint offset for position.
        position_control_low: Lower bound of the position control signal (action space).
            Defaults to ``-inf``.
        position_control_high: Upper bound of the position control signal (action space).
            Defaults to ``+inf``.
        velocity_multiplier: Scalar or ``(n_dofs,)`` per-joint gain for velocity.
        velocity_shift: Scalar or ``(n_dofs,)`` per-joint offset for velocity.
        velocity_control_low: Lower bound of the velocity control signal (action space).
            Defaults to ``-inf``.
        velocity_control_high: Upper bound of the velocity control signal (action space).
            Defaults to ``+inf``.

    All other args are forwarded to :class:`JointPositionAndVelocityController`.
    """

    controller_type = "mimic_joint_position_and_velocity"

    def __init__(
        self,
        world: GenesisWorld,
        entity: gs_entities.RigidEntity,
        position_multiplier: Union[float, PyTorchArrayType],
        velocity_multiplier: Union[float, PyTorchArrayType],
        Kp: Optional[Union[float, PyTorchArrayType]] = None,
        Kd: Optional[Union[float, PyTorchArrayType]] = None,
        joint_indexes: Optional[Sequence[int]] = None,
        max_joint_vel: Optional[Union[float, PyTorchArrayType]] = None,
        max_joint_pos_vel: Optional[Union[float, PyTorchArrayType]] = None,
        position_shift: Union[float, PyTorchArrayType] = 0.0,
        position_control_low: Optional[Union[float, PyTorchArrayType]] = None,
        position_control_high: Optional[Union[float, PyTorchArrayType]] = None,
        velocity_shift: Union[float, PyTorchArrayType] = 0.0,
        velocity_control_low: Optional[Union[float, PyTorchArrayType]] = None,
        velocity_control_high: Optional[Union[float, PyTorchArrayType]] = None,
        joint_names: Optional[Sequence[str]] = None,
        is_grasping_fn = None,
        **kwargs,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes, max_joint_vel, max_joint_pos_vel=max_joint_pos_vel, joint_names=joint_names, is_grasping_fn=is_grasping_fn, **kwargs)
        n_dofs = len(self.joint_indexes) if self.joint_indexes is not None else entity.n_dofs
        dev = world.device

        # Normalize control-signal bounds for device safety before BoxSpace.
        position_control_low = position_control_low.to(device=dev, dtype=torch.float32) if isinstance(position_control_low, torch.Tensor) else position_control_low
        position_control_high = position_control_high.to(device=dev, dtype=torch.float32) if isinstance(position_control_high, torch.Tensor) else position_control_high
        velocity_control_low = velocity_control_low.to(device=dev, dtype=torch.float32) if isinstance(velocity_control_low, torch.Tensor) else velocity_control_low
        velocity_control_high = velocity_control_high.to(device=dev, dtype=torch.float32) if isinstance(velocity_control_high, torch.Tensor) else velocity_control_high

        self._pos_multiplier = _prepare_1d(position_multiplier, n_dofs, "position_multiplier", dev)
        self._pos_shift = _prepare_1d(position_shift, n_dofs, "position_shift", dev)
        self._vel_multiplier = _prepare_1d(velocity_multiplier, n_dofs, "velocity_multiplier", dev)
        self._vel_shift = _prepare_1d(velocity_shift, n_dofs, "velocity_shift", dev)

        # Capture output clip bounds from parent's action space before overriding it.
        # Position: always available from DiscreteEntityController.
        self._pos_clip_low = self.joint_position_low
        self._pos_clip_high = self.joint_position_high
        # Velocity: from the velocity BoxSpace built by JointPositionAndVelocityController.
        vel_space = self.action_space['target_joint_velocity']
        self._vel_clip_low = vel_space.low
        self._vel_clip_high = vel_space.high

        self.action_space = DictSpace(
            PyTorchComputeBackend,
            {
                "target_joint_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=position_control_low if position_control_low is not None else -torch.inf,
                    high=position_control_high if position_control_high is not None else torch.inf,
                    shape=get_batch_aware_shape(world.batch_size, 1),
                    dtype=torch.float32,
                    device=dev,
                    metadata=_mimic_metadata(
                        self, "joint_position", self._pos_multiplier, self._pos_shift,
                        position_control_low, position_control_high,
                    ),
                ),
                "target_joint_velocity": BoxSpace(
                    PyTorchComputeBackend,
                    low=velocity_control_low if velocity_control_low is not None else -torch.inf,
                    high=velocity_control_high if velocity_control_high is not None else torch.inf,
                    shape=get_batch_aware_shape(world.batch_size, 1),
                    dtype=torch.float32,
                    device=dev,
                    metadata=_mimic_metadata(
                        self, "joint_velocity", self._vel_multiplier, self._vel_shift,
                        velocity_control_low, velocity_control_high,
                    ),
                ),
            },
            device=dev,
        )

    def apply_action(self, action, dt: Optional[float] = None):
        pos = _apply_mimic(action['target_joint_position'], self._pos_multiplier, self._pos_shift)
        vel = _apply_mimic(action['target_joint_velocity'], self._vel_multiplier, self._vel_shift)
        pos = torch.clamp(pos, self._pos_clip_low, self._pos_clip_high)
        vel = torch.clamp(vel, self._vel_clip_low, self._vel_clip_high)
        # Apply rate-limit / velocity clipping from base
        pos = self._clip_target_qpos(pos, dt)
        vel = self._clip_target_qvel(vel)
        mimic_action = {
            'target_joint_position': pos,
            'target_joint_velocity': vel,
        }
        super().apply_action(mimic_action, dt=None)  # already clipped

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return structured action dict with **actual** mapped targets.

        Applies the mimic mapping (multiplier + shift + clip) to the scalar
        control signal, then applies rate-limit / velocity clipping, and
        returns the resulting per-joint position and velocity targets, plus
        optional FK-based EEF pose.
        """
        pos = torch.clamp(
            _apply_mimic(action['target_joint_position'], self._pos_multiplier, self._pos_shift),
            self._pos_clip_low,
            self._pos_clip_high,
        )
        vel = torch.clamp(
            _apply_mimic(action['target_joint_velocity'], self._vel_multiplier, self._vel_shift),
            self._vel_clip_low,
            self._vel_clip_high,
        )
        pos = self._clip_target_qpos(pos, dt)
        vel = self._clip_target_qvel(vel)
        # Delegate to JointPositionAndVelocityController.get_action_dict
        # which builds the structured dict (position + optional EEF + velocity).
        return super().get_action_dict(
            {'target_joint_position': pos, 'target_joint_velocity': vel},
            dt=None,  # already clipped
        )

    def get_action_dict_space(self) -> DictSpace:
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        The output schema mirrors :class:`JointPositionAndVelocityController`:
        ``target_joint_position``, ``target_joint_velocity``, and optional
        EEF keys.  Bounds reflect the entity's joint/velocity limits (the
        post-mapping output range), not the scalar control-signal bounds.
        """
        return super().get_action_dict_space()


class MimicJointPositionController(JointPositionController):
    """Joint position controller driven by a single scalar control signal.

    The command sent to the entity is::

        actual_position = clip(action * multiplier + shift,
                               joint_position_low, joint_position_high)

    where ``action`` has shape ``(n_envs, 1)`` (or ``(1,)`` unbatched).

    Args:
        multiplier: Scalar or ``(n_dofs,)`` per-joint gain.
        shift: Scalar or ``(n_dofs,)`` per-joint offset.
        control_low: Lower bound of the control signal (action space). Defaults to ``-inf``.
        control_high: Upper bound of the control signal (action space). Defaults to ``+inf``.

    All other args are forwarded to :class:`JointPositionController`.
    """

    controller_type = "mimic_joint_position"

    def __init__(
        self,
        world: GenesisWorld,
        entity: gs_entities.RigidEntity,
        multiplier: Union[float, PyTorchArrayType],
        Kp: Optional[Union[float, PyTorchArrayType]] = None,
        Kd: Optional[Union[float, PyTorchArrayType]] = None,
        joint_indexes: Optional[Sequence[int]] = None,
        max_joint_pos_vel: Optional[Union[float, PyTorchArrayType]] = None,
        shift: Union[float, PyTorchArrayType] = 0.0,
        control_low: Optional[Union[float, PyTorchArrayType]] = None,
        control_high: Optional[Union[float, PyTorchArrayType]] = None,
        joint_names: Optional[Sequence[str]] = None,
        is_grasping_fn = None,
        **kwargs,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes, max_joint_pos_vel=max_joint_pos_vel, joint_names=joint_names, is_grasping_fn=is_grasping_fn, **kwargs)
        n_dofs = len(self.joint_indexes) if self.joint_indexes is not None else entity.n_dofs
        dev = world.device

        # Normalize control-signal bounds for device safety before BoxSpace.
        control_low = control_low.to(device=dev, dtype=torch.float32) if isinstance(control_low, torch.Tensor) else control_low
        control_high = control_high.to(device=dev, dtype=torch.float32) if isinstance(control_high, torch.Tensor) else control_high

        self._multiplier = _prepare_1d(multiplier, n_dofs, "multiplier", dev)
        self._shift = _prepare_1d(shift, n_dofs, "shift", dev)

        # Output clip bounds: entity's joint position limits.
        self._clip_low = self.joint_position_low
        self._clip_high = self.joint_position_high

        self.action_space = BoxSpace(
            PyTorchComputeBackend,
            low=control_low if control_low is not None else -torch.inf,
            high=control_high if control_high is not None else torch.inf,
            shape=get_batch_aware_shape(world.batch_size, 1),
            dtype=torch.float32,
            device=dev,
            metadata=_mimic_metadata(
                self, "joint_position", self._multiplier, self._shift, control_low, control_high,
            ),
        )

    def apply_action(self, action, dt: Optional[float] = None):
        actual = torch.clamp(
            _apply_mimic(action, self._multiplier, self._shift),
            self._clip_low,
            self._clip_high,
        )
        actual = self._clip_target_qpos(actual, dt)
        super().apply_action(actual, dt=None)  # already clipped

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return structured action dict with the **actual** mapped joint positions.

        Applies the mimic mapping (multiplier + shift + clip) to the scalar
        control signal, then applies rate-limit clipping, and returns the
        resulting per-joint position target, plus optional FK-based EEF pose.
        """
        actual = torch.clamp(
            _apply_mimic(action, self._multiplier, self._shift),
            self._clip_low,
            self._clip_high,
        )
        actual = self._clip_target_qpos(actual, dt)
        # Delegate to JointPositionController.get_action_dict which builds
        # the structured dict (position + optional EEF keys).
        return super().get_action_dict(actual, dt=None)  # already clipped

    def get_action_dict_space(self) -> DictSpace:
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        The output schema mirrors :class:`JointPositionController`:
        ``target_joint_position`` and optional EEF keys.  Bounds reflect
        the entity's joint position limits (the post-mapping output range).
        """
        return super().get_action_dict_space()


class MimicJointVelocityController(JointVelocityController):
    """Joint velocity controller driven by a single scalar control signal.

    The command sent to the entity is::

        actual_velocity = clip(action * multiplier + shift, -max_joint_vel, max_joint_vel)

    where ``action`` has shape ``(n_envs, 1)`` (or ``(1,)`` unbatched).

    Args:
        multiplier: Scalar or ``(n_dofs,)`` per-joint gain.
        shift: Scalar or ``(n_dofs,)`` per-joint offset.
        control_low: Lower bound of the control signal (action space). Defaults to ``-inf``.
        control_high: Upper bound of the control signal (action space). Defaults to ``+inf``.

    All other args are forwarded to :class:`JointVelocityController`.
    """

    controller_type = "mimic_joint_velocity"

    def __init__(
        self,
        world: GenesisWorld,
        entity: gs_entities.RigidEntity,
        multiplier: Union[float, PyTorchArrayType],
        Kp: Optional[Union[float, PyTorchArrayType]] = None,
        Kd: Optional[Union[float, PyTorchArrayType]] = None,
        joint_indexes: Optional[Sequence[int]] = None,
        max_joint_vel: Optional[Union[float, PyTorchArrayType]] = None,
        shift: Union[float, PyTorchArrayType] = 0.0,
        control_low: Optional[Union[float, PyTorchArrayType]] = None,
        control_high: Optional[Union[float, PyTorchArrayType]] = None,
        joint_names: Optional[Sequence[str]] = None,
        is_grasping_fn = None,
        **kwargs,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes, max_joint_vel, joint_names=joint_names, is_grasping_fn=is_grasping_fn, **kwargs)
        n_dofs = len(self.joint_indexes) if self.joint_indexes is not None else entity.n_dofs
        dev = world.device

        # Normalize control-signal bounds for device safety before BoxSpace.
        control_low = control_low.to(device=dev, dtype=torch.float32) if isinstance(control_low, torch.Tensor) else control_low
        control_high = control_high.to(device=dev, dtype=torch.float32) if isinstance(control_high, torch.Tensor) else control_high

        self._multiplier = _prepare_1d(multiplier, n_dofs, "multiplier", dev)
        self._shift = _prepare_1d(shift, n_dofs, "shift", dev)

        # Capture output clip bounds from the velocity BoxSpace set by the parent
        # before overriding the action space.
        self._clip_low = self.action_space.low
        self._clip_high = self.action_space.high

        self.action_space = BoxSpace(
            PyTorchComputeBackend,
            low=control_low if control_low is not None else -torch.inf,
            high=control_high if control_high is not None else torch.inf,
            shape=get_batch_aware_shape(world.batch_size, 1),
            dtype=torch.float32,
            device=dev,
            metadata=_mimic_metadata(
                self, "joint_velocity", self._multiplier, self._shift, control_low, control_high,
            ),
        )

    def apply_action(self, action, dt: Optional[float] = None):
        actual = torch.clamp(
            _apply_mimic(action, self._multiplier, self._shift),
            self._clip_low,
            self._clip_high,
        )
        actual = self._clip_target_qvel(actual)
        super().apply_action(actual, dt=None)  # already clipped

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return structured action dict with the **actual** mapped joint velocities.

        Applies the mimic mapping (multiplier + shift + clip) to the scalar
        control signal, then applies velocity clipping, and returns the
        resulting per-joint velocity target.
        """
        actual = torch.clamp(
            _apply_mimic(action, self._multiplier, self._shift),
            self._clip_low,
            self._clip_high,
        )
        actual = self._clip_target_qvel(actual)
        return {"target_joint_velocity": actual}

    def get_action_dict_space(self) -> DictSpace:
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        Contains a single key ``target_joint_velocity``.  Bounds reflect
        the entity's velocity limits (the post-mapping output range), not
        the scalar control-signal bounds.
        """
        batch_size = self.world.batch_size
        n_dofs = len(self.joint_indexes) if self.joint_indexes is not None else self.entity.n_dofs
        device = self.world.device
        return DictSpace(
            PyTorchComputeBackend,
            {
                "target_joint_velocity": BoxSpace(
                    PyTorchComputeBackend,
                    low=self._clip_low,
                    high=self._clip_high,
                    shape=get_batch_aware_shape(batch_size, n_dofs),
                    dtype=self._clip_low.dtype,
                    device=device,
                    metadata=self._joint_target_metadata("joint_velocity"),
                ),
            },
            device=device,
        )
