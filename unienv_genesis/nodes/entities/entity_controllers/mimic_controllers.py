from typing import Optional, Union, Sequence
import torch
from genesis.engine import entities as gs_entities
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
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


class MimicJointPositionAndVelocityController(JointPositionAndVelocityController):
    """Joint position+velocity controller driven by a single scalar control signal.

    The command sent to the entity is::

        actual_position = clip(action['target_joint_position'] * position_multiplier + position_shift,
                               joint_position_low, joint_position_high)
        actual_velocity = clip(action['target_joint_velocity'] * velocity_multiplier + velocity_shift,
                               min_joint_vel, max_joint_vel)

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
        min_joint_vel: Optional[Union[float, PyTorchArrayType]] = None,
        max_joint_vel: Optional[Union[float, PyTorchArrayType]] = None,
        position_shift: Union[float, PyTorchArrayType] = 0.0,
        position_control_low: Optional[Union[float, PyTorchArrayType]] = None,
        position_control_high: Optional[Union[float, PyTorchArrayType]] = None,
        velocity_shift: Union[float, PyTorchArrayType] = 0.0,
        velocity_control_low: Optional[Union[float, PyTorchArrayType]] = None,
        velocity_control_high: Optional[Union[float, PyTorchArrayType]] = None,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes, min_joint_vel, max_joint_vel)
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        dev = world.device

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
                ),
                "target_joint_velocity": BoxSpace(
                    PyTorchComputeBackend,
                    low=velocity_control_low if velocity_control_low is not None else -torch.inf,
                    high=velocity_control_high if velocity_control_high is not None else torch.inf,
                    shape=get_batch_aware_shape(world.batch_size, 1),
                    dtype=torch.float32,
                    device=dev,
                ),
            },
            device=dev,
        )

    def apply_action(self, action):
        pos = _apply_mimic(action['target_joint_position'], self._pos_multiplier, self._pos_shift)
        vel = _apply_mimic(action['target_joint_velocity'], self._vel_multiplier, self._vel_shift)
        mimic_action = {
            'target_joint_position': torch.clamp(pos, self._pos_clip_low, self._pos_clip_high),
            'target_joint_velocity': torch.clamp(vel, self._vel_clip_low, self._vel_clip_high),
        }
        super().apply_action(mimic_action)


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
        shift: Union[float, PyTorchArrayType] = 0.0,
        control_low: Optional[Union[float, PyTorchArrayType]] = None,
        control_high: Optional[Union[float, PyTorchArrayType]] = None,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes)
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        dev = world.device

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
        )

    def apply_action(self, action):
        actual = torch.clamp(
            _apply_mimic(action, self._multiplier, self._shift),
            self._clip_low,
            self._clip_high,
        )
        # super().apply_action(actual)


class MimicJointVelocityController(JointVelocityController):
    """Joint velocity controller driven by a single scalar control signal.

    The command sent to the entity is::

        actual_velocity = clip(action * multiplier + shift, min_joint_vel, max_joint_vel)

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
        min_joint_vel: Optional[Union[float, PyTorchArrayType]] = None,
        max_joint_vel: Optional[Union[float, PyTorchArrayType]] = None,
        shift: Union[float, PyTorchArrayType] = 0.0,
        control_low: Optional[Union[float, PyTorchArrayType]] = None,
        control_high: Optional[Union[float, PyTorchArrayType]] = None,
    ) -> None:
        super().__init__(world, entity, Kp, Kd, joint_indexes, min_joint_vel, max_joint_vel)
        n_dofs = len(joint_indexes) if joint_indexes is not None else entity.n_dofs
        dev = world.device

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
        )

    def apply_action(self, action):
        actual = torch.clamp(
            _apply_mimic(action, self._multiplier, self._shift),
            self._clip_low,
            self._clip_high,
        )
        super().apply_action(actual)
