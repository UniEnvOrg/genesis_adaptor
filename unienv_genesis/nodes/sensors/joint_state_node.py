"""Joint state sensor node for Genesis.

Reads joint state directly from a :class:`RigidEntity` (not a Genesis sensor).
This is valid immediately after build/reset because entity state queries read
the solver state directly.
"""

from __future__ import annotations

from typing import Optional, Dict, Sequence, Union
import torch
from dataclasses import dataclass

from unienv_interface.space import Space, DictSpace, BoxSpace
from unienv_interface.backends.pytorch import (
    PyTorchComputeBackend,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape, broadcast_to_batch
from unienv_genesis.options.attach_option import AttachmentProvider
from unienv_genesis.utils.space_metadata import (
    joint_proprio_metadata,
    position_metadata,
    quaternion_metadata,
)
from ..entities.entity_controllers.base import resolve_joint_indexes

from .base import SensorNode

__all__ = [
    "JointStateSensorNodeOptions",
    "JointStateSensorNode",
]


@dataclass(frozen=True)
class JointStateSensorNodeOptions:
    """Options for :class:`JointStateSensorNode`.

    The entity is resolved via ``attach``.  The link returned by the attachment
    provider is ignored — joint state is read at the entity level.
    """

    attach: AttachmentProvider
    """Provider that resolves the rigid entity whose joint state is read."""

    joint_indexes: Optional[Sequence[int]] = None
    """Optional explicit DOF indexes.  ``None`` means all DOFs."""

    joint_names: Optional[Sequence[str]] = None
    """Optional joint names.  At most one of ``joint_indexes``/``joint_names``."""

    read_position: bool = True
    """Expose ``joint_position`` (from ``get_dofs_position``)."""

    read_velocity: bool = True
    """Expose ``joint_velocity`` (from ``get_dofs_velocity``)."""

    read_torque: bool = True
    """Expose ``joint_torque`` (applied force from ``get_dofs_force``)."""

    read_motor_torque: bool = False
    """Expose ``joint_motor_torque`` (control force from ``get_dofs_control_force``)."""

    read_root_pose: bool = False
    """Expose ``root_pos`` and ``root_quat`` (entity base pose)."""


class JointStateSensorNode(SensorNode[None, None, Dict[str, PyTorchArrayType]]):
    """Sensor node that reads joint state directly from a :class:`RigidEntity`.

    Unlike the IMU/Contact sensors which wrap Genesis sensor objects, this node
    queries the entity's solver state directly via ``get_dofs_position`` etc.,
    so the observation is valid immediately after build/reset (no need to wait
    for the first ``env.step()``).
    """

    reload_priorities = {0}
    after_reload_priorities = {-50}
    reset_priorities = {0}
    after_reset_priorities = {-50}
    post_environment_step_priorities = {0}

    def __init__(
        self,
        world: GenesisWorld,
        name: str,
        option: JointStateSensorNodeOptions,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
    ):
        super().__init__(
            world=world,
            name=name,
            control_timestep=control_timestep,
            update_timestep=update_timestep,
        )
        self.world: GenesisWorld = world

        self.option = option
        assert (
            option.read_position
            or option.read_velocity
            or option.read_torque
            or option.read_motor_torque
            or option.read_root_pose
        ), "At least one of read_position/read_velocity/read_torque/read_motor_torque/read_root_pose must be True"

        self.sensor = None
        self._entity = None
        self._joint_indexes = None
        self._n_dofs: Optional[int] = None

        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        # Built in after_reload once the entity is built (limits require build).
        self.observation_space: Optional[
            Space[PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]
        ] = None

    def reload(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs,
    ) -> None:
        """Resolve the entity and joint selection before the scene is built."""
        if priority != 0:
            return
        entity, _link = self.option.attach.get_attachment(self.world)
        self._entity = entity
        self._joint_indexes = resolve_joint_indexes(
            entity, self.option.joint_indexes, self.option.joint_names,
        )
        self._n_dofs = (
            len(self._joint_indexes)
            if self._joint_indexes is not None
            else entity.n_dofs
        )

    def reset(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs,
    ) -> None:
        """No-op; the cache is refreshed in ``after_reset``."""
        pass

    def after_reload(
        self,
        *,
        priority: int = 0,
        mask: Optional[PyTorchArrayType] = None,
    ) -> None:
        if priority in self.after_reload_priorities:
            self.observation_space = self._get_observation_space()
            self.update_observation()

    def after_reset(
        self,
        *,
        priority: int = 0,
        mask: Optional[PyTorchArrayType] = None,
    ) -> None:
        if priority in self.after_reset_priorities:
            self.update_observation()

    def post_environment_step(
        self,
        dt: Union[float, PyTorchArrayType],
        *,
        priority: int = 0,
    ) -> None:
        if priority == 0:
            self.update_observation()

    def update_observation(self) -> None:
        """Query the entity solver state and cache it."""
        entity = self._entity
        dofs_idx = self._joint_indexes
        dtype = self.world.dtype
        device = self.world.device

        def _t(x: PyTorchArrayType) -> PyTorchArrayType:
            return torch.as_tensor(x, dtype=dtype, device=device)

        obs: Dict[str, PyTorchArrayType] = {}
        if self.option.read_position:
            obs["joint_position"] = _t(entity.get_dofs_position(dofs_idx_local=dofs_idx))
        if self.option.read_velocity:
            obs["joint_velocity"] = _t(entity.get_dofs_velocity(dofs_idx_local=dofs_idx))
        if self.option.read_torque:
            obs["joint_torque"] = _t(entity.get_dofs_force(dofs_idx_local=dofs_idx))
        if self.option.read_motor_torque:
            obs["joint_motor_torque"] = _t(
                entity.get_dofs_control_force(dofs_idx_local=dofs_idx)
            )
        if self.option.read_root_pose:
            obs["root_pos"] = _t(entity.get_pos())
            obs["root_quat"] = _t(entity.get_quat())

        self._last_observation = obs

    def _get_observation_space(
        self,
    ) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        """Build the observation space from the built entity's limits.

        Per-DOF readings carry ``units`` / ``names`` lists from the entity's
        joints (``rad`` / ``rad/s`` / ``N*m`` for rotational DOFs, ``m`` /
        ``m/s`` / ``N`` for translational ones); the root pose is in the
        world frame.
        """
        entity = self._entity
        dofs_idx = self._joint_indexes
        n_dofs = self._n_dofs
        B = self.world.batch_size
        dtype = self.world.dtype
        device = self.world.device

        spaces: Dict[str, Space] = {}

        if self.option.read_position:
            pos_low, pos_high = entity.get_dofs_limit(dofs_idx_local=dofs_idx)
            # Genesis returns per-DOF bounds (no batch dim); broadcast to the
            # batch shape so the BoxSpace shape matches the observation shape.
            pos_low = broadcast_to_batch(pos_low, B, n_dofs, device=device)
            pos_high = broadcast_to_batch(pos_high, B, n_dofs, device=device)
            spaces["joint_position"] = BoxSpace(
                PyTorchComputeBackend,
                low=pos_low,
                high=pos_high,
                dtype=dtype,
                device=device,
                metadata=joint_proprio_metadata(entity, dofs_idx, "position"),
            )

        if self.option.read_velocity:
            spaces["joint_velocity"] = BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=get_batch_aware_shape(B, n_dofs),
                dtype=dtype,
                device=device,
                metadata=joint_proprio_metadata(entity, dofs_idx, "velocity"),
            )

        if self.option.read_torque:
            spaces["joint_torque"] = BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=get_batch_aware_shape(B, n_dofs),
                dtype=dtype,
                device=device,
                metadata=joint_proprio_metadata(entity, dofs_idx, "force"),
            )

        if self.option.read_motor_torque:
            force_low, force_high = entity.get_dofs_force_range(dofs_idx_local=dofs_idx)
            force_low = broadcast_to_batch(force_low, B, n_dofs, device=device)
            force_high = broadcast_to_batch(force_high, B, n_dofs, device=device)
            spaces["joint_motor_torque"] = BoxSpace(
                PyTorchComputeBackend,
                low=force_low,
                high=force_high,
                dtype=dtype,
                device=device,
                metadata=joint_proprio_metadata(entity, dofs_idx, "force"),
            )

        if self.option.read_root_pose:
            spaces["root_pos"] = BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=get_batch_aware_shape(B, 3),
                dtype=dtype,
                device=device,
                metadata=position_metadata(frame="world"),
            )
            spaces["root_quat"] = BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=1,
                shape=get_batch_aware_shape(B, 4),
                dtype=dtype,
                device=device,
                metadata=quaternion_metadata(frame="world"),
            )

        return DictSpace(
            PyTorchComputeBackend,
            spaces,
            device=device,
        )

    def get_observation(self) -> Dict[str, PyTorchArrayType]:
        if self._last_observation is None:
            raise RuntimeError(
                f"JointStateSensorNode '{self.name}' has no observation yet. "
                f"Ensure the node has been initialised via after_reload/after_reset."
            )
        return self._last_observation
