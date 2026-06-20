"""IMU sensor node for Genesis.

Wraps ``gs.options.sensors.IMU`` via ``scene.add_sensor()``.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, Union, Sequence
import torch
from dataclasses import dataclass

import genesis as gs
from genesis.engine.sensors.imu import IMUSensor

from unienv_interface.space import Space, DictSpace, BoxSpace
from unienv_interface.backends.pytorch import (
    PyTorchComputeBackend,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
from unienv_genesis.options.attach_option import AttachmentProvider

from .base import SensorNode

__all__ = [
    "IMUSensorNodeOptions",
    "IMUSensorNode",
]


@dataclass(frozen=True)
class IMUSensorNodeOptions:
    """Options for :class:`IMUSensorNode`."""

    attach: AttachmentProvider
    """Provider that resolves the entity/link the IMU is attached to."""

    pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the IMU in the link local frame."""

    euler_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Euler offset (degrees) of the IMU in the link local frame."""

    acc_noise: Union[float, Sequence[float]] = 0.0
    gyro_noise: Union[float, Sequence[float]] = 0.0
    mag_noise: Union[float, Sequence[float]] = 0.0
    acc_bias: Union[float, Sequence[float]] = 0.0
    gyro_bias: Union[float, Sequence[float]] = 0.0
    mag_bias: Union[float, Sequence[float]] = 0.0

    magnetic_field: Optional[Tuple[float, float, float]] = None
    """Reference magnetic field.  When ``None`` the Genesis default is used."""

    draw_debug: bool = False


class IMUSensorNode(SensorNode[IMUSensor, None, Dict[str, PyTorchArrayType]]):
    """Sensor node wrapping the Genesis IMU sensor.

    The sensor reports ``lin_acc``, ``ang_vel`` and ``mag`` — each a ``(3,)``
    tensor (unbatched) or ``(B, 3)`` tensor (batched).  All measurements are
    expressed in the sensor/link local frame.

    .. note::

       Genesis only populates its sensor cache during ``scene.step()``.
       :class:`GenesisWorld` runs a no-advance cache refresh (and optionally
       ``n_settle_steps`` physics steps) at the end of ``after_reload`` /
       ``after_reset`` so that this node returns a meaningful observation
       (the accelerometer reads gravity) on the first ``env.reset()`` call.
       Configure ``GenesisWorld(n_settle_steps=...)`` if contact-based
       sensors in the same scene also need populated data immediately.
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
        option: IMUSensorNodeOptions,
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
        self.sensor: Optional[IMUSensor] = None
        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        self.observation_space: Space[
            PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
        ] = self._get_observation_space()

    def reload(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs,
    ) -> None:
        """Add the IMU sensor to the (unbuilt) scene."""
        if priority != 0:
            return
        entity, link = self.option.attach.get_attachment(self.world)
        link = link if link is not None else entity.base_link

        imu_kwargs = dict(
            entity_idx=entity.idx,
            link_idx_local=link.idx_local,
            pos_offset=self.option.pos_offset,
            euler_offset=self.option.euler_offset,
            acc_noise=self.option.acc_noise,
            gyro_noise=self.option.gyro_noise,
            mag_noise=self.option.mag_noise,
            acc_bias=self.option.acc_bias,
            gyro_bias=self.option.gyro_bias,
            mag_bias=self.option.mag_bias,
            draw_debug=self.option.draw_debug,
        )
        if self.option.magnetic_field is not None:
            imu_kwargs["magnetic_field"] = self.option.magnetic_field

        opts = gs.options.sensors.IMU(**imu_kwargs)
        self.sensor = self.world.scene.add_sensor(opts)

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
        """Read the IMU sensor and cache the observation."""
        data = self.sensor.read()
        dtype = self.world.dtype
        device = self.world.device
        obs: Dict[str, PyTorchArrayType] = {
            "lin_acc": torch.as_tensor(data.lin_acc, dtype=dtype, device=device),
            "ang_vel": torch.as_tensor(data.ang_vel, dtype=dtype, device=device),
            "mag": torch.as_tensor(data.mag, dtype=dtype, device=device),
        }
        self._last_observation = obs

    def _get_observation_space(
        self,
    ) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        B = self.world.batch_size
        dtype = self.world.dtype
        device = self.world.device
        shape = get_batch_aware_shape(B, 3)
        spaces: Dict[str, Space] = {
            "lin_acc": BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=shape,
                dtype=dtype,
                device=device,
            ),
            "ang_vel": BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=shape,
                dtype=dtype,
                device=device,
            ),
            "mag": BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=shape,
                dtype=dtype,
                device=device,
            ),
        }
        return DictSpace(
            PyTorchComputeBackend,
            spaces,
            device=device,
        )

    def get_observation(self) -> Dict[str, PyTorchArrayType]:
        if self._last_observation is None:
            raise RuntimeError(
                f"IMUSensorNode '{self.name}' has no observation yet. "
                f"Ensure the node has been initialised via after_reload/after_reset."
            )
        return self._last_observation
