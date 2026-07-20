"""Contact force sensor node for Genesis.

Wraps ``gs.options.sensors.ContactForce`` via ``scene.add_sensor()``.
"""

from __future__ import annotations

from typing import Optional, Dict, Union, Sequence
import torch
from dataclasses import dataclass

import genesis as gs
from genesis.engine.sensors.contact_force import ContactForceSensor

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
    "ContactForceSensorNodeOptions",
    "ContactForceSensorNode",
]


@dataclass(frozen=True)
class ContactForceSensorNodeOptions:
    """Options for :class:`ContactForceSensorNode`."""

    attach: AttachmentProvider
    """Provider that resolves the entity/link whose contact force is measured."""

    min_force: Union[float, Sequence[float]] = 0.0
    max_force: Union[float, Sequence[float]] = float("inf")

    draw_debug: bool = False


class ContactForceSensorNode(SensorNode[ContactForceSensor, None, Dict[str, PyTorchArrayType]]):
    """Sensor node wrapping the Genesis ContactForce sensor.

    Reports ``contact_force`` — a ``(3,)`` tensor (unbatched) or ``(B, 3)``
    tensor (batched) giving the contact force on the link in the **link local
    frame**.

    .. note::

       Genesis detects contacts during the collider pass inside
       ``scene.step()`` and clears the contact list on ``scene.reset()``.
       :class:`GenesisWorld` performs a no-advance sensor-cache refresh after
       full resets/reloads, but it never advances the physics automatically,
       so contact forces will read **zero** until the collider has run —
       i.e. until after the first one or two ``env.step()`` calls.
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
        option: ContactForceSensorNodeOptions,
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
        self.sensor: Optional[ContactForceSensor] = None
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
        """Add the ContactForce sensor to the (unbuilt) scene."""
        if priority != 0:
            return
        entity, link = self.option.attach.get_attachment(self.world)
        link = link if link is not None else entity.base_link

        opts = gs.options.sensors.ContactForce(
            entity_idx=entity.idx,
            link_idx_local=link.idx_local,
            min_force=self.option.min_force,
            max_force=self.option.max_force,
            draw_debug=self.option.draw_debug,
        )
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
        """Read the ContactForce sensor and cache the observation."""
        data = self.sensor.read()
        obs: Dict[str, PyTorchArrayType] = {
            "contact_force": torch.as_tensor(
                data, dtype=self.world.dtype, device=self.world.device,
            ),
        }
        self._last_observation = obs

    def _get_observation_space(
        self,
    ) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        B = self.world.batch_size
        dtype = self.world.dtype
        device = self.world.device
        spaces: Dict[str, Space] = {
            "contact_force": BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=get_batch_aware_shape(B, 3),
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
                f"ContactForceSensorNode '{self.name}' has no observation yet. "
                f"Ensure the node has been initialised via after_reload/after_reset."
            )
        return self._last_observation
