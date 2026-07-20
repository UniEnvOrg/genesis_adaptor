"""Contact sensor node for Genesis.

Wraps ``gs.options.sensors.Contact`` via ``scene.add_sensor()``.
"""

from __future__ import annotations

from typing import Optional, Dict, Union, Sequence
import torch
from dataclasses import dataclass

import genesis as gs
from genesis.engine.sensors.contact_force import ContactSensor

from unienv_interface.space import Space, DictSpace, BoxSpace, BinarySpace
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
    "ContactSensorNodeOptions",
    "ContactSensorNode",
]


@dataclass(frozen=True)
class ContactSensorNodeOptions:
    """Options for :class:`ContactSensorNode`."""

    attach: AttachmentProvider
    """Provider that resolves the entity/link whose contact state is queried."""

    threshold: float = 0.0
    """Contact-force threshold above which the link is considered in contact."""

    filter_link_idx: Sequence[int] = ()
    """Global rigid link indices to filter out of the contact query."""

    draw_debug: bool = False


class ContactSensorNode(SensorNode[ContactSensor, None, Dict[str, PyTorchArrayType]]):
    """Sensor node wrapping the Genesis Contact sensor.

    Reports ``in_contact`` — a boolean scalar (unbatched, shape ``()``) or
    ``(B,)`` boolean tensor (batched) indicating whether the link is in contact
    with any other (non-filtered) link above ``threshold``.

    .. note::

       Genesis detects contacts during the collider pass inside
       ``scene.step()`` and clears the contact list on ``scene.reset()``.
       :class:`GenesisWorld` performs a no-advance sensor-cache refresh after
       full resets/reloads, but it never advances the physics automatically,
       so ``in_contact`` will read **False** until the collider has run —
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
        option: ContactSensorNodeOptions,
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
        self.sensor: Optional[ContactSensor] = None
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
        """Add the Contact sensor to the (unbuilt) scene."""
        if priority != 0:
            return
        entity, link = self.option.attach.get_attachment(self.world)
        link = link if link is not None else entity.base_link

        opts = gs.options.sensors.Contact(
            entity_idx=entity.idx,
            link_idx_local=link.idx_local,
            threshold=self.option.threshold,
            filter_link_idx=self.option.filter_link_idx,
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
        """Read the Contact sensor, squeeze the trailing dim, and cache it."""
        data = self.sensor.read()
        # Genesis returns (1,) unbatched / (B, 1) batched; squeeze the trailing
        # dim to expose () / (B,) bool.
        obs: Dict[str, PyTorchArrayType] = {
            "in_contact": torch.as_tensor(
                data.squeeze(-1), dtype=torch.bool, device=self.world.device,
            ),
        }
        self._last_observation = obs

    def _get_observation_space(
        self,
    ) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        B = self.world.batch_size
        device = self.world.device
        spaces: Dict[str, Space] = {
            "in_contact": BinarySpace(
                PyTorchComputeBackend,
                shape=get_batch_aware_shape(B, ()),
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
                f"ContactSensorNode '{self.name}' has no observation yet. "
                f"Ensure the node has been initialised via after_reload/after_reset."
            )
        return self._last_observation
