"""End-effector state sensor node for Genesis.

Reads the end-effector link pose directly from a :class:`RigidLink`, with an
optional **virtual link offset** and optional base-frame transform.  This is
the sensor-node analogue of the EEF observation exposed by
:class:`DiscreteEntityController`.
"""

from __future__ import annotations

from typing import Optional, Dict, Sequence, Tuple, Union
import torch
from dataclasses import dataclass

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
)

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
from unienv_genesis.utils.space_metadata import (
    pose_frame,
    position_metadata,
    quaternion_metadata,
)

from .base import SensorNode

__all__ = [
    "EndEffectorStateSensorNodeOptions",
    "EndEffectorStateSensorNode",
]


@dataclass(frozen=True)
class EndEffectorStateSensorNodeOptions:
    """Options for :class:`EndEffectorStateSensorNode`."""

    attach: AttachmentProvider
    """Provider that resolves the target link whose pose is reported.

    Use ``AttachFromName(entity_name, link_name)``.  When the provider returns
    ``link=None`` the entity's ``base_link`` is used.  When the provider
    exposes ``offset_pos`` / ``offset_quat`` attributes (e.g.
    ``AttachFromName(..., offset_pos=...)``) they describe a virtual EEF
    offset in the resolved link's local frame.
    """

    base_frame: Optional[AttachmentProvider] = None
    """Optional base frame provider (resolves to a ``RigidLink``).

    When ``None`` the reported pose is in the world frame.  When the provider
    exposes ``offset_pos`` / ``offset_quat`` attributes they describe a
    virtual base-frame offset in the resolved base link's local frame.
    """

    rotation_in_quaternion: bool = True
    """When True, expose ``eef_quaternion``.  The observation space always
    includes both ``eef_position`` and ``eef_quaternion`` (currently only
    quaternion rotation is supported)."""


class EndEffectorStateSensorNode(SensorNode[None, None, Dict[str, PyTorchArrayType]]):
    """Sensor node that reads the end-effector pose with a virtual EEF offset.

    Reads the target link pose directly (NOT a Genesis sensor), so the
    observation is valid immediately after build/reset.

    .. note:: **Virtual EEF point convention**

       When the ``attach`` provider exposes an ``offset_pos`` attribute, the
       reported pose refers to a *virtual* EEF point, **not** the physical
       ``target_link`` origin.

       The offset should place this virtual point at the **gripper fingertip**
       — i.e. the farthest extent along the approach direction, centred in
       the opening direction.  The offset is expressed in the target_link's
       local frame: ``eef_pos = link_pos + quaternion_apply(link_quat,
       offset_pos)`` and ``eef_quat = quaternion_multiply(link_quat,
       offset_quat)``.

       When ``base_frame`` is configured the resulting world-frame pose is
       transformed into the (possibly offset) base frame.
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
        option: EndEffectorStateSensorNodeOptions,
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

        # Read the virtual offsets from the attachment providers (defensively,
        # since AttachmentProvider is a Protocol and not every implementation
        # exposes these attributes).
        eef_offset_pos = getattr(option.attach, 'offset_pos', None)
        eef_offset_quat = getattr(option.attach, 'offset_quat', None)
        if option.base_frame is not None:
            base_frame_offset_pos = getattr(option.base_frame, 'offset_pos', None)
            base_frame_offset_quat = getattr(option.base_frame, 'offset_quat', None)
        else:
            base_frame_offset_pos = None
            base_frame_offset_quat = None

        # Precompute the virtual EEF offset tensors (in target_link local frame).
        if eef_offset_pos is not None:
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

        # Precompute the virtual base-frame offset tensors (in the base_frame
        # link's local frame).  Mirrors DiscreteEntityController.
        if base_frame_offset_pos is not None:
            self._base_frame_offset_pos = torch.as_tensor(
                base_frame_offset_pos, dtype=torch.float32, device=world.device,
            )
            if base_frame_offset_quat is not None:
                self._base_frame_offset_quat = torch.as_tensor(
                    base_frame_offset_quat, dtype=torch.float32, device=world.device,
                )
            else:
                self._base_frame_offset_quat = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=world.device,
                )
        else:
            self._base_frame_offset_pos = None
            self._base_frame_offset_quat = None

        self.sensor = None
        self._target_link = None
        self._base_frame_link = None

        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        # Shapes are fixed (3 and 4) so the space can be built up-front.
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
        """Resolve the target link and optional base frame link."""
        if priority != 0:
            return
        entity, link = self.option.attach.get_attachment(self.world)
        self._target_link = link if link is not None else entity.base_link

        if self.option.base_frame is not None:
            base_entity, base_link = self.option.base_frame.get_attachment(self.world)
            self._base_frame_link = (
                base_link if base_link is not None else base_entity.base_link
            )
        else:
            self._base_frame_link = None

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

    # ------------------------------------------------------------------
    # Virtual offset / base-frame math (mirrors DiscreteEntityController)
    # ------------------------------------------------------------------

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

    def _convert_absolute_to_base(
        self,
        pos_abs: PyTorchArrayType,
        quat_abs: PyTorchArrayType,
    ) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Transform a pose from world frame to the configured base frame.

        Mirrors :meth:`DiscreteEntityController.get_base_transform` /
        :meth:`DiscreteEntityController.convert_absolute_to_base`: when
        ``base_frame_offset_pos`` is set, the base frame is a *virtual* frame
        offset from the physical ``base_frame`` link origin.
        """
        if self._base_frame_link is None:
            return pos_abs, quat_abs
        base_pos = self._base_frame_link.get_pos()
        base_quat = self._base_frame_link.get_quat()
        # Apply the virtual base-frame offset (if configured).
        if self._base_frame_offset_pos is not None:
            base_pos = base_pos + quaternion_apply(base_quat, self._base_frame_offset_pos)
            base_quat = quaternion_multiply(base_quat, self._base_frame_offset_quat)
        pos_base = quaternion_apply(
            quaternion_invert(base_quat), pos_abs - base_pos,
        )
        quat_base = quaternion_multiply(quaternion_invert(base_quat), quat_abs)
        return pos_base, quat_base

    def update_observation(self) -> None:
        """Read the target link pose, apply the virtual offset and base frame."""
        link_pos = self._target_link.get_pos()
        link_quat = self._target_link.get_quat()
        virt_pos, virt_quat = self._link_to_virtual(link_pos, link_quat)
        eef_pos, eef_quat = self._convert_absolute_to_base(virt_pos, virt_quat)

        dtype = self.world.dtype
        device = self.world.device
        obs: Dict[str, PyTorchArrayType] = {
            "eef_position": torch.as_tensor(eef_pos, dtype=dtype, device=device),
            "eef_quaternion": torch.as_tensor(eef_quat, dtype=dtype, device=device),
        }
        self._last_observation = obs

    def _get_observation_space(
        self,
    ) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        B = self.world.batch_size
        dtype = self.world.dtype
        device = self.world.device
        frame = pose_frame(self.option.base_frame is not None)
        extra = {"source": self.name}
        link_name = getattr(self.option.attach, "link_name", None)
        if isinstance(link_name, str):
            extra["link"] = link_name
        spaces: Dict[str, Space] = {
            "eef_position": BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf,
                high=torch.inf,
                shape=get_batch_aware_shape(B, 3),
                dtype=dtype,
                device=device,
                metadata=position_metadata(frame=frame, **extra),
            ),
            "eef_quaternion": BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=1,
                shape=get_batch_aware_shape(B, 4),
                dtype=dtype,
                device=device,
                metadata=quaternion_metadata(frame=frame, **extra),
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
                f"EndEffectorStateSensorNode '{self.name}' has no observation yet. "
                f"Ensure the node has been initialised via after_reload/after_reset."
            )
        return self._last_observation
