"""Tactile (touch) sensor node for Genesis: one probe per link, observation keyed by link name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import trimesh

import genesis as gs
from genesis.engine.sensors.base_sensor import Sensor
from genesis.utils.geom import trans_quat_to_T

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
    "TactileSensorNodeOptions",
    "TactileSensorNode",
]

_SENSOR_TYPES = ("contact_depth", "kinematic")  # depth (1,) m | link-frame force/torque (3,)

_FACE_AXES = {
    "+x": (0, 1.0), "-x": (0, -1.0),
    "+y": (1, 1.0), "-y": (1, -1.0),
    "+z": (2, 1.0), "-z": (2, -1.0),
}


def _inner_face_point(link, face_axis: str) -> np.ndarray:
    """Probe position on the ``face_axis`` face of the link's collision bounds, in link frame."""
    if face_axis not in _FACE_AXES:
        raise ValueError(
            f"face_axis must be one of {sorted(_FACE_AXES)}, got {face_axis!r}."
        )
    axis, sign = _FACE_AXES[face_axis]

    meshes = []
    for geom in link.geoms:
        mesh = geom.get_trimesh().copy()
        mesh.apply_transform(trans_quat_to_T(geom.init_pos, geom.init_quat))  # rest, known pre-build
        meshes.append(mesh)
    if not meshes:
        raise ValueError(
            f"Link '{getattr(link, 'name', link)}' has no collision geometry to "
            f"place a tactile probe on."
        )
    lo, hi = trimesh.util.concatenate(meshes).bounds

    point = 0.5 * (np.asarray(lo) + np.asarray(hi))
    point[axis] = hi[axis] if sign > 0 else lo[axis]
    return point.astype(np.float64)


@dataclass(frozen=True)
class TactileSensorNodeOptions:
    """Options for :class:`TactileSensorNode`."""

    attaches: Sequence[AttachmentProvider]
    """One provider per instrumented link; each must resolve a named link, used as the sub-key."""

    sensor_type: str = "contact_depth"
    """One of ``"contact_depth"`` or ``"kinematic"`` (see module docstring)."""

    face_axis: str = "+x"
    """Link-local box face every probe sits on, same for all links; ``"+x"`` is the Wuji palm side."""

    probe_radius: float = 0.012
    """Probe sensing radius in metres; deliberately wide to cover a broad patch of the link face."""

    draw_debug: bool = False


class TactileSensorNode(SensorNode[Sensor, None, Dict[str, PyTorchArrayType]]):
    """Sensor node wrapping one Genesis probe sensor per link."""

    reload_priorities = {0}
    after_reload_priorities = {-50}
    reset_priorities = {0}
    after_reset_priorities = {-50}
    post_environment_step_priorities = {0}  # zero until the collider runs: 1-2 env.step() calls

    def __init__(
        self,
        world: GenesisWorld,
        name: str,
        option: TactileSensorNodeOptions,
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

        if option.sensor_type not in _SENSOR_TYPES:
            raise ValueError(
                f"TactileSensorNode '{name}': sensor_type must be one of "
                f"{_SENSOR_TYPES}, got {option.sensor_type!r}."
            )
        self.option = option

        self._keys: list[str] = [a.link_name for a in option.attaches]
        if len(set(self._keys)) != len(self._keys):
            raise ValueError(
                f"TactileSensorNode '{name}' has duplicate link names: {self._keys}"
            )

        self._sensors: Dict[str, Sensor] = {}
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
        """Add one probe sensor per link to the (unbuilt) scene."""
        if priority != 0:
            return
        cls = (
            gs.options.sensors.KinematicTaxel
            if self.option.sensor_type == "kinematic"
            else gs.options.sensors.ContactDepthProbe
        )
        self._sensors = {}
        for key, attach in zip(self._keys, self.option.attaches):
            entity, link = attach.get_attachment(self.world)
            link = link if link is not None else entity.base_link
            probe = _inner_face_point(link, self.option.face_axis)
            self._sensors[key] = self.world.scene.add_sensor(
                cls(
                    entity_idx=entity.idx,
                    link_idx_local=link.idx_local,
                    probe_local_pos=[tuple(probe)],
                    probe_radius=self.option.probe_radius,
                    draw_debug=self.option.draw_debug,
                )
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

    def _to_tensor(self, data) -> PyTorchArrayType:
        return torch.as_tensor(
            data, dtype=self.world.dtype, device=self.world.device
        )

    def update_observation(self) -> None:
        """Read every link's probe and cache the observation."""
        obs: Dict[str, PyTorchArrayType] = {}
        for key, sensor in self._sensors.items():
            data = sensor.read()
            if self.option.sensor_type == "kinematic":
                obs[key] = {  # (1, 3) unbatched / (B, 1, 3) batched -> drop the probe axis
                    "force": self._to_tensor(data.force).squeeze(-2),
                    "torque": self._to_tensor(data.torque).squeeze(-2),
                }
            else:
                obs[key] = self._to_tensor(data)
        self._last_observation = obs

    def _get_observation_space(
        self,
    ) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        B = self.world.batch_size
        dtype = self.world.dtype
        device = self.world.device

        def _link_space() -> Space:
            if self.option.sensor_type == "kinematic":
                vec = BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=get_batch_aware_shape(B, 3),
                    dtype=dtype,
                    device=device,
                )
                return DictSpace(
                    PyTorchComputeBackend,
                    {"force": vec, "torque": vec},
                    device=device,
                )
            return BoxSpace(
                PyTorchComputeBackend,
                low=0.0,
                high=torch.inf,
                shape=get_batch_aware_shape(B, 1),
                dtype=dtype,
                device=device,
            )

        spaces: Dict[str, Space] = {key: _link_space() for key in self._keys}
        return DictSpace(PyTorchComputeBackend, spaces, device=device)

    def get_observation(self) -> Dict[str, PyTorchArrayType]:
        if self._last_observation is None:
            raise RuntimeError(
                f"TactileSensorNode '{self.name}' has no observation yet. "
                f"Ensure the node has been initialised via after_reload/after_reset."
            )
        return self._last_observation
