"""Tactile (touch) sensor node for Genesis: a band of taxels per link, keyed by link name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union

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

_SENSOR_TYPES = ("contact_depth", "kinematic")

_FACE_AXES = {
    "+x": (0, 1.0), "-x": (0, -1.0),
    "+y": (1, 1.0), "-y": (1, -1.0),
    "+z": (2, 1.0), "-z": (2, -1.0),
}


def _build_link_mesh(link) -> trimesh.Trimesh:
    """The link's collision geometry in link frame, available before the scene is built."""
    meshes = []
    for geom in link.geoms:
        mesh = geom.get_trimesh().copy()
        mesh.apply_transform(trans_quat_to_T(geom.init_pos, geom.init_quat))  # rest, known pre-build
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"Link '{getattr(link, 'name', link)}' has no collision geometry to place tactile probes on.")
    return trimesh.util.concatenate(meshes)


def _compute_arc_probes(link, face_axis: str, arc_degrees: float, n_around: int, n_along: int, margin: float):

    """Return taxel positions and outward normals (N, 3) each, in link frame, N = n_around * n_along."""
    if face_axis not in _FACE_AXES:
        raise ValueError(f"face_axis must be one of {sorted(_FACE_AXES)}; got {face_axis!r}.")
    if n_around < 1 or n_along < 1:
        raise ValueError(f"n_around and n_along must be >= 1; got {n_around} and {n_along}.")
    axis, sign = _FACE_AXES[face_axis]

    mesh = _build_link_mesh(link)
    lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    long_axis = int(np.argmax(hi - lo))
    if axis == long_axis:
        raise ValueError(f"face_axis {face_axis!r} points along link '{getattr(link, 'name', link)}'s long axis; the band would wrap its end cap. Pick a face on one of the two cross axes.")
    cross = 3 - long_axis - axis

    centre = 0.5 * (lo + hi)
    span = hi[long_axis] - lo[long_axis]
    stations = (
        np.linspace(lo[long_axis] + margin * span, hi[long_axis] - margin * span, n_along)
        if n_along > 1
        else np.array([centre[long_axis]])
    )
    # arc_degrees spans probe centres; each probe also senses its own footprint,
    # so the band actually sensed is wider by roughly one footprint.
    half = np.radians(arc_degrees) / 2.0
    angles = np.linspace(-half, half, n_around) if n_around > 1 else np.array([0.0])

    origins = np.tile(centre, (n_along, n_around, 1))
    origins[..., long_axis] = stations[:, None]
    dirs = np.zeros((n_along, n_around, 3))
    dirs[..., axis] = sign * np.cos(angles)
    dirs[..., cross] = np.sin(angles)
    origins, dirs = origins.reshape(-1, 3), dirs.reshape(-1, 3)

    pos = np.empty((len(origins), 3))
    nrm = np.empty((len(origins), 3))
    hit = np.zeros(len(origins), dtype=bool)
    loc, ray_idx, tri_idx = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True
    )
    if len(ray_idx):
        t = np.einsum("ij,ij->i", loc - origins[ray_idx], dirs[ray_idx])
        order = np.lexsort((-t, ray_idx))
        rows = order[np.concatenate(([True], np.diff(ray_idx[order]) != 0))]
        hit[ray_idx[rows]] = True
        pos[ray_idx[rows]] = loc[rows]
        nrm[ray_idx[rows]] = mesh.face_normals[tri_idx[rows]]

    if not hit.all():  # concave pocket: fall back to the nearest point on the surface
        reach = 4.0 * float(np.linalg.norm(hi - lo))
        near, _, tri = trimesh.proximity.closest_point(
            mesh, origins[~hit] + dirs[~hit] * reach
        )
        pos[~hit] = near
        nrm[~hit] = mesh.face_normals[tri]

    flip = np.einsum("ij,ij->i", nrm, pos - origins) < 0
    nrm[flip] *= -1.0
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pos, nrm


@dataclass(frozen=True)
class TactileSensorNodeOptions:
    attaches: Sequence[AttachmentProvider]
    """One provider per instrumented link; each must resolve a named link, used as the sub-key."""

    sensor_type: str = "contact_depth"
    """One of ``"contact_depth"`` or ``"kinematic"`` (see module docstring)."""

    face_axis: str = "+x"
    """Link-local face the taxel band is centred on; ``"+x"`` is the Wuji palm side.

    The default for every link. Links whose compute_frame is rotated relative to the rest
    -- a thumb base, typically -- need :attr:`face_axis_overrides` instead.
    """

    face_axis_overrides: Optional[Mapping[str, str]] = None
    """Per-link ``face_axis``, keyed by link name; links absent from it use :attr:`face_axis`."""

    probe_radius: float = 0.004
    """Per-taxel sensing radius in metres; keep it well under the link's cross-section radius."""

    arc_degrees: float = 120.0
    """Angular span of the taxel band, measured between probe centres."""

    n_around: int = 4
    """Taxels across the band."""

    n_along: int = 5
    """Taxel rows along the link's long axis. ``N = n_around * n_along`` per link."""

    margin: float = 0.12
    """Fraction of the link's length left clear at each end."""

    draw_debug: bool = False


class TactileSensorNode(SensorNode[Sensor, None, Dict[str, PyTorchArrayType]]):
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
            raise ValueError(f"TactileSensorNode '{name}': sensor_type must be one of {_SENSOR_TYPES}; got {option.sensor_type!r}.")
        self.option = option

        self._keys: list[str] = [a.link_name for a in option.attaches]
        if len(set(self._keys)) != len(self._keys):
            raise ValueError(f"TactileSensorNode '{name}' has duplicate link names: {self._keys}")

        unknown = set(option.face_axis_overrides or {}) - set(self._keys)
        if unknown:
            raise ValueError(f"TactileSensorNode '{name}': face_axis_overrides names links that are not attached: {sorted(unknown)}. Attached links: {self._keys}.")

        self.n_taxels: int = option.n_around * option.n_along
        self._sensors: Dict[str, Sensor] = {}
        self._positions: Dict[str, np.ndarray] = {}
        self._normals: Dict[str, np.ndarray] = {}
        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        self.observation_space: Space[PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType] = self._get_observation_space()

    @property
    def probe_positions(self) -> Dict[str, np.ndarray]:
        """Per-link taxel positions ``(N, 3)`` in link frame. Populated by ``reload``."""
        return self._positions

    @property
    def probe_normals(self) -> Dict[str, np.ndarray]:
        """Per-link outward taxel normals ``(N, 3)`` in link frame; ContactDepthProbe has none of its own."""
        return self._normals

    def reload(self, *, priority: int = 0, seed: Optional[int] = None,
               mask: Optional[PyTorchArrayType] = None, **kwargs) -> None:
        """Add one probe sensor per link to the (unbuilt) scene."""
        if priority != 0:
            return
        cls = (
            gs.options.sensors.KinematicTaxel
            if self.option.sensor_type == "kinematic"
            else gs.options.sensors.ContactDepthProbe
        )
        overrides = self.option.face_axis_overrides or {}
        self._sensors, self._positions, self._normals = {}, {}, {}
        for attach in self.option.attaches:
            key = attach.link_name
            entity, link = attach.get_attachment(self.world)
            link = link if link is not None else entity.base_link
            probe, normal = _compute_arc_probes(
                link,
                overrides.get(key, self.option.face_axis),
                self.option.arc_degrees,
                self.option.n_around,
                self.option.n_along,
                self.option.margin,
            )
            self._positions[key], self._normals[key] = probe, normal
            self._sensors[key] = self.world.scene.add_sensor(
                cls(
                    entity_idx=entity.idx,
                    link_idx_local=link.idx_local,
                    probe_local_pos=[tuple(p) for p in probe],
                    probe_radius=self.option.probe_radius,
                    draw_debug=self.option.draw_debug,
                )
            )

    def reset(self, *, priority: int = 0, seed: Optional[int] = None,
              mask: Optional[PyTorchArrayType] = None, **kwargs) -> None:
        pass

    def after_reload(self, *, priority: int = 0, mask: Optional[PyTorchArrayType] = None) -> None:
        if priority in self.after_reload_priorities:
            self.update_observation()

    def after_reset(self, *, priority: int = 0, mask: Optional[PyTorchArrayType] = None) -> None:
        if priority in self.after_reset_priorities:
            self.update_observation()

    def post_environment_step(self, dt: Union[float, PyTorchArrayType], *, priority: int = 0) -> None:
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
                obs[key] = {
                    "force": self._to_tensor(data.force),
                    "torque": self._to_tensor(data.torque),
                }
            else:
                obs[key] = self._to_tensor(data)
        self._last_observation = obs

    def _get_observation_space(self) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        B = self.world.batch_size
        dtype = self.world.dtype
        device = self.world.device

        def _create_link_space() -> Space:
            if self.option.sensor_type == "kinematic":
                vec = BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=get_batch_aware_shape(B, (self.n_taxels, 3)),
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
                shape=get_batch_aware_shape(B, self.n_taxels),
                dtype=dtype,
                device=device,
            )

        spaces: Dict[str, Space] = {key: _create_link_space() for key in self._keys}
        return DictSpace(PyTorchComputeBackend, spaces, device=device)

    def get_observation(self) -> Dict[str, PyTorchArrayType]:
        if self._last_observation is None:
            raise RuntimeError(
                f"TactileSensorNode '{self.name}' has no observation yet. "
                f"Ensure the node has been initialised via after_reload/after_reset."
            )
        return self._last_observation
