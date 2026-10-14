"""Camera sensor node for Genesis."""

from typing import Optional, Dict, Any, List, Tuple, Union, Literal, Sequence, TYPE_CHECKING
import sys
import torch
import numpy as np
from dataclasses import dataclass, field
import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from genesis.options.options import Options
from genesis.vis.camera import Camera
from genesis.vis.batch_renderer import BatchRenderer

# DSO note: ``gs_nyx`` bundles its own monolithic OpenUSD runtime
# (``gs_nyx/libusd_ms.so``), which conflicts with the ``usd-core`` one the
# ``pxr`` python modules load — importing both in one process corrupts the
# Tf notice registry and segfaults at the next Usd stage open (both import
# orders; the ArtVIP/USD geometry helpers open pxr stages at call time).
# The Nyx symbols are therefore imported **lazily** on first use of a Nyx
# render path (``_ensure_nyx``); camera_node stays importable next to
# USD-using code when Nyx is never actually rendered from.

_NYX_STATE = {"available": None, "npr": None, "nps": None,
              "NyxCameraOptions": None, "NyxCameraSensor": None}


def _ensure_nyx():
    """Lazily import the gs_nyx renderer symbols; ``False`` when unavailable.

    Refuses when ``pxr`` is already imported: gs_nyx bundles its own
    monolithic OpenUSD runtime (``libusd_ms.so``), and the two runtimes
    corrupt each other's Tf notice registry in one process (segfault/abort
    at the next stage open — *both* import orders).  Callers that opened a
    USD stage get an explicit Nyx-unavailable error instead of a crash.
    """
    if _NYX_STATE["available"] is None:
        if "pxr" in sys.modules:
            _NYX_STATE["available"] = False
            return False
        try:
            import gs_nyx.nyx_py_renderer as npr
            import gs_nyx.nyx_py_sdk as nps
            from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
            from gs_nyx_plugin.nyx_camera_sensor import NyxCameraSensor
            _NYX_STATE.update(
                available=True, npr=npr, nps=nps,
                NyxCameraOptions=NyxCameraOptions, NyxCameraSensor=NyxCameraSensor,
            )
        except Exception:
            _NYX_STATE["available"] = False
    return _NYX_STATE["available"]


try:
    NYX_AVAILABLE: bool = None  # replaced at function scope; see _ensure_nyx
except Exception:  # pragma: no cover
    NYX_AVAILABLE = False


def __getattr__(name):
    """Module-level lazy attributes (``NYX_AVAILABLE`` and the Nyx classes)."""
    global NYX_AVAILABLE
    if name in ("NYX_AVAILABLE", "npr", "nps", "NyxCameraOptions", "NyxCameraSensor"):
        state = _ensure_nyx()
        if not state:
            if name in ("npr", "nps", "NyxCameraOptions", "NyxCameraSensor"):
                raise ImportError(
                    "gs_nyx is unavailable (not installed, or its import "
                    "failed); Nyx camera options/remotes are disabled."
                )
        value = _NYX_STATE.get(name if name != "NYX_AVAILABLE" else "available")
        if name == "NYX_AVAILABLE":
            NYX_AVAILABLE = bool(_NYX_STATE["available"])
            return NYX_AVAILABLE
        return _NYX_STATE[name]
    raise AttributeError(name)

from xbarray.transformations.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_quaternion
)
from xbarray.transformations.pointcloud.pytorch import (
    depth_image_to_world,
)
from unienv_interface.space import Space, DictSpace, BoxSpace
from unienv_interface.backends.pytorch import (
    PyTorchComputeBackend,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)
from unienv_interface.backends.numpy import (
    NumpyComputeBackend,
    NumpyArrayType,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
)
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
from unienv_genesis.options.attach_option import AttachmentProvider

from .base import SensorNode, ContextType, ObsType

__all__ = [
    "CameraSensorNodeOptions",
    "CameraSensorNode",
]
CameraSensor = Union[Camera, BatchRenderer]  # type: ignore[misc]

@dataclass(frozen=True)
class CameraSensorNodeOptions:
    # ── Backend selection ──────────────────────────────────────────────
    backend : Literal["genesis", "nyx"] = "genesis"
    """Rendering backend to use. ``"genesis"`` uses the scene's configured renderer
    (Rasterizer, RayTracer, or BatchRenderer) via ``scene.add_camera()``.
    ``"nyx"`` uses the Nyx path tracer plugin via ``scene.add_sensor()``.
    """
    
    model : Literal['pinhole', 'thinlens', 'fisheye'] = "pinhole"
    """Specifies the camera model. Options are 'pinhole', 'thinlens', or 'fisheye'."""
    res : Tuple[int, int] = (320, 320)
    """The resolution of the camera, specified as a tuple (width, height)."""
    pos : Tuple[float, float, float] = (0.5, 2.5, 3.5)
    """The position of the camera in the scene, specified as (x, y, z)."""
    quat : Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Camera orientation quaternion ``(w, x, y, z)`` in OpenGL convention.

    OpenGL camera axes are:
      - ``+X`` right
      - ``+Y`` up
      - ``-Z`` forward (view direction)
    """
    fov : float = 30
    """The vertical field of view of the camera in degrees."""
    aperture : float = 2.0
    """The aperture size of the camera, controlling depth of field."""
    focus_dist : Optional[float] = None
    """The focus distance of the camera. If None, it will be auto-computed using `pos` and `lookat`."""
    GUI : bool = False
    """Whether to display the camera's rendered image in a separate GUI window."""
    spp : int = 256
    """Samples per pixel. Only available when using RayTracer / NYX renderer. Defaults to 256."""
    denoise : Optional[bool] = None
    """Whether to denoise the camera's rendered image. Only available when using the RayTracer / NYX renderer. Defaults to `None` which means true for ray tracers and false for rasterizers."""
    near : float = 0.1
    """Distance from camera center to near plane in meters."""
    far : float = 20.0
    """Distance from camera center to far plane in meters."""
    env_idx : Optional[int] = None
    """The specific environment index to bind to the camera. This option must be specified if and only if a
    non-batched renderer is being used. If provided, only this environment will be taken into account when
    following a rigid entity via 'follow_entity' and when being attached to some rigid link via 'attach'. Note
    that this option is unrelated to which environment is being rendering on the scene. Default to None for
    batched renderers (ie BatchRender), 'rendered_envs_idx[0]' otherwise (ie Raytracer or Rasterizer).
    """
    debug : bool = False
    """Whether to use the debug camera for monitoring/debugging without interfering with robot perception."""
    attach : Optional[AttachmentProvider] = None
    """Provider that resolves the entity/link to attach the camera to.
    If ``None``, the camera will not be attached to any entity.
    Use ``AttachFromName(entity_name, link_name)`` for name-based attachment,
    or implement a custom ``AttachmentProvider``.
    """
    # ========== NYX-specific options ==========
    render_mode : Optional[Any] = None
    """``npr.ERenderMode`` value (e.g. ``npr.ERenderMode.FastPathTracer``).
    If ``None``, uses ``NyxCameraOptions`` default. Only used when ``backend="nyx"``.
    """
    anti_aliasing : Optional[Any] = None
    """``nps.EAntiAliasing`` value. Forwarded to ``NyxCameraOptions.anti_aliasing``.
    Only used when ``backend="nyx"``.
    """
    debug_view : Optional[Any] = None
    """``npr.EDebugView`` value. Forwarded to ``NyxCameraOptions.debug_view``.
    Only used when ``backend="nyx"``.
    """
    light_fields : Tuple[Any, ...] = ()
    """Additional per-camera Nyx ``LightFieldAsset`` objects.

    These are merged with the shared light fields configured on
    ``GenesisWorld(light_fields=...)`` and passed to
    ``NyxCameraOptions(light_fields=...)``. Only used when
    ``backend="nyx"``; ignored by other backends. Typed as a tuple of
    ``Any`` so the option remains usable even when the Nyx plugin (and
    therefore ``gs_nyx.nyx_py_sdk.LightFieldAsset``) is not installed.
    """

    @staticmethod
    def quat_from_lookat(
        pos: Tuple[float, float, float],
        lookat: Tuple[float, float, float],
        up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> Tuple[float, float, float, float]:
        """Compute OpenGL-convention quaternion from look-at parameters.

        Returns ``(w, x, y, z)`` where the camera frame follows OpenGL:
        ``+Y`` is up and ``-Z`` points toward ``lookat``.
        """
        pos = np.array(pos, dtype=np.float32)
        lookat = np.array(lookat, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        forward = lookat - pos
        forward /= np.linalg.norm(forward)
        Z_axis = -forward  # OpenGL camera forward is -Z
        Y_axis = up
        Y_axis -= Y_axis.dot(Z_axis) * Z_axis  # Make orthogonal to Z
        Y_axis /= np.linalg.norm(Y_axis)
        X_axis = np.cross(Y_axis, Z_axis)
        rot_matrix = np.stack([X_axis, Y_axis, Z_axis], axis=1)  # camera-to-world
        quat = matrix_to_quaternion(NumpyComputeBackend, rot_matrix)
        return tuple(quat)
    
    def get_lookat(self) -> Tuple[float, float, float]:
        """Compute look-at point from OpenGL-convention ``quat``."""
        rot_matrix = quaternion_to_matrix(
            NumpyComputeBackend,
            np.array(self.quat, dtype=np.float32),
        )
        forward_vector = -rot_matrix[:, 2]  # OpenGL forward is -Z
        lookat = np.array(self.pos, dtype=np.float32) + forward_vector
        return tuple(lookat)
    
    def get_up(self) -> Tuple[float, float, float]:
        """Compute OpenGL up vector from ``quat``."""
        rot_matrix = quaternion_to_matrix(
            NumpyComputeBackend,
            np.array(self.quat, dtype=np.float32),
        )
        up_vector = rot_matrix[:, 1]
        return tuple(up_vector)

class CameraSensorNode(SensorNode[
    CameraSensor,
    None,
    Dict[str, PyTorchArrayType],
]):
    """Camera sensor node for rendering RGB and depth images.

    This node wraps Genesis's Camera sensor which provides visual
    observations from a specified viewpoint in the scene.

    When ``render_rgb=True`` (the default), the node declares
    ``render_mode = "rgb_array"`` so it participates in the standard
    ``env.render()`` pipeline and is usable with
    ``EpisodeVideoWrapper``.
    """
    supported_render_modes = ("rgb_array",)

    reload_priorities = {0}
    after_reload_priorities = {-50}
    reset_priorities = {0}
    after_reset_priorities = {-50}
    post_environment_step_priorities = {0}

    def __init__(
        self,
        world: GenesisWorld,
        name: str,
        option : CameraSensorNodeOptions,
        render_rgb: bool = True,
        render_depth: bool = False,
        render_seg: bool = False,
        render_normal: bool = False,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
        render_position: bool = False,
    ):
        """Initialize camera sensor node.
        
        Args:
            world: GenesisWorld instance
            name: Unique name for this sensor
            option: Camera options
            attach_entity_name: Name of the entity to attach the camera to
            attach_entity_link: Name of the link to attach the camera to
            control_timestep: Control timestep (defaults to world timestep)
            update_timestep: Update timestep (defaults to world timestep)
        """
        
        super().__init__(
            world=world,
            name=name,
            control_timestep=control_timestep,
            update_timestep=update_timestep,
        )
        self.world : GenesisWorld = world
        
        self.sensor: CameraSensor | "NyxCameraSensor" | None = None
        self._genesis_sensor: CameraSensor | None = None  # Secondary genesis camera for depth/seg/normal when using NYX
        self.option = option
        assert (render_rgb or render_depth or render_seg or render_normal or render_position), (
            "At least one of render_rgb, render_depth, render_seg, render_normal, or render_position must be True"
        )
        if option.backend == "nyx":
            if not _ensure_nyx():
                raise ImportError(
                    "NYX backend requires gs-nyx-plugin to be installed. "
                    "Install it according to https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/nyx_renderer.html"
                )
            if not render_rgb:
                raise ValueError(
                    "NYX backend requires render_rgb=True."
                )
        self.render_rgb = render_rgb
        self.render_depth = render_depth
        self.render_seg = render_seg
        self.render_normal = render_normal
        self.render_position = render_position
        # Position map is derived from depth. We may render depth internally
        # even when it is not explicitly exposed in observations.
        self._render_depth_internal = self.render_depth or self.render_position
        
        # Attachment state (resolved during _init_sensor)
        self._attach_entity = None
        self._attach_link = None
        self._offset_T = None  # 4x4 camera-to-link transform in OpenGL convention
        # Per-env camera-to-world pose (OpenGL convention, ``(B, 4, 4)``) set by
        # :meth:`set_pose`; ``None`` while the camera keeps its option pose.
        self._world_pose_T: Optional[torch.Tensor] = None
        # Genesis caches rendered frames per simulation step; a pose change
        # within the step must bypass that cache once.
        self._force_genesis_render = False

        # Expose render() when RGB is available.
        self.render_mode = "rgb_array" if render_rgb else None

        # Cached observation
        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        
        # Build observation space
        self.observation_space = self._get_observation_space()

    def _init_sensor(self) -> None:
        """Initialize the camera sensor according to the specified backend."""
        # Resolve the attachment once for all backends.  An entity-only
        # attachment (``link=None``) is normalized to the entity's base link
        # so that the link handed to the renderer and the link used to derive
        # reported extrinsics are always the same link.
        self._attach_entity = None
        self._attach_link = None
        self._world_pose_T = None  # a rebuilt scene starts from the option pose
        if self.option.attach is not None:
            entity, link = self.option.attach.get_attachment(self.world)
            if entity is not None and link is None:
                link = entity.base_link
            self._attach_entity = entity
            self._attach_link = link

        if self.option.backend == "nyx":
            self.sensor = self._init_sensor_nyx()
            # If depth/seg/normal/position is requested, also create a genesis camera
            if self._render_depth_internal or self.render_seg or self.render_normal:
                self._genesis_sensor = self._init_sensor_genesis()
        else:
            self.sensor = self._init_sensor_genesis()

    def _init_sensor_genesis(self) -> CameraSensor:
        """Initialize via Genesis ``scene.add_camera()``."""

        camera = self.world.scene.add_camera(
            model=self.option.model,
            res=self.option.res,
            pos=self.option.pos,
            lookat=self.option.get_lookat(),
            up=self.option.get_up(),
            fov=self.option.fov,
            aperture=self.option.aperture,
            focus_dist=self.option.focus_dist,
            GUI=self.option.GUI,
            spp=self.option.spp,
            denoise=self.option.denoise,
            near=self.option.near,
            far=self.option.far,
            env_idx=self.option.env_idx,
            debug=self.option.debug
        )
        if self._attach_entity is not None:
            # ``_attach_link`` was normalized to the entity's base link in
            # ``_init_sensor`` when the provider gave an entity-only attach.
            attach_link = self._attach_link
            relative_T = np.eye(4, dtype=np.float32)
            relative_T[:3, :3] = quaternion_to_matrix(
                NumpyComputeBackend,
                np.array(self.option.quat, dtype=np.float32),
            )
            relative_T[:3, 3] = np.array(self.option.pos, dtype=np.float32)
            self._offset_T = relative_T.copy()
            camera.attach(attach_link, relative_T)
        else:
            # Static camera: offset_T is the camera's own pose in world frame
            self._offset_T = np.eye(4, dtype=np.float32)
            self._offset_T[:3, :3] = quaternion_to_matrix(
                NumpyComputeBackend,
                np.array(self.option.quat, dtype=np.float32),
            )
            self._offset_T[:3, 3] = np.array(self.option.pos, dtype=np.float32)
        
        return camera

    def _init_sensor_nyx(self) -> CameraSensor:
        """Initialize via NYX ``scene.add_sensor(NyxCameraOptions(...))``."""

        # Resolve attachment to entity_idx / link_idx_local / offset_T.
        # ``_attach_link`` was normalized to the entity's base link in
        # ``_init_sensor`` for entity-only attachments, so the link passed
        # to NYX here is the same link used by ``_get_extrinsics``.
        entity_idx = None
        link_idx_local = None
        offset_T = None
        if self._attach_entity is not None:
            entity_idx = self._attach_entity.idx
            link_idx_local = self._attach_link.idx_local
            # Build the 4x4 offset transform from quat + pos
            offset_T = np.eye(4, dtype=np.float64)
            offset_T[:3, :3] = quaternion_to_matrix(
                NumpyComputeBackend,
                np.array(self.option.quat, dtype=np.float32),
            ).astype(np.float64)
            offset_T[:3, 3] = np.array(self.option.pos, dtype=np.float64)
            self._offset_T = offset_T.astype(np.float32)
        else:
            # Static camera: offset_T is the camera's own pose in world frame
            self._offset_T = np.eye(4, dtype=np.float32)
            self._offset_T[:3, :3] = quaternion_to_matrix(
                NumpyComputeBackend,
                np.array(self.option.quat, dtype=np.float32),
            )
            self._offset_T[:3, 3] = np.array(self.option.pos, dtype=np.float32)

        lights = []
        env_map = None
        for light in self.world.lights.values():
            lights.append(light.to_light_dict())
        
        if self.world.env_map is not None:
            env_map = self.world.env_map.to_hyx_asset()

        # Merge shared world light fields with per-camera option light fields.
        # Both are Nyx LightFieldAsset objects forwarded to NyxCameraOptions.
        light_fields = tuple(self.world.light_fields) + tuple(self.option.light_fields)

        # Build NyxCameraOptions kwargs
        nyx_kwargs = dict(
            res=self.option.res,
            pos=self.option.pos,
            lookat=self.option.get_lookat(),
            up=self.option.get_up(),
            fov=self.option.fov,
            near=self.option.near,
            far=self.option.far,
            spp=self.option.spp,
            lights=lights,
            env_maps=(env_map,) if env_map is not None else (),
            light_fields=light_fields,
        )
        
        # Only pass render_mode if explicitly set
        if self.option.render_mode is not None:
            nyx_kwargs["render_mode"] = self.option.render_mode

        # Optional NYX fields — only pass when explicitly set
        if self.option.denoise is not None:
            nyx_kwargs["denoise"] = self.option.denoise
        if entity_idx is not None:
            nyx_kwargs["entity_idx"] = entity_idx
        if link_idx_local is not None:
            nyx_kwargs["link_idx_local"] = link_idx_local
        if offset_T is not None:
            nyx_kwargs["offset_T"] = offset_T
        if self.option.anti_aliasing is not None:
            nyx_kwargs["anti_aliasing"] = self.option.anti_aliasing
        if self.option.debug_view is not None:
            nyx_kwargs["debug_view"] = self.option.debug_view

        assert _ensure_nyx(), "the Nyx path requires gs_nyx (was unresolvable)"
        camera = self.world.scene.add_sensor(_NYX_STATE["NyxCameraOptions"](**nyx_kwargs))
        return camera

    def set_pose(
        self,
        pos: Any,
        lookat: Any,
        up: Any = (0.0, 0.0, 1.0),
        envs_idx: Optional[Any] = None,
    ) -> None:
        """Move a static camera at runtime, per environment.

        Call after the scene is built (e.g. from an ``after_reload`` hook).
        The pose persists across resets and is dropped when the scene is
        rebuilt. Rendering and the reported ``extrinsics_cv`` follow it.

        Args:
            pos / lookat / up: ``(3,)`` or ``(N, 3)`` world-frame camera
                position, look-at point and up vector.
            envs_idx: Environments to move (``N`` of them); all when ``None``.
        """
        if self.option.attach is not None:
            raise ValueError(f"camera {self.name!r} is attached to a link; it cannot be posed")
        if self.sensor is None:
            raise RuntimeError(f"camera {self.name!r} is not built yet")
        import genesis.utils.geom as gu

        device = self.world.device
        n_envs = self.world.batch_size if self.world.batch_size is not None else 1
        if envs_idx is None:
            envs_idx = torch.arange(n_envs, device=device)
        envs_idx = torch.as_tensor(envs_idx, dtype=torch.long, device=device).reshape(-1)
        n = envs_idx.numel()

        def rows(value):
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
            return value.reshape(-1, 3).expand(n, 3)

        pose_T = gu.pos_lookat_up_to_T(rows(pos), rows(lookat), rows(up))  # (N, 4, 4) cam->world, OpenGL

        if self._world_pose_T is None:
            genesis_camera = self._genesis_sensor if self._genesis_sensor is not None else self.sensor
            if hasattr(genesis_camera, "get_transform"):
                current = torch.as_tensor(genesis_camera.get_transform(), dtype=torch.float32, device=device)
            else:
                current = torch.as_tensor(self._offset_T, dtype=torch.float32, device=device)
            self._world_pose_T = current.reshape(-1, 4, 4).expand(n_envs, 4, 4).clone()
        self._world_pose_T[envs_idx] = pose_T.to(self._world_pose_T.dtype)

        # Genesis camera (the primary one, or the NYX depth/seg/normal one).
        genesis_camera = self._genesis_sensor if self._genesis_sensor is not None else (
            self.sensor if self.option.backend != "nyx" else None
        )
        if genesis_camera is not None:
            if self.world.batch_size is None:
                genesis_camera.set_pose(transform=self._world_pose_T[0])
            else:
                genesis_camera.set_pose(transform=pose_T, envs_idx=envs_idx)
            self._force_genesis_render = True
        if self.option.backend == "nyx":
            # Per-env static poses through the plugin's attachment path (the
            # public ``update_camera_pose`` applies one pose to every env);
            # static NYX cameras have no link, so nothing overrides it.
            self.sensor._apply_camera_transform(self._world_pose_T)
            self._invalidate_nyx_render_cache()

    def reload(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs
    ) -> None:
        """Add camera to the scene."""
        if priority == 0:
            self._init_sensor()

    def reset(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs
    ) -> None:
        """Reset camera state if needed.

        Intentionally a no-op: camera/sensor state is owned by Genesis's
        reset machinery — framework sensors (Nyx via ``scene.add_sensor``)
        are reset by the ``SensorManager``, and visualizer cameras / the
        BatchRenderer by ``scene.reset()`` -> ``visualizer.reset()``. Calling
        their ``reset()`` directly is redundant and signature-incompatible
        with the current Genesis API (framework ``Sensor.reset`` is a
        classmethod taking shared metadata/cache; ``BatchRenderer.reset``
        takes no ``envs_idx``).
        """
    
    def after_reset(
        self,
        *,
        priority: int = 0,
        mask: Optional[PyTorchArrayType] = None,
    ) -> None:
        """Update observation after reset."""
        if priority in self.after_reset_priorities:
            self._invalidate_nyx_render_cache()
            self.update_observation()

    def _invalidate_nyx_render_cache(self) -> None:
        """Force the Nyx path tracer to re-render on the next ``read()``.

        Genesis camera sensors only invalidate their cached frame when
        ``scene.t`` advances. A reset that follows an episode with zero
        elapsed steps leaves ``scene.t`` unchanged, so ``read()`` would
        return the pre-reset image even though entities have moved. Marking
        the sensor stale makes the post-reset ``update_observation()``
        render the freshly-reset scene. (The Nyx renderer re-renders all
        sibling cameras in one pass, so flagging this sensor is enough.)
        """
        if self.option.backend == "nyx" and self.sensor is not None:
            self.sensor._stale = True

    def after_reload(
        self,
        *,
        priority: int = 0,
        mask: Optional[PyTorchArrayType] = None,
    ) -> None:
        """Update observation after reload."""
        if priority in self.after_reload_priorities:
            self.update_observation()
    
    def post_environment_step(
        self,
        dt: Union[float, PyTorchArrayType],
        *,
        priority: int = 0
    ) -> None:
        """Update observation after environment step."""
        if priority == 0:
            self.update_observation()

    def _get_observation_space(self) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        """Get the observation space for this camera sensor."""
        spaces: Dict[str, Space] = {}

        W, H = self.option.res
        B = self.world.batch_size
        if self.render_rgb:
            spaces["rgb"] = BoxSpace(
                PyTorchComputeBackend,
                low=0,
                high=255,
                shape=get_batch_aware_shape(B, (H, W, 3)),
                dtype=torch.uint8,
                device=self.world.device,
            )

        if self.render_depth:
            spaces["depth"] = BoxSpace(
                PyTorchComputeBackend,
                low=0.0,
                high=float("inf"),
                shape=get_batch_aware_shape(B, (H, W)),
                dtype=self.world.dtype,
                device=self.world.device,
            )

        if self.render_seg:
            spaces["seg"] = BoxSpace(
                PyTorchComputeBackend,
                low=0,
                high=float("inf"),
                shape=get_batch_aware_shape(B, (H, W)),
                dtype=torch.int32,
                device=self.world.device,
            )

        if self.render_normal:
            spaces["normal"] = BoxSpace(
                PyTorchComputeBackend,
                low=-1.0,
                high=1.0,
                shape=get_batch_aware_shape(B, (H, W, 3)),
                dtype=self.world.dtype,
                device=self.world.device,
            )

        if self.render_position:
            spaces["position"] = BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=get_batch_aware_shape(B, (H, W, 3)),
                dtype=self.world.dtype,
                device=self.world.device,
            )

        spaces["intrinsics_cv"] = BoxSpace(
            PyTorchComputeBackend,
            low=-float("inf"),
            high=float("inf"),
            shape=get_batch_aware_shape(B, (3, 3)),
            dtype=self.world.dtype,
            device=self.world.device,
        )
        spaces["extrinsics_cv"] = BoxSpace(
            PyTorchComputeBackend,
            low=-float("inf"),
            high=float("inf"),
            shape=get_batch_aware_shape(B, (3, 4)),
            dtype=self.world.dtype,
            device=self.world.device,
        )
        spaces["local_extrinsics_cv"] = BoxSpace(
            PyTorchComputeBackend,
            low=-float("inf"),
            high=float("inf"),
            shape=get_batch_aware_shape(B, (3, 4)),
            dtype=self.world.dtype,
            device=self.world.device,
        )

        return DictSpace(
            PyTorchComputeBackend,
            spaces,
            device=self.world.device,
        )

    def _get_intrinsics(self) -> torch.Tensor:
        """Build intrinsic matrix in standard OpenCV convention.

        For the Genesis backend, ``cx``/``cy`` are read from the sensor with a
        half-pixel offset correction.  For the NYX backend, intrinsics are
        computed analytically from ``fov`` and ``res``.
        
        If a secondary genesis sensor exists (NYX backend with depth/seg/normal),
        prefer its intrinsics for consistency with the depth/seg/normal data.

        Returns ``(3, 3)`` for a single camera, ``(B, 3, 3)`` for batched.
        """
        # Prefer genesis sensor if available (for consistency with depth/seg/normal)
        if self._genesis_sensor is not None:
            sensor = self._genesis_sensor
        elif self.option.backend == "nyx":
            return self._get_intrinsics_from_fov()
        else:
            sensor = self.sensor

        fx = float(sensor.f)
        fy = fx  # Genesis uses a single focal length
        cx = float(sensor.cx)
        cy = float(sensor.cy)
        K = torch.tensor(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=self.world.dtype,
            device=self.world.device,
        )
        B = self.world.batch_size
        if B is not None:
            K = K.unsqueeze(0).expand(B, -1, -1)
        return K

    def _get_intrinsics_from_fov(self) -> torch.Tensor:
        """Compute intrinsic matrix from ``fov`` and ``res`` (NYX backend).

        Uses the standard pinhole relation ``f = (H/2) / tan(fov/2)`` and
        places the principal point at the image centre (OpenCV convention).
        """
        W, H = self.option.res
        fov_rad = np.deg2rad(self.option.fov)
        fy = (H / 2.0) / np.tan(fov_rad / 2.0)
        fx = fy
        cx = W / 2.0
        cy = H / 2.0
        K = torch.tensor(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=self.world.dtype,
            device=self.world.device,
        )
        B = self.world.batch_size
        if B is not None:
            K = K.unsqueeze(0).expand(B, -1, -1)
        return K

    def _get_extrinsics(self) -> torch.Tensor:
        """Build ``(..., 3, 4)`` world-to-camera extrinsic matrix (OpenCV).

        Returns the ``[R | t]`` matrix such that ``cam = R @ world + t``.
        Shape is ``(3, 4)`` for a single camera, ``(B, 3, 4)`` for batched.

        For the Genesis backend the transform is read from ``sensor.get_transform()``.
        For the NYX backend, if the camera is attached to a link, the world extrinsics
        are computed from the link's current world transform composed with the stored
        ``_offset_T``.  Entity-only attachments were normalized to the entity's base
        link in ``_init_sensor``, so this is the same link NYX renders from.  If a
        secondary genesis sensor exists (NYX backend with depth/seg/normal), prefer
        its extrinsics for consistency.
        """
        T_OPENGL_TO_OPENCV = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=self.world.dtype,
            device=self.world.device,
        )

        # Prefer genesis sensor if available (for consistency with depth/seg/normal)
        sensor = self._genesis_sensor if self._genesis_sensor is not None else self.sensor

        if hasattr(sensor, "get_transform"):
            camera_transform = torch.as_tensor(
                sensor.get_transform(),
                dtype=self.world.dtype,
                device=self.world.device,
            )
        elif self.option.backend == "nyx" and self._attach_link is not None:
            # NYX attached camera: compose link world transform with offset_T
            camera_transform = self._get_nyx_attached_transform()
        elif self.option.backend == "nyx" and self._world_pose_T is not None:
            # NYX static camera moved by set_pose
            camera_transform = self._world_pose_T.to(dtype=self.world.dtype)
        elif self.option.backend == "nyx":
            # NYX static camera: use stored offset_T as world pose
            camera_transform = torch.as_tensor(
                self._offset_T, dtype=self.world.dtype, device=self.world.device,
            )
        else:
            raise RuntimeError(
                "Sensor does not expose get_transform(); cannot compute extrinsics."
            )

        cam_to_world = camera_transform @ T_OPENGL_TO_OPENCV
        world_to_cam = torch.linalg.inv(cam_to_world)
        return world_to_cam[..., :3, :]

    def _get_nyx_attached_transform(self) -> torch.Tensor:
        """Compute the camera's world transform for an NYX attached camera.

        Reads the link's current ``pos`` / ``quat``, builds the link-to-world
        matrix, and composes it with the stored ``_offset_T``.
        """
        link_pos = self._attach_link.get_pos()   # (3,) or (B, 3) tensor
        link_quat = self._attach_link.get_quat()  # (4,) or (B, 4) tensor (w, x, y, z)

        device = self.world.device
        link_pos = torch.as_tensor(link_pos, dtype=self.world.dtype, device=device)
        link_quat = torch.as_tensor(link_quat, dtype=self.world.dtype, device=device)

        offset_T = torch.as_tensor(
            self._offset_T, dtype=self.world.dtype, device=device,
        )

        R_link = quaternion_to_matrix(PyTorchComputeBackend, link_quat).to(
            dtype=self.world.dtype, device=device,
        )
        T_link = torch.eye(4, dtype=self.world.dtype, device=device)
        if link_pos.ndim > 1:
            T_link = T_link.unsqueeze(0).expand(link_pos.shape[0], -1, -1).clone()
        T_link[..., :3, :3] = R_link
        T_link[..., :3, 3] = link_pos
        if link_pos.ndim == 1:
            return T_link @ offset_T
        else:
            return T_link @ offset_T.unsqueeze(0)

    def _get_local_extrinsics(self) -> torch.Tensor:
        """Build ``(..., 3, 4)`` link-to-camera extrinsic matrix (OpenCV).

        This is the fixed transform from the attached link's frame to the
        camera frame, expressed as ``[R | t]`` such that
        ``p_cam = R @ p_link + t``.

        For a static camera (no attachment), the "link" frame is the world
        frame, so this equals the world extrinsics.

        Shape is ``(3, 4)`` for a single camera, ``(B, 3, 4)`` for batched.
        """
        T_OPENGL_TO_OPENCV = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=self.world.dtype,
            device=self.world.device,
        )

        if self._world_pose_T is not None:
            # A static camera moved by set_pose: its "link" is the world.
            return self._get_extrinsics()

        offset_T = torch.from_numpy(self._offset_T).to(dtype=self.world.dtype, device=self.world.device)

        # offset_T is camera-to-link in OpenGL convention.
        # Convert to OpenCV and invert to get link-to-camera.
        cam_to_link_cv = offset_T @ T_OPENGL_TO_OPENCV
        link_to_cam_cv = torch.linalg.inv(cam_to_link_cv)
        result = link_to_cam_cv[:3, :]

        B = self.world.batch_size
        if B is not None:
            result = result.unsqueeze(0).expand(B, -1, -1)
        return result

    def _depth_to_position_map(self, depth: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert per-pixel depth to per-pixel 3D world positions.

        Args:
            depth: Depth map with shape ``(H, W)`` or ``(B, H, W)``.

        Returns:
            Position map with shape ``(H, W, 3)`` or ``(B, H, W, 3)``.
        """
        depth_t = torch.as_tensor(depth, dtype=self.world.dtype, device=self.world.device)
        squeeze_batch = depth_t.ndim == 2
        if squeeze_batch:
            depth_t = depth_t.unsqueeze(0)
        elif depth_t.ndim != 3:
            raise ValueError(f"Depth map must have shape (H, W) or (B, H, W), got {tuple(depth_t.shape)}")

        intrinsics = self._get_intrinsics()   # (..., 3, 3)
        extrinsics = self._get_extrinsics()   # (..., 3, 4)

        # depth_image_to_world returns (..., H, W, 4) with last dim = (x, y, z, valid)
        world_coords = depth_image_to_world(depth_t, intrinsics, extrinsics)
        position = world_coords[..., :3]

        if squeeze_batch:
            position = position[0]

        return position

    def depth_to_position_map(self, depth: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Public helper to convert depth map to world position map."""
        return self._depth_to_position_map(depth)

    def update_observation(self) -> None:
        """Render and cache camera observations."""
        device = self.world.device

        if self.option.backend == "nyx":
            self._update_observation_nyx(device)
        else:
            self._update_observation_genesis(device)

    def _update_observation_genesis(self, device) -> None:
        """Render via Genesis Camera / BatchRenderer."""
        # Move camera to attached link's current position before rendering
        if self.option.attach is not None:
            self.sensor.move_to_attach()
        
        rgb, depth, segmentation, normal = self.sensor.render(
            rgb=self.render_rgb,
            depth=self._render_depth_internal,
            segmentation=self.render_seg,
            normal=self.render_normal,
            antialiasing=True,
            force_render=self._force_genesis_render,
        )
        self._force_genesis_render = False

        def _to_tensor(x, **kwargs):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, **kwargs)
            return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, **kwargs)

        obs: Dict[str, PyTorchArrayType] = {}
        if self.render_rgb:
            obs["rgb"] = _to_tensor(rgb)

        if self.render_depth:
            obs["depth"] = _to_tensor(depth, dtype=self.world.dtype)

        if self.render_position:
            obs["position"] = self._depth_to_position_map(depth)

        if self.render_seg:
            obs["seg"] = _to_tensor(segmentation)

        if self.render_normal:
            obs["normal"] = _to_tensor(normal, dtype=self.world.dtype)

        obs["intrinsics_cv"] = self._get_intrinsics()
        obs["extrinsics_cv"] = self._get_extrinsics()
        obs["local_extrinsics_cv"] = self._get_local_extrinsics()

        self._last_observation = obs

    def _update_observation_nyx(self, device) -> None:
        """Render via NYX path tracer (``sensor.read().rgb``).
        
        If depth/seg/normal/position is also requested, those come from the
        secondary genesis camera (``self._genesis_sensor``).
        """
        # NYX handles move_to_attach() internally during read().
        assert _ensure_nyx() and isinstance(
            self.sensor, _NYX_STATE["NyxCameraSensor"]
        ), "Expected sensor to be a NyxCameraSensor"
        data = self.sensor.read()
        
        rgb_tensor = torch.as_tensor(data.rgb, dtype=torch.uint8, device=device)

        # NYX always returns (B, H, W, 3); squeeze for unbatched worlds.
        if not self.world.is_batched:
            rgb_tensor = rgb_tensor.squeeze(0)

        # Clone the RGB tensor so the observation owns its storage.  The NYX
        # renderer reuses a shared image cache (``image_cache``) that is
        # overwritten on every ``render()`` call; without cloning, successive
        # ``_last_observation["rgb"]`` values would all alias the same buffer
        # and resolve to the final frame's data when read later (e.g. by a
        # video recording wrapper that accumulates frames across steps).
        rgb_tensor = rgb_tensor.clone()

        obs: Dict[str, PyTorchArrayType] = {}
        obs["rgb"] = rgb_tensor

        # If we have a secondary genesis sensor, get depth/seg/normal from it
        if self._genesis_sensor is not None:
            # Move genesis camera to attached link's current position
            if self.option.attach is not None:
                self._genesis_sensor.move_to_attach()
            
            _, depth, segmentation, normal = self._genesis_sensor.render(
                rgb=False,
                depth=self._render_depth_internal,
                segmentation=self.render_seg,
                normal=self.render_normal,
                force_render=self._force_genesis_render,
            )
            self._force_genesis_render = False

            def _to_tensor(x, **kwargs):
                if isinstance(x, torch.Tensor):
                    return x.to(device=device, **kwargs)
                return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, **kwargs)

            if self.render_depth:
                obs["depth"] = _to_tensor(depth, dtype=self.world.dtype)

            if self.render_position:
                obs["position"] = self._depth_to_position_map(depth)

            if self.render_seg:
                obs["seg"] = _to_tensor(segmentation)

            if self.render_normal:
                obs["normal"] = _to_tensor(normal, dtype=self.world.dtype)

        obs["intrinsics_cv"] = self._get_intrinsics()
        obs["extrinsics_cv"] = self._get_extrinsics()
        obs["local_extrinsics_cv"] = self._get_local_extrinsics()

        self._last_observation = obs
    
    @property
    def segmentation_sensor(self) -> CameraSensor:
        """Return the Genesis-backed sensor that provides segmentation metadata.

        When using the NYX backend with a secondary Genesis camera for
        depth/seg/normal, this returns the secondary ``_genesis_sensor``.
        Otherwise it returns the primary ``sensor``.

        This is the correct source for ``_visualizer.segmentation_idx_dict``
        and other Genesis-specific segmentation metadata.
        """
        if self._genesis_sensor is not None:
            return self._genesis_sensor
        return self.sensor

    def get_observation(self) -> ObsType:
        """Get the latest camera observation."""
        return self._last_observation

    def render(self) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """Return the latest RGB frame as a ``(H, W, 3)`` uint8 pytorch array.

        Returns ``None`` if no frame has been captured yet or if
        ``render_rgb`` is ``False``.
        """
        if self._last_observation is None or "rgb" not in self._last_observation:
            return None
        rgb = self._last_observation["rgb"]
        return rgb
