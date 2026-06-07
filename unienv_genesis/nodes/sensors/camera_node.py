"""Camera sensor node for Genesis."""

from typing import Optional, Dict, Any, List, Tuple, Union, Literal, Sequence, TYPE_CHECKING
import torch
import numpy as np
from dataclasses import dataclass, field
import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from genesis.options.options import Options
from genesis.vis.camera import Camera
from genesis.vis.batch_renderer import BatchRenderer

try:
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx_plugin.nyx_camera_sensor import NyxCameraSensor
    NYX_AVAILABLE = True
except Exception:
    NYX_AVAILABLE = False

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
            if not NYX_AVAILABLE:
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

        # Expose render() when RGB is available.
        self.render_mode = "rgb_array" if render_rgb else None

        # Cached observation
        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        
        # Build observation space
        self.observation_space = self._get_observation_space()

    def _init_sensor(self) -> None:
        """Initialize the camera sensor according to the specified backend."""
        if self.option.backend == "nyx":
            self.sensor = self._init_sensor_nyx()
            # If depth/seg/normal/position is requested, also create a genesis camera
            if self._render_depth_internal or self.render_seg or self.render_normal:
                self._genesis_sensor = self._init_sensor_genesis()
        else:
            self.sensor = self._init_sensor_genesis()

    def _init_sensor_genesis(self) -> CameraSensor:
        """Initialize via Genesis ``scene.add_camera()``."""

        if self.option.attach is not None:
            self._attach_entity, self._attach_link = self.option.attach.get_attachment(self.world)
        
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
            attach_link = self._attach_link if self._attach_link is not None else self._attach_entity.base_link
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

        # Resolve attachment to entity_idx / link_idx_local / offset_T
        entity_idx = None
        link_idx_local = None
        offset_T = None
        if self.option.attach is not None:
            self._attach_entity, self._attach_link = self.option.attach.get_attachment(self.world)
            entity_idx = self._attach_entity.idx
            if self._attach_link is not None:
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

        camera = self.world.scene.add_sensor(NyxCameraOptions(**nyx_kwargs))
        return camera

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
        """Reset camera state if needed."""
        if mask is not None:
            env_indices = mask.nonzero(as_tuple=True)[0]
        else:
            env_indices = None
        # BatchRenderer exposes reset(); the plain Camera class does not.
        if hasattr(self.sensor, "reset"):
            self.sensor.reset(envs_idx=env_indices)
        # Also reset the secondary genesis sensor if it exists
        if self._genesis_sensor is not None and hasattr(self._genesis_sensor, "reset"):
            self._genesis_sensor.reset(envs_idx=env_indices)
    
    def after_reset(
        self,
        *,
        priority: int = 0,
        mask: Optional[PyTorchArrayType] = None,
    ) -> None:
        """Update observation after reset."""
        if priority in self.after_reset_priorities:
            self.update_observation()

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
                dtype=torch.float32,
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
                dtype=torch.float32,
                device=self.world.device,
            )

        if self.render_position:
            spaces["position"] = BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=get_batch_aware_shape(B, (H, W, 3)),
                dtype=torch.float32,
                device=self.world.device,
            )

        spaces["intrinsics_cv"] = BoxSpace(
            PyTorchComputeBackend,
            low=-float("inf"),
            high=float("inf"),
            shape=get_batch_aware_shape(B, (3, 3)),
            dtype=torch.float32,
            device=self.world.device,
        )
        spaces["extrinsics_cv"] = BoxSpace(
            PyTorchComputeBackend,
            low=-float("inf"),
            high=float("inf"),
            shape=get_batch_aware_shape(B, (3, 4)),
            dtype=torch.float32,
            device=self.world.device,
        )
        spaces["local_extrinsics_cv"] = BoxSpace(
            PyTorchComputeBackend,
            low=-float("inf"),
            high=float("inf"),
            shape=get_batch_aware_shape(B, (3, 4)),
            dtype=torch.float32,
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
            dtype=torch.float32,
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
            dtype=torch.float32,
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
        ``_offset_T``.  If a secondary genesis sensor exists (NYX backend with
        depth/seg/normal), prefer its extrinsics for consistency.
        """
        T_OPENGL_TO_OPENCV = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.world.device,
        )

        # Prefer genesis sensor if available (for consistency with depth/seg/normal)
        sensor = self._genesis_sensor if self._genesis_sensor is not None else self.sensor

        if hasattr(sensor, "get_transform"):
            camera_transform = torch.as_tensor(
                sensor.get_transform(),
                dtype=torch.float32,
                device=self.world.device,
            )
        elif self.option.backend == "nyx" and self._attach_link is not None:
            # NYX attached camera: compose link world transform with offset_T
            camera_transform = self._get_nyx_attached_transform()
        elif self.option.backend == "nyx":
            # NYX static camera: use stored offset_T as world pose
            camera_transform = torch.as_tensor(
                self._offset_T, dtype=torch.float32, device=self.world.device,
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
        link_pos = torch.as_tensor(link_pos, dtype=torch.float32, device=device)
        link_quat = torch.as_tensor(link_quat, dtype=torch.float32, device=device)

        offset_T = torch.as_tensor(
            self._offset_T, dtype=torch.float32, device=device,
        )

        R_link = quaternion_to_matrix(PyTorchComputeBackend, link_quat).to(
            dtype=torch.float32, device=device,
        )
        T_link = torch.eye(4, dtype=torch.float32, device=device)
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
            dtype=torch.float32,
            device=self.world.device,
        )

        offset_T = torch.from_numpy(self._offset_T).to(device=self.world.device)

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
        depth_t = torch.as_tensor(depth, dtype=torch.float32, device=self.world.device)
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
            antialiasing=True
        )

        def _to_tensor(x, **kwargs):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, **kwargs)
            return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, **kwargs)

        obs: Dict[str, PyTorchArrayType] = {}
        if self.render_rgb:
            obs["rgb"] = _to_tensor(rgb)

        if self.render_depth:
            obs["depth"] = _to_tensor(depth, dtype=torch.float32)

        if self.render_position:
            obs["position"] = self._depth_to_position_map(depth)

        if self.render_seg:
            obs["seg"] = _to_tensor(segmentation)

        if self.render_normal:
            obs["normal"] = _to_tensor(normal, dtype=torch.float32)

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
        assert isinstance(self.sensor, NyxCameraSensor), "Expected sensor to be a NyxCameraSensor"
        data = self.sensor.read()
        
        rgb_tensor = torch.as_tensor(data.rgb, dtype=torch.uint8, device=device)

        # NYX always returns (B, H, W, 3); squeeze for unbatched worlds.
        if not self.world.is_batched:
            rgb_tensor = rgb_tensor.squeeze(0)

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
            )

            def _to_tensor(x, **kwargs):
                if isinstance(x, torch.Tensor):
                    return x.to(device=device, **kwargs)
                return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, **kwargs)

            if self.render_depth:
                obs["depth"] = _to_tensor(depth, dtype=torch.float32)

            if self.render_position:
                obs["position"] = self._depth_to_position_map(depth)

            if self.render_seg:
                obs["seg"] = _to_tensor(segmentation)

            if self.render_normal:
                obs["normal"] = _to_tensor(normal, dtype=torch.float32)

        obs["intrinsics_cv"] = self._get_intrinsics()
        obs["extrinsics_cv"] = self._get_extrinsics()
        obs["local_extrinsics_cv"] = self._get_local_extrinsics()

        self._last_observation = obs
    
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
