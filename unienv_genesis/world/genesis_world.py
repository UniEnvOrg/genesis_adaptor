"""Genesis World backend for UniEnv."""

from typing import Optional, Any, Dict, Union, List, Sequence
import time
import logging
import genesis as gs
import torch
from unienv_interface.backends.pytorch import (
    PyTorchComputeBackend,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)
from unienv_interface.world import World
from unienv_genesis.options.light_option import GenesisWorldLightOption, GenesisWorldEnvMapOption

logger = logging.getLogger(__name__)

class GenesisWorld(World[PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]):
    """World backend using Genesis physics simulator with PyTorch backend.
    
    This wraps a Genesis scene and provides UniEnv World interface.
    Supports parallel environments via Genesis's built-in batching.
    
    Note:
    When batch_size is None, the environment is unbatched (single environment).
    Genesis uses n_envs=0 for this case, and tensors returned by Genesis will
    not have a batch dimension. The adaptor code handles both batched and 
    unbatched cases through utility functions in batch_dim_utils.
    """
    
    backend = PyTorchComputeBackend

    def __init__(
        self,
        batch_size: Optional[int] = None,
        scene_init_kwargs : Optional[Dict[str, Any]] = None,
        scene_build_kwargs : Optional[Dict[str, Any]] = None,
        lights : Optional[Dict[str, GenesisWorldLightOption]] = None,
        env_map : Optional[GenesisWorldEnvMapOption] = None,
        light_fields : Optional[Sequence[Any]] = None,
    ):
        """Initialize Genesis World.
        
        Args:
            batch_size: Number of parallel environments, or None for single (unbatched) environment.
            scene_init_kwargs: Kwargs passed to gs.Scene constructor.
            scene_build_kwargs: Kwargs passed to scene.build() method.
            lights: Dictionary of light options keyed by name.
            env_map: Optional environment map option.
            light_fields: Optional sequence of Nyx ``LightFieldAsset`` objects
                that are shared across all Nyx cameras created within this
                world. These are forwarded to every Nyx camera in addition to
                any per-camera light fields configured on
                ``CameraSensorNodeOptions``. Ignored by non-Nyx backends.
                Stored immutably as a tuple on ``self.light_fields``.
        """
        self._batch_size = batch_size
        # Copy caller-supplied dicts so we never mutate the caller's objects.
        self.scene_init_kwargs = dict(scene_init_kwargs) if scene_init_kwargs is not None else {}
        self.scene_build_kwargs = dict(scene_build_kwargs) if scene_build_kwargs is not None else {}
        if lights is None:
            self.lights = {
                "default": GenesisWorldLightOption(
                    light_type="directional",
                    direction=(0.0, 0.0, -1.0),
                    color=(1.0, 1.0, 1.0),
                    intensity=2.0,
                    cast_shadow=True,
                )
            }
        else:
            self.lights = dict(lights)
        self.env_map = env_map
        # Shared Nyx LightFieldAsset objects applied to every Nyx camera.
        # Stored as an immutable tuple so callers cannot mutate the shared set
        # after construction.
        self.light_fields = tuple(light_fields) if light_fields is not None else ()
        self.scene = self.create_scene()
    
    def create_scene(
        self,
        disable_viewer: bool = False,
    ) -> gs.Scene:
        # Per-call viewer suppression must NOT mutate scene_init_kwargs:
        # RobotNode's throwaway space-discovery build passes disable_viewer=True,
        # and a persistent mutation here would silently disable the viewer on
        # every subsequent (re)built scene as well.
        scene_init_kwargs = self.scene_init_kwargs
        if disable_viewer and "show_viewer" in scene_init_kwargs:
            scene_init_kwargs = {**scene_init_kwargs, "show_viewer": False}
        
        if isinstance(scene_init_kwargs.get("renderer", None), gs.options.renderers.RayTracer):
            renderer_option : gs.options.renderers.RayTracer = scene_init_kwargs["renderer"]
            new_lights = []
            for light in self.lights.values():
                if light.light_type != "point":
                    logger.warning(f"Light {light} of type {light.light_type} is not supported by RayTracer and will be ignored.")
                    continue
                new_lights.append(gs.options.renderers.SphereLight(
                    pos=light.pos,
                    color=light.color,
                    intensity=light.intensity,
                    radius=light.range,
                ))
            renderer_option.lights = new_lights
            if self.env_map is not None and self.env_map.png_path is not None:
                renderer_option.env_surface = gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path=self.env_map.png_path,
                    ),
                )
                renderer_option.env_pos=self.env_map.png_pos
                renderer_option.env_euler=self.env_map.png_euler
                renderer_option.env_radius=self.env_map.png_radius

        scene = gs.Scene(
            **scene_init_kwargs
        )

        return scene

    def build_scene(self, scene : gs.Scene, n_envs : Optional[int] = None) -> None:
        """Build the Genesis scene."""
        if not scene.is_built:
            # Handle lights — the scene's renderer is read from the kwargs the
            # scene was created with (same source as ``create_scene``): Genesis
            # 1.4+ does not expose a ``scene.renderer_options`` attribute; the
            # absent ``renderer=`` kwarg means the default (Rasterizer), which
            # is not a BatchRenderer.
            renderer = self.scene_init_kwargs.get("renderer")
            if isinstance(renderer, gs.options.renderers.BatchRenderer):
                for light in self.lights.values():
                    if not light.light_type in ["directional", "spot"]:
                        logger.warning(f"Light {light} of type {light.light_type} is not supported by BatchRenderer and will be ignored.")
                        continue
                    scene.add_light(
                        # Genesis's BatchRenderer light records require a
                        # concrete pos/dir tensor row (``_make_tensor`` chokes
                        # on None) even for directional lights (whose pos is
                        # ignored by the shading) — default the missing
                        # endpoint to the world origin.
                        pos=(0.0, 0.0, 0.0) if light.pos is None else light.pos,
                        dir=(0.0, 0.0, -1.0) if light.direction is None else light.direction,
                        color=light.color,
                        intensity=light.intensity,
                        directional=(light.light_type == "directional"),
                        castshadow=light.cast_shadow,
                        cutoff=light.spot_outer_cone_angle if light.light_type == "spot" else 90,
                        attenuation=light.attenuation,
                    )
            scene.build(
                n_envs=0 if n_envs is None else n_envs,
                **self.scene_build_kwargs
            )

    @property
    def device(self) -> PyTorchDeviceType:
        """PyTorch device for tensor operations."""
        return gs.device

    @property
    def dtype(self) -> PyTorchDtypeType:
        """PyTorch dtype matching Genesis's initialized floating precision.
        
        Returns Genesis's canonical runtime float dtype (gs.tc_float), which
        reflects the precision set during gs.init(precision="32"|"64").
        
        Raises:
            RuntimeError: If Genesis has not been initialized via gs.init().
        """
        if not getattr(gs, "_initialized", False):
            raise RuntimeError(
                "Genesis is not initialized. Call gs.init() before accessing world.dtype."
            )
        return gs.tc_float

    @property
    def batch_size(self) -> Optional[int]:
        """Number of parallel environments, or None for unbatched."""
        return self._batch_size
    
    @property
    def is_batched(self) -> bool:
        """Whether the environment is batched (has a batch dimension)."""
        from unienv_genesis.utils.batch_dim_utils import is_batched as _is_batched
        return _is_batched(self._batch_size)
    
    @batch_size.setter
    def batch_size(self, value: Optional[int]):
        assert value is None or value > 0, "Batch size must be None or non-negative"
        assert not self.scene.is_built, "Cannot change batch size after scene is built"
        self._batch_size = value

    @property
    def dofs_info_batched(self) -> bool:
        """Whether the scene stores per-env DOF info (gains, limits, ...).

        Genesis only allows per-env values for ``set_dofs_kp`` / ``set_dofs_kv``
        etc. when the scene was created with
        ``RigidOptions(batch_dofs_info=True)``.  Otherwise DOF info is a single
        ``(n_dofs,)`` table shared by all envs.  Controllers use this to decide
        whether to broadcast gains over the batch dimension.
        """
        if not self.is_batched:
            return False
        return bool(self.scene.options.rigid.batch_dofs_info)

    @property
    def world_timestep(self) -> float:
        """Simulation timestep in seconds."""
        return self.scene.sim.dt
    
    @property
    def world_subtimestep(self) -> float:
        """Substep timestep in seconds.
        
        Returns world_timestep divided by number of substeps.
        """
        return self.world_timestep / self.scene.sim.substeps
        
    def step(self) -> float:
        """Step the Genesis simulation.

        Returns:
            float: Simulation timestep (world_timestep)
        """
        self.scene.step()

        return self.world_timestep
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs
    ) -> None:
        """Reset the Genesis simulation.
        
        Args:
            seed: Random seed for reproducibility
            mask: Boolean mask for which environments to reset (batch_size,)
            **kwargs: Additional arguments passed to scene.reset()
        """  
        if seed is not None:
            gs.set_random_seed(seed)

        if mask is not None:
            # Genesis accepts boolean masks for indexing
            # env_indices = mask.nonzero(as_tuple=True)[0]
            env_indices = mask
            self.scene.reset(envs_idx=env_indices)
        else:
            self.scene.reset()

    def after_reset(self, *, seed = None, mask = None, **kwargs):
        if not self.scene.is_built:
            self.reload(seed=seed, mask=mask, **kwargs)
            return
        if seed is not None:
            gs.set_random_seed(seed)
        # Refresh sensor caches (without advancing the physics) so sensor
        # nodes return meaningful observations on the very first
        # `env.reset()`/`env.step()` read.  See `_refresh_sensor_caches` for
        # details.
        #
        # Masked resets skip this: the refresh re-reads sensor state for the
        # *entire* batch — Genesis has no per-env cache refresh — which would
        # needlessly overwrite the still-valid cached observations of the
        # environments that were not reset.
        if mask is None:
            self._refresh_sensor_caches()

    def reload(
        self,
        *,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        disable_viewer: bool = False,
        **kwargs
    ) -> None:
        """Prepare scene for reloading. Clears old entities but does NOT compile.
        
        This is called BEFORE nodes add entities. The scene is compiled
        in after_reload() after all nodes have added their entities.
        """
        assert mask is None, "Masking not supported in reload()"
        if seed is not None:
            gs.set_random_seed(seed)
        
        # Clear scene if it was previously built
        if self.scene is not None:
            self.scene.destroy()
        self.scene = self.create_scene(disable_viewer=disable_viewer)
    
    def after_reload(
        self,
        *,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        temporary : bool = False,
        **kwargs
    ) -> None:
        """Compile the scene AFTER all nodes have added their entities."""
        assert mask is None, "Masking not supported in after_reload()"
        if seed is not None:
            gs.set_random_seed(seed)
        
        if temporary and (
            self.scene.visualizer._batch_renderer is not None and
            len(self.scene.visualizer.cameras) == 0
        ):
            self.scene.add_camera()

        self.build_scene(self.scene, n_envs=self.batch_size)
        # Refresh sensor caches (without advancing the physics) so sensor
        # nodes return meaningful observations on the very first
        # `env.reset()`/`env.step()` read.  See `_refresh_sensor_caches` for
        # details.
        self._refresh_sensor_caches()

    def _refresh_sensor_caches(self) -> None:
        """Refresh Genesis sensor caches without advancing the simulation.

        Genesis only fills its sensor cache (IMU accelerations, contact
        forces, …) during ``scene.step()`` — ``sim.step()`` runs the physics
        substeps and *then* calls ``sensor_manager.step()`` to read the solver
        state into the cache.  Right after ``scene.build()`` or
        ``scene.reset()`` the cache is empty and the collider has not yet
        detected any contacts, so ``sensor.read()`` returns zeros.

        This performs a *no-advance* refresh by calling
        ``sensor_manager.step()`` directly: it re-reads the current solver
        state without stepping the physics, which populates sensors that read
        retained solver quantities (e.g. the IMU accelerometer, which reflects
        gravity on subsequent resets).  Contact-based sensors still read zero
        until the first real ``env.step()`` because the collider's contact
        list is cleared on reset.  This method never calls ``scene.step()`` —
        the world deliberately does not advance the simulation outside of
        ``env.step()``.

        It is only invoked on full (unmasked) resets and reloads: the refresh
        re-reads sensor state for the *entire* batch (Genesis has no per-env
        cache refresh), so running it on a masked reset would needlessly
        overwrite the still-valid cached observations of the environments
        that were not reset.

        ``BaseCameraSensor._update_shared_cache`` is a no-op (cameras render
        lazily on ``read()``), so this never triggers spurious rendering.
        """
        if self.scene is None or not getattr(self.scene, "is_built", False):
            return
        sm = getattr(getattr(self.scene, "_sim", None), "_sensor_manager", None)
        if sm is not None:
            sm.step()
        
    def close(self) -> None:
        """Close the Genesis simulation and cleanup."""
        # Genesis handles cleanup automatically
        self.scene.destroy()
        self.scene = None
