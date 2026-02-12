"""Genesis World backend for UniEnv."""

from typing import Optional, Any, Dict, Union
import time
import genesis as gs
import genesis.engine.entities as gs_entities
import torch
from unienv_interface.backends.pytorch import (
    PyTorchComputeBackend,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)
from unienv_interface.world import World

class GenesisWorld(World[PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]):
    """World backend using Genesis physics simulator with PyTorch backend.
    
    This wraps a Genesis scene and provides UniEnv World interface.
    Supports parallel environments via Genesis's built-in batching.
    
    Example:
        # Using explicit scene creation
        scene = gs.Scene(...)
        world = GenesisWorld(scene=scene, batch_size=4)
        
        # Or using the create() factory method with all Genesis options
        world = GenesisWorld.create(
            batch_size=4,
            sim_options=gs.options.SimOptions(dt=0.02),
            show_viewer=True,
        )
    """
    GS_INIT_KWARGS_DEFAULT = {
        "backend": gs.gs_backend.gpu,
        "performance_mode": True
    }
    
    def __init__(
        self,
        scene: gs.Scene,
        batch_size: int = 1,
    ):
        """Initialize Genesis World.
        
        Args:
            scene: Pre-configured Genesis scene. Use GenesisWorld.create() 
                   to create a scene with default simulation options.
            batch_size: Number of parallel environments
        """
        self.scene = scene
        self._batch_size = batch_size
        
        # Track if scene is built
        self._scene_built = False
    
    @property
    def backend(self):
        """Compute backend for tensor operations."""
        return self._backend

    @property
    def device(self) -> PyTorchDeviceType:
        """PyTorch device for tensor operations."""
        return gs.device

    @property
    def batch_size(self) -> int:
        """Number of parallel environments."""
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value: int):
        assert value > 0, "Batch size must be positive"
        assert not self._scene_built, "Cannot change batch size after scene is built"
        self._batch_size = value

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

    @staticmethod
    def create(
        batch_size: int = 1,
        # gs.Scene parameters (exact match to gs.Scene.__init__)
        sim_options: Optional[gs.options.SimOptions] = None,
        coupler_options: Optional[gs.options.BaseCouplerOptions] = None,
        tool_options: Optional[gs.options.ToolOptions] = None,
        rigid_options: Optional[gs.options.RigidOptions] = None,
        mpm_options: Optional[gs.options.MPMOptions] = None,
        sph_options: Optional[gs.options.SPHOptions] = None,
        fem_options: Optional[gs.options.FEMOptions] = None,
        sf_options: Optional[gs.options.SFOptions] = None,
        pbd_options: Optional[gs.options.PBDOptions] = None,
        vis_options: Optional[gs.options.VisOptions] = None,
        viewer_options: Optional[gs.options.ViewerOptions] = None,
        profiling_options: Optional[gs.options.ProfilingOptions] = None,
        renderer: Optional[gs.renderers.RendererOptions] = None,
        show_viewer: bool = False,
        show_FPS: Optional[bool] = None,  # deprecated, use profiling_options.show_FPS instead
    ) -> "GenesisWorld":
        """Factory method to create a GenesisWorld with all Genesis Scene options.
        
        This is a direct wrapper around gs.Scene.__init__. See Genesis documentation
        for details on each option type.
        
        Args:
            batch_size: Number of parallel environments
            sim_options: Simulation options (dt, substeps, gravity, etc.)
            coupler_options: Coupler options
            tool_options: Tool options
            rigid_options: Rigid body physics options
            mpm_options: Material Point Method options
            sph_options: Smoothed Particle Hydrodynamics options
            fem_options: Finite Element Method options
            sf_options: Stable Fluids options
            pbd_options: Position Based Dynamics options
            vis_options: Visualization options
            viewer_options: Interactive viewer options
            profiling_options: Profiling options
            renderer: Custom renderer
            show_viewer: Show interactive viewer window
            show_FPS: Deprecated, use profiling_options.show_FPS instead
            
        Returns:
            GenesisWorld instance with configured scene
        """
        scene = gs.Scene(
            sim_options=sim_options,
            coupler_options=coupler_options,
            tool_options=tool_options,
            rigid_options=rigid_options,
            mpm_options=mpm_options,
            sph_options=sph_options,
            fem_options=fem_options,
            sf_options=sf_options,
            pbd_options=pbd_options,
            vis_options=vis_options,
            viewer_options=viewer_options,
            profiling_options=profiling_options,
            renderer=renderer,
            show_viewer=show_viewer,
            show_FPS=show_FPS,
        )
        
        return GenesisWorld(scene=scene, batch_size=batch_size)
    
    def build(self) -> None:
        if not gs._initialized:
            gs.init(**self.GS_INIT_KWARGS_DEFAULT)
        
        """Build the Genesis scene. Must be called before simulation."""
        if not self._scene_built:
            self.scene.build(n_envs=self.batch_size)
            self._scene_built = True
        
    def step(self) -> float:
        """Step the Genesis simulation.
        
        Returns:
            float: Simulation timestep (world_timestep)
        """
        if not self._scene_built:
            self.build()
            
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
        if not self._scene_built:
            self.reload(seed=seed, mask=mask, **kwargs)
            
        if seed is not None:
            gs.set_random_seed(seed)
        
        if mask is not None:
            env_indices = torch.nonzero(mask)[0]
            self.scene.reset(envs_idx=env_indices)
        else:
            self.scene.reset()
        
    def reload(
        self,
        *,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
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
        if self._scene_built:
            self.scene.destroy()
            self._scene_built = False
    
    def after_reload(
        self,
        *,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs
    ) -> None:
        """Compile the scene AFTER all nodes have added their entities."""
        assert mask is None, "Masking not supported in after_reload()"
        if not self._scene_built:
            self.build()
        
        if seed is not None:
            gs.set_random_seed(seed)
        
    def close(self) -> None:
        """Close the Genesis simulation and cleanup."""
        # Genesis handles cleanup automatically
        if self._scene_built:
            self.scene.destroy()
            self._scene_built = False

