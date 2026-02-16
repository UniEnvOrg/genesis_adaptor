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
    
    Note:
    This implementation does not support `n_envs = 0` because it adds significant complexity to handle when the return values of various functions (like forward kinematics) can be either batched or unbatched. By enforcing `n_envs > 0`, we can ensure that all return values are consistently batched, simplifying the implementation and reducing potential bugs.
    """
    GS_INIT_KWARGS_DEFAULT = {
        "backend": gs.gs_backend.gpu,
        "performance_mode": True
    }
    
    def __init__(
        self,
        batch_size: int = 1,
        scene_init_kwargs : Dict[str, Any] = {},
        scene_build_kwargs : Dict[str, Any] = {},
    ):
        """Initialize Genesis World.
        
        Args:
            scene: Pre-configured Genesis scene. Use GenesisWorld.create() 
                   to create a scene with default simulation options.
            batch_size: Number of parallel environments
        """
        if not gs._initialized:
            gs.init(**self.GS_INIT_KWARGS_DEFAULT)
        
        self.batch_size = batch_size
        self.scene_init_kwargs = scene_init_kwargs
        self.scene_build_kwargs = scene_build_kwargs
        self.scene = self.create_scene()
    
    def create_scene(
        self,
    ) -> gs.Scene:
        scene = gs.Scene(
            **self.scene_init_kwargs
        )
        return scene

    def build_scene(self) -> None:
        """Build the Genesis scene."""
        if not self.scene.is_built:
            self.scene.build(
                n_envs=self.batch_size,
                **self.scene_build_kwargs
            )

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
            env_indices = torch.nonzero(mask)[0]
            self.scene.reset(envs_idx=env_indices)
        else:
            self.scene.reset()

    def after_reset(self, *, seed = None, mask = None, **kwargs):
        if not self.scene.is_built:
            self.reload(seed=seed, mask=mask, **kwargs)
            return
        if seed is not None:
            gs.set_random_seed(seed)

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
        if self.scene.is_built:
            self.scene.destroy()
        
        self.scene = self.create_scene()
    
    def after_reload(
        self,
        *,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs
    ) -> None:
        """Compile the scene AFTER all nodes have added their entities."""
        assert mask is None, "Masking not supported in after_reload()"
        if seed is not None:
            gs.set_random_seed(seed)
        
        self.build_scene()
        
    def close(self) -> None:
        """Close the Genesis simulation and cleanup."""
        # Genesis handles cleanup automatically
        self.scene.destroy()
