"""Genesis World backend for UniEnv."""

from typing import Optional, Any, Dict, Union
import time
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
    GS_INIT_KWARGS_DEFAULT = {
        "backend": gs.gpu,
        "performance_mode": True
    }
    
    backend = PyTorchComputeBackend

    def __init__(
        self,
        batch_size: Optional[int] = None,
        scene_init_kwargs : Dict[str, Any] = {},
        scene_build_kwargs : Dict[str, Any] = {},
    ):
        """Initialize Genesis World.
        
        Args:
            batch_size: Number of parallel environments, or None for single (unbatched) environment.
            scene_init_kwargs: Kwargs passed to gs.Scene constructor.
            scene_build_kwargs: Kwargs passed to scene.build() method.
        """
        if not gs._initialized:
            gs.init(**self.GS_INIT_KWARGS_DEFAULT)
        
        self._batch_size = batch_size
        self.scene_init_kwargs = scene_init_kwargs
        self.scene_build_kwargs = scene_build_kwargs
        self.scene = self.create_scene()
    
    def create_scene(
        self,
        disable_viewer: bool = False,
    ) -> gs.Scene:
        if disable_viewer and "show_viewer" in self.scene_init_kwargs:
            self.scene_init_kwargs["show_viewer"] = False
        
        scene = gs.Scene(
            **self.scene_init_kwargs
        )
        return scene

    def build_scene(self, scene : gs.Scene, n_envs : Optional[int] = None) -> None:
        """Build the Genesis scene."""
        if not scene.is_built:
            scene.build(
                n_envs=0 if n_envs is None else n_envs,
                **self.scene_build_kwargs
            )

    @property
    def device(self) -> PyTorchDeviceType:
        """PyTorch device for tensor operations."""
        return gs.device

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
        if self.scene.is_built:
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
        
    def close(self) -> None:
        """Close the Genesis simulation and cleanup."""
        # Genesis handles cleanup automatically
        self.scene.destroy()
