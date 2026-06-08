from typing import Any, Dict, Optional, Tuple, Union
import math
import torch

from unienv_interface.backends.pytorch import *
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world.node import WorldNode
from unienv_genesis.world import GenesisWorld
from unienv_genesis_collection.objects.base_object import ManipulatableObject, bbox_size
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_genesis_collection.scenes.cluttered_table_manager import ClutteredTableManager

_PARK_POS = (0.0, 0.0, -1000.0)
_PARK_QUAT = (1.0, 0.0, 0.0, 0.0)

class ObjectRelocateTask(WorldNode):
    DEFAULT_RESET_PRIORITY = 25
    DEFAULT_POSTSTEP_PRIORITY = 0

    """
    Task: Move a target object from a random positions on the table to a goal location.

    Provides:
    - **observation**: ``{"target_position": (3,), "goal_position": (3,)}``
    - **reward**: ``-distance(target, goal)`` (dense)
    - **termination**: ``True`` when ``distance < success_threshold``

    The task node's reset lifecycle (`reload_priorities` and `reset_priorities`) runs at one priority higher than ``ClutteredTableManager``, which runs before the cluttered objects are placed.
    
    After placing the target object it calls :meth:`ClutteredTableManager.set_next_reset_avoid_spaces` so clutter objects don't overlap with the target target.
    """

    has_reward = True
    has_termination_signal = True
    has_truncation_signal = False

    def __init__(
        self,
        world: GenesisWorld,
        table_builder : TableBuilder,
        clutter_manager: Optional[ClutteredTableManager],
        target_object: ManipulatableObject,
        name: str = "task",
        placement_margin: Union[float, Tuple[float, float]] = 0.05,
        success_threshold: float = 0.03,
        reload_priority: Optional[int] = None,
        reset_priority: Optional[int] = None,
        after_reload_priority: Optional[int] = None,
        post_environment_step_priority: int = DEFAULT_POSTSTEP_PRIORITY,
    ):
        self.world = world
        self._name = name
        self.table_builder = table_builder
        self.clutter_manager = clutter_manager
        self.target_object = target_object

        # Register priorities relative to clutter manager if available, otherwise use defaults.
        if self.clutter_manager is not None:
            self.reload_priorities = {reload_priority or (next(iter(clutter_manager.reload_priorities)) + 1)}
            self.after_reload_priorities = {after_reload_priority or (next(iter(clutter_manager.after_reload_priorities)) + 1)}
            self.reset_priorities = {reset_priority or (next(iter(clutter_manager.reset_priorities)) + 1)}
        else:
            self.reload_priorities = {reload_priority or self.DEFAULT_RESET_PRIORITY}
            self.after_reload_priorities = {after_reload_priority or self.DEFAULT_RESET_PRIORITY}
            self.reset_priorities = {reset_priority or self.DEFAULT_RESET_PRIORITY}
        self.post_environment_step_priorities = {post_environment_step_priority}

        # Task parameters.
        self.placement_margin = placement_margin
        self.success_threshold = success_threshold

        # Random state.
        self.rng: torch.Generator = torch.Generator(device=world.device)
        self.rng.seed()

        # Task state (persistent across episodes).
        self.target_object_entity = None

        # Task state (per-episode)
        self._goal_pos: Optional[torch.Tensor] = None # (B, 3)
        self._reward : Union[float, torch.Tensor] = 0.0 # scalar or (B,)
        self._terminated : Union[bool, torch.Tensor] = False # scalar or (B,)

        # Observation space — known at construction time.
        self.observation_space = DictSpace(
            PyTorchComputeBackend,
            {
                "target_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(3,),
                    dtype=torch.float32,
                ),
                "goal_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(3,),
                    dtype=torch.float32,
                ),
            },
        )

    @property
    def name(self) -> str:
        return self._name

    def _reseed(self, seed : int) -> None:
        self.rng.manual_seed(seed)

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority not in self.reload_priorities:
            return
        if seed is not None:
            self._reseed(seed)

        batch_fixed_verts = self.world.batch_size is not None
        self.target_object_entity = self.target_object.load_into_scene(
            world=self.world,
            pos=_PARK_POS,
            quat=_PARK_QUAT,
            batch_fixed_verts=batch_fixed_verts,
        )

        # Cache object dimensions for placement logic.
        self._target_object_bbox_local = self.target_object.get_local_bounding_box(self.target_object_entity).to(device=self.world.device) # BoxSpace with low and high of shape (3, )
        self._target_object_radius = self.target_object.get_footprint_radius(self.target_object_entity) # scalar
        self._target_object_extent_local = bbox_size(self._target_object_bbox_local) # (3,)

    def after_reload(self, *, priority=0, mask=None):
        if priority not in self.after_reload_priorities:
            return
        # Initialise placement for the first time.
        self.reset(priority=next(iter(self.reset_priorities)), seed=None, mask=None)

    def sample_target_placement(self, n_samples: int) -> torch.Tensor:
        """
        Sample random target placements on the table (world frame).
        Returns:
            Tensor of shape ([B], n_samples, 3) with XYZ world positions.
        """
        # --- Compute valid half extents for placement in table's local XY plane ---
        table_half_xy = (self.table_builder.size[..., :2] / 2.0) # ([B], 2)
        if isinstance(self.placement_margin, (int, float)):
            table_half_xy -= self.placement_margin
        else:
            placement_margin = torch.tensor(self.placement_margin, device=table_half_xy.device)
            if table_half_xy.ndim > 1:
                placement_margin = placement_margin.unsqueeze(0)  # (1, 2)
            table_half_xy -= placement_margin
        
        placement_half_xy = table_half_xy - self._target_object_radius # ([B], 2)

        # --- Compute Z from bbox ---
        placement_z_scalar = self.table_builder.pos[2] - self._target_object_bbox_local.low[2] # scalar

        target_xy_shape = (n_samples, 2) if self.world.batch_size is None else (self.world.batch_size, n_samples, 2)
        xy = torch.rand(target_xy_shape, generator=self.rng, device=self.world.device) * 2 - 1 # uniform in [-1,1], shape ([B], n_samples, 2)
        target_xy = xy * placement_half_xy[..., None, :] # scale to valid range, shape ([B], n_samples, 2)
        target_world_xy = self.table_builder.xy_to_world(target_xy) # ([B], n_samples, 2)
        target_world_z = torch.full(target_world_xy.shape[:-1] + (1,), placement_z_scalar, device=target_world_xy.device) # ([B], n_samples, 1)
        target_world_pos = torch.cat([target_world_xy, target_world_z], dim=-1) # ([B], n_samples, 3)
        return target_world_pos

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority not in self.reset_priorities:
            return
        
        assert self.target_object_entity is not None, "Target object must be loaded before reset"

        if seed is not None:
            self._reseed(seed)
        
        # Valid placement half-extents (table half minus margin minus object half-extent on each axis).
        placement_chunk = self.sample_target_placement(n_samples=2)
        initial_placement_xyz = placement_chunk[..., 0, :]  # ([n_envs], 3)
        self._goal_pos = placement_chunk[..., 1, :]  # ([n_envs], 3)

        self.target_object.set_pos_3d(
            self.target_object_entity,
            initial_placement_xyz[mask] if mask is not None else initial_placement_xyz,
            random_quaternion=False,
            rng=self.rng,
            # envs_idx=mask.nonzero(as_tuple=True)[0] if mask is not None else None,
            envs_idx=mask if mask is not None else None,
        )
        avoid_xy, avoid_radius = self.target_object.get_footprint(self.target_object_entity) # ([n_envs], 2), ([n_envs], )
        self.clutter_manager.set_next_reset_avoid_spaces(
            avoid_xy[..., None, :], avoid_radius[..., None]
        ) # Only one zone to avoid

        if mask is None:
            self._reward = 0.0 if self.world.batch_size is None else torch.zeros(self.world.batch_size, device=self.world.device)
            self._terminated = False if self.world.batch_size is None else torch.zeros(self.world.batch_size, dtype=torch.bool, device=self.world.device)
        else:
            assert self.world.batch_size is not None, "Masking only supported in batched environments"
            self._reward.masked_fill_(mask, 0.0)
            self._terminated.masked_fill_(mask, False)

    def post_environment_step(self, dt, *, priority=0):
        if priority not in self.post_environment_step_priorities:
            return
        
        cur_pos = self.target_object_entity.get_pos() # ([n_envs],3)
        
        dist = torch.linalg.vector_norm(
            cur_pos - self._goal_pos, dim=-1,
        )  # ([n_envs],)
        if self.world.batch_size is None:
            dist = float(dist.item())

        self._reward = -dist
        self._terminated = dist < self.success_threshold

    def get_observation(self):
        return {
            "target_position": self.target_object_entity.get_pos(),
            "goal_position": self._goal_pos,
        }

    def get_reward(self):
        return self._reward

    def get_termination(self):
        return self._terminated

    def get_info(self) -> Dict[str, Any]:
        """Return auxiliary information about the current episode.

        Exposes the target object's technical name and its language
        annotations (if any) so that downstream consumers (logging,
        language-conditioned policies, etc.) can refer to the object in
        natural language.

        Returns:
            A dictionary with keys:

            - ``"target_object_name"`` — technical identifier
              (:attr:`ManipulatableObject.name`).
            - ``"target_object_annotation"`` — the first annotation in
              :attr:`ManipulatableObject.language_annotations`, or
              ``None`` if the tuple is empty.
            - ``"target_object_annotations"`` — the full
              :attr:`ManipulatableObject.language_annotations` tuple
              (may be empty).
        """
        annotations = self.target_object.language_annotations
        return {
            "target_object_name": self.target_object.name,
            "target_object_annotation": annotations[0] if annotations else None,
            "target_object_annotations": annotations,
        }
