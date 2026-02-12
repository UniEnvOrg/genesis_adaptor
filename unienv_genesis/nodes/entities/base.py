"""Abstract entity node for Genesis."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union, TypeVar, Generic
import genesis as gs
from genesis.engine.entities.base_entity import Entity as BaseEntity
from genesis.engine import entities as gs_entities
import torch
from unienv_interface.world.node import WorldNode, ContextType, ObsType, ActType
from unienv_interface.space import Space, DictSpace, BoxSpace
from unienv_interface.backends.pytorch import (
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)

from unienv_genesis.world.genesis_world import GenesisWorld

EntityType = TypeVar('EntityType', bound=BaseEntity)
class EntityNode(ABC, WorldNode[
    ContextType,
    ObsType,
    ActType,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType
], Generic[
    EntityType,
    ContextType,
    ObsType,
    ActType
]):
    """Abstract base class for Genesis Entity nodes.
    
    Entity nodes add entities to the scene during reload() and cache
    references in after_reload().
    """
    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False
    
    def __init__(
        self,
        world: GenesisWorld,
        name: str,
        control_timestep : Optional[float] = None,
        update_timestep : Optional[float] = None,
    ):
        self.name = name
        self.world = world
        
        self.entity: Optional[EntityType] = None
        
        self._control_timestep = control_timestep
        self._update_timestep = update_timestep
    
    @property
    def control_timestep(self) -> float:
        return self._control_timestep or self.world.world_timestep

    @control_timestep.setter
    def control_timestep(self, value: float) -> None:
        self._control_timestep = value
    
    @property
    def update_timestep(self) -> float:
        return self._update_timestep or self.world.world_timestep
    
    @update_timestep.setter
    def update_timestep(self, value: float) -> None:
        self._update_timestep = value