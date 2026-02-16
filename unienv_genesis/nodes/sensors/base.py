"""Abstract sensor node for Genesis."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, Dict, Any, List, Tuple, TypeVar, Union
import genesis as gs
from genesis.engine.sensors.base_sensor import Sensor as BaseSensor
import torch
from unienv_interface.world.node import WorldNode, ContextType, ObsType
from unienv_interface.space import Space, DictSpace, BoxSpace
from unienv_interface.backends.pytorch import (
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)

from unienv_genesis.world.genesis_world import GenesisWorld

SensorType = TypeVar('SensorType')
class SensorNode(ABC, WorldNode[
    ContextType,
    ObsType,
    None,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType
],Generic[
    SensorType,
    ContextType,
    ObsType
]):
    """Abstract base class for Genesis Sensor nodes.
    
    Sensor nodes read data from the simulation during update_observation()
    and return sensor readings via get_observation().
    """
    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False
    
    def __init__(
        self,
        world: GenesisWorld,
        name: str,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
    ):
        self.name = name
        self.world = world
        
        self.sensor : Optional[SensorType] = None

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
    