from typing import Optional, Union, Generic, TypeVar

import torch
import abc

from unienv_genesis.world.genesis_world import GenesisWorld

ActT = TypeVar('ActT')

class BaseRefPolicy(abc.ABC, Generic[ActT]):
    def __init__(
        self,
        world : GenesisWorld,
    ):
        self.world = world

    # ========== Lifecycle methods ==========

    def reset(
        self,
        *,
        mask : Optional[torch.Tensor] = None,
        seed : Optional[int] = None,
    ) -> None:
        pass

    def reload(
        self,
        *,
        mask : Optional[torch.Tensor] = None,
        seed : Optional[int] = None,
    ) -> None:
        self.reset(mask=mask, seed=seed)
    
    @abc.abstractmethod
    def step(
        self,
        *,
        mask : Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError
    
    # ========== Methods for querying policy state ==========
    @abc.abstractmethod
    def get_action(self) -> ActT:
        raise NotImplementedError

    @abc.abstractmethod
    def get_success(self) -> Union[torch.Tensor, bool]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_failure(self) -> Union[torch.Tensor, bool]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_timeout(self) -> Union[torch.Tensor, bool]:
        raise NotImplementedError