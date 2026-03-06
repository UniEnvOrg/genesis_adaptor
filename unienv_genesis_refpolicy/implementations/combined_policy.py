"""CombinedRefPolicy — combine named sub-policies into a Dict action."""

from typing import Dict, Optional, Sequence, Union

import torch

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis_refpolicy.base.policy import BaseRefPolicy
from unienv_genesis.utils.batch_dim_utils import (
    reduce_and,
    reduce_or
)

class CombinedRefPolicy(BaseRefPolicy[Dict[str, torch.Tensor]]):
    """Run named sub-policies in parallel and combine their actions.

    Each sub-policy produces a :class:`torch.Tensor` action.  The
    combined action is a ``Dict[str, Tensor]`` keyed by sub-policy name,
    matching the format expected by
    :class:`~unienv_genesis.nodes.entities.entity_controllers.combined_controller.CombinedController`.

    Args:
        world: Genesis world instance.
        policies: Mapping from name to sub-policy.
        success_keys: Names of sub-policies whose success determines the
            overall success (logical AND).  ``None`` means **all** sub-
            policies must succeed.
    """

    def __init__(
        self,
        world: GenesisWorld,
        policies: Dict[str, BaseRefPolicy[torch.Tensor]],
        success_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(world)
        self._policies = dict(policies)
        self._success_keys = (
            list(success_keys) if success_keys is not None
            else list(self._policies.keys())
        )

    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        for policy in self._policies.values():
            policy.reset(mask=mask, seed=seed)

    def reload(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        for policy in self._policies.values():
            policy.reload(mask=mask, seed=seed)

    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        for policy in self._policies.values():
            policy.step(mask=mask)

    def get_action(self) -> Dict[str, torch.Tensor]:
        return {
            name: policy.get_action()
            for name, policy in self._policies.items()
        }

    def get_success(self) -> Union[torch.Tensor, bool]:
        results = [self._policies[k].get_success() for k in self._success_keys]
        return reduce_and(*results)

    def get_failure(self) -> Union[torch.Tensor, bool]:
        results = [p.get_failure() for p in self._policies.values()]
        return reduce_or(*results)

    def get_timeout(self) -> Union[torch.Tensor, bool]:
        results = [self._policies[k].get_timeout() for k in self._success_keys]
        return reduce_or(*results)
