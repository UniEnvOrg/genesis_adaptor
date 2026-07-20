"""CombinedRefPolicy — combine named sub-policies into a Dict action."""

from typing import Any, Dict, Optional, Sequence, Union

import torch

from unienv_interface.space import DictSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend
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

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        for policy in self._policies.values():
            policy.reset(mask=mask, seed=seed)

    @torch.inference_mode()
    def reload(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        for policy in self._policies.values():
            policy.reload(mask=mask, seed=seed)

    @torch.inference_mode()
    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        dt: float,
    ) -> None:
        """Step every sub-policy, forwarding the explicit control-step dt."""
        for policy in self._policies.values():
            policy.step(mask=mask, dt=dt)

    @torch.inference_mode()
    def get_action(self) -> Dict[str, torch.Tensor]:
        return {
            name: policy.get_action()
            for name, policy in self._policies.items()
        }

    @torch.inference_mode()
    def convert_action_dict(
        self, action: Dict[str, torch.Tensor], dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a raw action dict by delegating to each sub-policy.

        The result is keyed by sub-policy name, with each value being the
        controller-native action dict produced by that sub-policy's
        :meth:`convert_action_dict`.
        """
        return {
            name: policy.convert_action_dict(action[name], dt=dt)
            for name, policy in self._policies.items()
        }

    def get_action_dict_space(self) -> DictSpace:
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        The schema is ``{name: policy.get_action_dict_space()}`` for each
        sub-policy, preserving the controller-native nested structure.
        """
        return DictSpace(
            PyTorchComputeBackend,
            {
                name: policy.get_action_dict_space()
                for name, policy in self._policies.items()
            },
            device=self.world.device,
        )

    @torch.inference_mode()
    def get_success(self) -> Union[torch.Tensor, bool]:
        results = [self._policies[k].get_success() for k in self._success_keys]
        return reduce_and(*results)

    @torch.inference_mode()
    def get_failure(self) -> Union[torch.Tensor, bool]:
        results = [p.get_failure() for p in self._policies.values()]
        return reduce_or(*results)

    @torch.inference_mode()
    def get_timeout(self) -> Union[torch.Tensor, bool]:
        results = [self._policies[k].get_timeout() for k in self._success_keys]
        return reduce_or(*results)
