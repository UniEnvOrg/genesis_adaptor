"""RepeatActionRefPolicy — output a fixed action for N steps."""

from typing import Any, Dict, Optional, Union

import torch

from unienv_interface.space import DictSpace, BoxSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers.base import EntityController
from unienv_genesis.utils.batch_dim_utils import (
    is_batched,
    get_batch_aware_shape,
    broadcast_to_batch,
    try_scalar
)
from unienv_genesis_refpolicy.base.policy import BaseRefPolicy


class RepeatActionRefPolicy(BaseRefPolicy[torch.Tensor]):
    """Output a fixed action tensor every step.

    Useful for holding a gripper open/closed or any constant-action phase.

    Args:
        world: Genesis world instance.
        action: The fixed action tensor to return on every
            :meth:`get_action` call (e.g. ``torch.tensor([-1.0])`` for
            gripper open).  Automatically broadcast to
            ``(batch_size, ...)`` for batched worlds.
        steps: Number of steps after which :meth:`get_success` returns
            ``True``.  ``0`` means **infinite** — the policy never
            reports success on its own (useful as a "background" policy
            inside :class:`CombinedRefPolicy`).
        controller: Optional controller the action is destined for.  When
            given, :meth:`convert_action_dict` / :meth:`get_action_dict_space`
            delegate to it, producing the controller-native action-dict
            schema instead of the generic ``{"action": ...}`` wrapper.
            Pass it whenever this policy stands in for another policy that
            drives the same controller (e.g. as a graph node alongside a
            grasp policy), so all nodes share one action-dict schema.
    """

    def __init__(
        self,
        world: GenesisWorld,
        action: torch.Tensor,
        steps: int = 0,
        controller: Optional[EntityController] = None,
    ) -> None:
        super().__init__(world)
        self.steps = steps
        self.controller = controller

        batch_size = world.batch_size
        device = world.device

        # Store the action in native shape: (D,) unbatched, (B, D) batched.
        self._action = broadcast_to_batch(
            action.to(device=device, dtype=torch.float32),
            batch_size, action.shape[-1], device=device,
        )

        # Per-env step counter — () unbatched, (B,) batched.
        scalar_shape = get_batch_aware_shape(batch_size, ())
        self._step_count = torch.zeros(scalar_shape, dtype=torch.long, device=device)

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        if mask is not None:
            self._step_count[mask] = 0
        else:
            self._step_count.zero_()

    @torch.inference_mode()
    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if mask is not None:
            self._step_count[mask] += 1
        else:
            self._step_count += 1

    @torch.inference_mode()
    def get_action(self) -> torch.Tensor:
        return self._action

    @torch.inference_mode()
    def convert_action_dict(
        self, action: torch.Tensor, dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a raw action to an action dict.

        Delegates to the configured controller's :meth:`get_action_dict`
        (controller-native schema) when one was provided; otherwise falls
        back to the generic ``{"action": ...}`` wrapper.
        """
        if self.controller is not None:
            return self.controller.get_action_dict(action, dt=dt)
        return super().convert_action_dict(action, dt=dt)

    def get_action_dict_space(self) -> DictSpace:
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        With a configured controller this is the controller-native schema;
        otherwise the output is ``{"action": self._action}``, described with
        an unbounded :class:`BoxSpace` matching the action shape.
        """
        if self.controller is not None:
            return self.controller.get_action_dict_space()
        action_dim = self._action.shape[-1]
        batch_size = self.world.batch_size
        device = self.world.device
        return DictSpace(
            PyTorchComputeBackend,
            {
                "action": BoxSpace(
                    PyTorchComputeBackend,
                    low=-torch.inf,
                    high=torch.inf,
                    shape=get_batch_aware_shape(batch_size, action_dim),
                    dtype=self._action.dtype,
                    device=device,
                ),
            },
            device=device,
        )

    @torch.inference_mode()
    def get_success(self) -> Union[torch.Tensor, bool]:
        if self.steps <= 0:
            return self._unwrap_false()
        return try_scalar(self._step_count >= self.steps)

    @torch.inference_mode()
    def get_failure(self) -> Union[torch.Tensor, bool]:
        return self._unwrap_false()

    @torch.inference_mode()
    def get_timeout(self) -> Union[torch.Tensor, bool]:
        return self._unwrap_false()

    def _unwrap_false(self) -> Union[torch.Tensor, bool]:
        if is_batched(self.world.batch_size):
            return torch.zeros_like(self._step_count, dtype=torch.bool)
        return False
