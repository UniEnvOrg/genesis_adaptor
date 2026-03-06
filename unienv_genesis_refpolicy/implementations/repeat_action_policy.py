"""RepeatActionRefPolicy — output a fixed action for N steps."""

from typing import Optional, Union

import torch

from unienv_genesis.world.genesis_world import GenesisWorld
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
    """

    def __init__(
        self,
        world: GenesisWorld,
        action: torch.Tensor,
        steps: int = 0,
    ) -> None:
        super().__init__(world)
        self.steps = steps

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

    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if mask is not None:
            self._step_count[mask] += 1
        else:
            self._step_count += 1

    def get_action(self) -> torch.Tensor:
        return self._action

    def get_success(self) -> Union[torch.Tensor, bool]:
        if self.steps <= 0:
            return self._unwrap_false()
        return try_scalar(self._step_count >= self.steps)

    def get_failure(self) -> Union[torch.Tensor, bool]:
        return self._unwrap_false()

    def get_timeout(self) -> Union[torch.Tensor, bool]:
        return self._unwrap_false()

    def _unwrap_false(self) -> Union[torch.Tensor, bool]:
        if is_batched(self.world.batch_size):
            return torch.zeros_like(self._step_count, dtype=torch.bool)
        return False
