"""RepeatActionRefPolicy — output a fixed action every step."""

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
from unienv_genesis_refpolicy.base.policy import (
    BaseRefPolicy,
    DURATION_COMPARISON_EPS,
)


class RepeatActionRefPolicy(BaseRefPolicy[torch.Tensor]):
    """Output a fixed action tensor every step.

    Useful for holding a gripper open/closed or any constant-action phase.

    Args:
        world: Genesis world instance.
        action: The fixed action tensor to return on every
            :meth:`get_action` call (e.g. ``torch.tensor([-1.0])`` for
            gripper open).  Automatically broadcast to
            ``(batch_size, ...)`` for batched worlds.
        duration_seconds: Optional wall-clock duration in seconds after
            which :meth:`get_success` returns ``True``.  Elapsed time is
            accumulated per :meth:`step` from the explicit ``dt`` argument
            (the control/policy timestep), so it stays correct for explicit
            or variable control-step durations.
            ``None`` (default) holds indefinitely — the policy never
            reports success on its own (useful as a "background" policy
            inside :class:`CombinedRefPolicy`).  A supplied value must be
            strictly positive.
        controller: Optional controller the action is destined for.  When
            given, :meth:`convert_action_dict` / :meth:`get_action_dict_space`
            delegate to it, producing the controller-native action-dict
            schema instead of the generic ``{"action": ...}`` wrapper.
            Pass it whenever this policy stands in for another policy that
            drives the same controller (e.g. as a graph node alongside a
            grasp policy), so all nodes share one action-dict schema.
    """

    #: Tolerance (seconds) guarding the duration comparison against
    #: floating-point accumulation error when summing ``dt`` repeatedly.
    _DURATION_EPS = DURATION_COMPARISON_EPS

    def __init__(
        self,
        world: GenesisWorld,
        action: torch.Tensor,
        controller: Optional[EntityController] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(world)
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError(
                "RepeatActionRefPolicy: 'duration_seconds' must be strictly "
                f"positive or None, got {duration_seconds}."
            )
        self.duration_seconds = (
            None if duration_seconds is None else float(duration_seconds)
        )
        self.controller = controller

        batch_size = world.batch_size
        device = world.device

        # Store the action in native shape: (D,) unbatched, (B, D) batched.
        self._action = broadcast_to_batch(
            action.to(device=device, dtype=torch.float32),
            batch_size, action.shape[-1], device=device,
        )

        # Per-env elapsed-time counter — () unbatched, (B,) batched.
        # Tracked in float64: float32 accumulation error at common control
        # rates (e.g. 60 Hz / 100 Hz) can leave the counter a hair below
        # the duration threshold at the boundary step, delaying the
        # transition by one full step.
        scalar_shape = get_batch_aware_shape(batch_size, ())
        self._elapsed_seconds = torch.zeros(
            scalar_shape, dtype=torch.float64, device=device,
        )

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        if mask is not None:
            self._elapsed_seconds[mask] = 0.0
        else:
            self._elapsed_seconds.zero_()

    @torch.inference_mode()
    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        dt: float,
    ) -> None:
        step_dt = float(dt)
        if mask is not None:
            self._elapsed_seconds[mask] += step_dt
        else:
            self._elapsed_seconds += step_dt

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
        if self.duration_seconds is None:
            return self._unwrap_false()
        # The small epsilon guards against float accumulation error when
        # summing dt (e.g. 15 * (1/15) = 0.9999999999999999 < 1.0).
        return try_scalar(
            self._elapsed_seconds >= self.duration_seconds - self._DURATION_EPS
        )

    @torch.inference_mode()
    def get_failure(self) -> Union[torch.Tensor, bool]:
        return self._unwrap_false()

    @torch.inference_mode()
    def get_timeout(self) -> Union[torch.Tensor, bool]:
        return self._unwrap_false()

    def _unwrap_false(self) -> Union[torch.Tensor, bool]:
        if is_batched(self.world.batch_size):
            return torch.zeros_like(self._elapsed_seconds, dtype=torch.bool)
        return False
