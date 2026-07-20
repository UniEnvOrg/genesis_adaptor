"""GoToLambdaRefPolicy — GoToRefPolicy with callable target resolution.

A subclass of :class:`GoToRefPolicy` that resolves the target position and
orientation from **callables** at :meth:`reset` time, instead of taking
fixed tensors at construction time.  This allows the target to depend on
runtime state — e.g. a task's per-episode randomised goal, or an object's
current position after grasping.

When ``target_quat_fn`` is ``None`` (or returns ``None``), the EEF's
current orientation at reset time is used as the target orientation
(orientation-free reaching).

Example — companion for :class:`ReachTargetTask`::

    from unienv_genesis_refpolicy.implementations.goto_lambda_policy import GoToLambdaRefPolicy

    task = ReachTargetTask(world, eef_controller=arm, target_workspace=(...))
    policy = GoToLambdaRefPolicy(
        world, controller=arm,
        target_pos_fn=lambda: task._target_pos,
    )

Example — lift after grasp (reads object position at reset time)::

    lift_policy = GoToLambdaRefPolicy(
        world, controller=arm,
        target_pos_fn=lambda: target_entity.get_pos() + torch.tensor([0, 0, 0.15]),
    )
"""

from typing import Any, Callable, Dict, Optional, Union

import torch

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers.end_effector_pose import (
    EndEffectorPoseController,
)
from unienv_genesis.utils.batch_dim_utils import broadcast_to_batch
from unienv_genesis_refpolicy.base.policy import require_eef_controller
from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy

__all__ = ["GoToLambdaRefPolicy"]


class GoToLambdaRefPolicy(GoToRefPolicy):
    """Move an end-effector to a target pose resolved from callables at reset.

    Args:
        world: Genesis world instance.
        controller: The EEF pose controller to query / produce actions for.
        target_pos_fn: Callable returning the target position tensor
            ``(3,)`` or ``(B, 3)``.  Called at every :meth:`reset`.
        target_quat_fn: Callable returning the target quaternion
            ``(w, x, y, z)`` as ``(4,)`` or ``(B, 4)``, or ``None`` to use
            the EEF's current orientation.  When this argument itself is
            ``None`` (the default), the EEF's current orientation is always
            used.
        **kwargs: Forwarded to :class:`GoToRefPolicy` (thresholds, stall
            settings, max_seconds, …).
    """

    def __init__(
        self,
        world: GenesisWorld,
        controller: EndEffectorPoseController,
        target_pos_fn: Callable[[], torch.Tensor],
        target_quat_fn: Optional[Callable[[], Optional[torch.Tensor]]] = None,
        **kwargs,
    ) -> None:
        require_eef_controller(controller, policy_name="GoToLambdaRefPolicy")
        self._target_pos_fn = target_pos_fn
        self._target_quat_fn = target_quat_fn

        # Initialise the parent with dummy zero targets — the real targets
        # and action are resolved in reset().
        batch_size = world.batch_size
        device = world.device
        dummy_pos = torch.zeros(
            (3,) if batch_size is None else (batch_size, 3),
            dtype=torch.float32, device=device,
        )
        dummy_quat = torch.zeros(
            (4,) if batch_size is None else (batch_size, 4),
            dtype=torch.float32, device=device,
        )
        dummy_quat[..., 0] = 1.0  # identity quaternion

        super().__init__(
            world=world,
            controller=controller,
            target_pos=dummy_pos,
            target_quat=dummy_quat,
            **kwargs,
        )

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        # --- Resolve fresh targets from the callables ---
        target_pos = self._target_pos_fn().detach().to(
            device=self.world.device, dtype=torch.float32,
        )

        if self._target_quat_fn is not None:
            target_quat = self._target_quat_fn()
        else:
            target_quat = None

        if target_quat is None:
            # Use the EEF's current orientation.
            _, eef_quat = self.controller.get_eef_pose_absolute()
            target_quat = eef_quat.detach().to(
                device=self.world.device, dtype=torch.float32,
            )
        else:
            target_quat = target_quat.detach().to(
                device=self.world.device, dtype=torch.float32,
            )

        # Broadcast to batch if needed.
        target_pos = broadcast_to_batch(
            target_pos, self.world.batch_size, 3, device=self.world.device,
        )
        target_quat = broadcast_to_batch(
            target_quat, self.world.batch_size, 4, device=self.world.device,
        )

        # Store the fresh targets.  The frame-dependent action is recomputed
        # from them exactly once in the parent reset below (which also covers
        # base-frame changes), so we deliberately do not convert here.
        if mask is not None:
            self.target_pos[mask] = target_pos[mask]
            self.target_quat[mask] = target_quat[mask]
        else:
            self.target_pos.copy_(target_pos)
            self.target_quat.copy_(target_quat)

        # Reset tracking state and recompute the action via the parent.
        super().reset(mask=mask, seed=seed)
