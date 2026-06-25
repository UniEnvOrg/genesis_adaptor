"""MoveObjectLambdaPolicy — MoveObjectPolicy with callable target resolution."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers.end_effector_pose import (
    EndEffectorPoseController,
)
from unienv_genesis.utils.batch_dim_utils import broadcast_to_batch
from unienv_genesis_refpolicy.base.policy import require_eef_controller
from unienv_genesis_refpolicy.implementations.move_object_policy import MoveObjectPolicy

__all__ = ["MoveObjectLambdaPolicy"]


class MoveObjectLambdaPolicy(MoveObjectPolicy):
    """Move a grasped object toward a target object pose resolved at reset."""

    def __init__(
        self,
        world: GenesisWorld,
        controller: EndEffectorPoseController,
        object_entity_fn: Callable[[], object],
        target_pos_fn: Callable[[], torch.Tensor],
        target_quat_fn: Optional[Callable[[], Optional[torch.Tensor]]] = None,
        gripper_controller=None,
        fail_on_gripper_contact_loss: bool = False,
        slip_position_threshold: float = 0.01,
        slip_orientation_threshold: float = 0.05,
        **kwargs,
    ) -> None:
        require_eef_controller(controller, policy_name="MoveObjectLambdaPolicy")
        self._object_entity_fn = object_entity_fn
        self._target_pos_fn = target_pos_fn
        self._target_quat_fn = target_quat_fn

        batch_size = world.batch_size
        device = world.device
        dummy_pos = torch.zeros(
            (3,) if batch_size is None else (batch_size, 3),
            dtype=torch.float32,
            device=device,
        )

        super().__init__(
            world=world,
            controller=controller,
            object_entity=object_entity_fn(),
            target_pos=dummy_pos,
            target_quat=None,
            gripper_controller=gripper_controller,
            fail_on_gripper_contact_loss=fail_on_gripper_contact_loss,
            slip_position_threshold=slip_position_threshold,
            slip_orientation_threshold=slip_orientation_threshold,
            **kwargs,
        )

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.object_entity = self._object_entity_fn()
        self._object_target_pos = self._target_pos_fn().detach().to(
            device=self.world.device, dtype=torch.float32,
        )
        if self._target_quat_fn is not None:
            target_quat = self._target_quat_fn()
            if target_quat is not None:
                target_quat = target_quat.detach().to(
                    device=self.world.device, dtype=torch.float32,
                )
        else:
            target_quat = None
        self._object_target_quat = target_quat
        super().reset(mask=mask, seed=seed)
