"""MoveObjectPolicy — move an object by preserving the current grasp transform.

This policy resolves an end-effector target pose from a desired **object** pose
by preserving the current object→EEF transform measured at :meth:`reset` time.
It is useful after grasp acquisition, where commanding the EEF directly to the
object origin is incorrect for arbitrary grasps.
"""

from __future__ import annotations

from typing import Optional

import torch

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
)

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers.end_effector_pose import (
    EndEffectorPoseController,
)
from unienv_genesis.utils.batch_dim_utils import broadcast_to_batch
from unienv_genesis_refpolicy.base.policy import (
    require_eef_controller,
    DURATION_COMPARISON_EPS,
)
from unienv_genesis_refpolicy.base.tip_frame import tip_frame_controller
from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy
from unienv_genesis.utils.batch_dim_utils import try_scalar

__all__ = ["MoveObjectPolicy"]


class MoveObjectPolicy(GoToRefPolicy):
    """Move a grasped object toward a target object pose.

    The policy reads the current object pose and EEF pose at :meth:`reset`,
    computes the current object-frame grasp transform, and preserves that
    transform while moving the object toward the requested target pose.

    Args:
        world: Genesis world instance.
        controller: EEF pose controller.
        object_entity: The currently grasped object entity.
        target_pos: Desired object position, shape ``(3,)`` or ``(B, 3)``.
        target_quat: Optional desired object quaternion. If ``None``, the
            object's current quaternion at reset is preserved.
        gripper_controller: Optional gripper controller used for contact-based
            failure detection. With an ``eef_tip_offset`` it also makes the
            policy move the gripper's tip, as the grasp policy does.
        fail_on_gripper_contact_loss: Whether losing object contact should fail
            the policy.
        slip_position_threshold: Maximum allowed drift in the cached object→EEF
            relative position before declaring slip.
        slip_orientation_threshold: Maximum allowed quaternion distance in the
            cached object→EEF relative orientation before declaring slip.
        track_relative_pose: When ``True`` (default), :meth:`step` recomputes
            the current object→EEF relative pose every frame and updates the
            slip-drift flags. When ``False``, the per-step recomputation and
            slip-drift update are skipped entirely, relying on the transform
            cached at :meth:`reset`. Contact-loss checking still runs if
            enabled.
        **kwargs: Forwarded to :class:`GoToRefPolicy`.
    """

    def __init__(
        self,
        world: GenesisWorld,
        controller: EndEffectorPoseController,
        object_entity,
        target_pos: torch.Tensor,
        target_quat: Optional[torch.Tensor] = None,
        gripper_controller=None,
        fail_on_gripper_contact_loss: bool = False,
        slip_position_threshold: float = 0.01,
        slip_orientation_threshold: float = 0.05,
        track_relative_pose: bool = True,
        **kwargs,
    ) -> None:
        controller = tip_frame_controller(controller, gripper_controller)
        require_eef_controller(controller, policy_name="MoveObjectPolicy")
        if fail_on_gripper_contact_loss and gripper_controller is None:
            raise ValueError(
                "MoveObjectPolicy: gripper_controller is required when "
                "fail_on_gripper_contact_loss=True"
            )
        self.object_entity = object_entity
        self._object_target_pos = target_pos
        self._object_target_quat = target_quat
        self.gripper_controller = gripper_controller
        self.fail_on_gripper_contact_loss = fail_on_gripper_contact_loss
        self.slip_position_threshold = float(slip_position_threshold)
        self.slip_orientation_threshold = float(slip_orientation_threshold)
        self.track_relative_pose = bool(track_relative_pose)

        batch_size = world.batch_size
        device = world.device
        dummy_pos = torch.zeros(
            (3,) if batch_size is None else (batch_size, 3),
            dtype=torch.float32,
            device=device,
        )
        dummy_quat = torch.zeros(
            (4,) if batch_size is None else (batch_size, 4),
            dtype=torch.float32,
            device=device,
        )
        dummy_quat[..., 0] = 1.0

        super().__init__(
            world=world,
            controller=controller,
            target_pos=dummy_pos,
            target_quat=dummy_quat,
            **kwargs,
        )
        scalar_shape = self._elapsed_seconds.shape
        pos_shape = self._last_eef_pos.shape
        quat_shape = self._last_eef_quat.shape
        device = world.device
        self._initial_rel_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._initial_rel_quat = torch.zeros(quat_shape, dtype=torch.float32, device=device)
        self._initial_rel_quat[..., 0] = 1.0
        self._slipped = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._lost_grasp_contact = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._grasping_at_reset = torch.zeros(scalar_shape, dtype=torch.bool, device=device)

    @staticmethod
    def _bool_tensor(value, like: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=like.device, dtype=torch.bool)
        return torch.full_like(like, bool(value), dtype=torch.bool)

    def _current_relative_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_pos = self.object_entity.get_pos().detach().to(
            device=self.world.device, dtype=torch.float32,
        )
        object_quat = self.object_entity.get_quat().detach().to(
            device=self.world.device, dtype=torch.float32,
        )
        eef_pos, eef_quat = self.controller.get_eef_pose_absolute()
        eef_pos = eef_pos.detach().to(device=self.world.device, dtype=torch.float32)
        eef_quat = eef_quat.detach().to(device=self.world.device, dtype=torch.float32)
        q_obj_inv = quaternion_invert(object_quat)
        rel_quat = quaternion_multiply(q_obj_inv, eef_quat)
        rel_pos = quaternion_apply(q_obj_inv, eef_pos - object_pos)
        return rel_pos, rel_quat

    @staticmethod
    def _compute_target_eef_pose(
        *,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
        target_object_pos: torch.Tensor,
        target_object_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_obj_inv = quaternion_invert(object_quat)
        rel_quat = quaternion_multiply(q_obj_inv, eef_quat)
        rel_pos = quaternion_apply(q_obj_inv, eef_pos - object_pos)
        target_eef_pos = target_object_pos + quaternion_apply(target_object_quat, rel_pos)
        target_eef_quat = quaternion_multiply(target_object_quat, rel_quat)
        return target_eef_pos, target_eef_quat

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        object_pos = self.object_entity.get_pos().detach().to(
            device=self.world.device, dtype=torch.float32,
        )
        object_quat = self.object_entity.get_quat().detach().to(
            device=self.world.device, dtype=torch.float32,
        )
        eef_pos, eef_quat = self.controller.get_eef_pose_absolute()
        eef_pos = eef_pos.detach().to(device=self.world.device, dtype=torch.float32)
        eef_quat = eef_quat.detach().to(device=self.world.device, dtype=torch.float32)

        target_object_pos = self._object_target_pos.detach().to(
            device=self.world.device, dtype=torch.float32,
        )
        target_object_pos = broadcast_to_batch(
            target_object_pos, self.world.batch_size, 3, device=self.world.device,
        )

        if self._object_target_quat is None:
            target_object_quat = object_quat
        else:
            target_object_quat = self._object_target_quat.detach().to(
                device=self.world.device, dtype=torch.float32,
            )
            target_object_quat = broadcast_to_batch(
                target_object_quat, self.world.batch_size, 4, device=self.world.device,
            )

        target_pos, target_quat = self._compute_target_eef_pose(
            object_pos=object_pos,
            object_quat=object_quat,
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            target_object_pos=target_object_pos,
            target_object_quat=target_object_quat,
        )

        # Store the fresh EEF targets.  The frame-dependent action is
        # recomputed from them exactly once in the parent reset below (which
        # also covers base-frame changes), so we deliberately do not convert
        # here.
        if mask is not None:
            self.target_pos[mask] = target_pos[mask]
            self.target_quat[mask] = target_quat[mask]
        else:
            self.target_pos.copy_(target_pos)
            self.target_quat.copy_(target_quat)

        rel_pos, rel_quat = self._current_relative_pose()
        if mask is not None:
            self._initial_rel_pos[mask] = rel_pos[mask]
            self._initial_rel_quat[mask] = rel_quat[mask]
            self._slipped[mask] = False
            self._lost_grasp_contact[mask] = False
        else:
            self._initial_rel_pos.copy_(rel_pos)
            self._initial_rel_quat.copy_(rel_quat)
            self._slipped.zero_()
            self._lost_grasp_contact.zero_()

        if self.fail_on_gripper_contact_loss:
            grasping_now = self.gripper_controller.is_grasping(self.object_entity)
            grasping_now = self._bool_tensor(grasping_now, self._grasping_at_reset)
            if mask is not None:
                self._grasping_at_reset[mask] = grasping_now[mask]
            else:
                self._grasping_at_reset.copy_(grasping_now)
        else:
            if mask is not None:
                self._grasping_at_reset[mask] = False
            else:
                self._grasping_at_reset.zero_()

        super().reset(mask=mask, seed=seed)

    @torch.inference_mode()
    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        dt: float,
    ) -> None:
        super().step(mask=mask, dt=dt)

        if self.track_relative_pose:
            rel_pos, rel_quat = self._current_relative_pose()
            pos_drift = torch.norm(rel_pos - self._initial_rel_pos, dim=-1)
            quat_dot = torch.abs(torch.sum(rel_quat * self._initial_rel_quat, dim=-1))
            quat_dist = 1.0 - quat_dot.clamp(max=1.0)
            pos_slip = (
                pos_drift > self.slip_position_threshold
                if self.slip_position_threshold > 0
                else torch.zeros_like(pos_drift, dtype=torch.bool)
            )
            quat_slip = (
                quat_dist > self.slip_orientation_threshold
                if self.slip_orientation_threshold > 0
                else torch.zeros_like(quat_dist, dtype=torch.bool)
            )
            slipped = pos_slip | quat_slip
            if mask is not None:
                slipped = slipped & mask
                self._slipped = self._slipped | slipped
            else:
                self._slipped = self._slipped | slipped

        if self.fail_on_gripper_contact_loss:
            grasping_now = self.gripper_controller.is_grasping(self.object_entity)
            grasping_now = self._bool_tensor(grasping_now, self._grasping_at_reset)
            lost_contact = self._grasping_at_reset & ~grasping_now
            if mask is not None:
                lost_contact = lost_contact & mask
                self._lost_grasp_contact = self._lost_grasp_contact | lost_contact
            else:
                self._lost_grasp_contact = self._lost_grasp_contact | lost_contact

    @torch.inference_mode()
    def get_success(self):
        success = self._reached & ~self._slipped & ~self._lost_grasp_contact
        return try_scalar(success)

    @torch.inference_mode()
    def get_failure(self):
        failed = self._stalled | self._slipped | self._lost_grasp_contact
        return try_scalar(failed)

    @torch.inference_mode()
    def get_timeout(self):
        if self.max_seconds <= 0:
            return try_scalar(torch.zeros_like(self._reached))
        timed_out = (
            # Shared epsilon keeps exact duration boundaries from slipping a
            # step, consistent with GoToRefPolicy.get_timeout (this override
            # only adds the slip/contact-loss exclusions).
            (self._elapsed_seconds >= self.max_seconds - DURATION_COMPARISON_EPS)
            & ~self._reached
            & ~self._stalled
            & ~self._slipped
            & ~self._lost_grasp_contact
        )
        return try_scalar(timed_out)
