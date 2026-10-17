"""The arm's EEF controller seen in the gripper's tip frame.

An arm controller's end effector is its ``target_link`` (e.g. a robot flange),
but grasp candidates place the gripper's *tip* (the point between the
fingertips). A gripper controller describes where its tip sits relative to the
arm's EEF with the optional ``eef_tip_offset`` (``(3,)`` position, metres, in
the EEF frame) and ``eef_tip_offset_quat`` (``(w, x, y, z)``, default
identity) attributes (see ``EntityController.extra_attributes``).

:func:`tip_frame_controller` wraps the arm controller in a
:class:`TipFrameEEFController` whose EEF poses, IK targets and world-frame
action targets are tip poses; the policies that receive both controllers
(``BoxGripperGraspPolicy``, ``PickPlaceRefPolicy``, ``MoveObjectPolicy``) wrap
it themselves, so their logic is frame-agnostic. Without ``eef_tip_offset`` the
tip is the EEF and the controller is used as-is.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
)

__all__ = ["TipFrameEEFController", "tip_frame_controller"]


class TipFrameEEFController:
    """A view of an EEF pose controller whose end effector is a tool tip.

    The tip is the controller's EEF composed with a fixed local offset::

        tip_pos  = eef_pos + eef_quat * offset_pos
        tip_quat = eef_quat * offset_quat

    :meth:`get_eef_pose_absolute` / :meth:`get_eef_pose_base` report the tip,
    and :meth:`solve_ik_absolute` / :meth:`solve_ik_base` /
    :meth:`get_action_world` take tip targets, converting them to EEF targets
    of the wrapped controller. Everything else (``has_eef``, ``world``,
    ``get_action_dict``, ``apply_action``, ...) is the wrapped controller's,
    so actions stay in the controller's format and frame. The IK error is the
    wrapped controller's (the error of its physical target link).
    """

    def __init__(
        self,
        controller: Any,
        offset_pos: Sequence[float],
        offset_quat: Optional[Sequence[float]] = None,
    ) -> None:
        device = controller.world.device
        self.controller = controller
        self.tip_offset_pos = torch.as_tensor(offset_pos, dtype=torch.float32, device=device)
        self.tip_offset_quat = torch.as_tensor(
            offset_quat if offset_quat is not None else (1.0, 0.0, 0.0, 0.0),
            dtype=torch.float32,
            device=device,
        )

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the view does not define.
        if name == "controller":  # not set yet (e.g. while unpickling)
            raise AttributeError(name)
        return getattr(self.controller, name)

    def eef_to_tip(
        self, position: torch.Tensor, quaternion: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """EEF pose -> tip pose (in the same frame)."""
        tip_quat = quaternion_multiply(quaternion, self.tip_offset_quat)
        tip_pos = position + quaternion_apply(quaternion, self.tip_offset_pos)
        return tip_pos, tip_quat

    def tip_to_eef(
        self, position: torch.Tensor, quaternion: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tip pose -> EEF pose (in the same frame)."""
        eef_quat = quaternion_multiply(quaternion, quaternion_invert(self.tip_offset_quat))
        eef_pos = position - quaternion_apply(eef_quat, self.tip_offset_pos)
        return eef_pos, eef_quat

    # The offset is local (right-multiplied), so it commutes with the
    # base <-> world transform: the conversions apply in either frame.

    def get_eef_pose_absolute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The tip pose in the world frame."""
        return self.eef_to_tip(*self.controller.get_eef_pose_absolute())

    def get_eef_pose_base(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The tip pose in the controller's base frame."""
        return self.eef_to_tip(*self.controller.get_eef_pose_base())

    def solve_ik_absolute(
        self,
        target_position: torch.Tensor,
        target_quaternion: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """IK for a world-frame tip target."""
        position, quaternion = self.tip_to_eef(target_position, target_quaternion)
        return self.controller.solve_ik_absolute(position, quaternion, envs_idx=envs_idx)

    def solve_ik_base(
        self, target_position: torch.Tensor, target_quaternion: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """IK for a base-frame tip target."""
        return self.controller.solve_ik_base(*self.tip_to_eef(target_position, target_quaternion))

    def get_action_world(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """The controller action that moves the tip to a world-frame pose."""
        return self.controller.get_action_world(*self.tip_to_eef(pos, quat))


def tip_frame_controller(eef_controller: Any, gripper_controller: Any = None) -> Any:
    """``eef_controller`` in the tip frame of ``gripper_controller``.

    Returns a :class:`TipFrameEEFController` when the gripper controller has an
    ``eef_tip_offset``, else ``eef_controller`` itself (also when it is already
    a tip-frame view).
    """
    if gripper_controller is None or isinstance(eef_controller, TipFrameEEFController):
        return eef_controller
    offset_pos = getattr(gripper_controller, "eef_tip_offset", None)
    if offset_pos is None:
        return eef_controller
    return TipFrameEEFController(
        eef_controller, offset_pos, getattr(gripper_controller, "eef_tip_offset_quat", None),
    )
