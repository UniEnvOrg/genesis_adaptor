"""Smooth commanded-pose motion for the interpolating reference policies.

Two opt-in additions to the straight-line Cartesian interpolation of
:func:`~unienv_genesis_refpolicy.implementations.goto_policy.step_pose_toward`
(constant speed: the commanded velocity jumps from zero to full speed at the
start of a motion and back to zero at its end):

* :func:`ramped_step_pose_toward` — a trapezoidal speed profile. The commanded
  pose accelerates to the cruise speeds within ``ramp_seconds`` and decelerates
  to stop at the target. Position and rotation stay synchronized: the profile
  runs on the motion's normalized length (seconds at cruise speed).
* The command hand-off — a policy that ends stores its commanded pose, speed
  and direction on the arm controller (:func:`store_command`); the next one
  starts from it (:func:`load_command`) instead of from the achieved EEF pose,
  keeping the speed in proportion to how well the new direction continues the
  old one. Consecutive phases then join without a stop-and-snap at the
  achieved pose. The stored command is in the underlying controller's EEF
  frame, so tip-frame views (:mod:`~unienv_genesis_refpolicy.base.tip_frame`)
  and plain controllers share it; a stored command far from the achieved pose
  (a new episode, a long hold, a failed phase) is ignored.
* The command leash (:func:`leash_command`) — the commanded pose is kept within
  ``max_command_lead`` (m) / ``max_command_lead_angle`` (rad) of the *achieved*
  EEF pose, so the command never runs far ahead of an arm that cannot follow
  (joint-rate limits, contact, a phase that starts far away). Like a teleop
  step on the real robot, the command then leads the arm by a bounded amount.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

__all__ = [
    "HANDOFF_MAX_DISTANCE",
    "HANDOFF_MAX_ANGLE",
    "ramped_step_pose_toward",
    "store_command",
    "load_command",
    "leash_command",
    "leashed_speed",
]

#: A stored command farther than this from the achieved EEF pose is not continued (m / rad).
HANDOFF_MAX_DISTANCE = 0.05
HANDOFF_MAX_ANGLE = 0.5

_STATE_ATTR = "_refpolicy_command_handoff"


def _quat_angle(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(q0 * q1, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def ramped_step_pose_toward(
    pos: torch.Tensor,
    quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    speed: torch.Tensor,
    dt: float,
    max_linear_speed: Optional[float],
    max_angular_speed: Optional[float],
    ramp_seconds: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step of a trapezoidal-profile straight-line motion toward a target pose.

    ``speed`` ``(...)`` is the current fraction of the cruise speeds (``0`` at
    rest, ``1`` at cruise). The motion's remaining length is measured in
    seconds at cruise speed (the larger of the translational and rotational
    parts); the speed rises by ``dt / ramp_seconds`` per step up to cruise and
    is capped so the command can still stop at the target with the same
    deceleration. Returns the new ``(pos, quat, speed)``.
    """
    from unienv_genesis_refpolicy.implementations.goto_policy import _quat_slerp

    remaining = torch.zeros_like(speed)
    if max_linear_speed is not None:
        remaining = torch.maximum(remaining, torch.norm(target_pos - pos, dim=-1) / max_linear_speed)
    if max_angular_speed is not None:
        remaining = torch.maximum(remaining, _quat_angle(quat, target_quat) / max_angular_speed)
    accel = 1.0 / float(ramp_seconds)
    stoppable = torch.sqrt(2.0 * accel * remaining)
    new_speed = torch.minimum(torch.minimum(speed + accel * dt, stoppable), torch.ones_like(speed))
    # At least one acceleration step's worth of motion, so a motion that starts
    # within the stopping band still makes progress.
    new_speed = torch.where(remaining > 0, torch.maximum(new_speed, torch.full_like(speed, accel * dt)), new_speed * 0.0)
    alpha = torch.clamp(new_speed * dt / remaining.clamp(min=1e-9), max=1.0)
    new_speed = torch.where(alpha >= 1.0, torch.zeros_like(new_speed), new_speed)
    return pos + alpha.unsqueeze(-1) * (target_pos - pos), _quat_slerp(quat, target_quat, alpha), new_speed


def _root_controller(controller: Any) -> Any:
    """The arm controller under a tip-frame view (or the controller itself)."""
    return controller.controller if hasattr(controller, "tip_to_eef") else controller


def store_command(
    controller: Any,
    pos: torch.Tensor,
    quat: torch.Tensor,
    speed: torch.Tensor,
    direction: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Store a policy's commanded pose (in ``controller``'s frame), speed and unit direction."""
    if hasattr(controller, "tip_to_eef"):
        pos, quat = controller.tip_to_eef(pos, quat)
    root = _root_controller(controller)
    state = getattr(root, _STATE_ATTR, None)
    if state is None or mask is None:
        state = {
            "pos": pos.detach().clone(), "quat": quat.detach().clone(),
            "speed": speed.detach().clone(), "direction": direction.detach().clone(),
        }
        setattr(root, _STATE_ATTR, state)
        return
    state["pos"][mask] = pos.detach()[mask]
    state["quat"][mask] = quat.detach()[mask]
    state["speed"][mask] = speed.detach()[mask]
    state["direction"][mask] = direction.detach()[mask]


def load_command(
    controller: Any,
    eef_pos: torch.Tensor,
    eef_quat: torch.Tensor,
    target_pos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Start ``(pos, quat, speed)`` of a new motion in ``controller``'s frame.

    The stored command where it is close to the achieved pose
    (``eef_pos`` / ``eef_quat``), else the achieved pose at rest. The carried
    speed is scaled by the cosine between the stored direction and the new
    motion's (``target_pos`` minus the start), clamped at zero.
    """
    speed = torch.zeros(eef_pos.shape[:-1], dtype=eef_pos.dtype, device=eef_pos.device)
    state = getattr(_root_controller(controller), _STATE_ATTR, None)
    if state is None:
        return eef_pos.clone(), eef_quat.clone(), speed
    pos, quat = state["pos"], state["quat"]
    if hasattr(controller, "eef_to_tip"):
        pos, quat = controller.eef_to_tip(pos, quat)
    close = (torch.norm(pos - eef_pos, dim=-1) < HANDOFF_MAX_DISTANCE) & (
        _quat_angle(quat, eef_quat) < HANDOFF_MAX_ANGLE
    )
    new_direction = target_pos - pos
    new_norm = torch.norm(new_direction, dim=-1)
    cosine = torch.sum(state["direction"] * new_direction, dim=-1) / new_norm.clamp(min=1e-9)
    carried = torch.where(new_norm > 1e-3, state["speed"] * cosine.clamp(0.0, 1.0), torch.zeros_like(speed))
    return (
        torch.where(close.unsqueeze(-1), pos, eef_pos),
        torch.where(close.unsqueeze(-1), quat, eef_quat),
        torch.where(close, carried, speed),
    )


def leash_command(
    controller: Any,
    cmd_pos: torch.Tensor,
    cmd_quat: torch.Tensor,
    eef_pos: torch.Tensor,
    eef_quat: torch.Tensor,
    max_distance: Optional[float],
    max_angle: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamp a commanded pose onto the ball / cone around the achieved EEF pose.

    ``cmd_*`` and ``eef_*`` are poses in ``controller``'s frame (a tip-frame
    view's tip, or the plain controller's EEF). The lead is measured on the
    underlying arm controller's EEF (e.g. the flange: the pose the controller
    is actually commanded with), so a rotation about a tool tip that swings
    the flange counts toward the position lead. There:

    * the position is clipped to ``max_distance`` from the achieved position
      (along the line to the command);
    * the rotation is slerped from the achieved rotation toward the command,
      by at most ``max_angle``.

    ``None`` leaves that part unbounded. Returns ``(pos, quat, clamped)``:
    the leashed pose in ``controller``'s frame and a bool ``(...)`` mask of the
    envs whose command was moved (unmoved envs return the command unchanged,
    bit for bit).
    """
    from unienv_genesis_refpolicy.implementations.goto_policy import _quat_slerp

    clamped = torch.zeros(cmd_pos.shape[:-1], dtype=torch.bool, device=cmd_pos.device)
    if max_distance is None and max_angle is None:
        return cmd_pos, cmd_quat, clamped
    tip_view = hasattr(controller, "tip_to_eef")
    if tip_view:
        c_pos, c_quat = controller.tip_to_eef(cmd_pos, cmd_quat)
        a_pos, a_quat = controller.tip_to_eef(eef_pos, eef_quat)
    else:
        c_pos, c_quat, a_pos, a_quat = cmd_pos, cmd_quat, eef_pos, eef_quat
    new_quat = c_quat
    if max_angle is not None:
        angle = _quat_angle(a_quat, c_quat)
        over = angle > float(max_angle)
        fraction = torch.where(over, float(max_angle) / angle.clamp(min=1e-9), torch.ones_like(angle))
        new_quat = torch.where(over.unsqueeze(-1), _quat_slerp(a_quat, c_quat, fraction), c_quat)
        clamped = clamped | over
    new_pos = c_pos
    if max_distance is not None:
        delta = c_pos - a_pos
        distance = torch.norm(delta, dim=-1)
        over = distance > float(max_distance)
        scale = torch.where(over, float(max_distance) / distance.clamp(min=1e-9), torch.ones_like(distance))
        new_pos = torch.where(over.unsqueeze(-1), a_pos + scale.unsqueeze(-1) * delta, c_pos)
        clamped = clamped | over
    if tip_view:
        new_pos, new_quat = controller.eef_to_tip(new_pos, new_quat)
    mask = clamped.unsqueeze(-1)
    return torch.where(mask, new_pos, cmd_pos), torch.where(mask, new_quat, cmd_quat), clamped


def leashed_speed(
    speed: torch.Tensor,
    clamped: torch.Tensor,
    prev_pos: torch.Tensor,
    prev_quat: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    dt: float,
    max_linear_speed: Optional[float],
    max_angular_speed: Optional[float],
) -> torch.Tensor:
    """The speed state (fraction of cruise) of a ramped motion after a leash step.

    Where the command was clamped (``clamped``) it restarts from rest, so it
    accelerates again once the arm has caught up instead of resuming at the
    planned speed. ``prev_*`` / ``pos`` / ``quat`` (the command before / after
    the step) and the speeds are given for alternative policies.
    """
    return torch.where(clamped, torch.zeros_like(speed), speed)
