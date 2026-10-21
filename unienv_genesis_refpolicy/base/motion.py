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
