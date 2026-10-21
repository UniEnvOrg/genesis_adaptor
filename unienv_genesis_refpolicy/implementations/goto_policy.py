"""GoToRefPolicy — move an end-effector to a target world-frame pose."""

from typing import Any, Dict, Optional, Union

import torch

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers.end_effector_pose import (
    EndEffectorPoseController,
)
from unienv_genesis.utils.batch_dim_utils import (
    is_batched,
    try_scalar,
    get_batch_aware_shape,
    broadcast_to_batch,
)
from unienv_genesis_refpolicy.base.policy import (
    BaseRefPolicy,
    require_eef_controller,
    DURATION_COMPARISON_EPS,
)
from unienv_genesis_refpolicy.base.motion import load_command, ramped_step_pose_toward, store_command

def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation of ``(..., 4)`` ``(w, x, y, z)`` quaternions at ``t`` ``(...)``."""
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin = torch.sin(theta)
    t = t.unsqueeze(-1)
    small = sin < 1e-6
    safe = torch.where(small, torch.ones_like(sin), sin)
    w0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / safe)
    w1 = torch.where(small, t, torch.sin(t * theta) / safe)
    q = w0 * q0 + w1 * q1
    return q / torch.norm(q, dim=-1, keepdim=True)


def step_pose_toward(
    pos: torch.Tensor,
    quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    dt: float,
    max_linear_speed: Optional[float],
    max_angular_speed: Optional[float],
):
    """One straight-line step of a commanded pose toward a target pose.

    Position and rotation (slerp) advance by the same fraction of the remaining
    motion, the largest one that keeps both within ``max_linear_speed * dt`` /
    ``max_angular_speed * dt`` (a ``None`` speed is unbounded). Returns the new
    ``(pos, quat)``.
    """
    delta = target_pos - pos
    distance = torch.norm(delta, dim=-1)
    dot = torch.sum(quat * target_quat, dim=-1)
    angle = 2.0 * torch.acos(dot.abs().clamp(max=1.0))
    alpha = torch.ones_like(distance)
    if max_linear_speed is not None:
        alpha = torch.minimum(alpha, max_linear_speed * dt / distance.clamp(min=1e-9))
    if max_angular_speed is not None:
        alpha = torch.minimum(alpha, max_angular_speed * dt / angle.clamp(min=1e-9))
    return pos + alpha.unsqueeze(-1) * delta, _quat_slerp(quat, target_quat, alpha)


#: Default Cartesian interpolation speeds of :class:`GoToRefPolicy` (m/s, rad/s).
#: The angular speed is one a joint-rate-limited arm (e.g. the FR3 under the
#: DROID limits) can follow; a faster commanded rotation only makes the EEF lag.
DEFAULT_MAX_LINEAR_SPEED = 0.15
DEFAULT_MAX_ANGULAR_SPEED = 0.4


class GoToRefPolicy(BaseRefPolicy[torch.Tensor]):
    """Move an end-effector to a target world-frame pose.

    Produces action tensors compatible with
    :class:`EndEffectorPoseController`.  The observation argument to
    :meth:`get_action` is unused — the target is predetermined.

    Supports both batched and unbatched worlds.

    Args:
        world: Genesis world instance.
        controller: The EEF pose controller to query / produce actions for.
        target_pos: Target position in world frame, ``(3,)`` (unbatched)
            or ``(batch_size, 3)`` (batched).
        target_quat: Target quaternion ``(w, x, y, z)`` in world frame,
            ``(4,)`` or ``(batch_size, 4)``.
        position_threshold: Position tolerance in metres for success.
        orientation_threshold: Quaternion distance tolerance for success,
            measured as ``1 - |q1 · q2|``.
        position_threshold_stall: Per-step translational motion threshold
            in metres. If the EEF moves less than this value, the step
            is considered translationally stalled.
        orientation_threshold_stall: Per-step rotational motion threshold
            measured as ``1 - |q_t · q_{t-1}|``. If the orientation change
            is below this value, the step is considered rotationally stalled.
        stall_seconds: Continuous stalled duration before declaring failure.
            ``0`` disables stall detection.
        max_seconds: Maximum policy runtime before timeout. ``0`` disables
            timeout.
        max_linear_speed / max_angular_speed: Cartesian interpolation speeds
            (m/s, rad/s; default :data:`DEFAULT_MAX_LINEAR_SPEED` /
            :data:`DEFAULT_MAX_ANGULAR_SPEED`). The commanded pose starts at
            the EEF pose on :meth:`reset` and advances toward the target along
            the straight line (slerp for the rotation, position and rotation
            synchronized) by at most these speeds per :meth:`step`. A ``None``
            speed is unbounded; with both ``None`` the action is the target
            pose itself, so the EEF follows whatever path the arm's
            joint-space motion takes — an arc, which e.g. lifts a tool off a
            table over a long horizontal move. Success / stall checks use the
            actual EEF pose against the final target either way.
        ramp_seconds: With interpolation, a trapezoidal speed profile: the
            commanded pose accelerates to the speeds above within this many
            seconds and decelerates to stop at the target
            (:func:`~unienv_genesis_refpolicy.base.motion.ramped_step_pose_toward`).
            ``None`` keeps the constant-speed motion.
        continue_command: With interpolation, start from the commanded pose
            (and speed) the previous interpolating policy on this arm left
            behind, when it is close to the EEF, instead of from the EEF pose
            (:func:`~unienv_genesis_refpolicy.base.motion.load_command`).

    Notes:
        Time accounting always uses the explicit ``dt`` passed to
        :meth:`step` (the control/policy timestep in seconds).

        The frame-dependent action (world-frame target converted through the
        controller's current base frame) is computed at construction and
        **recomputed on every :meth:`reset`**, selectively for the reset
        environments, so randomized or moving base frames do not retain
        construction-time conversions.
    """

    def __init__(
        self,
        world: GenesisWorld,
        controller: EndEffectorPoseController,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        position_threshold: float = 0.01,
        orientation_threshold: float = 0.05,
        position_threshold_stall: float = 1e-4,
        orientation_threshold_stall: float = 1e-3,
        stall_seconds: float = 1.0,
        max_seconds: float = 0.0,
        max_linear_speed: Optional[float] = DEFAULT_MAX_LINEAR_SPEED,
        max_angular_speed: Optional[float] = DEFAULT_MAX_ANGULAR_SPEED,
        ramp_seconds: Optional[float] = None,
        continue_command: bool = False,
    ) -> None:
        super().__init__(world)
        require_eef_controller(controller, policy_name="GoToRefPolicy")
        self.controller = controller
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.max_seconds = max_seconds

        self.position_threshold_stall = position_threshold_stall
        self.orientation_threshold_stall = orientation_threshold_stall
        self.stall_seconds = stall_seconds
        self.max_linear_speed = None if max_linear_speed is None else float(max_linear_speed)
        self.max_angular_speed = None if max_angular_speed is None else float(max_angular_speed)
        self._interpolate = self.max_linear_speed is not None or self.max_angular_speed is not None
        self.ramp_seconds = None if ramp_seconds is None else float(ramp_seconds)
        self.continue_command = bool(continue_command)

        batch_size = world.batch_size
        device = world.device

        # Store targets in native shape: (3,)/(4,) unbatched, (B,3)/(B,4) batched.
        self.target_pos = broadcast_to_batch(
            target_pos.to(device=device, dtype=torch.float32),
            batch_size, 3, device=device,
        )
        self.target_quat = broadcast_to_batch(
            target_quat.to(device=device, dtype=torch.float32),
            batch_size, 4, device=device,
        )

        # Pre-compute the action so get_action() is valid before the first
        # reset; reset() recomputes it from the stored targets against the
        # controller's current base frame.
        self._action = controller.get_action_world(
            self.target_pos, self.target_quat,
        )

        # Mutable per-env state — native shapes via get_batch_aware_shape.
        scalar_shape = get_batch_aware_shape(batch_size, ())  # () or (B,)
        pos_shape = get_batch_aware_shape(batch_size, 3)       # (3,) or (B, 3)
        quat_shape = get_batch_aware_shape(batch_size, 4)      # (4,) or (B, 4)

        # Elapsed-time counters use float64: float32 accumulation error at
        # common control rates (e.g. 60 Hz / 100 Hz) can leave the counter
        # just below the threshold at the exact boundary step, delaying
        # timeout/stall transitions by one full step.
        self._elapsed_seconds = torch.zeros(
            scalar_shape, dtype=torch.float64, device=device,
        )
        self._reached = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._stall_elapsed_seconds = torch.zeros(
            scalar_shape, dtype=torch.float64, device=device,
        )
        self._stalled = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._last_eef_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._last_eef_quat = torch.zeros(quat_shape, dtype=torch.float32, device=device)

        # Initialise last EEF pose from the current EEF pose.
        eef_pos, eef_quat = controller.get_eef_pose_absolute()
        self._last_eef_pos.copy_(eef_pos.detach().float())
        self._last_eef_quat.copy_(eef_quat.detach().float())

        # Commanded (interpolated) pose; only used with Cartesian interpolation.
        self._cmd_pos = self._last_eef_pos.clone()
        self._cmd_quat = self._last_eef_quat.clone()
        self._cmd_speed = torch.zeros(scalar_shape, dtype=torch.float32, device=device)  # fraction of cruise
        self._cmd_dir = torch.zeros(pos_shape, dtype=torch.float32, device=device)  # last unit step direction
        if self._interpolate:
            self._action = controller.get_action_world(self._cmd_pos, self._cmd_quat)

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        # Recompute the frame-dependent action from the stored targets so a
        # changed base frame (e.g. a randomized or moving robot base) does
        # not retain a stale construction-time conversion.  Subclasses that
        # resolve fresh targets in their own reset() write them into
        # ``self.target_pos`` / ``self.target_quat`` before delegating here,
        # so this single recomputation also covers them.
        if self._interpolate:
            # Start the straight-line motion from where the EEF is now (or from
            # the previous policy's command, continue_command).
            eef_pos, eef_quat = self.controller.get_eef_pose_absolute()
            start_pos, start_quat = eef_pos.detach().float(), eef_quat.detach().float()
            start_speed = torch.zeros_like(self._cmd_speed)
            if self.continue_command:
                start_pos, start_quat, start_speed = load_command(
                    self.controller, start_pos, start_quat, self.target_pos,
                )
            if mask is not None:
                self._cmd_pos[mask] = start_pos[mask]
                self._cmd_quat[mask] = start_quat[mask]
                self._cmd_speed[mask] = start_speed[mask]
            else:
                self._cmd_pos.copy_(start_pos)
                self._cmd_quat.copy_(start_quat)
                self._cmd_speed.copy_(start_speed)
            action = self.controller.get_action_world(self._cmd_pos, self._cmd_quat)
        else:
            action = self.controller.get_action_world(self.target_pos, self.target_quat)
        if mask is not None:
            self._action[mask] = action[mask]
            self._elapsed_seconds[mask] = 0.0
            self._reached[mask] = False
            self._stall_elapsed_seconds[mask] = 0.0
            self._stalled[mask] = False
            eef_pos, eef_quat = self.controller.get_eef_pose_absolute()
            self._last_eef_pos[mask] = eef_pos.detach().float()[mask]
            self._last_eef_quat[mask] = eef_quat.detach().float()[mask]
        else:
            self._action.copy_(action)
            self._elapsed_seconds.zero_()
            self._reached.zero_()
            self._stall_elapsed_seconds.zero_()
            self._stalled.zero_()
            eef_pos, eef_quat = self.controller.get_eef_pose_absolute()
            self._last_eef_pos.copy_(eef_pos.detach().float())
            self._last_eef_quat.copy_(eef_quat.detach().float())

    @torch.inference_mode()
    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        dt: float,
    ) -> None:
        step_dt = float(dt)

        # Increment elapsed time.
        if mask is not None:
            self._elapsed_seconds[mask] += step_dt
        else:
            self._elapsed_seconds += step_dt

        # Query current EEF pose (native shapes from controller).
        eef_pos, eef_quat = self.controller.get_eef_pose_absolute()
        eef_pos = eef_pos.detach().float()
        eef_quat = eef_quat.detach().float()

        # --- Success check (position + orientation) ---
        pos_err = torch.norm(eef_pos - self.target_pos, dim=-1)
        quat_dot = torch.abs(torch.sum(eef_quat * self.target_quat, dim=-1))
        quat_dist = 1.0 - quat_dot.clamp(max=1.0)

        newly_reached = (pos_err < self.position_threshold) & (quat_dist < self.orientation_threshold)
        if mask is not None:
            newly_reached = newly_reached & mask
        self._reached = self._reached | newly_reached

        # --- Stall detection ---
        if self.stall_seconds > 0:
            pos_motion = torch.norm(eef_pos - self._last_eef_pos, dim=-1)
            quat_motion_dot = torch.abs(torch.sum(eef_quat * self._last_eef_quat, dim=-1))
            quat_motion = 1.0 - quat_motion_dot.clamp(max=1.0)
            stalled_this_step = (
                (pos_motion < self.position_threshold_stall)
                & (quat_motion < self.orientation_threshold_stall)
            )

            if mask is not None:
                updated = torch.where(
                    stalled_this_step[mask],
                    self._stall_elapsed_seconds[mask] + step_dt,
                    torch.zeros_like(self._stall_elapsed_seconds[mask]),
                )
                self._stall_elapsed_seconds[mask] = updated
            else:
                self._stall_elapsed_seconds = torch.where(
                    stalled_this_step,
                    self._stall_elapsed_seconds + step_dt,
                    torch.zeros_like(self._stall_elapsed_seconds),
                )

            # The epsilon guards against float accumulation error when
            # summing dt (e.g. 100 * 0.01 = 0.9999999999999999 < 1.0 in
            # float64); it is far below any realistic control timestep.
            stall_trigger = (
                (self._stall_elapsed_seconds >= self.stall_seconds - DURATION_COMPARISON_EPS)
                & ~self._reached
            )
            if mask is not None:
                stall_trigger = stall_trigger & mask
            self._stalled = self._stalled | stall_trigger

        if mask is not None:
            self._last_eef_pos[mask] = eef_pos[mask]
            self._last_eef_quat[mask] = eef_quat[mask]
        else:
            self._last_eef_pos.copy_(eef_pos)
            self._last_eef_quat.copy_(eef_quat)

        if self._interpolate:
            self._advance_command(step_dt, mask)

    def _advance_command(self, dt: float, mask: Optional[torch.Tensor]) -> None:
        """Move the commanded pose toward the target along the straight line."""
        if self.ramp_seconds is not None:
            new_pos, new_quat, new_speed = ramped_step_pose_toward(
                self._cmd_pos, self._cmd_quat, self.target_pos, self.target_quat, self._cmd_speed,
                dt, self.max_linear_speed, self.max_angular_speed, self.ramp_seconds,
            )
        else:
            new_pos, new_quat = step_pose_toward(
                self._cmd_pos, self._cmd_quat, self.target_pos, self.target_quat,
                dt, self.max_linear_speed, self.max_angular_speed,
            )
            new_speed = self._cmd_speed
        moved = new_pos - self._cmd_pos
        moved_norm = torch.norm(moved, dim=-1, keepdim=True)
        new_dir = torch.where(moved_norm > 1e-6, moved / moved_norm.clamp(min=1e-9), self._cmd_dir)
        if mask is not None:
            self._cmd_pos[mask] = new_pos[mask]
            self._cmd_quat[mask] = new_quat[mask]
            self._cmd_speed[mask] = new_speed[mask]
            self._cmd_dir[mask] = new_dir[mask]
            self._action[mask] = self.controller.get_action_world(self._cmd_pos, self._cmd_quat)[mask]
        else:
            self._cmd_pos.copy_(new_pos)
            self._cmd_quat.copy_(new_quat)
            self._cmd_speed.copy_(new_speed)
            self._cmd_dir.copy_(new_dir)
            self._action.copy_(self.controller.get_action_world(self._cmd_pos, self._cmd_quat))
        if self.continue_command:
            store_command(self.controller, self._cmd_pos, self._cmd_quat, self._cmd_speed, self._cmd_dir, mask)

    @torch.inference_mode()
    def get_action(self) -> torch.Tensor:
        return self._action

    @torch.inference_mode()
    def convert_action_dict(
        self, action: torch.Tensor, dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a raw action to a controller-native action dict.

        Delegates to the EEF pose controller's :meth:`get_action_dict`,
        producing keys such as ``target_joint_position``,
        ``target_eef_position``, and ``target_eef_quaternion``.
        """
        return self.controller.get_action_dict(action, dt=dt)

    def get_action_dict_space(self):
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        Delegates to the EEF pose controller's :meth:`get_action_dict_space`,
        preserving the controller-native schema.
        """
        return self.controller.get_action_dict_space()

    @torch.inference_mode()
    def get_success(self) -> Union[torch.Tensor, bool]:
        return try_scalar(self._reached)

    @torch.inference_mode()
    def get_failure(self) -> Union[torch.Tensor, bool]:
        return try_scalar(self._stalled)

    @torch.inference_mode()
    def get_timeout(self) -> Union[torch.Tensor, bool]:
        if self.max_seconds <= 0:
            return try_scalar(
                torch.zeros_like(self._reached),
            )
        timed_out = (
            (self._elapsed_seconds >= self.max_seconds - DURATION_COMPARISON_EPS)
            & ~self._reached & ~self._stalled
        )
        return try_scalar(timed_out)

    @staticmethod
    def check_reachable_error(
        eef_controller: EndEffectorPoseController,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute IK error for one or more target poses.

        Calls :meth:`EndEffectorPoseController.solve_ik_absolute` and
        returns the achieved 6-D error
        ``[pos_err_x, pos_err_y, pos_err_z, rot_err_x, rot_err_y, rot_err_z]``.

        Supports an optional *N* (candidate) dimension.  Because
        ``solve_ik_absolute`` only accepts ``([B], dim)`` inputs, this
        method loops over *N* when present.

        Args:
            eef_controller: The EEF pose controller to query.  Must have
                ``has_eef=True`` (i.e. configured with ``target_link``).
            target_pos: Target positions, shape ``([B], [N], 3)``.
                When *envs_idx* is given the batch dimension must be the
                number of selected envs ``M`` instead of ``B``.
            target_quat: Target quaternions ``(w,x,y,z)``,
                shape ``([B], [N], 4)``.  Same sub-batch convention.
            envs_idx: Optional env selection (index tensor or boolean mask)
                forwarded to :meth:`solve_ik_absolute`, restricting the IK
                solve to a sub-batch of envs.  Batched worlds only.

        Returns:
            Achieved IK error, shape ``([B], [N], 6)`` (``[M]`` with *envs_idx*).

        Raises:
            RuntimeError: If *eef_controller* does not expose EEF semantics.
        """
        require_eef_controller(
            eef_controller,
            policy_name="GoToRefPolicy.check_reachable_error",
        )
        batched = is_batched(eef_controller.world.batch_size)
        base_ndim = 2 if batched else 1  # expected ndim for ([B], dim)
        has_n = target_pos.ndim > base_ndim

        if not has_n:
            _, error = eef_controller.solve_ik_absolute(
                target_pos, target_quat, envs_idx=envs_idx,
            )
            return error

        n_candidates = target_pos.shape[-2]
        errors = []
        for i in range(n_candidates):
            if batched:
                pos_i = target_pos[:, i]    # (B, 3)
                quat_i = target_quat[:, i]  # (B, 4)
            else:
                pos_i = target_pos[i]    # (3,)
                quat_i = target_quat[i]  # (4,)
            _, error_i = eef_controller.solve_ik_absolute(
                pos_i, quat_i, envs_idx=envs_idx,
            )
            errors.append(error_i)

        # Stack: unbatched (N, 6), batched (B, N, 6).
        if batched:
            return torch.stack(errors, dim=1)
        return torch.stack(errors, dim=0)
