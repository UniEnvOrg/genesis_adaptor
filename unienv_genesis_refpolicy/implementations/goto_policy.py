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
from unienv_genesis_refpolicy.base.policy import BaseRefPolicy, require_eef_controller

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

    Notes:
        Time accounting is based on :attr:`GenesisWorld.world_timestep`
        added once per :meth:`step` call.
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

        # Pre-compute the action (constant for the lifetime of the policy).
        self._action = controller.get_action_world(
            self.target_pos, self.target_quat,
        )

        # Mutable per-env state — native shapes via get_batch_aware_shape.
        scalar_shape = get_batch_aware_shape(batch_size, ())  # () or (B,)
        pos_shape = get_batch_aware_shape(batch_size, 3)       # (3,) or (B, 3)
        quat_shape = get_batch_aware_shape(batch_size, 4)      # (4,) or (B, 4)

        self._elapsed_seconds = torch.zeros(
            scalar_shape, dtype=torch.float32, device=device,
        )
        self._reached = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._stall_elapsed_seconds = torch.zeros(
            scalar_shape, dtype=torch.float32, device=device,
        )
        self._stalled = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._last_eef_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._last_eef_quat = torch.zeros(quat_shape, dtype=torch.float32, device=device)

        # Initialise last EEF pose from the current EEF pose.
        eef_pos, eef_quat = controller.get_eef_pose_absolute()
        self._last_eef_pos.copy_(eef_pos.detach().float())
        self._last_eef_quat.copy_(eef_quat.detach().float())

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        if mask is not None:
            self._elapsed_seconds[mask] = 0.0
            self._reached[mask] = False
            self._stall_elapsed_seconds[mask] = 0.0
            self._stalled[mask] = False
            eef_pos, eef_quat = self.controller.get_eef_pose_absolute()
            self._last_eef_pos[mask] = eef_pos.detach().float()[mask]
            self._last_eef_quat[mask] = eef_quat.detach().float()[mask]
        else:
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
    ) -> None:
        dt = float(self.world.world_timestep)

        # Increment elapsed time.
        if mask is not None:
            self._elapsed_seconds[mask] += dt
        else:
            self._elapsed_seconds += dt

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
                    self._stall_elapsed_seconds[mask] + dt,
                    torch.zeros_like(self._stall_elapsed_seconds[mask]),
                )
                self._stall_elapsed_seconds[mask] = updated
            else:
                self._stall_elapsed_seconds = torch.where(
                    stalled_this_step,
                    self._stall_elapsed_seconds + dt,
                    torch.zeros_like(self._stall_elapsed_seconds),
                )

            stall_trigger = (
                (self._stall_elapsed_seconds >= self.stall_seconds) & ~self._reached
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

    @torch.inference_mode()
    def get_action(self) -> torch.Tensor:
        return self._action

    @torch.inference_mode()
    def get_action_dict(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return a controller-native action dict for the current action.

        Delegates to the EEF pose controller's :meth:`get_action_dict`,
        producing keys such as ``target_joint_position``,
        ``target_eef_position``, and ``target_eef_quaternion``.
        """
        return self.controller.get_action_dict(self._action, dt=dt)

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
            (self._elapsed_seconds >= self.max_seconds) & ~self._reached & ~self._stalled
        )
        return try_scalar(timed_out)

    @staticmethod
    def check_reachable_error(
        eef_controller: EndEffectorPoseController,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
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
            target_quat: Target quaternions ``(w,x,y,z)``,
                shape ``([B], [N], 4)``.

        Returns:
            Achieved IK error, shape ``([B], [N], 6)``.

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
            _, error = eef_controller.solve_ik_absolute(target_pos, target_quat)
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
            _, error_i = eef_controller.solve_ik_absolute(pos_i, quat_i)
            errors.append(error_i)

        # Stack: unbatched (N, 6), batched (B, N, 6).
        if batched:
            return torch.stack(errors, dim=1)
        return torch.stack(errors, dim=0)
