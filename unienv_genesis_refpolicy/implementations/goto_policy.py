"""GoToRefPolicy — move an end-effector to a target world-frame pose."""

from typing import Optional, Union

import torch

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers.end_effector_pose import (
    EndEffectorPoseController,
)
from unienv_genesis.utils.batch_dim_utils import (
    is_batched,
    get_batch_aware_shape,
    broadcast_to_batch,
)
from unienv_genesis_refpolicy.base.policy import BaseRefPolicy


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
        max_steps: Maximum steps before timeout.  ``0`` means infinite
            (no timeout).
        stall_steps: Number of consecutive steps with motion below
            *stall_threshold* before declaring failure.  ``0`` disables
            stall detection.
        stall_threshold: Motion magnitude (metres) below which a step
            counts as stalled.
    """

    def __init__(
        self,
        world: GenesisWorld,
        controller: EndEffectorPoseController,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        position_threshold: float = 0.01,
        orientation_threshold: float = 0.05,
        max_steps: int = 200,
        stall_steps: int = 20,
        stall_threshold: float = 1e-4,
    ) -> None:
        super().__init__(world)
        self.controller = controller
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.max_steps = max_steps
        self.stall_steps = stall_steps
        self.stall_threshold = stall_threshold

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

        self._step_count = torch.zeros(scalar_shape, dtype=torch.long, device=device)
        self._reached = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._stall_count = torch.zeros(scalar_shape, dtype=torch.long, device=device)
        self._stalled = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._last_eef_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)

        # Initialise _last_eef_pos from the current EEF position.
        eef_pos, _ = controller.get_eef_pose_absolute()
        self._last_eef_pos.copy_(eef_pos.detach().float())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        if mask is not None:
            self._step_count[mask] = 0
            self._reached[mask] = False
            self._stall_count[mask] = 0
            self._stalled[mask] = False
            eef_pos, _ = self.controller.get_eef_pose_absolute()
            self._last_eef_pos[mask] = eef_pos.detach().float()[mask]
        else:
            self._step_count.zero_()
            self._reached.zero_()
            self._stall_count.zero_()
            self._stalled.zero_()
            eef_pos, _ = self.controller.get_eef_pose_absolute()
            self._last_eef_pos.copy_(eef_pos.detach().float())

    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        # Increment step counter.
        if mask is not None:
            self._step_count[mask] += 1
        else:
            self._step_count += 1

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
        if self.stall_steps > 0:
            motion = torch.norm(eef_pos - self._last_eef_pos, dim=-1)
            new_stall = torch.where(
                motion < self.stall_threshold,
                self._stall_count + 1,
                torch.zeros_like(self._stall_count),
            )
            if mask is not None:
                self._stall_count[mask] = new_stall[mask]
            else:
                self._stall_count = new_stall

            stall_trigger = (self._stall_count >= self.stall_steps) & ~self._reached
            if mask is not None:
                stall_trigger = stall_trigger & mask
            self._stalled = self._stalled | stall_trigger

        if mask is not None:
            self._last_eef_pos[mask] = eef_pos[mask]
        else:
            self._last_eef_pos.copy_(eef_pos)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self._action

    def get_success(self) -> Union[torch.Tensor, bool]:
        return self._unwrap(self._reached)

    def get_failure(self) -> Union[torch.Tensor, bool]:
        return self._unwrap(self._stalled)

    def get_timeout(self) -> Union[torch.Tensor, bool]:
        if self.max_steps <= 0:
            return self._unwrap(
                torch.zeros_like(self._reached),
            )
        timed_out = (
            (self._step_count >= self.max_steps) & ~self._reached & ~self._stalled
        )
        return self._unwrap(timed_out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            eef_controller: The EEF pose controller to query.
            target_pos: Target positions, shape ``([B], [N], 3)``.
            target_quat: Target quaternions ``(w,x,y,z)``,
                shape ``([B], [N], 4)``.

        Returns:
            Achieved IK error, shape ``([B], [N], 6)``.
        """
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

    def _unwrap(self, t: torch.Tensor) -> Union[torch.Tensor, bool]:
        """Return a Python bool for unbatched worlds, else the tensor."""
        if is_batched(self.world.batch_size):
            return t
        return bool(t.item())
