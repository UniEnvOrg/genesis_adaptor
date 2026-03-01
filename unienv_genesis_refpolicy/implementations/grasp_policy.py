"""BoxGripperGraspPolicy — grasp a ManipulatableObject with a parallel-jaw gripper."""

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from xbarray.transformations.rotation_conversions.pytorch import (
    matrix_to_quaternion,
    quaternion_multiply,
)

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers.end_effector_pose import (
    EndEffectorPoseController,
)
from unienv_genesis.nodes.entities.entity_controllers.base import EntityController
from unienv_genesis.utils.batch_dim_utils import (
    is_batched,
    get_batch_aware_shape,
    broadcast_to_batch,
)
from unienv_genesis_collection.objects.base_object import ManipulatableObject
from unienv_genesis_refpolicy.base.policy import BaseRefPolicy
from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy


class GraspPhase:
    OPEN_GRIPPER = 0
    PRE_GRASP = 1
    APPROACH = 2
    CLOSE_GRIPPER = 3
    DONE = 4
    _NUM_PHASES = 5


class BoxGripperGraspPolicy(BaseRefPolicy[Dict[str, torch.Tensor]]):
    """Grasp a :class:`ManipulatableObject` using a parallel-jaw gripper.

    Takes an EEF-pose controller for the arm and a separate controller
    for the gripper.  The returned action is a ``Dict[str, Tensor]``
    keyed by *eef_name* and *gripper_name*.

    Phase sequence::

        OPEN_GRIPPER → PRE_GRASP → APPROACH → CLOSE_GRIPPER → DONE

    The best grasp candidate is selected automatically using IK error:
    all candidates from :meth:`ManipulatableObject.list_box_gripper_grasping_candidates`
    are evaluated via :meth:`GoToRefPolicy.check_reachable_error` and the
    one with the lowest position error is chosen per environment.

    Args:
        world: Genesis world instance.
        eef_controller: :class:`EndEffectorPoseController` for the arm.
        gripper_controller: Controller for the gripper (any
            :class:`EntityController`).
        target_object: The object descriptor to grasp.
        target_entity: The loaded Genesis entity for *target_object*.
        gripper_width: Maximum gripper opening in metres.
        gripper_depth: Depth of the gripper fingers in metres.
        approach_distance: Pre-grasp offset along the approach direction.
        position_threshold: EEF position tolerance for phase transitions.
        orientation_threshold: Quaternion distance tolerance (``1 - |q1·q2|``).
        open_gripper_steps: Steps to hold the gripper open before moving.
        close_gripper_steps: Steps to hold the gripper closed before success.
        phase_patience: Max steps per motion phase before declaring failure.
        max_steps: Overall timeout (``0`` = infinite).
        eef_name: Key used for the arm action in the returned dict.
        gripper_name: Key used for the gripper action in the returned dict.
        gripper_open_action: Gripper action value for fully open.
        gripper_close_action: Gripper action value for fully closed.
        approach_axis: ``(3,)`` direction in the **EEF frame** that should
            align with the grasp approach direction.
        open_axis: ``(3,)`` direction in the **EEF frame** that should
            align with the gripper opening direction.
    """

    def __init__(
        self,
        world: GenesisWorld,
        eef_controller: EndEffectorPoseController,
        gripper_controller: EntityController,
        target_object: ManipulatableObject,
        target_entity,
        approach_axis: torch.Tensor,
        open_axis: torch.Tensor,
        gripper_width: float,
        gripper_depth: float = 0.01,
        approach_distance: float = 0.10,
        position_threshold: float = 0.01,
        orientation_threshold: float = 0.05,
        open_gripper_steps: int = 10,
        close_gripper_steps: int = 30,
        phase_patience: int = 50,
        max_steps: int = 300,
        eef_name: str = "arm",
        gripper_name: str = "gripper",
        gripper_open_action: float = -1.0,
        gripper_close_action: float = 1.0,
        ik_error_threshold: float = 0.05,
    ) -> None:
        super().__init__(world)

        self._eef_ctrl = eef_controller
        self._gripper_ctrl = gripper_controller
        self.target_object = target_object
        self.target_entity = target_entity
        self.gripper_width = gripper_width
        self.gripper_depth = gripper_depth
        self.approach_distance = approach_distance
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.open_gripper_steps = open_gripper_steps
        self.close_gripper_steps = close_gripper_steps
        self.phase_patience = phase_patience
        self.max_steps = max_steps
        self._eef_name = eef_name
        self._gripper_name = gripper_name
        self.gripper_open_action = gripper_open_action
        self.gripper_close_action = gripper_close_action
        self.ik_error_threshold = ik_error_threshold

        batch_size = world.batch_size
        device = world.device

        # Build the EEF-frame axis mapping.  The grasp frame convention is
        # z = approach, y = opening, x = cross(y, z).  When the EEF frame
        # uses a different convention (e.g. approach along -Z), approach_axis
        # and open_axis describe that mapping.
        z_e = F.normalize(approach_axis.to(device=device, dtype=torch.float32), dim=-1)
        y_e = open_axis.to(device=device, dtype=torch.float32)
        y_e = y_e - (y_e * z_e).sum(-1, keepdim=True) * z_e
        y_e = F.normalize(y_e, dim=-1)
        x_e = torch.cross(y_e, z_e, dim=-1)
        R_eef = torch.stack([x_e, y_e, z_e], dim=-1)  # (3, 3)
        q_eef = matrix_to_quaternion(R_eef)  # (4,) wxyz
        # Store the conjugate (= inverse for unit quat): maps grasp frame → EEF frame.
        self._axes_to_eef_quat = q_eef * torch.tensor(
            [1.0, -1.0, -1.0, -1.0], device=device,
        )

        # Allocate mutable state in native shapes.
        scalar_shape = get_batch_aware_shape(batch_size, ())
        pos_shape = get_batch_aware_shape(batch_size, 3)
        quat_shape = get_batch_aware_shape(batch_size, 4)

        self._phase = torch.zeros(scalar_shape, dtype=torch.long, device=device)
        self._step_in_phase = torch.zeros(scalar_shape, dtype=torch.long, device=device)
        self._global_step = torch.zeros(scalar_shape, dtype=torch.long, device=device)
        self._failed = torch.zeros(scalar_shape, dtype=torch.bool, device=device)

        self._target_grasp_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._target_grasp_quat = torch.zeros(quat_shape, dtype=torch.float32, device=device)
        self._pre_grasp_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._initial_eef_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._initial_eef_quat = torch.zeros(quat_shape, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        batch_size = self.world.batch_size

        # 1. Query grasp candidates (world coordinates).
        candidates = self.target_object.list_box_gripper_grasping_candidates(
            self.target_entity,
            self.gripper_width,
            gripper_depth=self.gripper_depth,
        )

        # Number of candidates: unbatched (N, 3) → shape[0], batched (B, N, 3) → shape[1].
        n_candidates = (
            candidates.position.shape[1] if is_batched(batch_size)
            else candidates.position.shape[0]
        )

        if n_candidates == 0:
            # No valid grasps — mark as failed.
            print(f"  [GraspPolicy] No grasp candidates found — marking as failed")
            self._zero_counters(mask)
            if mask is not None:
                self._failed[mask] = True
            else:
                self._failed.fill_(True)
            return

        print(f"  [GraspPolicy] Found {n_candidates} grasp candidates")

        # 2. Select the best candidate per env using IK error.
        #    Compute EEF quaternions for ALL candidates: ([B], N, 4).
        all_quats = self._build_eef_quaternion(
            candidates.approach_direction, candidates.opening_direction,
        )
        #    Evaluate IK error for each candidate: ([B], N, 6).
        ik_errors = GoToRefPolicy.check_reachable_error(
            self._eef_ctrl, candidates.position, all_quats,
        )
        #    Rank by position error norm: ([B], N).
        pos_errors = torch.norm(ik_errors[..., :3], dim=-1)
        #    Pick the candidate with the lowest error per env.
        best_idx = pos_errors.argmin(dim=-1)  # () or (B,)

        print(f"  [GraspPolicy] IK pos errors per candidate: {pos_errors.tolist()}")
        if not is_batched(batch_size):
            print(f"  [GraspPolicy] Best candidate idx={best_idx.item()}, "
                  f"pos_error={pos_errors[best_idx].item():.6f}")

        # IK quality gate: fail early if best candidate is unreachable.
        if not is_batched(batch_size):
            best_error = pos_errors[best_idx]
            if best_error.item() > self.ik_error_threshold:
                print(f"  [GraspPolicy] Best IK error {best_error.item():.4f} > "
                      f"threshold {self.ik_error_threshold} — unreachable")
                self._zero_counters(mask)
                if mask is not None:
                    self._failed[mask] = True
                else:
                    self._failed.fill_(True)
                return
        else:
            best_error = pos_errors.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
            unreachable = best_error > self.ik_error_threshold
            fail_mask = unreachable if mask is None else (mask & unreachable)
            if fail_mask.any():
                print(f"  [GraspPolicy] IK quality gate: {fail_mask.sum().item()} envs unreachable")
                self._failed = self._failed | fail_mask

        if is_batched(batch_size):
            gather_idx = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
            grasp_pos = candidates.position.gather(1, gather_idx).squeeze(1)
            approach_dir = candidates.approach_direction.gather(1, gather_idx).squeeze(1)
            opening_dir = candidates.opening_direction.gather(1, gather_idx).squeeze(1)
        else:
            idx = best_idx.item()
            grasp_pos = candidates.position[idx]
            approach_dir = candidates.approach_direction[idx]
            opening_dir = candidates.opening_direction[idx]

        # 3. EEF quaternion from selected candidate's directions.
        grasp_quat = self._build_eef_quaternion(approach_dir, opening_dir)

        # 4. Pre-grasp position.
        pre_grasp_pos = grasp_pos - self.approach_distance * approach_dir

        # 5. Capture current EEF pose.
        eef_pos, eef_quat = self._eef_ctrl.get_eef_pose_absolute()
        eef_pos = eef_pos.detach().float()
        eef_quat = eef_quat.detach().float()

        if not is_batched(batch_size):
            print(f"  [GraspPolicy] EEF current  pos={eef_pos.tolist()}")
            print(f"  [GraspPolicy] EEF current quat={eef_quat.tolist()}")
            print(f"  [GraspPolicy] Grasp target pos={grasp_pos.tolist()}")
            print(f"  [GraspPolicy] Grasp target quat={grasp_quat.tolist()}")
            print(f"  [GraspPolicy] Pre-grasp    pos={pre_grasp_pos.tolist()}")
            print(f"  [GraspPolicy] Approach dir    ={approach_dir.tolist()}")

        # 6. Store targets and zero counters.
        if mask is not None:
            self._target_grasp_pos[mask] = grasp_pos[mask]
            self._target_grasp_quat[mask] = grasp_quat[mask]
            self._pre_grasp_pos[mask] = pre_grasp_pos[mask]
            self._initial_eef_pos[mask] = eef_pos[mask]
            self._initial_eef_quat[mask] = eef_quat[mask]
        else:
            self._target_grasp_pos.copy_(grasp_pos)
            self._target_grasp_quat.copy_(grasp_quat)
            self._pre_grasp_pos.copy_(pre_grasp_pos)
            self._initial_eef_pos.copy_(eef_pos)
            self._initial_eef_quat.copy_(eef_quat)

        self._zero_counters(mask)

    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        # 1. Increment counters.
        if mask is not None:
            self._global_step[mask] += 1
            self._step_in_phase[mask] += 1
        else:
            self._global_step += 1
            self._step_in_phase += 1

        # 2. Current EEF pose.
        eef_pos, eef_quat = self._eef_ctrl.get_eef_pose_absolute()
        eef_pos = eef_pos.detach().float()
        eef_quat = eef_quat.detach().float()

        # 3. Compute target pos for the current phase (for reach checks).
        target_pos = self._get_phase_target_pos()
        target_quat = self._get_phase_target_quat()

        pos_err = torch.norm(eef_pos - target_pos, dim=-1)
        quat_dot = torch.abs(torch.sum(eef_quat * target_quat, dim=-1))
        quat_dist = 1.0 - quat_dot.clamp(max=1.0)
        reached = (pos_err < self.position_threshold) & (
            quat_dist < self.orientation_threshold
        )

        _PHASE_NAMES = ["OPEN_GRIPPER", "PRE_GRASP", "APPROACH", "CLOSE_GRIPPER", "DONE"]
        if not is_batched(self.world.batch_size):
            phase_idx = self._phase.item()
            step_in = self._step_in_phase.item()
            if step_in % 10 == 0 or reached.item():
                print(f"  [GraspPolicy] phase={_PHASE_NAMES[phase_idx]} "
                      f"step_in_phase={step_in} "
                      f"pos_err={pos_err.item():.4f} "
                      f"quat_dist={quat_dist.item():.4f} "
                      f"reached={reached.item()}")

        # 4. Phase transitions.
        #    Only one transition per step() call to avoid cascading: after a
        #    transition the ``reached`` flag is stale (computed against the old
        #    phase target), so further checks would use invalid data.
        alive = mask if mask is not None else ~self._failed

        # Track which envs already transitioned this step (batched) so we
        # don't cascade.
        transitioned = torch.zeros_like(alive)

        # OPEN_GRIPPER → PRE_GRASP  (timer-based, not reach-based)
        trans_open = (
            alive & ~transitioned
            & (self._phase == GraspPhase.OPEN_GRIPPER)
            & (self._step_in_phase >= self.open_gripper_steps)
        )
        if not is_batched(self.world.batch_size) and trans_open.item():
            print(f"  [GraspPolicy] TRANSITION: OPEN_GRIPPER -> PRE_GRASP")
        self._advance(trans_open, GraspPhase.PRE_GRASP)
        transitioned = transitioned | trans_open

        # PRE_GRASP → APPROACH
        in_pre = alive & ~transitioned & (self._phase == GraspPhase.PRE_GRASP)
        trans_pre = in_pre & reached
        if not is_batched(self.world.batch_size) and trans_pre.item():
            print(f"  [GraspPolicy] TRANSITION: PRE_GRASP -> APPROACH")
        self._advance(trans_pre, GraspPhase.APPROACH)
        transitioned = transitioned | trans_pre
        # patience failure
        fail_pre = in_pre & ~reached & (self._step_in_phase >= self.phase_patience)
        if not is_batched(self.world.batch_size) and fail_pre.item():
            print(f"  [GraspPolicy] FAILURE: PRE_GRASP patience exceeded "
                  f"(step_in_phase={self._step_in_phase.item()}, "
                  f"pos_err={pos_err.item():.4f}, quat_dist={quat_dist.item():.4f})")
            print(f"  [GraspPolicy]   eef_pos={eef_pos.tolist()}")
            print(f"  [GraspPolicy]   target ={target_pos.tolist()}")
        self._failed = self._failed | fail_pre

        # APPROACH → CLOSE_GRIPPER  (on reach OR patience timeout — try closing anyway)
        in_app = alive & ~transitioned & (self._phase == GraspPhase.APPROACH)
        timed_out_app = in_app & ~reached & (self._step_in_phase >= self.phase_patience)
        trans_app = in_app & (reached | timed_out_app)
        if not is_batched(self.world.batch_size) and trans_app.item():
            if timed_out_app.item():
                print(f"  [GraspPolicy] TRANSITION: APPROACH -> CLOSE_GRIPPER (patience timeout, "
                      f"pos_err={pos_err.item():.4f} — will verify grasp)")
            else:
                print(f"  [GraspPolicy] TRANSITION: APPROACH -> CLOSE_GRIPPER")
        self._advance(trans_app, GraspPhase.CLOSE_GRIPPER)
        transitioned = transitioned | trans_app

        # CLOSE_GRIPPER → DONE (with optional grasp verification)
        trans_close = (
            alive & ~transitioned
            & (self._phase == GraspPhase.CLOSE_GRIPPER)
            & (self._step_in_phase >= self.close_gripper_steps)
        )
        if trans_close.any():
            grasping_result = self._eef_ctrl.is_grasping(self.target_entity)
            if grasping_result is not None:
                # Grasp verification available — succeed only if contact detected.
                if not is_batched(self.world.batch_size):
                    if grasping_result:
                        print(f"  [GraspPolicy] TRANSITION: CLOSE_GRIPPER -> DONE (grasp verified)")
                        self._advance(trans_close, GraspPhase.DONE)
                    else:
                        print(f"  [GraspPolicy] FAILURE: CLOSE_GRIPPER grasp verification failed")
                        self._failed = self._failed | trans_close
                else:
                    grasped = _to_bool_tensor(grasping_result, trans_close)
                    self._advance(trans_close & grasped, GraspPhase.DONE)
                    self._failed = self._failed | (trans_close & ~grasped)
            else:
                # No verification — backwards compatible.
                if not is_batched(self.world.batch_size) and trans_close.item():
                    print(f"  [GraspPolicy] TRANSITION: CLOSE_GRIPPER -> DONE (no verification)")
                self._advance(trans_close, GraspPhase.DONE)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_action(
        self, observation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_pos = self._get_phase_target_pos()
        target_quat = self._get_phase_target_quat()

        arm_action = self._eef_ctrl.get_action_world(target_pos, target_quat)

        # Gripper: open for phases 0–2, close for phases 3–4.
        if is_batched(self.world.batch_size):
            gripper_val = torch.where(
                self._phase >= GraspPhase.CLOSE_GRIPPER,
                self.gripper_close_action,
                self.gripper_open_action,
            ).unsqueeze(-1)  # (B, 1)
        else:
            if self._phase.item() >= GraspPhase.CLOSE_GRIPPER:
                gripper_val = torch.tensor(
                    [self.gripper_close_action],
                    device=self.world.device,
                    dtype=torch.float32,
                )
            else:
                gripper_val = torch.tensor(
                    [self.gripper_open_action],
                    device=self.world.device,
                    dtype=torch.float32,
                )

        return {self._eef_name: arm_action, self._gripper_name: gripper_val}

    def get_success(self) -> Union[torch.Tensor, bool]:
        return self._unwrap(self._phase == GraspPhase.DONE)

    def get_failure(self) -> Union[torch.Tensor, bool]:
        return self._unwrap(self._failed)

    def get_timeout(self) -> Union[torch.Tensor, bool]:
        if self.max_steps <= 0:
            return self._unwrap(torch.zeros_like(self._failed))
        timed_out = (
            (self._global_step >= self.max_steps)
            & (self._phase != GraspPhase.DONE)
            & ~self._failed
        )
        return self._unwrap(timed_out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_eef_quaternion(
        self,
        approach_dir: torch.Tensor,
        opening_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Construct EEF quaternion from grasp approach/opening directions.

        Builds a grasp-frame rotation (z = approach, y = opening, x = y × z)
        then applies the ``approach_axis`` / ``open_axis`` mapping to obtain
        the final EEF orientation.
        """
        z = F.normalize(approach_dir, dim=-1)
        # Orthogonalise opening_dir w.r.t. z.
        y = opening_dir - (opening_dir * z).sum(-1, keepdim=True) * z
        y = F.normalize(y, dim=-1)
        x = torch.cross(y, z, dim=-1)
        R = torch.stack([x, y, z], dim=-1)  # (..., 3, 3)
        quat = matrix_to_quaternion(R)       # (..., 4) wxyz

        # Map from grasp frame to EEF frame via the axis mapping.
        quat = quaternion_multiply(quat, self._axes_to_eef_quat)
        return quat

    def _get_phase_target_pos(self) -> torch.Tensor:
        """Return the arm target position for the current phase."""
        if is_batched(self.world.batch_size):
            all_pos = torch.stack([
                self._initial_eef_pos,   # OPEN_GRIPPER
                self._pre_grasp_pos,     # PRE_GRASP
                self._target_grasp_pos,  # APPROACH
                self._target_grasp_pos,  # CLOSE_GRIPPER
                self._target_grasp_pos,  # DONE
            ], dim=1)  # (B, 5, 3)
            idx = self._phase.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)  # (B, 1, 3)
            return all_pos.gather(1, idx).squeeze(1)  # (B, 3)
        # Unbatched: simple index.
        phase = self._phase.item()
        if phase == GraspPhase.OPEN_GRIPPER:
            return self._initial_eef_pos
        if phase == GraspPhase.PRE_GRASP:
            return self._pre_grasp_pos
        return self._target_grasp_pos

    def _get_phase_target_quat(self) -> torch.Tensor:
        """Return the arm target quaternion for the current phase."""
        if is_batched(self.world.batch_size):
            all_quat = torch.stack([
                self._initial_eef_quat,    # OPEN_GRIPPER
                self._target_grasp_quat,   # PRE_GRASP
                self._target_grasp_quat,   # APPROACH
                self._target_grasp_quat,   # CLOSE_GRIPPER
                self._target_grasp_quat,   # DONE
            ], dim=1)  # (B, 5, 4)
            idx = self._phase.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 4)  # (B, 1, 4)
            return all_quat.gather(1, idx).squeeze(1)  # (B, 4)
        phase = self._phase.item()
        if phase == GraspPhase.OPEN_GRIPPER:
            return self._initial_eef_quat
        return self._target_grasp_quat

    def _advance(self, trans_mask: torch.Tensor, next_phase: int) -> None:
        """Advance envs in *trans_mask* to *next_phase*, zeroing step counter."""
        if not trans_mask.any():
            return
        if is_batched(self.world.batch_size):
            self._phase[trans_mask] = next_phase
            self._step_in_phase[trans_mask] = 0
        else:
            if trans_mask.item():
                self._phase.fill_(next_phase)
                self._step_in_phase.zero_()

    def _zero_counters(self, mask: Optional[torch.Tensor]) -> None:
        if mask is not None:
            self._phase[mask] = 0
            self._step_in_phase[mask] = 0
            self._global_step[mask] = 0
            self._failed[mask] = False
        else:
            self._phase.zero_()
            self._step_in_phase.zero_()
            self._global_step.zero_()
            self._failed.zero_()

    def _unwrap(self, t: torch.Tensor) -> Union[torch.Tensor, bool]:
        if is_batched(self.world.batch_size):
            return t
        return bool(t.item())


def _to_bool_tensor(value: Union[torch.Tensor, bool], like: torch.Tensor) -> torch.Tensor:
    """Coerce a bool or Tensor to a bool Tensor matching *like*'s shape/device."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.full_like(like, value, dtype=torch.bool)
