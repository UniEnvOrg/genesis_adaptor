"""BoxGripperGraspPolicy — grasp a ManipulatableObject with a parallel-jaw gripper."""

from typing import Any, Callable, Dict, Optional, Union
import json
from pathlib import Path

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
    try_scalar
)
from unienv_genesis_collection.objects.base_object import ManipulatableObject, GraspCandidates
from unienv_genesis_refpolicy.base.policy import (
    BaseRefPolicy,
    require_eef_controller,
    DURATION_COMPARISON_EPS,
)
from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider
from unienv_genesis_refpolicy.grasp_providers.object_geometry import (
    ObjectGeometryGraspCandidateProvider,
)
from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy, step_pose_toward

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
    one with the lowest combined position + orientation error is chosen
    per environment.

    Args:
        world: Genesis world instance.
        eef_controller: :class:`EndEffectorPoseController` for the arm.
        gripper_controller: Controller for the gripper (any
            :class:`EntityController`).
        target_object: The object descriptor to grasp.
        target_entity: The loaded Genesis entity for *target_object*.  May be
            ``None`` when ``target_entity_fn`` is given.
        target_entity_fn: Optional zero-argument callable returning the current
            (single, batched) entity.  When given it is re-read at every
            :meth:`reset`, so the same policy instance can be re-targeted at a
            newly-loaded entity after a scene reload instead of being rebuilt.
            Takes precedence over ``target_entity``.
        gripper_width: Maximum gripper opening in metres.
        gripper_depth: Depth of the gripper fingers in metres.
        gripper_open_action: Gripper action value for fully open.
        gripper_close_action: Gripper action value for fully closed.
        approach_distance: Pre-grasp offset along the approach direction.
        position_threshold: EEF position tolerance for phase transitions.
        orientation_threshold: Quaternion distance tolerance (``1 - |q1·q2|``).
        open_gripper_seconds: Seconds to hold the gripper open before moving.
        close_gripper_seconds: Seconds to hold the gripper closed before success.
        phase_patience_seconds: Max seconds per motion phase before declaring failure.
        max_seconds: Overall timeout in seconds (``0`` = infinite).
        eef_name: Key used for the arm action in the returned dict.
        gripper_name: Key used for the gripper action in the returned dict.
        approach_axis: ``(3,)`` direction in the **EEF frame** that should
            align with the grasp approach direction. If ``None``, read from
            ``gripper_controller.approach_direction``.
        open_axis: ``(3,)`` direction in the **EEF frame** that should
            align with the gripper opening direction. If ``None``, read from
            ``gripper_controller.open_direction``.
        gripper_width: Maximum opening width in metres. If ``None``, read from
            ``gripper_controller.gripper_width``.
        gripper_depth: Finger depth in metres. If ``None``, read from
            ``gripper_controller.gripper_depth``.
        filter_candidate_fn: Optional callable
            ``(GraspCandidates) -> BoolTensor`` that returns a mask over the
            candidate dimension (``True`` = keep).  Applied in world
            coordinates before IK evaluation.  Useful to restrict grasps
            to top-down only, avoid certain orientations, etc.  When
            ``None``, the candidate provider's
            :meth:`~unienv_genesis_refpolicy.grasp_providers.base.GraspCandidateProvider.filter`
            is used.
        max_linear_speed / max_angular_speed: Opt-in Cartesian interpolation
            of the PRE_GRASP and APPROACH motions (m/s, rad/s), as
            :class:`GoToRefPolicy`'s: the commanded pose starts at the EEF pose
            on :meth:`reset` and advances toward the phase target along the
            straight line, so the fingers descend straight onto the grasp
            instead of following the arm's joint-space arc (which can land a
            finger on top of a thin object). Reach checks still use the actual
            EEF pose against the phase target. ``None`` (default) commands the
            phase target directly.

    The provider's
    :meth:`~unienv_genesis_refpolicy.grasp_providers.base.GraspCandidateProvider.cost`,
    when it returns one, orders the *reachable* candidates (IK error within
    ``ik_error_threshold``) by IK error + cost; it never makes an unreachable
    candidate win, and the IK gate still sees the IK error alone.  Otherwise
    the reachable candidate with the lowest IK error is chosen.

    With ``selection_temperature`` set (and a provider cost), the grasp is
    *sampled* instead of taking the best one — e.g. for data collection, so a
    poor best grasp is not repeated every episode: each candidate is weighted
    ``exp(-(score - best) / selection_temperature)`` with ``score`` = IK error
    + cost, filtered-out and unreachable candidates getting zero weight.  The
    draw uses the policy's generator, reseeded by :meth:`reset`'s ``seed``.

    Notes:
        Time accounting (``open_gripper_seconds``, ``close_gripper_seconds``,
        ``phase_patience_seconds``, ``max_seconds``) always uses the explicit
        ``dt`` passed to :meth:`step` (the control/policy timestep in
        seconds).
    """

    def __init__(
        self,
        world: GenesisWorld,
        eef_controller: EndEffectorPoseController,
        gripper_controller: EntityController,
        target_object: ManipulatableObject,
        target_entity,
        approach_axis: Optional[torch.Tensor] = None,
        open_axis: Optional[torch.Tensor] = None,
        gripper_width: Optional[float] = None,
        gripper_depth: Optional[float] = None,
        gripper_open_action: Optional[float] = None,
        gripper_close_action: Optional[float] = None,
        approach_distance: float = 0.10,
        position_threshold: float = 0.01,
        orientation_threshold: float = 0.05,
        open_gripper_seconds: float = 1.0,
        close_gripper_seconds: float = 1.0,
        phase_patience_seconds: float = 3.0,
        max_seconds: float = 10.0,
        eef_name: str = "arm",
        gripper_name: str = "gripper",
        ik_error_threshold: float = 0.05,
        filter_candidate_fn : Optional[Callable[[GraspCandidates], torch.Tensor]] = None,
        candidate_provider: Optional[GraspCandidateProvider] = None,
        target_entity_fn: Optional[Callable[[], Any]] = None,
        selection_temperature: Optional[float] = None,
        max_linear_speed: Optional[float] = None,
        max_angular_speed: Optional[float] = None,
    ) -> None:
        super().__init__(world)

        require_eef_controller(eef_controller, policy_name="BoxGripperGraspPolicy")
        self._eef_ctrl = eef_controller
        self._gripper_ctrl = gripper_controller
        self.target_object = target_object
        self.target_entity = target_entity
        self.target_entity_fn = target_entity_fn
        self.approach_distance = approach_distance
        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.open_gripper_seconds = open_gripper_seconds
        self.close_gripper_seconds = close_gripper_seconds
        self.phase_patience_seconds = phase_patience_seconds
        self.max_seconds = max_seconds
        self._eef_name = eef_name
        self._gripper_name = gripper_name
        self.ik_error_threshold = ik_error_threshold
        self.selection_temperature = selection_temperature
        self.max_linear_speed = None if max_linear_speed is None else float(max_linear_speed)
        self.max_angular_speed = None if max_angular_speed is None else float(max_angular_speed)
        self._interpolate = self.max_linear_speed is not None or self.max_angular_speed is not None
        self._selection_rng = torch.Generator(device=world.device)
        self._filter_candidate_fn = filter_candidate_fn

        # Resolve optional parameters from gripper controller attributes.
        if gripper_width is None:
            gripper_width = gripper_controller.gripper_width
        if gripper_depth is None:
            gripper_depth = gripper_controller.gripper_depth
        if approach_axis is None:
            approach_axis = torch.tensor(gripper_controller.approach_direction, dtype=torch.float32)
        if open_axis is None:
            open_axis = torch.tensor(gripper_controller.open_direction, dtype=torch.float32)
        if gripper_open_action is None:
            gripper_open_action = gripper_controller.gripper_open_action
        if gripper_close_action is None:
            gripper_close_action = gripper_controller.gripper_close_action

        self.gripper_width = gripper_width
        self.gripper_depth = gripper_depth
        self.gripper_open_action = gripper_open_action
        self.gripper_close_action = gripper_close_action
        # The default provider shares the policy's entity source, so a
        # rebindable policy also re-resolves the entity inside propose().  A
        # caller-supplied provider is responsible for its own binding.
        self._candidate_provider = candidate_provider or ObjectGeometryGraspCandidateProvider(
            target_object=target_object,
            target_entity=target_entity,
            target_entity_fn=target_entity_fn,
            gripper_width=gripper_width,
            gripper_depth=gripper_depth,
        )

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
        # Elapsed-time counters use float64: float32 accumulation error at
        # common control rates (e.g. 60 Hz / 100 Hz) can leave the counter
        # just below the threshold at the exact boundary step, delaying
        # phase transitions/timeouts by one full step.
        self._phase_elapsed_seconds = torch.zeros(
            scalar_shape, dtype=torch.float64, device=device,
        )
        self._total_elapsed_seconds = torch.zeros(
            scalar_shape, dtype=torch.float64, device=device,
        )
        self._failed = torch.zeros(scalar_shape, dtype=torch.bool, device=device)

        self._target_grasp_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._target_grasp_quat = torch.zeros(quat_shape, dtype=torch.float32, device=device)
        self._pre_grasp_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._initial_eef_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._initial_eef_quat = torch.zeros(quat_shape, dtype=torch.float32, device=device)
        # Commanded (interpolated) pose; only used with Cartesian interpolation.
        self._cmd_pos = torch.zeros(pos_shape, dtype=torch.float32, device=device)
        self._cmd_quat = torch.zeros(quat_shape, dtype=torch.float32, device=device)
        self._debug_last = {}

    def save_debug(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {}
        arrays = {}
        for key, value in self._debug_last.items():
            if isinstance(value, torch.Tensor):
                arrays[key] = value.detach().to("cpu").numpy()
            else:
                meta[key] = value
        if arrays:
            import numpy as np

            np.savez(output_dir / "grasp_policy_debug.npz", **arrays)
        if meta:
            (output_dir / "grasp_policy_debug.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        batch_size = self.world.batch_size
        batched = is_batched(batch_size)
        # Batched partial reset: compress candidate selection + IK to the
        # masked sub-batch so the cost scales with the number of resetting
        # envs, not the full batch.
        use_sub = batched and mask is not None

        # 0. Re-resolve the target entity from its (possibly rebindable) source.
        #    A scene reload swaps the entity id, so reading the callable here
        #    re-targets the policy — and step()'s ``is_grasping(...)`` check
        #    below — without rebuilding the policy.
        self.target_entity = (
            self.target_entity_fn()
            if self.target_entity_fn is not None
            else self.target_entity
        )

        if seed is not None:
            self._selection_rng.manual_seed(int(seed))

        # 1. Query grasp candidates (world coordinates).
        candidates = self._candidate_provider.propose(
            mask=mask,
            seed=seed,
        )
        self._debug_last = {
            "raw_candidate_count": int(
                candidates.position.shape[1] if batched else candidates.position.shape[0]
            ),
            "raw_candidate_position": candidates.position.clone(),
            "raw_candidate_approach": candidates.approach_direction.clone(),
            "raw_candidate_opening": candidates.opening_direction.clone(),
        }

        # 1b. Apply optional filter to remove unwanted candidates.
        filter_fn = self._filter_candidate_fn or self._candidate_provider.filter
        keep: Optional[torch.Tensor] = filter_fn(candidates)
        if keep is not None:
            self._debug_last["filter_keep_mask"] = keep.clone() if isinstance(keep, torch.Tensor) else keep
            if not batched:
                # keep: (N,) bool → index into candidate dim
                idx = torch.where(keep)[0]
                candidates = GraspCandidates(
                    position=candidates.position[idx],
                    approach_direction=candidates.approach_direction[idx],
                    opening_direction=candidates.opening_direction[idx],
                )
                keep = None
            # Batched filtering: mask out rejected candidates by setting
            # their error to inf later (per-env candidate sets differ).
        self._debug_last["post_filter_candidate_count"] = int(
            candidates.position.shape[1] if batched else candidates.position.shape[0]
        )
        self._debug_last["post_filter_candidate_position"] = candidates.position.clone()
        self._debug_last["post_filter_candidate_approach"] = candidates.approach_direction.clone()
        self._debug_last["post_filter_candidate_opening"] = candidates.opening_direction.clone()

        # Number of candidates: unbatched (N, 3) → shape[0], batched (B, N, 3) → shape[1].
        n_candidates = (
            candidates.position.shape[1] if batched
            else candidates.position.shape[0]
        )

        if n_candidates == 0:
            # No valid grasps — mark as failed.
            self._zero_counters(mask)
            self._debug_last["status"] = "no_candidates"
            if mask is not None:
                self._failed[mask] = True
            else:
                self._failed.fill_(True)
            return

        # 1c. Optional grasp-quality cost over the (filtered) candidates.
        cost: Optional[torch.Tensor] = self._candidate_provider.cost(candidates)
        if cost is not None:
            self._debug_last["candidate_cost"] = cost.clone()

        if use_sub:
            cand_pos = candidates.position[mask]                # (M, N, 3)
            cand_approach = candidates.approach_direction[mask]  # (M, N, 3)
            cand_opening = candidates.opening_direction[mask]    # (M, N, 3)
            if keep is not None:
                keep = keep[mask]                                # (M, N)
            if cost is not None:
                cost = cost[mask]                                # (M, N)
            envs_idx = mask
        else:
            cand_pos = candidates.position
            cand_approach = candidates.approach_direction
            cand_opening = candidates.opening_direction
            envs_idx = None

        # 2. Select the best candidate per (resetting) env using IK error.
        #    Compute EEF quaternions for ALL candidates: ([M|B], N, 4).
        all_quats = self._build_eef_quaternion(cand_approach, cand_opening)
        #    Evaluate IK error for each candidate: ([M|B], N, 6).
        ik_errors = GoToRefPolicy.check_reachable_error(
            self._eef_ctrl, cand_pos, all_quats, envs_idx=envs_idx,
        )
        #    Rank by combined position + orientation error: ([M|B], N).
        pos_errors = torch.norm(ik_errors[..., :3], dim=-1)
        rot_errors = torch.norm(ik_errors[..., 3:], dim=-1)
        combined_errors = pos_errors + rot_errors
        self._debug_last["combined_errors"] = combined_errors.clone()
        #    Apply candidate filter mask (batched path): set rejected
        #    candidates to inf so they are never selected.
        if keep is not None:
            combined_errors[~keep] = float('inf')
        #    Pick the candidate with the lowest combined error per env.  A
        #    quality cost only orders the reachable candidates: unreachable
        #    ones keep their raw error behind a large offset, so they still
        #    win only when nothing is reachable (and then fail the gate).
        ranking = combined_errors
        if cost is not None:
            cost = cost.to(device=combined_errors.device, dtype=combined_errors.dtype)
            reachable = combined_errors <= self.ik_error_threshold
            ranking = torch.where(
                reachable, combined_errors + cost, combined_errors + 1e6,
            )
        best_idx = ranking.argmin(dim=-1)  # () or (M|B,)
        if cost is not None and self.selection_temperature:
            best_idx = self._sample_candidate(ranking, best_idx)
        self._debug_last["best_idx"] = best_idx.clone() if isinstance(best_idx, torch.Tensor) else int(best_idx)

        # IK quality gate: fail if the best candidate is unreachable.  The
        # flag is applied AFTER _zero_counters below so it isn't wiped.
        unreachable: Optional[torch.Tensor] = None
        if not batched:
            best_error = combined_errors[best_idx]
            self._debug_last["best_error"] = float(best_error.item())
            if best_error.item() > self.ik_error_threshold:
                self._zero_counters(mask)
                self._debug_last["status"] = "ik_unreachable"
                if mask is not None:
                    self._failed[mask] = True
                else:
                    self._failed.fill_(True)
                return
        else:
            best_error = combined_errors.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
            unreachable = best_error > self.ik_error_threshold  # (M|B,)
            self._debug_last["best_error"] = best_error.clone()

        if batched:
            gather_idx = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
            grasp_pos = cand_pos.gather(1, gather_idx).squeeze(1)
            approach_dir = cand_approach.gather(1, gather_idx).squeeze(1)
            opening_dir = cand_opening.gather(1, gather_idx).squeeze(1)
        else:
            idx = best_idx.item()
            grasp_pos = cand_pos[idx]
            approach_dir = cand_approach[idx]
            opening_dir = cand_opening[idx]
        self._debug_last["selected_grasp_pos"] = grasp_pos.clone()
        self._debug_last["selected_approach_dir"] = approach_dir.clone()
        self._debug_last["selected_opening_dir"] = opening_dir.clone()

        # 3. EEF quaternion from selected candidate's directions.
        grasp_quat = self._build_eef_quaternion(approach_dir, opening_dir)
        self._debug_last["selected_grasp_quat"] = grasp_quat.clone()

        # 4. Pre-grasp position.
        pre_grasp_pos = grasp_pos - self.approach_distance * approach_dir
        self._debug_last["selected_pre_grasp_pos"] = pre_grasp_pos.clone()

        # 5. Capture current EEF pose.
        eef_pos, eef_quat = self._eef_ctrl.get_eef_pose_absolute()
        eef_pos = eef_pos.detach().float()
        eef_quat = eef_quat.detach().float()

        # 6. Store targets and zero counters.  Index assignment does not cast
        # like ``copy_`` does, and the candidate providers may produce float64.
        grasp_pos = grasp_pos.to(self._target_grasp_pos.dtype)
        grasp_quat = grasp_quat.to(self._target_grasp_quat.dtype)
        pre_grasp_pos = pre_grasp_pos.to(self._pre_grasp_pos.dtype)
        if use_sub:
            # Sub-batch tensors are already compressed to the masked envs.
            self._target_grasp_pos[mask] = grasp_pos
            self._target_grasp_quat[mask] = grasp_quat
            self._pre_grasp_pos[mask] = pre_grasp_pos
            self._initial_eef_pos[mask] = eef_pos[mask]
            self._initial_eef_quat[mask] = eef_quat[mask]
        elif mask is not None:
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

        # The straight-line motion starts from where the EEF is now.
        if mask is not None:
            self._cmd_pos[mask] = self._initial_eef_pos[mask]
            self._cmd_quat[mask] = self._initial_eef_quat[mask]
        else:
            self._cmd_pos.copy_(self._initial_eef_pos)
            self._cmd_quat.copy_(self._initial_eef_quat)

        self._zero_counters(mask)
        self._debug_last["status"] = "ok"

        # Apply the IK quality gate after counters were zeroed.
        if unreachable is not None:
            if use_sub:
                self._failed[mask] = unreachable
            else:
                self._failed.copy_(unreachable)

    @torch.inference_mode()
    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        dt: float,
    ) -> None:
        step_dt = float(dt)

        # 1. Increment elapsed-time counters.
        if mask is not None:
            self._total_elapsed_seconds[mask] += step_dt
            self._phase_elapsed_seconds[mask] += step_dt
        else:
            self._total_elapsed_seconds += step_dt
            self._phase_elapsed_seconds += step_dt

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

        # 4. Phase transitions.
        #    Only one transition per step() call to avoid cascading: after a
        #    transition the ``reached`` flag is stale (computed against the old
        #    phase target), so further checks would use invalid data.
        #    The small epsilon in every duration comparison guards against
        #    float accumulation error when summing dt repeatedly (e.g.
        #    100 * 0.01 = 0.9999999999999999 < 1.0 in float64); it is far
        #    below any realistic control timestep.
        eps = DURATION_COMPARISON_EPS
        alive = mask if mask is not None else ~self._failed

        # Track which envs already transitioned this step (batched) so we
        # don't cascade.
        transitioned = torch.zeros_like(alive)

        # OPEN_GRIPPER → PRE_GRASP  (timer-based, not reach-based)
        trans_open = (
            alive & ~transitioned
            & (self._phase == GraspPhase.OPEN_GRIPPER)
            & (self._phase_elapsed_seconds >= self.open_gripper_seconds - eps)
        )
        self._advance(trans_open, GraspPhase.PRE_GRASP)
        transitioned = transitioned | trans_open

        # PRE_GRASP → APPROACH
        in_pre = alive & ~transitioned & (self._phase == GraspPhase.PRE_GRASP)
        trans_pre = in_pre & reached
        self._advance(trans_pre, GraspPhase.APPROACH)
        transitioned = transitioned | trans_pre
        # patience failure
        fail_pre = in_pre & ~reached & (
            self._phase_elapsed_seconds >= self.phase_patience_seconds - eps
        )
        self._failed = self._failed | fail_pre

        # APPROACH → CLOSE_GRIPPER  (on reach OR patience timeout — try closing anyway)
        in_app = alive & ~transitioned & (self._phase == GraspPhase.APPROACH)
        timed_out_app = in_app & ~reached & (
            self._phase_elapsed_seconds >= self.phase_patience_seconds - eps
        )
        trans_app = in_app & (reached | timed_out_app)
        self._advance(trans_app, GraspPhase.CLOSE_GRIPPER)
        transitioned = transitioned | trans_app

        # CLOSE_GRIPPER → DONE (with optional grasp verification)
        trans_close = (
            alive & ~transitioned
            & (self._phase == GraspPhase.CLOSE_GRIPPER)
            & (self._phase_elapsed_seconds >= self.close_gripper_seconds - eps)
        )
        if trans_close.any():
            grasping_result = self._gripper_ctrl.is_grasping(self.target_entity)
            if grasping_result is not None:
                # Grasp verification available — succeed only if contact detected.
                if not is_batched(self.world.batch_size):
                    if grasping_result:
                        self._advance(trans_close, GraspPhase.DONE)
                    else:
                        self._failed = self._failed | trans_close
                else:
                    grasped = _to_bool_tensor(grasping_result, trans_close)
                    self._advance(trans_close & grasped, GraspPhase.DONE)
                    self._failed = self._failed | (trans_close & ~grasped)
            else:
                # No verification — backwards compatible.
                self._advance(trans_close, GraspPhase.DONE)

        # 5. Advance the commanded pose toward the (possibly new) phase target.
        if self._interpolate:
            new_pos, new_quat = step_pose_toward(
                self._cmd_pos, self._cmd_quat,
                self._get_phase_target_pos(), self._get_phase_target_quat(),
                step_dt, self.max_linear_speed, self.max_angular_speed,
            )
            if mask is not None:
                self._cmd_pos[mask] = new_pos[mask]
                self._cmd_quat[mask] = new_quat[mask]
            else:
                self._cmd_pos.copy_(new_pos)
                self._cmd_quat.copy_(new_quat)

    @torch.inference_mode()
    def get_action(
        self, 
    ) -> Dict[str, torch.Tensor]:
        if self._interpolate:
            target_pos, target_quat = self._cmd_pos, self._cmd_quat
        else:
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

    @torch.inference_mode()
    def convert_action_dict(
        self, action: Dict[str, torch.Tensor], dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a raw action dict to a controller-native nested action dict.

        Delegates the arm action to the EEF pose controller's
        :meth:`get_action_dict` and the gripper action to the gripper
        controller's :meth:`get_action_dict`.  The result is a dict keyed
        by ``eef_name`` and ``gripper_name``, matching the format produced
        by :class:`CombinedController.get_action_dict`.
        """
        return {
            self._eef_name: self._eef_ctrl.get_action_dict(
                action[self._eef_name], dt=dt,
            ),
            self._gripper_name: self._gripper_ctrl.get_action_dict(
                action[self._gripper_name], dt=dt,
            ),
        }

    def get_action_dict_space(self):
        """Return a :class:`DictSpace` matching :meth:`get_action_dict` output.

        The schema is ``{eef_name: eef_ctrl.get_action_dict_space(),
        gripper_name: gripper_ctrl.get_action_dict_space()}``, mirroring
        the controller-native nested structure.
        """
        from unienv_interface.space import DictSpace
        from unienv_interface.backends.pytorch import PyTorchComputeBackend
        return DictSpace(
            PyTorchComputeBackend,
            {
                self._eef_name: self._eef_ctrl.get_action_dict_space(),
                self._gripper_name: self._gripper_ctrl.get_action_dict_space(),
            },
            device=self.world.device,
        )

    @torch.inference_mode()
    def get_success(self) -> Union[torch.Tensor, bool]:
        return try_scalar(self._phase == GraspPhase.DONE)

    @torch.inference_mode()
    def get_failure(self) -> Union[torch.Tensor, bool]:
        return try_scalar(self._failed)

    @torch.inference_mode()
    def get_timeout(self) -> Union[torch.Tensor, bool]:
        if self.max_seconds <= 0:
            return try_scalar(torch.zeros_like(self._failed))
        timed_out = (
            (self._total_elapsed_seconds >= self.max_seconds - DURATION_COMPARISON_EPS)
            & (self._phase != GraspPhase.DONE)
            & ~self._failed
        )
        return try_scalar(timed_out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_candidate(
        self, ranking: torch.Tensor, best_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Draw a candidate per env with weight ``exp(-(score - best) / T)``.

        Rows without any finite weight (nothing kept) keep ``best_idx``.
        """
        shifted = ranking - ranking.min(dim=-1, keepdim=True).values
        weights = torch.exp(-shifted / float(self.selection_temperature))
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        flat = weights.reshape(-1, weights.shape[-1])
        empty = flat.sum(dim=-1) == 0
        flat = torch.where(
            empty[:, None],
            F.one_hot(best_idx.reshape(-1), flat.shape[-1]).to(flat),
            flat,
        )
        sampled = torch.multinomial(flat, 1, generator=self._selection_rng)
        return sampled.reshape(best_idx.shape)

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
        """Advance envs in *trans_mask* to *next_phase*, zeroing phase timer."""
        if not trans_mask.any():
            return
        if is_batched(self.world.batch_size):
            self._phase[trans_mask] = next_phase
            self._phase_elapsed_seconds[trans_mask] = 0.0
        else:
            if trans_mask.item():
                self._phase.fill_(next_phase)
                self._phase_elapsed_seconds.zero_()

    def _zero_counters(self, mask: Optional[torch.Tensor]) -> None:
        if mask is not None:
            self._phase[mask] = 0
            self._phase_elapsed_seconds[mask] = 0.0
            self._total_elapsed_seconds[mask] = 0.0
            self._failed[mask] = False
        else:
            self._phase.zero_()
            self._phase_elapsed_seconds.zero_()
            self._total_elapsed_seconds.zero_()
            self._failed.zero_()
    
    @staticmethod
    def filter_fn_top_down(candidates : GraspCandidates):
        """Keep only grasps that approach from above (EEF approach-axis points down).
        """
        return candidates.approach_direction[..., 2] < 0

    @staticmethod
    def filter_fn_prefer_top_down(candidates : GraspCandidates) -> torch.Tensor:
        """Prefer top-down grasps, falling back to all candidates per row.

        See :func:`filter_fn_prefer_top_down` (module level) for the full
        rationale.  This static form mirrors :meth:`filter_fn_top_down`.
        """
        return filter_fn_prefer_top_down(candidates)


def filter_fn_prefer_top_down(candidates: GraspCandidates) -> torch.Tensor:
    """Grasp-candidate filter that *prefers* top-down grasps without excluding all.

    ``BoxGripperGraspPolicy.filter_fn_top_down`` keeps only candidates whose
    approach axis points down.  Catalog objects whose grasp candidates come from
    mesh sampling (GSO/YCB) often expose side approaches only — e.g. a hammer
    lying flat — and filtering them out would leave the grasp policy with no
    candidate at all (every candidate gets an ``inf`` IK error and the phase
    fails immediately).

    This filter keeps the candidates approaching from above
    (``approach_direction[..., 2] < 0``) and, for any batch row where that set
    is empty, falls back to keeping *all* candidates in that row.  The fallback
    is applied per row on a batched ``(B, N)`` mask; for an unbatched ``(N,)``
    mask the top-down mask is returned unchanged (matching the reference
    implementation this is ported from).

    Args:
        candidates: ``GraspCandidates`` in world coordinates.

    Returns:
        ``(..., n_candidates)`` bool mask (``True`` = keep).
    """
    keep = candidates.approach_direction[..., 2] < 0
    if isinstance(keep, torch.Tensor) and keep.dim() > 1:
        return keep | ~keep.any(dim=-1, keepdim=True)
    return keep


def filter_fn_clean_top_down(
    candidates: GraspCandidates,
    *,
    min_down_cos: float = 0.5,
    max_opening_alignment: float = 0.5,
) -> torch.Tensor:
    """Grasp-candidate filter that keeps only *clean* top-down grasps.

    Unlike :func:`filter_fn_prefer_top_down` (a soft preference that accepts any
    downward-leaning approach), this is a **restrictive** filter: a candidate is
    kept only when

    1. its approach is genuinely downward —
       ``approach_direction[..., 2] <= -min_down_cos`` (default ``-0.5``,
       i.e. within ~60° of straight down), **and**
    2. its grasp frame is well-conditioned — the approach and opening axes are
       near-orthogonal, ``|approach · opening| <= max_opening_alignment``
       (default ``0.5``).  A candidate source may emit frames in which the
       approach is almost *collinear* with the opening
       (``|approach · opening| ≈ 1``); building an EEF frame from such a pair
       leaves the closing direction ill-defined and the resulting pose
       unreachable, so those candidates are rejected here even when their
       approach happens to point down.  (:class:`MeshObject`'s antipodal
       sampler emits exactly orthogonal frames.)

    Per-row fallback: for a batched ``(B, N)`` mask, any row with no surviving
    candidate falls back to keeping *all* candidates in that row (so a feasible
    grasp is still attempted rather than failing the phase outright).  For an
    unbatched ``(N,)`` mask the restrictive mask is returned unchanged.

    Args:
        candidates: ``GraspCandidates`` in world coordinates.
        min_down_cos: Minimum ``-approach_z`` (cosine of the approach to
            straight-down) for a candidate to count as top-down.
        max_opening_alignment: Maximum ``|approach · opening|`` for the grasp
            frame to count as well-conditioned.

    Returns:
        ``(..., n_candidates)`` bool mask (``True`` = keep).
    """
    approach = candidates.approach_direction
    opening = candidates.opening_direction
    downward = approach[..., 2] <= -float(min_down_cos)
    alignment = (approach * opening).sum(dim=-1).abs()
    clean = alignment <= float(max_opening_alignment)
    keep = downward & clean
    if isinstance(keep, torch.Tensor) and keep.dim() > 1:
        return keep | ~keep.any(dim=-1, keepdim=True)
    return keep


def _to_bool_tensor(value: Union[torch.Tensor, bool], like: torch.Tensor) -> torch.Tensor:
    """Coerce a bool or Tensor to a bool Tensor matching *like*'s shape/device."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.full_like(like, value, dtype=torch.bool)
