"""The command leash of the interpolating reference policies.

``max_command_lead`` / ``max_command_lead_angle`` keep the commanded pose within
a ball / cone around the *achieved* EEF pose
(:func:`unienv_genesis_refpolicy.base.motion.leash_command`). CPU only, with mock
controllers whose EEF is moved by hand (a blocked arm, an arm that follows).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from unienv_genesis_collection.objects.base_object import GraspCandidates
from unienv_genesis_refpolicy.base.motion import (
    HANDOFF_MAX_DISTANCE,
    _quat_angle,
    leash_command,
    store_command,
)
from unienv_genesis_refpolicy.base.tip_frame import TipFrameEEFController
from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider
from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy
from unienv_genesis_refpolicy.implementations.grasp_policy import BoxGripperGraspPolicy, GraspPhase

IDENTITY = (1.0, 0.0, 0.0, 0.0)
DT = 1.0 / 15.0


def _quat_z(angle: float):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def _world(batch_size=None):
    return SimpleNamespace(batch_size=batch_size, device=torch.device("cpu"))


class _MockArm:
    """EEF pose controller mock; the EEF stays where the test puts it."""

    has_eef = True

    def __init__(self, world, pos, quat=IDENTITY):
        self.world = world
        shape = () if world.batch_size is None else (world.batch_size,)
        self.pos = torch.tensor(pos, dtype=torch.float32).expand(*shape, 3).clone()
        self.quat = torch.tensor(quat, dtype=torch.float32).expand(*shape, 4).clone()

    def get_eef_pose_absolute(self):
        return self.pos.clone(), self.quat.clone()

    def get_action_world(self, pos, quat):
        return torch.cat([pos, quat], dim=-1)

    def get_action_dict(self, action, dt=None):
        return {"action": action}


def _lead(arm, action):
    """Position / rotation lead of an ``[pos, quat]`` action over the arm's EEF."""
    return (
        torch.norm(action[..., :3] - arm.pos, dim=-1),
        _quat_angle(action[..., 3:7], arm.quat),
    )


# ----------------------------------------------------------------------
# leash_command
# ----------------------------------------------------------------------


def test_leash_none_is_a_noop():
    cmd = torch.tensor([1.0, 0.0, 0.0])
    q = torch.tensor(_quat_z(1.0))
    pos, quat, clamped = leash_command(None, cmd, q, torch.zeros(3), torch.tensor(IDENTITY), None, None)
    assert pos is cmd and quat is q
    assert not bool(clamped)


def test_leash_inside_the_ball_returns_the_command_unchanged():
    cmd = torch.tensor([0.01, 0.02, 0.0])
    q = torch.tensor(_quat_z(0.1))
    pos, quat, clamped = leash_command(None, cmd, q, torch.zeros(3), torch.tensor(IDENTITY), 0.03, 0.15)
    assert torch.equal(pos, cmd) and torch.equal(quat, q)
    assert not bool(clamped)


def test_leash_clips_position_onto_the_ball_along_the_lead():
    eef = torch.tensor([0.1, 0.2, 0.3])
    cmd = eef + torch.tensor([0.3, 0.4, 0.0])  # 0.5 m ahead
    pos, quat, clamped = leash_command(
        None, cmd, torch.tensor(IDENTITY), eef, torch.tensor(IDENTITY), 0.03, None,
    )
    assert bool(clamped)
    assert torch.allclose(pos, eef + torch.tensor([0.018, 0.024, 0.0]), atol=1e-6)
    assert torch.equal(quat, torch.tensor(IDENTITY))


def test_leash_slerps_rotation_onto_the_cone():
    eef_q = torch.tensor(_quat_z(0.2))
    cmd_q = torch.tensor(_quat_z(1.2))  # 1 rad ahead about z
    pos, quat, clamped = leash_command(None, torch.zeros(3), cmd_q, torch.zeros(3), eef_q, None, 0.15)
    assert bool(clamped)
    assert torch.allclose(quat, torch.tensor(_quat_z(0.35)), atol=1e-6)
    assert float(_quat_angle(quat, eef_q)) == pytest.approx(0.15, abs=1e-5)


def test_leash_takes_the_short_way_round_for_sign_flipped_quaternions():
    eef_q = torch.tensor(IDENTITY)
    cmd_q = -torch.tensor(_quat_z(0.5))  # same rotation, opposite hemisphere
    _, quat, clamped = leash_command(None, torch.zeros(3), cmd_q, torch.zeros(3), eef_q, None, 0.15)
    assert bool(clamped)
    assert float(_quat_angle(quat, torch.tensor(_quat_z(0.15)))) < 1e-5


def test_leash_batched_moves_only_the_envs_over_the_limit():
    eef = torch.zeros(3, 3)
    eef_q = torch.tensor(IDENTITY).expand(3, 4)
    cmd = torch.tensor([[0.01, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cmd_q = torch.stack([torch.tensor(IDENTITY), torch.tensor(IDENTITY), torch.tensor(_quat_z(0.6))])
    pos, quat, clamped = leash_command(None, cmd, cmd_q, eef, eef_q, 0.03, 0.15)
    assert clamped.tolist() == [False, True, True]
    assert torch.equal(pos[0], cmd[0]) and torch.equal(quat[0], cmd_q[0])
    assert torch.allclose(pos[1], torch.tensor([0.03, 0.0, 0.0]))
    assert torch.equal(pos[2], cmd[2])
    assert float(_quat_angle(quat[2], eef_q[2])) == pytest.approx(0.15, abs=1e-5)


def test_leash_of_a_tip_frame_view_bounds_the_flange():
    """A rotation about the tool tip swings the flange: the lead is the flange's."""
    arm = _MockArm(_world(), (0.4, 0.0, 0.5))
    view = TipFrameEEFController(arm, [0.0, 0.0, 0.159])
    tip_pos, tip_quat = view.get_eef_pose_absolute()
    # Same tip position, 0.5 rad about the x axis: the flange moves ~8 cm.
    cmd_q = torch.tensor([math.cos(0.25), math.sin(0.25), 0.0, 0.0])
    pos, quat, clamped = leash_command(view, tip_pos, cmd_q, tip_pos, tip_quat, 0.03, None)
    assert bool(clamped)
    flange_pos, flange_quat = view.tip_to_eef(pos, quat)
    assert float(torch.norm(flange_pos - arm.pos)) == pytest.approx(0.03, abs=1e-5)
    # The rotation itself is not limited (no angle leash).
    assert float(_quat_angle(flange_quat, cmd_q)) < 1e-5


# ----------------------------------------------------------------------
# GoToRefPolicy
# ----------------------------------------------------------------------


def _goto(arm, target_pos, target_quat=IDENTITY, **kwargs):
    options = dict(stall_seconds=0, ramp_seconds=0.5)
    options.update(kwargs)
    return GoToRefPolicy(
        arm.world, controller=arm,
        target_pos=torch.tensor(target_pos), target_quat=torch.tensor(target_quat),
        **options,
    )


def test_goto_without_leash_runs_ahead_of_a_blocked_arm():
    """Default (no leash): the command keeps going to the target."""
    arm = _MockArm(_world(), (0.0, 0.0, 0.5))
    policy = _goto(arm, (0.5, 0.0, 0.5))
    policy.reset()
    for _ in range(120):
        policy.step(dt=DT)
    lead, _ = _lead(arm, policy.get_action())
    assert float(lead) > 0.4


def test_goto_leash_bounds_the_lead_of_a_blocked_arm():
    arm = _MockArm(_world(), (0.0, 0.0, 0.5))
    policy = _goto(arm, (0.5, 0.0, 0.5), _quat_z(1.0), max_command_lead=0.03, max_command_lead_angle=0.15)
    policy.reset()
    for _ in range(120):
        policy.step(dt=DT)
        lead, angle = _lead(arm, policy.get_action())
        assert float(lead) <= 0.03 + 1e-6
        assert float(angle) <= 0.15 + 1e-5
    # Waiting on the ball for the arm, at rest (the ramp restarts once it moves).
    assert float(lead) == pytest.approx(0.03, abs=1e-6)
    assert float(policy._cmd_speed) == 0.0


def test_goto_leash_lets_a_following_arm_reach_the_target():
    arm = _MockArm(_world(), (0.0, 0.0, 0.5))
    target = torch.tensor([0.3, 0.1, 0.4])
    policy = _goto(arm, tuple(target.tolist()), max_command_lead=0.03, max_command_lead_angle=0.15)
    policy.reset()
    for _ in range(300):
        policy.step(dt=DT)
        action = policy.get_action()
        assert float(_lead(arm, action)[0]) <= 0.03 + 1e-6
        # The arm covers half of the remaining lead per step.
        arm.pos = arm.pos + 0.5 * (action[:3] - arm.pos)
        if policy.get_success():
            break
    assert policy.get_success()
    assert torch.allclose(policy._cmd_pos, target, atol=1e-5)


def test_goto_leash_batched_with_masked_steps():
    arm = _MockArm(_world(2), (0.0, 0.0, 0.5))
    policy = _goto(arm, (0.5, 0.0, 0.5), max_command_lead=0.03)
    policy.reset()
    mask = torch.tensor([True, False])
    for _ in range(60):
        policy.step(dt=DT, mask=mask)
    lead, _ = _lead(arm, policy.get_action())
    assert float(lead[0]) == pytest.approx(0.03, abs=1e-6)
    assert float(lead[1]) == 0.0  # never stepped


def test_goto_leash_without_interpolation_commands_the_clipped_target():
    arm = _MockArm(_world(), (0.0, 0.0, 0.5))
    policy = _goto(
        arm, (0.5, 0.0, 0.5), max_linear_speed=None, max_angular_speed=None, ramp_seconds=None,
        max_command_lead=0.03,
    )
    policy.reset()
    assert torch.allclose(policy.get_action()[:3], torch.tensor([0.03, 0.0, 0.5]), atol=1e-6)
    arm.pos = torch.tensor([0.2, 0.0, 0.5])
    policy.step(dt=DT)
    assert torch.allclose(policy.get_action()[:3], torch.tensor([0.23, 0.0, 0.5]), atol=1e-6)
    arm.pos = torch.tensor([0.49, 0.0, 0.5])
    policy.step(dt=DT)
    assert torch.allclose(policy.get_action()[:3], torch.tensor([0.5, 0.0, 0.5]), atol=1e-6)


def test_goto_leash_clamps_a_carried_command_and_stops_its_speed():
    """A hand-off within HANDOFF_MAX_DISTANCE but beyond the leash starts on the ball, at rest."""
    arm = _MockArm(_world(), (0.0, 0.0, 0.5))
    carried = torch.tensor([0.045, 0.0, 0.5])
    assert 0.045 < HANDOFF_MAX_DISTANCE
    store_command(arm, carried, torch.tensor(IDENTITY), torch.tensor(1.0), torch.tensor([1.0, 0.0, 0.0]))
    leashed = _goto(arm, (0.5, 0.0, 0.5), continue_command=True, max_command_lead=0.03)
    leashed.reset()
    assert torch.allclose(leashed.get_action()[:3], torch.tensor([0.03, 0.0, 0.5]), atol=1e-6)
    assert float(leashed._cmd_speed) == 0.0

    store_command(arm, carried, torch.tensor(IDENTITY), torch.tensor(1.0), torch.tensor([1.0, 0.0, 0.0]))
    free = _goto(arm, (0.5, 0.0, 0.5), continue_command=True)
    free.reset()
    assert torch.allclose(free.get_action()[:3], carried)
    assert float(free._cmd_speed) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# BoxGripperGraspPolicy
# ----------------------------------------------------------------------


class _MockGripper:
    has_eef = False
    gripper_open_action = -1.0
    gripper_close_action = 1.0
    gripper_width = 0.085
    gripper_depth = 0.035
    approach_direction = [0.0, 0.0, 1.0]
    open_direction = [0.0, 1.0, 0.0]

    def is_grasping(self, target_entity):
        return True

    def get_action_dict(self, action, dt=None):
        return {"action": action}


def _grasp(monkeypatch, arm, **kwargs):
    class _Provider(GraspCandidateProvider):
        def propose(self, *, mask=None, seed=None):
            return GraspCandidates(
                position=torch.tensor([[0.3, 0.1, 0.3]], dtype=torch.float32),
                approach_direction=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
                opening_direction=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            )

    def _reachable(_ctrl, target_pos, target_quat, envs_idx=None):
        return torch.zeros(target_pos.shape[:-1] + (6,), dtype=torch.float32)

    monkeypatch.setattr(GoToRefPolicy, "check_reachable_error", staticmethod(_reachable))
    return BoxGripperGraspPolicy(
        world=arm.world, eef_controller=arm, gripper_controller=_MockGripper(),
        target_object=object(), target_entity=object(), candidate_provider=_Provider(),
        open_gripper_seconds=0.2, close_gripper_seconds=0.2, phase_patience_seconds=30.0, max_seconds=100.0,
        **kwargs,
    )


@pytest.mark.parametrize("interpolate", [True, False])
def test_grasp_leash_bounds_the_pre_grasp_command(monkeypatch, interpolate):
    arm = _MockArm(_world(), (0.0, 0.0, 0.8))
    speeds = dict(max_linear_speed=0.15, max_angular_speed=0.4, ramp_seconds=0.5) if interpolate else {}
    policy = _grasp(monkeypatch, arm, max_command_lead=0.03, max_command_lead_angle=0.15, **speeds)
    policy.reset()
    saw_pre_grasp = False
    for _ in range(150):
        policy.step(dt=DT)
        action = policy.get_action()["arm"]
        lead, angle = _lead(arm, action)
        assert float(lead) <= 0.03 + 1e-6
        assert float(angle) <= 0.15 + 1e-5
        saw_pre_grasp |= policy._phase.item() == GraspPhase.PRE_GRASP
    assert saw_pre_grasp and policy._phase.item() == GraspPhase.PRE_GRASP  # the blocked arm never arrives


def test_grasp_leash_off_commands_the_pre_grasp_target_directly(monkeypatch):
    arm = _MockArm(_world(), (0.0, 0.0, 0.8))
    policy = _grasp(monkeypatch, arm)
    policy.reset()
    for _ in range(5):
        policy.step(dt=DT)
    assert policy._phase.item() == GraspPhase.PRE_GRASP
    action = policy.get_action()["arm"]
    assert torch.allclose(action[:3], torch.tensor([0.3, 0.1, 0.4]), atol=1e-6)
