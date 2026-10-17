"""The tip-frame view of an EEF controller (``unienv_genesis_refpolicy.base.tip_frame``).

CPU only, with a mock controller: the view converts tip poses to / from the
wrapped controller's EEF poses and delegates everything else.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from xbarray.transformations.rotation_conversions.pytorch import quaternion_apply

from unienv_genesis_refpolicy.base.tip_frame import TipFrameEEFController, tip_frame_controller

OFFSET = [0.0, 0.0, 0.159]
# 90 degrees about x: the EEF z axis points along world -y.
QUAT_X90 = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]


class _MockEEFController:
    has_eef = True
    rotation_representation = "quat"

    def __init__(self, pos, quat):
        self.world = SimpleNamespace(device=torch.device("cpu"), batch_size=None)
        self.pos = torch.tensor(pos, dtype=torch.float32)
        self.quat = torch.tensor(quat, dtype=torch.float32)
        self.ik_calls = []

    def get_eef_pose_absolute(self):
        return self.pos, self.quat

    def get_eef_pose_base(self):
        return self.pos - 1.0, self.quat  # a translated base frame

    def solve_ik_absolute(self, position, quaternion, envs_idx=None):
        self.ik_calls.append((position, quaternion, envs_idx))
        return torch.zeros(7), torch.full((6,), 0.5)

    def solve_ik_base(self, position, quaternion):
        self.ik_calls.append((position, quaternion, None))
        return torch.zeros(7), torch.full((6,), 0.25)

    def get_action_world(self, pos, quat):
        return torch.cat([pos, quat])

    def get_action_dict(self, action, dt=None):
        return {"action": action}


def _gripper(**attrs):
    return SimpleNamespace(**attrs)


def test_tip_pose_is_eef_plus_rotated_offset():
    ctrl = _MockEEFController([0.1, 0.2, 0.3], QUAT_X90)
    tip = TipFrameEEFController(ctrl, OFFSET)
    pos, quat = tip.get_eef_pose_absolute()
    assert torch.allclose(pos, torch.tensor([0.1, 0.2 - 0.159, 0.3]), atol=1e-6)
    assert torch.allclose(quat, ctrl.quat)
    base_pos, _ = tip.get_eef_pose_base()
    assert torch.allclose(base_pos, pos - 1.0, atol=1e-6)


@pytest.mark.parametrize("offset_quat", [None, QUAT_X90])
def test_tip_to_eef_inverts_eef_to_tip(offset_quat):
    ctrl = _MockEEFController([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    tip = TipFrameEEFController(ctrl, OFFSET, offset_quat)
    pos = torch.tensor([[0.4, -0.1, 0.2], [0.0, 0.3, 0.5]])
    quat = torch.nn.functional.normalize(torch.tensor([[0.9, 0.1, -0.3, 0.2], [0.2, 0.7, 0.1, 0.6]]), dim=-1)
    back_pos, back_quat = tip.tip_to_eef(*tip.eef_to_tip(pos, quat))
    assert torch.allclose(back_pos, pos, atol=1e-6)
    assert torch.allclose(back_quat, quat, atol=1e-6)


def test_ik_and_action_targets_are_converted_to_the_eef():
    ctrl = _MockEEFController([0.0, 0.0, 0.0], QUAT_X90)
    tip = TipFrameEEFController(ctrl, OFFSET)
    target_pos = torch.tensor([0.5, 0.0, 0.2])
    target_quat = torch.tensor(QUAT_X90)
    eef_pos = target_pos - quaternion_apply(target_quat, torch.tensor(OFFSET))

    _, error = tip.solve_ik_absolute(target_pos, target_quat, envs_idx="sel")
    position, quaternion, envs_idx = ctrl.ik_calls[-1]
    assert torch.allclose(position, eef_pos, atol=1e-6)
    assert torch.allclose(quaternion, target_quat, atol=1e-6)
    assert envs_idx == "sel"
    assert torch.equal(error, torch.full((6,), 0.5))  # the controller's error

    _, error = tip.solve_ik_base(target_pos, target_quat)
    assert torch.allclose(ctrl.ik_calls[-1][0], eef_pos, atol=1e-6)
    assert torch.equal(error, torch.full((6,), 0.25))

    action = tip.get_action_world(target_pos, target_quat)
    assert torch.allclose(action[:3], eef_pos, atol=1e-6)


def test_everything_else_is_delegated():
    ctrl = _MockEEFController([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    tip = TipFrameEEFController(ctrl, OFFSET)
    assert tip.has_eef is True
    assert tip.world is ctrl.world
    assert tip.rotation_representation == "quat"
    assert tip.get_action_dict("a") == {"action": "a"}
    with pytest.raises(AttributeError):
        tip.not_an_attribute


def test_tip_frame_controller_wraps_only_with_an_offset():
    ctrl = _MockEEFController([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert tip_frame_controller(ctrl) is ctrl
    assert tip_frame_controller(ctrl, _gripper()) is ctrl
    assert tip_frame_controller(ctrl, _gripper(eef_tip_offset=None)) is ctrl

    tip = tip_frame_controller(ctrl, _gripper(eef_tip_offset=OFFSET, eef_tip_offset_quat=QUAT_X90))
    assert isinstance(tip, TipFrameEEFController)
    assert torch.allclose(tip.tip_offset_pos, torch.tensor(OFFSET))
    assert torch.allclose(tip.tip_offset_quat, torch.tensor(QUAT_X90))
    # Idempotent: an already wrapped controller is not wrapped again.
    assert tip_frame_controller(tip, _gripper(eef_tip_offset=OFFSET)) is tip
