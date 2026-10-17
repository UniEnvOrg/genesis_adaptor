"""Unknown controller options are reported instead of silently swallowed (CPU).

``EntityController.__init__`` used to ``setattr`` every leftover keyword
argument, so a misspelled option (``max_joint_pos_vell=...``) or one meant
for another controller (``metadata=...``) was accepted and ignored.  Leftover
options outside ``EntityController.extra_attributes`` now raise an
``UnknownControllerOptionWarning`` (they are still stored, so code reading a
custom attribute keeps working).
"""

from __future__ import annotations

import os
import warnings

import pytest
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from genesis.utils.misc import get_assets_dir

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import (
    EntityController,
    UnknownControllerOptionWarning,
)

PANDA_XML = os.path.join(get_assets_dir(), "xml", "franka_emika_panda", "panda.xml")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]

GRIPPER_DESCRIPTION = dict(
    approach_direction=[0.0, 0.0, 1.0],
    open_direction=[0.0, 1.0, 0.0],
    gripper_width=0.08,
    gripper_depth=0.05,
    gripper_open_action=-1.0,
    gripper_close_action=1.0,
    eef_tip_offset=[0.0, 0.0, 0.1],
    eef_tip_offset_quat=[1.0, 0.0, 0.0, 0.0],
)


def _make_robot(**controller_kwargs) -> RobotDescriptionNode:
    return RobotDescriptionNode(
        GenesisWorld(), PANDA_XML, name="panda", **controller_kwargs,
    )


def test_unknown_option_warns_and_is_kept():
    with pytest.warns(UnknownControllerOptionWarning, match="max_joint_pos_vell"):
        robot = _make_robot(
            controller_type="joint_position",
            joint_names=ARM_JOINTS,
            max_joint_pos_vell=0.5,
        )
    # Still stored for backwards compatibility, but the real option is unset.
    assert robot.controller.max_joint_pos_vell == 0.5
    assert robot.controller.max_joint_pos_vel is None


def test_unknown_option_in_sub_controller_warns():
    with pytest.warns(UnknownControllerOptionWarning, match="metadata"):
        _make_robot(
            controller_type="combined",
            sub_controllers={
                "arm": {"controller_type": "joint_position", "joint_names": ARM_JOINTS},
                "gripper": {
                    "controller_type": "joint_position",
                    "joint_names": FINGER_JOINTS,
                    "metadata": {"label": "gripper"},
                },
            },
        )


def test_known_options_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownControllerOptionWarning)
        robot = _make_robot(
            controller_type="combined",
            sub_controllers={
                "arm": {
                    "controller_type": "eef_pose",
                    "joint_names": ARM_JOINTS,
                    "target_link": "hand",
                    "max_joint_pos_vel": 0.4,
                    "gravity_compensation": False,
                },
                "gripper": {
                    "controller_type": "mimic_joint_position",
                    "joint_names": FINGER_JOINTS,
                    "multiplier": -0.02,
                    "shift": 0.02,
                    "control_low": -1,
                    "control_high": 1,
                    **GRIPPER_DESCRIPTION,
                },
            },
        )
    gripper = robot.controller["gripper"]
    for key, value in GRIPPER_DESCRIPTION.items():
        assert getattr(gripper, key) == value
    assert set(GRIPPER_DESCRIPTION) <= EntityController.extra_attributes
