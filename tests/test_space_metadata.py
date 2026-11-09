"""Space metadata of the adaptor's controllers and sensors (CPU, no asset downloads).

Every space the adaptor declares carries unienv space metadata (``kind``,
``units``, ``frame``, ...). These tests build representative controllers and
sensor nodes on Genesis' bundled Franka Panda MJCF and a free box, check the
metadata against the vocabulary with ``validate_metadata`` and check the key
leaves: per-joint units follow the joint types, EEF actions name their
rotation representation, the gripper command is a normalized gripper signal
and camera images are ``rgb`` / ``HWC``.
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from genesis.utils.misc import get_assets_dir
from xbarray.transformations.rotation_conversions.pytorch import euler_angles_to_matrix

from unienv_interface.space import DictSpace
from unienv_interface.space.space_utils.metadata_utils import (
    find_spaces_by_metadata,
    validate_metadata,
)
from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers.end_effector_pose import (
    AbsoluteRotationConventionUtils,
)
from unienv_genesis.nodes.sensors import (
    CameraSensorNode,
    CameraSensorNodeOptions,
    ContactForceSensorNode,
    ContactForceSensorNodeOptions,
    ContactSensorNode,
    ContactSensorNodeOptions,
    EndEffectorStateSensorNode,
    EndEffectorStateSensorNodeOptions,
    IMUSensorNode,
    IMUSensorNodeOptions,
    JointStateSensorNode,
    JointStateSensorNodeOptions,
    KeypointTrackerNode,
)
from unienv_genesis.options.attach_option import AttachFromName
from unienv_genesis.utils.space_metadata import dof_names, dof_units

PANDA_XML = os.path.join(get_assets_dir(), "xml", "franka_emika_panda", "panda.xml")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]


def _expected_units(entity, joint_names, rotational, translational):
    """Units by joint type, looked up independently of the helper under test."""
    joints = {joint.name: joint for joint in entity.joints}
    units = []
    for name in joint_names:
        joint_type = joints[name].type
        assert joint_type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
        units.append(rotational if joint_type == gs.JOINT_TYPE.REVOLUTE else translational)
    return units


def _make_panda(batch_size=None, rotation_representation="euler", normalize_rotation_action=False):
    sub_controllers = {
        "arm": {
            "controller_type": "eef_pose",
            "joint_names": ARM_JOINTS,
            "target_link": "hand",
            "base_frame": "link0",
            "rotation_representation": rotation_representation,
            "normalize_rotation_action": normalize_rotation_action,
        },
        "gripper": {
            "controller_type": "mimic_joint_position",
            "joint_names": FINGER_JOINTS,
            "multiplier": -0.02,
            "shift": 0.02,
            "control_low": -1,
            "control_high": 1,
            "gripper_open_action": -1.0,
            "gripper_close_action": 1.0,
        },
        "all": {"controller_type": "joint_position_and_velocity"},
        "fingers_pv": {
            "controller_type": "mimic_joint_position_and_velocity",
            "joint_names": FINGER_JOINTS,
            "position_multiplier": 0.02,
            "velocity_multiplier": 0.05,
        },
        "fingers_vel": {
            "controller_type": "mimic_joint_velocity",
            "joint_names": FINGER_JOINTS,
            "multiplier": 0.05,
        },
    }
    return RobotDescriptionNode(
        GenesisWorld(batch_size=batch_size), PANDA_XML, name="panda",
        controller_type="combined", sub_controllers=sub_controllers,
    )


# ---------------------------------------------------------------------------
# Joint helpers
# ---------------------------------------------------------------------------


def test_dof_units_follow_joint_types():
    robot = _make_panda()
    entity = robot.entity
    assert dof_names(entity) == ARM_JOINTS + FINGER_JOINTS
    assert dof_units(entity) == ["rad"] * 7 + ["m"] * 2
    assert dof_units(entity, None, "velocity") == ["rad/s"] * 7 + ["m/s"] * 2
    assert dof_units(entity, None, "force") == ["N*m"] * 7 + ["N"] * 2
    # A DOF selection keeps the order of the selection.
    assert dof_units(entity, [8, 0]) == ["m", "rad"]
    assert dof_names(entity, [8, 0]) == ["finger_joint2", "joint1"]


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [None, 2])
def test_controller_spaces_are_described(batch_size):
    robot = _make_panda(batch_size=batch_size)
    controller = robot.controller
    validate_metadata(robot.action_space)
    validate_metadata(robot.observation_space)
    validate_metadata(controller.get_action_dict_space())

    entity = robot.entity
    all_joints = ARM_JOINTS + FINGER_JOINTS
    obs = robot.observation_space["all"]
    assert obs["joint_position"].metadata.to_dict()["units"] == _expected_units(entity, all_joints, "rad", "m")
    assert obs["joint_position"].metadata.to_dict()["names"] == all_joints
    assert obs["joint_velocity"].metadata.to_dict()["units"] == _expected_units(entity, all_joints, "rad/s", "m/s")
    assert obs["joint_torque"].metadata.to_dict()["units"] == _expected_units(entity, all_joints, "N*m", "N")
    assert obs["joint_motor_torque"].metadata.to_dict()["units"] == _expected_units(entity, all_joints, "N*m", "N")
    for key in ("joint_position", "joint_velocity", "joint_torque", "joint_motor_torque"):
        assert obs[key].metadata.to_dict()["kind"] == "proprio", key
    assert obs["root_pos"].metadata.to_dict() == {
        "kind": "proprio", "units": "m", "frame": "world", "names": ["x", "y", "z"],
    }
    assert obs["root_quat"].metadata.to_dict()["rotation"] == "quat_wxyz"

    # EEF observation of the arm: in the configured base frame.
    arm_obs = robot.observation_space["arm"]
    assert arm_obs["eef_position"].metadata.to_dict()["frame"] == "base"
    assert arm_obs["eef_position"].metadata.to_dict()["units"] == "m"
    assert arm_obs["eef_quaternion"].metadata.to_dict()["rotation"] == "quat_wxyz"
    assert arm_obs["eef_quaternion"].metadata.to_dict()["link"] == "hand"
    assert arm_obs["last_ik_solved_error"].metadata.to_dict()["units"] == ["m"] * 3 + ["rad"] * 3

    # Joint position / velocity targets.
    act = robot.action_space["all"]
    md = act["target_joint_position"].metadata.to_dict()
    assert (md["kind"], md["control"], md["mode"]) == ("action", "joint_position", "absolute")
    assert md["units"] == _expected_units(entity, all_joints, "rad", "m")
    assert md["names"] == all_joints
    md = act["target_joint_velocity"].metadata.to_dict()
    assert md["control"] == "joint_velocity"
    assert md["units"] == _expected_units(entity, all_joints, "rad/s", "m/s")

    # Single-joint subclasses keep the leaf metadata they extract.
    assert controller["arm"].get_action_dict_space()["target_joint_position"].metadata.to_dict()["names"] == ARM_JOINTS


def test_gripper_and_mimic_actions():
    robot = _make_panda()
    controller = robot.controller

    gripper = robot.action_space["gripper"].metadata.to_dict()
    assert gripper["kind"] == "action"
    assert gripper["control"] == "gripper" and gripper["mode"] == "absolute"
    assert gripper["normalized"] is True and gripper["units"] == "normalized"
    assert (gripper["open_value"], gripper["closed_value"]) == (-1.0, 1.0)
    assert gripper["mimic_joints"] == FINGER_JOINTS
    assert gripper["multiplier"] == [-0.02, -0.02] and gripper["shift"] == [0.02, 0.02]

    # A mimic controller without the gripper description is a plain joint command.
    pv = robot.action_space["fingers_pv"]
    assert pv["target_joint_position"].metadata.to_dict()["control"] == "joint_position"
    assert pv["target_joint_velocity"].metadata.to_dict()["control"] == "joint_velocity"
    assert "normalized" not in pv["target_joint_position"].metadata.to_dict()
    assert robot.action_space["fingers_vel"].metadata.to_dict()["control"] == "joint_velocity"

    # The action dicts describe the mapped per-joint targets (n_dofs entries),
    # not the scalar control signal.
    for name in ("gripper", "fingers_pv"):
        position = controller[name].get_action_dict_space()["target_joint_position"]
        assert position.shape == (2,), name
        assert position.metadata.to_dict()["units"] == ["m", "m"], name
    for name in ("fingers_pv", "fingers_vel"):
        velocity = controller[name].get_action_dict_space()["target_joint_velocity"]
        assert velocity.shape == (2,), name
        assert velocity.metadata.to_dict()["units"] == ["m/s", "m/s"], name


@pytest.mark.parametrize(
    "representation, normalize, rotation, rot_units",
    [
        ("euler", False, "euler_rxyz", ["rad"] * 3),
        ("euler", True, "euler_rxyz", ["normalized"] * 3),
        ("quat", False, "quat_wxyz", ["normalized"] * 4),
        ("rot6d", False, "rot6d", ["normalized"] * 6),
    ],
)
def test_eef_pose_action_rotation(representation, normalize, rotation, rot_units):
    robot = _make_panda(rotation_representation=representation, normalize_rotation_action=normalize)
    arm = robot.action_space["arm"]
    validate_metadata(arm)
    md = arm.metadata.to_dict()
    assert (md["kind"], md["control"], md["mode"]) == ("action", "eef_pose", "absolute")
    assert md["frame"] == "base" and md["link"] == "hand"
    assert md["rotation"] == rotation
    assert md["units"] == ["m", "m", "m", *rot_units]
    assert md["names"][:3] == ["x", "y", "z"] and len(md["names"]) == arm.shape[-1]
    if representation == "euler" and normalize:
        assert md["rotation_range"] == [-math.pi, math.pi]
    else:
        assert "rotation_range" not in md

    # The FK targets of the action dict are absolute EEF poses in the same frame.
    action_dict = robot.controller["arm"].get_action_dict_space()
    assert action_dict["target_eef_position"].metadata.to_dict()["frame"] == "base"
    assert action_dict["target_eef_quaternion"].metadata.to_dict()["rotation"] == "quat_wxyz"


def test_ref_policy_action_dict_space_keeps_controller_metadata():
    from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy

    robot = _make_panda()
    # The policy reads the current EEF pose, so the scene must exist and be built (Genesis >= 1.4.6 holds no state
    # before that): run the env's reload.
    WorldEnv(robot.world, [robot], render_mode=None).reset(seed=0, reload=True)
    arm = robot.controller["arm"]
    policy = GoToRefPolicy(
        robot.world, arm,
        target_pos=torch.tensor([0.4, 0.0, 0.4]),
        target_quat=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    space = policy.get_action_dict_space()
    validate_metadata(space)
    assert space["target_joint_position"].metadata.to_dict()["names"] == ARM_JOINTS
    assert space["target_eef_position"].metadata.to_dict()["control"] == "eef_pose"


def test_euler_convention_is_transforms3d_rxyz():
    """``euler_rxyz``: R = Rx(a) @ Ry(b) @ Rz(c), the convention the EEF controller decodes."""
    a, b, c = 0.3, -0.7, 1.1

    def rot(axis, angle):
        cos, sin = math.cos(angle), math.sin(angle)
        if axis == "x":
            return torch.tensor([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
        if axis == "y":
            return torch.tensor([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        return torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    expected = rot("x", a) @ rot("y", b) @ rot("z", c)
    decoded = euler_angles_to_matrix(torch.tensor([a, b, c]), convention="XYZ")
    assert torch.allclose(decoded, expected, atol=1e-6)
    assert AbsoluteRotationConventionUtils.METADATA_ROTATIONS["euler"] == "euler_rxyz"


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


class _BoxNode(WorldNode):
    """A plane and a free box (one 6-DOF free joint)."""

    reload_priorities = {20}
    reset_priorities = {20}
    name = "box_env"

    def __init__(self, world):
        self.world = world

    def reload(self, *, priority=0, **kwargs):
        if priority == 20:
            self.world.scene.add_entity(gs.morphs.Plane(), name="plane")
            self.world.scene.add_entity(
                gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05)), name="box",
            )

    def reset(self, **kwargs):
        pass

    def get_observation(self):
        return None


def test_sensor_spaces_are_described():
    world = GenesisWorld(
        scene_init_kwargs={"vis_options": gs.options.VisOptions(segmentation_level="link")},
    )
    link = AttachFromName("box", "box_baselink")
    cam_pos, cam_lookat = (1.0, 0.0, 0.6), (0.0, 0.0, 0.05)
    camera = CameraSensorNode(
        world, "front",
        CameraSensorNodeOptions(
            res=(64, 48), pos=cam_pos, quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
        ),
        render_depth=True, render_seg=True, render_normal=True, render_position=True,
    )
    nodes = [
        JointStateSensorNode(
            world, "joints",
            JointStateSensorNodeOptions(attach=AttachFromName("box"), read_motor_torque=True, read_root_pose=True),
        ),
        EndEffectorStateSensorNode(world, "eef", EndEffectorStateSensorNodeOptions(attach=link)),
        IMUSensorNode(world, "imu", IMUSensorNodeOptions(attach=link)),
        ContactForceSensorNode(world, "force", ContactForceSensorNodeOptions(attach=link)),
        ContactSensorNode(world, "contact", ContactSensorNodeOptions(attach=link)),
        camera,
        KeypointTrackerNode(world, "keypoints", camera_node=camera, max_points=8),
    ]
    env = WorldEnv(world, [_BoxNode(world), *nodes], render_mode=None)
    env.reset(seed=0, reload=True)
    for node in nodes:
        validate_metadata(node.observation_space)
    validate_metadata(env.observation_space)

    # Joint state of the free joint: 3 translational + 3 rotational DOFs.
    joints = nodes[0].observation_space
    assert joints["joint_position"].metadata.to_dict()["units"] == ["m"] * 3 + ["rad"] * 3
    assert joints["joint_velocity"].metadata.to_dict()["units"] == ["m/s"] * 3 + ["rad/s"] * 3
    assert joints["joint_motor_torque"].metadata.to_dict()["units"] == ["N"] * 3 + ["N*m"] * 3
    assert joints["root_quat"].metadata.to_dict()["rotation"] == "quat_wxyz"

    eef = nodes[1].observation_space
    assert (eef["eef_position"].metadata.to_dict()["frame"], eef["eef_position"].metadata.to_dict()["units"]) == ("world", "m")

    imu = nodes[2].observation_space
    assert imu["lin_acc"].metadata.to_dict()["units"] == "m/s^2"
    assert imu["ang_vel"].metadata.to_dict()["units"] == "rad/s"
    assert imu["lin_acc"].metadata.to_dict()["frame"] == "link_local"

    force = nodes[3].observation_space["contact_force"].metadata.to_dict()
    assert (force["kind"], force["units"], force["frame"]) == ("force", "N", "link_local")
    assert nodes[4].observation_space["in_contact"].metadata.to_dict()["kind"] == "mask"

    # Camera: what the leaves are, and that the data has the described layout.
    cam = camera.observation_space
    obs = camera.get_observation()
    rgb = cam["rgb"].metadata.to_dict()
    assert (rgb["kind"], rgb["format"], rgb["layout"], rgb["source"]) == ("image", "rgb", "HWC", "front")
    assert obs["rgb"].shape == (48, 64, 3) and obs["rgb"].dtype == torch.uint8
    assert "fps" not in rgb
    depth = cam["depth"].metadata.to_dict()
    assert (depth["kind"], depth["units"], depth["layout"]) == ("depth", "m", "HW")
    assert (depth["near"], depth["far"]) == (0.1, 20.0)
    seg = cam["seg"].metadata.to_dict()
    assert (seg["kind"], seg["layout"], seg["level"]) == ("segmentation", "HW", "link")
    assert cam["normal"].metadata.to_dict()["kind"] == "normal"
    position = cam["position"].metadata.to_dict()
    assert (position["kind"], position["units"], position["frame"], position["layout"]) == (
        "pointcloud", "m", "world", "HWC",
    )
    assert cam["intrinsics_cv"].metadata.to_dict()["kind"] == "intrinsics"
    for key, reference in (("extrinsics_cv", "world"), ("local_extrinsics_cv", "world")):
        md = cam[key].metadata.to_dict()
        assert (md["kind"], md["frame"], md["reference_frame"]) == ("extrinsics", "camera_cv", reference)
    assert [path for path, _ in find_spaces_by_metadata(env.observation_space, kind="image")] == [("front", "rgb")]

    keypoints = nodes[6].observation_space
    md = keypoints["point_positions"].metadata.to_dict()
    assert (md["kind"], md["units"], md["frame"]) == ("keypoints", "m", "world")
    assert keypoints["points_2d"].metadata.to_dict()["units"] == "px"
    assert keypoints["point_segmentations"].metadata.to_dict()["level"] == "link"
    assert keypoints["visibility_masks"].metadata.to_dict()["kind"] == "mask"
    env.close()


def test_attached_camera_records_its_link():
    world = GenesisWorld()
    camera = CameraSensorNode(
        world, "wrist",
        CameraSensorNodeOptions(res=(32, 24), pos=(0.0, 0.0, 0.1), attach=AttachFromName("box", "box_baselink")),
    )
    md = camera.observation_space["rgb"].metadata.to_dict()
    assert (md["source"], md["entity"], md["link"]) == ("wrist", "box", "box_baselink")
    assert camera.observation_space["local_extrinsics_cv"].metadata.to_dict()["reference_frame"] == "link_local"
    validate_metadata(camera.observation_space)
