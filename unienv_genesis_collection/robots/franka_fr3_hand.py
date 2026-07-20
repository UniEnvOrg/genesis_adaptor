"""Franka Research FR3 robot descriptor with Franka Hand gripper.

MJCF sources (Apache 2.0):
  Arm:     https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3
  Gripper: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

# Default home joint positions for FR3 arm (7 DOF) + Franka Hand (2 DOF, fully open)
ARM_HOME_JOINT_POSITIONS = torch.tensor([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRIPPER_HOME_JOINT_POSITIONS = torch.tensor([0.04, 0.04])
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, GRIPPER_HOME_JOINT_POSITIONS])

# End-effector link name for IK control
EE_LINK_NAME = "hand"

# Finger link names for grasp contact detection
FINGER_PAD_LINK_NAMES = ["left_finger", "right_finger"]

# DROID IK/controller velocity limit for the Franka arm (rad/s).
# This is the max joint velocity used by the DROID IK controller, not the
# per-joint hardware limits from franka_hardware.yaml.
# Ref: https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py#L10
DROID_FRANKA_ARM_IK_VEL_LIMIT = 0.2

# Franka hardware per-joint velocity limits (rad/s) from franka_hardware.yaml.
# Intentionally retained as hardware/reference documentation for future
# controller/planner work, even though the active controller uses the DROID
# 0.2 rad/s limit above; NOT wired into any active controller.
FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS = torch.tensor([2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51])

# Previous DROID_-prefixed name, intentionally retained alongside the
# reference constant above as searchable hardware documentation. Prefer
# ``FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS``; both names refer to the same
# reference tensor and neither feeds the active controller.
DROID_FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS = FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS


# Default combined sub-controller configuration
_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_indexes": list(range(7)),   # fr3_joint1 … fr3_joint7
        # Virtual EEF offset from the target_link (hand) origin to the
        # fingertip along the approach direction, centred in the open
        # direction.  This point is where grasp candidates place the EEF
        # (see base_object.list_box_gripper_grasping_candidates_local).
        #   finger base z = 0.0584 m + finger pad extent z = 0.053 m
        "target_link": LinkAttachment(
            link_name=EE_LINK_NAME, offset_pos=[0.0, 0.0, 0.1114],
        ),
        # DROID IK/controller velocity limit for the Franka arm (rad/s).
        "max_joint_pos_vel": DROID_FRANKA_ARM_IK_VEL_LIMIT,
    },
    "gripper": {
        "controller_type": "mimic_joint_position",
        "joint_indexes": [7, 8],           # finger_joint1, finger_joint2
        # Gripper action convention: -1 = fully open (0.04 m), +1 = fully closed (0.0 m)
        # Mapping applied by the mimic_joint_position controller:
        #   finger_position = action * GRIPPER_MULTIPLIER + GRIPPER_SHIFT
        "multiplier": -0.02,
        "shift": 0.02,
        "control_low": -1,
        "control_high": 1,
        "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
        # Grasp frame directions in the EEF frame (hand body frame).
        # approach_direction: axis along which the gripper approaches the
        #   object (fingertip direction); eef_offset is measured along this.
        # open_direction: axis along which the fingers separate.
        "approach_direction": [0.0, 0.0, 1.0],
        "open_direction": [0.0, 1.0, 0.0],
        "gripper_width": 0.08,    # 2 × 0.04 m max finger opening
        "gripper_depth": 0.02,    # fingertip pad depth
        "gripper_open_action": -1.0,  # action value for fully open gripper
        "gripper_close_action": 1.0,  # action value for fully closed gripper
    },
}


class FrankaFR3WithHand(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with Franka Hand gripper.

    The arm MJCF comes from the mujoco_menagerie ``franka_fr3`` model; the
    gripper comes from the mujoco_menagerie ``franka_emika_panda/hand.xml``.
    They are combined in ``assets/robots/franka_fr3_hand/fr3_hand.xml``.

    Default controller is ``"combined"``:
      - ``"arm"``     sub-controller: ``eef_pose`` (IK-based end-effector control)
      - ``"gripper"`` sub-controller: ``mimic_joint_position`` (single scalar)

    Gripper action convention::

        -1  →  fully open  (both fingers at 0.04 m)
        +1  →  fully closed (both fingers at 0.0 m)

    Args:
        world: GenesisWorld instance
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller.  For the default
            ``"combined"`` controller, pass ``sub_controllers=...`` to override
            the arm/gripper configuration.

    Example::

        from unienv_genesis import GenesisWorld
        from unienv_genesis_collection.robots.franka_fr3_hand import FrankaFR3WithHand

        world = GenesisWorld()
        robot = FrankaFR3WithHand(world, name="franka")

        # Action dict expected by the default combined controller:
        # {
        #   "arm":     <eef_pose action tensor>,
        #   "gripper": tensor([-1.0])   # open
        # }
    """

    keyframes = {
        "reset": RobotKeyframe(joint_position=HOME_JOINT_POSITIONS),
    }

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "franka",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        if controller_type == "eef_pose" and "target_link" not in controller_kwargs:
            controller_kwargs["target_link"] = LinkAttachment(link_name=EE_LINK_NAME)

        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_hand", "fr3_hand.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
