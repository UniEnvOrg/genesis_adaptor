"""Tianji Marvin dual-arm robot with Wuji 5-finger dexterous hands.

The Marvin robot (CCS 696 variant) is a 14-DOF dual-arm manipulator with a
fixed torso base. Each arm has 7 revolute joints and a Wuji 5-finger
dexterous hand attached at the wrist. Each hand has 5 fingers with 4 joints
each (20 joints per hand), every joint independently position-actuated.

DOF summary (qpos / actuator order):
  Left arm:    7 revolute joints  (Joint1_L .. Joint7_L)
  Left hand:   20 joints          (left_finger{1..5}_joint{1..4})
  Right arm:   7 revolute joints  (Joint1_R .. Joint7_R)
  Right hand:  20 joints          (right_finger{1..5}_joint{1..4})
  Total: 54 joints, 54 actuated DOF

MJCF source:
  cornell-grill/sim_infra: assets/robots/tianji_marvin_ccs_696
  (arm: tianji_marvin_ccs_696; hand meshes: ../wuji_hand)
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path

# End-effector link names (last actuated arm link, the wrist before the hand)
LEFT_EE_LINK_NAME = "Link7_L"
RIGHT_EE_LINK_NAME = "Link7_R"

# Arm joint names (resolved to indexes at controller init)
LEFT_ARM_JOINT_NAMES = [f"Joint{i}_L" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"Joint{i}_R" for i in range(1, 8)]

# Hand joint names: 5 fingers x 4 joints per hand, in qpos order
LEFT_HAND_JOINT_NAMES = [
    f"left_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]
RIGHT_HAND_JOINT_NAMES = [
    f"right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]

# Offset (in the wrist/EE link frame) from the Link7 origin to the hand palm
# origin. The palm body is mounted at pos="0 -0.09 0" relative to Link7.
# Tune toward the fingertips if grasp candidates should target a point further
# into the hand.
EE_PALM_OFFSET = [0.0, -0.09, 0.0]

# Default home position (54 DOF) matching the MJCF "home" keyframe:
# arms in a neutral folded pose, all fingers open (0).
# Joint1 is reduced from the MJCF value (0.712) to 0.2 so the hands rest
# higher, letting the base sit closer to the tabletop without the hands
# penetrating it.
# Left-arm Joint5 sign flipped (+0.356) to match the real left arm convention.
_ARM_HOME_LEFT = [0.2, -0.963, 0.0, -0.85, 0.356, 0.0, 0.0]
_ARM_HOME_RIGHT = [-0.2, -0.963, 0.0, -0.85, -0.356, 0.0, 0.0]
HOME_JOINT_POSITIONS = torch.zeros(54)
HOME_JOINT_POSITIONS[0:7] = torch.tensor(_ARM_HOME_LEFT)      # Joint1_L .. Joint7_L
# indices 7:27  -> left hand fingers (open, 0)
HOME_JOINT_POSITIONS[27:34] = torch.tensor(_ARM_HOME_RIGHT)   # Joint1_R .. Joint7_R
# indices 34:54 -> right hand fingers (open, 0)


# Default combined sub-controller configuration:
#   arms  -> eef_pose (IK-based end-effector control)
#   hands -> joint_position (direct per-joint position control, 20 DOF each)
_DEFAULT_SUB_CONTROLLERS = {
    "left_arm": {
        "controller_type": "eef_pose",
        "joint_names": LEFT_ARM_JOINT_NAMES,
        "target_link": LinkAttachment(
            link_name=LEFT_EE_LINK_NAME, offset_pos=EE_PALM_OFFSET,
        ),
    },
    "right_arm": {
        "controller_type": "eef_pose",
        "joint_names": RIGHT_ARM_JOINT_NAMES,
        "target_link": LinkAttachment(
            link_name=RIGHT_EE_LINK_NAME, offset_pos=EE_PALM_OFFSET,
        ),
    },
    "left_hand": {
        "controller_type": "joint_position",
        "joint_names": LEFT_HAND_JOINT_NAMES,
    },
    "right_hand": {
        "controller_type": "joint_position",
        "joint_names": RIGHT_HAND_JOINT_NAMES,
    },
}


class TianjiMarvinWuji(RobotDescriptionNode):
    """Tianji Marvin 14-DOF dual-arm robot with Wuji 5-finger hands.

    Each arm has 7 revolute joints; each Wuji hand has 5 fingers with 4
    independently position-actuated joints (20 DOF per hand).

    Default controller is ``"combined"``:
      - ``"left_arm"``   sub-controller: ``eef_pose``
      - ``"right_arm"``  sub-controller: ``eef_pose``
      - ``"left_hand"``  sub-controller: ``joint_position`` (20 finger joints)
      - ``"right_hand"`` sub-controller: ``joint_position`` (20 finger joints)

    Args:
        world: GenesisWorld instance
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller. For the default
            ``"combined"`` controller, pass ``sub_controllers=...`` to override.

    Example::

        from unienv_genesis import GenesisWorld
        from unienv_genesis_collection.robots.tianji_marvin_wuji import (
            TianjiMarvinWuji,
        )

        world = GenesisWorld()
        robot = TianjiMarvinWuji(world, name="tianji")

        # Action dict expected by the default combined controller:
        # {
        #   "left_arm":   <eef_pose action tensor>,
        #   "right_arm":  <eef_pose action tensor>,
        #   "left_hand":  <20-d joint position tensor>,
        #   "right_hand": <20-d joint position tensor>,
        # }
    """

    keyframes = {
        "reset": RobotKeyframe(joint_position=HOME_JOINT_POSITIONS),
    }

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "tianji",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        if controller_type == "eef_pose" and "target_link" not in controller_kwargs:
            controller_kwargs["target_link"] = LinkAttachment(link_name=LEFT_EE_LINK_NAME)

        super().__init__(
            world=world,
            description_file=str(get_asset_path(
                "robots", "tianji_marvin_ccs_696",
                "tianji_marvin_CCS_real_limit_wuji.xml",
            )),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
