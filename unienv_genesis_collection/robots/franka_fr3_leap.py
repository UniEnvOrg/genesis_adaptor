"""Franka Research FR3 robot descriptor with a LEAP Hand v2 Adv.

The hand (left- or right-hand model) is mounted on the back of the hand on the FR3 flange, its palm facing along the
flange axis (``scripts/build_fr3_leap_mjcf.py`` builds ``franka_fr3_leap/fr3_leap_<side>.xml`` from
``franka_fr3_hand/fr3_hand.xml`` and the upstream LEAP Hand v2 Adv URDF; the upstream repository has no license, so
the generated model stays local). A bimanual setup is two of these nodes.

DOF summary (qpos order):
  Arm:  7 revolute joints (fr3_joint1 .. fr3_joint7)
  Hand: 21 joints: per finger (index, middle, ring, pinky) mcpf (flexion), mcps (side), pip, dip; the thumb's
        thumb_mcpf, thumb_mcps, thumb_ip; the palm folds palm_4_finger, palm_thumb. Each finger's dip follows its
        pip (an equality constraint: one tendon on the hardware), so 17 joints are driven.

Default ``"combined"`` controller:
  - ``"arm"``:  ``eef_pose`` (IK) on the hand's ``palm`` frame (+x out of the palm, +z along the fingers).
  - ``"hand"``: ``joint_position``, the 17 driven joints (:data:`HAND_JOINT_NAMES`).
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

FINGERS = ("index", "middle", "ring", "pinky")
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
#: The driven hand joints (each finger's dip follows its pip).
HAND_JOINT_NAMES = [f"{f}_{j}" for f in FINGERS for j in ("mcpf", "mcps", "pip")] + [
    "thumb_mcpf", "thumb_mcps", "thumb_ip", "palm_4_finger", "palm_thumb"]
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_{p}" for f in (*FINGERS, "thumb") for p in ("proximal", "middle", "distal")
                              if not (f == "thumb" and p == "middle"))
#: Finger PD gains (N m / rad; the motors give ~0.5 N m, the palm folds 1-1.4 N m).
HAND_KP = [3.0] * 15 + [5.0, 5.0]
HAND_KD = [0.1] * 17
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, HAND_HOME_JOINT_POSITIONS])

_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_names": ARM_JOINT_NAMES,
        "target_link": LinkAttachment(link_name=PALM_LINK_NAME),
        "max_joint_pos_vel": DROID_FRANKA_ARM_IK_VEL_LIMIT,
    },
    "hand": {
        "controller_type": "joint_position",
        "joint_names": HAND_JOINT_NAMES,
        "Kp": HAND_KP,
        "Kd": HAND_KD,
        "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
    },
}


class FrankaFR3WithLeap(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with a LEAP Hand v2 Adv (21 joints, 17 driven).

    Args:
        world: GenesisWorld instance
        side: ``"left"`` or ``"right"``: which hand model is mounted.
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller; ``sub_controllers=...`` overrides the defaults.
    """

    keyframes = {"reset": RobotKeyframe(joint_position=HOME_JOINT_POSITIONS, joint_names=ARM_JOINT_NAMES + HAND_JOINT_NAMES)}

    def __init__(
        self,
        world: GenesisWorld,
        side: str = "right",
        name: str = "franka_leap",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        self.side = side
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_leap", f"fr3_leap_{side}.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
