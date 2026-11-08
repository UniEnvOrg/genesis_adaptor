"""Franka Research FR3 robot descriptor with a Sharpa Wave hand.

The hand (left- or right-hand model) is mounted by the back of its palm on the FR3 flange, its palm facing along the
flange axis (``scripts/build_fr3_sharpa_mjcf.py`` builds ``franka_fr3_sharpa/fr3_sharpa_<side>.xml`` from
``franka_fr3_hand/fr3_hand.xml`` and ``sharpa_wave/<side>_hand.xml``). A bimanual setup is two of these nodes.

DOF summary (qpos order):
  Arm:  7 revolute joints (fr3_joint1 .. fr3_joint7)
  Hand: 22 joints, each driven: the thumb's CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP; the index / middle / ring finger's
        MCP_FE (flexion), MCP_AA (side), PIP, DIP; the pinky's CMC, then the same four.

Default ``"combined"`` controller:
  - ``"arm"``:  ``eef_pose`` (IK) on the hand's ``palm`` frame (+x out of the palm, +z along the fingers).
  - ``"hand"``: ``joint_position``, the 22 joints (:data:`HAND_JOINT_NAMES`), the model's position gains.

MJCF sources: mujoco_menagerie ``franka_fr3`` and ``sharpa_wave`` (both Apache 2.0).
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
_FINGER_JOINTS = ("MCP_FE", "MCP_AA", "PIP", "DIP")
HAND_JOINT_NAMES = (
    [f"thumb_{j}" for j in ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP")]
    + [f"{f}_{j}" for f in ("index", "middle", "ring") for j in _FINGER_JOINTS]
    + ["pinky_CMC"] + [f"pinky_{j}" for j in _FINGER_JOINTS]
)
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_{p}" for f in ("index", "middle", "ring", "pinky") for p in ("PP", "MP", "DP")) + ("thumb_PP", "thumb_DP")
#: Position gains per joint type (N m / rad), the model's actuators'.
_KP = {"CMC_FE": 6.95, "CMC_AA": 13.2, "MCP_FE": 4.76, "MCP_AA": 6.62, "IP": 0.9, "PIP": 0.9, "DIP": 0.9, "CMC": 1.38}
HAND_KP = [_KP[name.split("_", 1)[1]] for name in HAND_JOINT_NAMES]
HAND_KD = [0.05 if kp < 2.0 else 0.2 for kp in HAND_KP]
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


class FrankaFR3WithSharpa(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with a 22-DOF Sharpa Wave hand.

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
        name: str = "franka_sharpa",
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
            description_file=str(get_asset_path("robots", "franka_fr3_sharpa", f"fr3_sharpa_{side}.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
