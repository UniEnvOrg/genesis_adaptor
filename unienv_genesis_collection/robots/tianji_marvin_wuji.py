"""Tianji Marvin dual-arm robot with Wuji 5-finger dexterous hands.

The Marvin robot (CCS 696 variant) is a 14-DOF dual-arm manipulator with a
fixed torso base. Each arm has 7 revolute joints and a Wuji 5-finger
dexterous hand attached at the wrist. Each hand has 5 fingers with 4 joints
each (20 joints per hand), every joint independently position-actuated.
The simulated arm sub-controllers limit joint-position targets to 0.2 rad/s,
aligned with the real hardware adaptor; the hand sub-controllers are unlimited.

The ``eef_pose`` sub-controllers drive a *virtual* end-effector frame that
coincides with the physical ``left_wuji_palm`` / ``right_wuji_palm`` body
frame (the palm inside the hand), **not** just the ``Link7`` wrist origin.
Both the palm's position *and* its orientation relative to ``Link7`` are
modelled via the ``offset_pos`` / ``offset_quat`` of the ``LinkAttachment``
(see ``EE_PALM_OFFSET`` and ``{LEFT,RIGHT}_EE_PALM_QUAT`` below).  This means
that commanding a target palm pose in world frame drives IK on the arm such
that ``Link7`` ends up at the pose that places the palm frame exactly at the
target.

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

from xbarray.transformations.rotation_conversions.pytorch import (
    matrix_to_quaternion,
)

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
MAX_JOINT_POS_VEL = 0.2  # rad/s — aligned with the real hardware adaptor (TianjiArmEefActor)
# Minimum per-step joint-position delta (rad) that the rate-limit clip allows
# through. Floors ``max_joint_pos_vel * dt`` so that a high physics-substep
# count (small control dt) cannot shrink the corrective PD error to zero —
# without this floor the clipped target tracks the sagging current pose and
# the sim arm drifts down under gravity instead of holding the commanded EEF
# pose. Set conservatively: only lifts the effective rate when dt is small.
MIN_JOINT_POS_DELTA_PER_STEP = 0.1
# Sim arm PD gains. The URDF defaults (Kp=[100,100,80,80,40,40,40],
# Kv=[10,10,8,8,4,4,4]) are too weak to hold the arm against gravity: a
# hold-home command sagged ~0.18 m and the rate-limit clip then tracked the
# sagging pose down. Raising both gains ~20x lets the PD hold/lift the arm
# (verified: hold sag < 0.01 m, +6 cm target tracked to ~+0.05 m, stable).
# Per-joint values follow the URDF joint order Joint1..Joint7 (shoulder->wrist),
# scaled from the defaults so the shoulder (heavier) joints get the most gain.
_ARM_KP_SCALE = 20.0
_ARM_KD_SCALE = 20.0
_ARM_KP = [100.0 * _ARM_KP_SCALE, 100.0 * _ARM_KP_SCALE, 80.0 * _ARM_KP_SCALE,
          80.0 * _ARM_KP_SCALE, 40.0 * _ARM_KP_SCALE, 40.0 * _ARM_KP_SCALE,
          40.0 * _ARM_KP_SCALE]
_ARM_KD = [10.0 * _ARM_KD_SCALE, 10.0 * _ARM_KD_SCALE, 8.0 * _ARM_KD_SCALE,
          8.0 * _ARM_KD_SCALE, 4.0 * _ARM_KD_SCALE, 4.0 * _ARM_KD_SCALE,
          4.0 * _ARM_KD_SCALE]

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
# origin. The palm body is mounted at pos="0 -0.09 0" relative to Link7 (see
# ``left_wuji_palm`` / ``right_wuji_palm`` in the MJCF, lines 131 and 301).
# Tune toward the fingertips if grasp candidates should target a point further
# into the hand.
EE_PALM_OFFSET = [0.0, -0.09, 0.0]


def _mjcf_xyaxes_to_palm_quat(xyaxes_x: str, xyaxes_y: str) -> torch.Tensor:
    """Build a (3, 3) rotation matrix from an MJCF ``xyaxes`` spec and
    convert it to a ``(w, x, y, z)`` quaternion (XBArray convention; the same
    convention consumed by ``LinkAttachment.offset_quat`` and the controller
    math in ``base._link_to_virtual`` / ``base._virtual_to_link`` via
    ``quaternion_multiply``).

    MJCF ``xyaxes`` encodes the rotated frame's x-axis and y-axis expressed in
    the parent (here ``Link7``) frame; the z-axis is taken as ``x cross y``
    to form a right-handed orthonormal basis.  The columns of the resulting
    matrix are ``(x | y | z)``.
    """
    x = torch.tensor([float(v) for v in xyaxes_x.split()], dtype=torch.float32)
    y = torch.tensor([float(v) for v in xyaxes_y.split()], dtype=torch.float32)
    x = x / x.norm()
    z = torch.linalg.cross(x, y)
    z = z / z.norm()
    y = torch.linalg.cross(z, x)  # re-orthogonalize for numerical safety
    rot = torch.stack([x, y, z], dim=-1)  # columns (x | y | z), shape (3, 3)
    quat = matrix_to_quaternion(rot)      # (w, x, y, z), real part first
    if quat[0] < 0:                       # canonicalize hemisphere
        quat = -quat
    return quat


# Orientation of the wuji palm body relative to the wrist (``Link7``) frame,
# for each hand.  Derived from the ``xyaxes`` attribute of the palm body in
# ``assets/robots/tianji_marvin_ccs_696/tianji_marvin_CCS_real_limit_wuji.xml``:
#
#   line 131 (left_wuji_palm):  xyaxes="-0.707107 0 0.707107  -0.707107 0 -0.707107"
#   line 301 (right_wuji_palm): xyaxes="0.707107 0 0.707107  -0.707107 0 0.707107"
#
# i.e. for the left palm:  x=(-0.707,0, 0.707), y=(-0.707,0,-0.707), z=x cross y=(0,-1,0);
# and for the right palm: x=( 0.707,0, 0.707), y=(-0.707,0, 0.707), z=x cross y=(0,-1,0).
#
# These are *mirrored* between the two hands (different x / y axes), so the
# two quaternions differ and must NOT be reused across left/right.  Both keep
# the palm's z-axis pointing along Link7's -y (the same direction the palm is
# translated), matching the physical hand mount.
LEFT_EE_PALM_QUAT = _mjcf_xyaxes_to_palm_quat(
    "-0.707107 0 0.707107", "-0.707107 0 -0.707107",
)
RIGHT_EE_PALM_QUAT = _mjcf_xyaxes_to_palm_quat(
    "0.707107 0 0.707107", "-0.707107 0 0.707107",
)

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
        "max_joint_pos_vel": MAX_JOINT_POS_VEL,
        "min_joint_pos_delta_per_step": MIN_JOINT_POS_DELTA_PER_STEP,
        "Kp": _ARM_KP,
        "Kd": _ARM_KD,
        "target_link": LinkAttachment(
            link_name=LEFT_EE_LINK_NAME,
            offset_pos=EE_PALM_OFFSET,
            offset_quat=LEFT_EE_PALM_QUAT,
        ),
    },
    "right_arm": {
        "controller_type": "eef_pose",
        "joint_names": RIGHT_ARM_JOINT_NAMES,
        "max_joint_pos_vel": MAX_JOINT_POS_VEL,
        "min_joint_pos_delta_per_step": MIN_JOINT_POS_DELTA_PER_STEP,
        "Kp": _ARM_KP,
        "Kd": _ARM_KD,
        "target_link": LinkAttachment(
            link_name=RIGHT_EE_LINK_NAME,
            offset_pos=EE_PALM_OFFSET,
            offset_quat=RIGHT_EE_PALM_QUAT,
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
    Arm joint-position targets are rate-limited to 0.2 rad/s to match the real
    hardware adaptor; the hand sub-controllers have no rate limit.

    Default controller is ``"combined"``:
      - ``"left_arm"``   sub-controller: ``eef_pose`` targeting the
        ``left_wuji_palm`` body frame (position ``EE_PALM_OFFSET`` and
        orientation ``LEFT_EE_PALM_QUAT`` relative to ``Link7_L``).
      - ``"right_arm"``  sub-controller: ``eef_pose`` targeting the
        ``right_wuji_palm`` body frame (position ``EE_PALM_OFFSET`` and
        orientation ``RIGHT_EE_PALM_QUAT`` relative to ``Link7_R``).
      - ``"left_hand"``  sub-controller: ``joint_position`` (20 finger joints)
      - ``"right_hand"`` sub-controller: ``joint_position`` (20 finger joints)

    Because the arm sub-controllers carry a non-identity ``offset_quat``, the
    commanded / observed EEF pose refers to the true palm frame, not just the
    wrist (``Link7``) origin with wrist orientation.

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
