"""Franka Research FR3 robot descriptor with Robotiq 2F-85 gripper.

MJCF sources (Apache 2.0):
  Arm:     https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3
  Gripper: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85_v4
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.sensors.camera_node import CameraSensorNodeOptions
from unienv_genesis.options.attach_option import AttachFromName, LinkAttachment
from unienv_genesis_collection.sensors.cameras.zed import get_zed_mini_camera_options, ZedSupportedResolution
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

# Default home joint positions: FR3 arm (7 DOF) + Robotiq 2F-85 (6 passive joints, open)
ARM_HOME_JOINT_POSITIONS = torch.tensor([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRIPPER_HOME_JOINT_POSITIONS = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, GRIPPER_HOME_JOINT_POSITIONS])

# End-effector link name for IK control (arm wrist, before gripper)
EE_LINK_NAME = "g_base"

# Joint names for arm and gripper (resolved to indexes at controller init)
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
GRIPPER_DRIVER_NAMES = ["g_left_driver_joint", "g_right_driver_joint"]

# Pad link names for grasp contact detection
GRIPPER_PAD_LINK_NAMES = ["g_left_pad", "g_right_pad"]

# Max joint velocity used by the arm's IK controller (rad/s).  This
# rate-limits the per-step IK joint-position targets.  Originally the DROID
# real-robot teleop limit (0.2 rad/s —
# Ref: https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py#L10
# ); raised to 0.4 rad/s for simulation, where the conservative teleop limit
# made the arm too slow to track transport waypoints (the Kp/Kd gains below are
# the MuJoCo-menagerie values, which tolerate this).
DROID_FRANKA_ARM_IK_VEL_LIMIT = 0.4

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
        "joint_names": ARM_JOINT_NAMES,
        # Virtual EEF offset from the target_link (g_base) origin to the gripper fingertip along the approach direction, centred in the open direction.  This point is where grasp candidates place the EEF (see base_object.list_box_gripper_grasping_candidates_local).
        # ~0.159 m (Robotiq base to fingertip)
        "target_link": LinkAttachment(
            link_name=EE_LINK_NAME, offset_pos=[0.0, 0.0, 0.159],
        ),
        "Kp": torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000], dtype=torch.float32),
        "Kd": torch.tensor([450, 450, 350, 350, 200, 200, 200], dtype=torch.float32),
        # DROID IK/controller velocity limit for the Franka arm (rad/s).
        "max_joint_pos_vel": DROID_FRANKA_ARM_IK_VEL_LIMIT,
    },
    "gripper": {
        "controller_type": "mimic_joint_position",
        "joint_names": GRIPPER_DRIVER_NAMES,
        "Kp": 100.0,
        "Kd": 10.0,
        # Gripper convention: -1 = fully open (driver=0), +1 = fully closed (driver=0.9)
        # finger_pos = action * multiplier + shift
        "multiplier": 0.45,
        "shift": 0.45,
        "control_low": -1,
        "control_high": 1,
        "is_grasping_fn": make_is_grasping(GRIPPER_PAD_LINK_NAMES),
        # Grasp frame directions in the EEF frame (g_base body frame).
        # approach_direction: axis along which the gripper approaches the
        #   object (fingertip direction); eef_offset is measured along this.
        # open_direction: axis along which the fingers separate.
        "approach_direction": [0.0, 0.0, 1.0],
        "open_direction": [1.0, 0.0, 0.0],
        "gripper_width": 0.085,
        "gripper_depth": 0.035,
        "gripper_open_action": -1.0,  # action value for fully open gripper
        "gripper_close_action": 1.0,  # action value for fully closed gripper
    },
}


class FrankaFR3WithRobotiq2F85(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with Robotiq 2F-85 gripper.

    The arm MJCF comes from the mujoco_menagerie ``franka_fr3`` model; the
    gripper comes from the mujoco_menagerie ``robotiq_2f85_v4`` model.
    They are combined in ``assets/robots/franka_fr3_robotiq_2f85/fr3_robotiq_2f85.xml``.

    Default controller is ``"combined"``:
      - ``"arm"``     sub-controller: ``eef_pose`` (IK-based end-effector control)
      - ``"gripper"`` sub-controller: ``mimic_joint_position`` (single scalar)

    Gripper action convention::

        -1  ->  fully open  (driver joints at 0 rad)
        +1  ->  fully closed (driver joints at 0.9 rad)

    Args:
        world: GenesisWorld instance
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller. For the default
            ``"combined"`` controller, pass ``sub_controllers=...`` to override
            the arm/gripper configuration.

    Example::

        from unienv_genesis import GenesisWorld
        from unienv_genesis_collection.robots.franka_fr3_robotiq_2f85 import (
            FrankaFR3WithRobotiq2F85,
        )

        world = GenesisWorld()
        robot = FrankaFR3WithRobotiq2F85(world, name="franka")

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

        # The MJCF references meshes in sibling asset directories
        # (``../franka_fr3/assets/`` and ``../robotiq_assets/``). ``get_asset_path``
        # only downloads the single requested description file, so the mesh
        # subtrees must be pre-fetched explicitly for Genesis to resolve the
        # relative mesh paths during MJCF parsing.
        get_asset_path("robots", "franka_fr3")
        get_asset_path("robots", "robotiq_assets")

        super().__init__(
            world=world,
            description_file=str(get_asset_path(
                "robots", "franka_fr3_robotiq_2f85",
                "fr3_robotiq_2f85.xml",
            )),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )

    def get_droid_gripper_cam_options(
        self,
        resolution : ZedSupportedResolution,
        **kwargs
    ) -> CameraSensorNodeOptions:
        # https://github.com/arhanjain/sim-evals/blob/main/src/sim_evals/environments/droid_environment.py
        orig_pos = (0.011, -0.031, -0.074)
        orig_quat = (-0.420, 0.570, 0.576, -0.409)
        """
        We apply
        R_usd_to_mjcf:
        [[0 -1 0]
        [0 0 -1]
        [1 0 0]]
        """
        
        pos  = (0.031, 0.074, -0.011)
        quat = (0.0025, 0.0085, 0.9873, 0.1585)  # (w, x, y, z)
        return get_zed_mini_camera_options(
            resolution,
            pos=pos,
            quat=quat,
            attach=AttachFromName(self.name, "g_base"),
            near=kwargs.pop("near", 0.01),
            **kwargs
        )
