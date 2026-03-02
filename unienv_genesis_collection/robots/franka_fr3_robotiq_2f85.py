"""Franka Research FR3 robot descriptor with Robotiq 2F-85 gripper.

MJCF sources (Apache 2.0):
  Arm:     https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3
  Gripper: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85_v4
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis_collection.asset_path import get_asset_path

# Default home joint positions: FR3 arm (7 DOF) + Robotiq 2F-85 (6 passive joints, open)
HOME_JOINT_POSITIONS = torch.tensor([
    0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853,  # arm
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,                      # gripper (open)
])

# End-effector link name for IK control (arm wrist, before gripper)
EE_LINK_NAME = "fr3_link7"

# Joint names for arm and gripper (resolved to indexes at controller init)
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
GRIPPER_DRIVER_NAMES = ["g_left_driver_joint", "g_right_driver_joint"]

# Pad link names for grasp contact detection
GRIPPER_PAD_LINK_NAMES = ["g_left_pad", "g_right_pad"]


def _make_is_grasping(pad_link_names):
    """Create a grasp-detection callable checking contacts on specific pad links."""
    _cached_pad_indices = {}  # id(entity) -> tensor of global link indices

    def is_grasping(robot_entity, target_entity):
        # Lazily resolve pad link global indices (cached per entity instance).
        key = id(robot_entity)
        if key not in _cached_pad_indices:
            indices = []
            for link in robot_entity.links:
                if link.name in pad_link_names:
                    indices.append(link.idx)
            _cached_pad_indices[key] = torch.tensor(indices, dtype=torch.long)
        pad_indices = _cached_pad_indices[key]

        contacts = robot_entity.get_contacts(with_entity=target_entity)

        # Check if any contact involves a pad link on the robot side.
        link_a = contacts["link_a"]  # (n_contacts,) or (B, n_contacts)
        link_b = contacts["link_b"]
        is_pad = (
            (link_a.unsqueeze(-1) == pad_indices).any(dim=-1)
            | (link_b.unsqueeze(-1) == pad_indices).any(dim=-1)
        )  # (n_contacts,) or (B, n_contacts)

        if contacts["force_a"].dim() == 2:
            # Unbatched: (n_contacts, 3) — is_pad is (n_contacts,)
            return is_pad.any()
        else:
            # Batched: valid_mask is (B, n_contacts)
            return (contacts["valid_mask"] & is_pad).any(dim=-1)

    return is_grasping


# Default combined sub-controller configuration
_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_names": ARM_JOINT_NAMES,
        "target_link": EE_LINK_NAME,
        # Virtual EEF offset from the target_link (fr3_link7) origin to
        # the gripper fingertip along the approach direction, centred in
        # the open direction.  This point is where grasp candidates place
        # the EEF (see base_object.list_box_gripper_grasping_candidates_local).
        #   0.107 m (FR3 flange) + ~0.159 m (Robotiq base to fingertip) = 0.266 m
        "eef_offset_pos": [0.0, 0.0, 0.266],
    },
    "gripper": {
        "controller_type": "mimic_joint_position",
        "joint_names": GRIPPER_DRIVER_NAMES,
        # Gripper convention: -1 = fully open (driver=0), +1 = fully closed (driver=0.9)
        # finger_pos = action * multiplier + shift
        "multiplier": 0.45,
        "shift": 0.45,
        "control_low": -1,
        "control_high": 1,
        "is_grasping_fn": _make_is_grasping(GRIPPER_PAD_LINK_NAMES),
        # Grasp frame directions in the EEF frame (fr3_link7 body frame).
        # approach_direction: axis along which the gripper approaches the
        #   object (fingertip direction); eef_offset is measured along this.
        # open_direction: axis along which the fingers separate.
        "approach_direction": [0.0, 0.0, 1.0],
        "open_direction": [0.0, 1.0, 0.0],
        "gripper_width": 0.085,
        "gripper_depth": 0.035,
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
            controller_kwargs["target_link"] = EE_LINK_NAME

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
