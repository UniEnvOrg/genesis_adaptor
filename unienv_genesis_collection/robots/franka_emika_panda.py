"""Franka Emika Panda robot descriptor with Franka Hand gripper.

MJCF sources (Apache 2.0):
  Full robot: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis_collection.asset_path import get_asset_path

# Default home joint positions for Panda arm (7 DOF) + Franka Hand (2 DOF, fully open)
HOME_JOINT_POSITIONS = torch.tensor([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])

# End-effector link name for IK control
EE_LINK_NAME = "hand"

# Finger link names for grasp contact detection
FINGER_PAD_LINK_NAMES = ["left_finger", "right_finger"]


def _make_is_grasping(pad_link_names):
    """Create a grasp-detection callable checking contacts on specific pad links."""
    _cached_pad_indices = {}  # id(entity) -> tensor of global link indices

    def is_grasping(robot_entity, target_entity):
        key = id(robot_entity)
        if key not in _cached_pad_indices:
            indices = []
            for link in robot_entity.links:
                if link.name in pad_link_names:
                    indices.append(link.idx)
            _cached_pad_indices[key] = torch.tensor(indices, dtype=torch.long)
        pad_indices = _cached_pad_indices[key]

        contacts = robot_entity.get_contacts(with_entity=target_entity)

        link_a = contacts["link_a"]
        link_b = contacts["link_b"]
        is_pad = (
            (link_a.unsqueeze(-1) == pad_indices).any(dim=-1)
            | (link_b.unsqueeze(-1) == pad_indices).any(dim=-1)
        )

        if contacts["force_a"].dim() == 2:
            return is_pad.any()
        else:
            return (contacts["valid_mask"] & is_pad).any(dim=-1)

    return is_grasping


# Default combined sub-controller configuration
_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_indexes": list(range(7)),   # joint1 … joint7
        "target_link": EE_LINK_NAME,
        # Virtual EEF offset from the target_link (hand) origin to the
        # fingertip along the approach direction, centred in the open
        # direction.  This point is where grasp candidates place the EEF
        # (see base_object.list_box_gripper_grasping_candidates_local).
        #   finger base z = 0.0584 m + finger pad extent z = 0.053 m
        "eef_offset_pos": [0.0, 0.0, 0.1114],
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
        "is_grasping_fn": _make_is_grasping(FINGER_PAD_LINK_NAMES),
        # Grasp frame directions in the EEF frame (hand body frame).
        # approach_direction: axis along which the gripper approaches the
        #   object (fingertip direction); eef_offset is measured along this.
        # open_direction: axis along which the fingers separate.
        "approach_direction": [0.0, 0.0, 1.0],
        "open_direction": [0.0, 1.0, 0.0],
        "gripper_width": 0.08,    # 2 × 0.04 m max finger opening
        "gripper_depth": 0.02,    # fingertip pad depth
    },
}


class FrankaEmikaPanda(RobotDescriptionNode):
    """Franka Emika Panda 7-DOF arm with Franka Hand gripper.

    The MJCF comes from the mujoco_menagerie ``franka_emika_panda`` model.

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
        from unienv_genesis_collection.robots.franka_emika_panda import FrankaEmikaPanda

        world = GenesisWorld()
        robot = FrankaEmikaPanda(world, name="franka")

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
            description_file=get_asset_path("robots", "franka_emika_panda", "panda.xml"),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
