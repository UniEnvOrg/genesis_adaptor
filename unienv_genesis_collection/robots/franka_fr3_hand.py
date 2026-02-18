"""Franka Research FR3 robot descriptor with Franka Hand gripper.

MJCF sources (Apache 2.0):
  Arm:     https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3
  Gripper: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe, RobotDescriptionNode
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis_collection.asset_path import get_asset_path

# Default home joint positions for FR3 arm (7 DOF) + Franka Hand (2 DOF, fully open)
HOME_JOINT_POSITIONS = torch.tensor([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04])

# End-effector link name for IK control
EE_LINK_NAME = "hand"

# Default combined sub-controller configuration
_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_indexes": list(range(7)),   # fr3_joint1 … fr3_joint7
        "target_link": EE_LINK_NAME,
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
            controller_kwargs["target_link"] = EE_LINK_NAME

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
