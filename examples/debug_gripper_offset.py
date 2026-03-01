"""Debug script: measure the correct eef_offset_pos for the Marvin robot.

Loads the robot at the home pose (all joints at zero), queries the world-frame
positions of left_link7 and the gripper pad links, and computes the offset
from link7 to the pad tip (tip of approach vector, centered between pads).
"""

from __future__ import annotations

import torch
import genesis as gs
from genesis.constants import backend as gs_backend

gs.init(backend=gs_backend.cpu)

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.robots.tianji_marvin_robotiq_2f85 import (
    TianjiMarvinWithRobotiq2F85,
)
from unienv_interface.world import WorldEnv


def main():
    world = GenesisWorld()
    robot = TianjiMarvinWithRobotiq2F85(world, name="tianji")

    env = WorldEnv(world, [robot])
    env.reset(seed=0, reload=True)

    entity = robot.entity

    # --- Retrieve link references ---
    link7 = entity.get_link("left_link7")
    lg_left_pad = entity.get_link("lg_left_pad")
    lg_right_pad = entity.get_link("lg_right_pad")

    link7_pos = link7.get_pos().detach()
    link7_quat = link7.get_quat().detach()

    left_pad_pos = lg_left_pad.get_pos().detach()
    left_pad_quat = lg_left_pad.get_quat().detach()
    right_pad_pos = lg_right_pad.get_pos().detach()
    right_pad_quat = lg_right_pad.get_quat().detach()

    print(f"link7 pos:      {link7_pos.tolist()}")
    print(f"link7 quat:     {link7_quat.tolist()}")
    print(f"left_pad pos:   {left_pad_pos.tolist()}")
    print(f"left_pad quat:  {left_pad_quat.tolist()}")
    print(f"right_pad pos:  {right_pad_pos.tolist()}")
    print(f"right_pad quat: {right_pad_quat.tolist()}")

    # --- Pad tip positions in world frame ---
    # From the MJCF, robotiq_pad_box2 is the outermost pad geom:
    #   pos="0.043258 0 0.13875", size="0.002 0.011 0.009375"
    # The tip (outermost point along the pad's local Z) is at z = 0.13875 + 0.009375 = 0.148125
    # The pad contact center in the pad's local frame (centered on opening, at tip of approach):
    pad_tip_local = torch.tensor([0.0, 0.0, 0.148125])

    left_tip_world = left_pad_pos + quaternion_apply(left_pad_quat, pad_tip_local)
    right_tip_world = right_pad_pos + quaternion_apply(right_pad_quat, pad_tip_local)

    print(f"\nleft pad tip (world):  {left_tip_world.tolist()}")
    print(f"right pad tip (world): {right_tip_world.tolist()}")

    # --- EEF point: midpoint between pad tips ---
    eef_world = (left_tip_world + right_tip_world) / 2.0
    print(f"EEF midpoint (world):  {eef_world.tolist()}")

    # --- Transform to link7's local frame ---
    link7_quat_inv = quaternion_invert(link7_quat)
    eef_local = quaternion_apply(link7_quat_inv, eef_world - link7_pos)

    print(f"\n=== Measured eef_offset_pos (link7 local frame): {[round(x, 6) for x in eef_local.tolist()]} ===")
    print(f"=== Current eef_offset_pos:                      [0.0, 0.0, 0.18] ===")
    print(f"=== Difference:                                  {[round(x - c, 6) for x, c in zip(eef_local.tolist(), [0.0, 0.0, 0.18])]} ===")

    # --- Also show pad center (not tip) for comparison ---
    pad_center_local = torch.tensor([0.0, 0.0, 0.129375])  # midpoint of pad_box1 and pad_box2
    left_center_world = left_pad_pos + quaternion_apply(left_pad_quat, pad_center_local)
    right_center_world = right_pad_pos + quaternion_apply(right_pad_quat, pad_center_local)
    center_world = (left_center_world + right_center_world) / 2.0
    center_local = quaternion_apply(link7_quat_inv, center_world - link7_pos)
    print(f"\nPad center offset (link7 local):  {[round(x, 6) for x in center_local.tolist()]}")

    env.close()


if __name__ == "__main__":
    main()
