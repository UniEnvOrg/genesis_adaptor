"""Example: Tianji Marvin with Robotiq grippers on a cluttered table.

Loads a Marvin dual-arm robot, a table, a ``CubeRelocateTask`` (target red
cube placed at a random position with a random goal), and a
``ClutteredTableManager`` with primitive shape distractors that avoids the
target cube.  Renders a video of the robot reaching toward the target cube.

Usage::

    python examples/cluttered_table_marvin.py
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple, TYPE_CHECKING

import torch

import genesis as gs
from genesis.constants import backend as gs_backend

gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import (
    CameraSensorNode,
    CameraSensorNodeOptions,
)
from unienv_genesis_collection.objects.ycb_parser import YCBParser
from unienv_genesis_collection.objects import BoxObject, SphereObject, CylinderObject
from unienv_genesis_collection.objects.base_object import (
    ManipulatableObject,
    bbox_size,
    bbox_center,
)
from unienv_genesis_collection.robots.tianji_marvin_robotiq_2f85 import (
    TianjiMarvinWithRobotiq2F85,
)
from unienv_genesis_collection.scenes.cluttered_table_manager import (
    ClutteredTableManager,
)
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper

if TYPE_CHECKING:
    from unienv_genesis.world.genesis_world import GenesisWorld as GenesisWorldT

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_cluttered_marvin")

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
TABLE_SURFACE_Z = 0.6
TABLE_SIZE = (1.0, 0.6)
TABLE_POS = (0.0, 0.0, TABLE_SURFACE_Z)
ROBOT_Z = TABLE_SURFACE_Z + 0.35  # base above table for downward reach
TOTAL_STEPS = 350
N_EPISODES = 2

_PARK_POS = (0.0, 0.0, -1000.0)
_PARK_QUAT = (1.0, 0.0, 0.0, 0.0)


# ============================================================================
# CubeRelocateTask — task node defined directly in the example
# ============================================================================


class CubeRelocateTask(WorldNode):
    """Place a target object at a random table position and provide a random
    goal location.

    Provides:
    - **observation**: ``{"target_position": (3,), "goal_position": (3,)}``
    - **reward**: ``-distance(target, goal)`` (dense)
    - **termination**: ``True`` when ``distance < success_threshold``

    The task node runs at priority **26** (reload & reset), which is
    **before** ``ClutteredTableManager`` (priority 25).  After placing the
    target it calls
    :meth:`ClutteredTableManager.set_next_reset_avoid_spaces` so clutter
    objects don't overlap with the target.
    """

    has_reward = True
    has_termination_signal = True
    has_truncation_signal = False

    reload_priorities = {26}
    after_reload_priorities = {26}
    reset_priorities = {26}
    post_environment_step_priorities = {0}

    def __init__(
        self,
        world: "GenesisWorldT",
        name: str,
        clutter_manager: ClutteredTableManager,
        target_object: ManipulatableObject,
        table_size: Tuple[float, float],
        table_pos: Tuple[float, float, float],
        placement_margin: float = 0.05,
        success_threshold: float = 0.03,
    ):
        self.world = world
        self._name = name
        self.clutter_manager = clutter_manager
        self.target_object = target_object
        self.table_size = table_size
        self.table_pos_t = torch.tensor(table_pos, dtype=torch.float32)
        self.placement_margin = placement_margin
        self.success_threshold = success_threshold

        self.rng: torch.Generator = torch.Generator()
        self.rng.seed()

        # Populated during reload.
        self._target_entity = None
        self._target_bbox = None
        self._target_dims: Optional[torch.Tensor] = None
        self._target_center: Optional[torch.Tensor] = None
        self._target_fp_r: float = 0.0

        # State (per-episode).
        self._goal_pos: Optional[torch.Tensor] = None
        self._reward = 0.0
        self._terminated = False

        # Observation space — known at construction time.
        self.observation_space = DictSpace(
            PyTorchComputeBackend,
            {
                "target_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(3,),
                    dtype=torch.float32,
                ),
                "goal_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(3,),
                    dtype=torch.float32,
                ),
            },
        )

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority != 26:
            return
        if seed is not None:
            self.rng.manual_seed(seed)

        batch_fixed_verts = self.world.batch_size is not None
        self._target_entity = self.target_object.load_into_scene(
            world=self.world,
            pos=_PARK_POS,
            quat=_PARK_QUAT,
            name_override=f"{self._name}_target",
            batch_fixed_verts=batch_fixed_verts,
        )

        # Pre-compute bounding box info.
        self._target_bbox = self.target_object.get_local_bounding_box(
            self._target_entity,
        )
        self._target_dims = bbox_size(self._target_bbox)     # (3,)
        self._target_center = bbox_center(self._target_bbox) # (3,)
        self._target_fp_r = float(
            torch.linalg.norm(self._target_dims[:2]) / 2.0
        )

    def after_reload(self, *, priority=0, mask=None):
        if priority != 26:
            return
        # Initialise placement for the first time.
        self.reset(priority=26, seed=None, mask=None)

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority != 26:
            return
        if self._target_entity is None:
            return
        if seed is not None:
            self.rng.manual_seed(seed)

        n_envs = self.world.batch_size

        # Valid placement half-extents (table half minus margin minus object
        # half-extent on each axis).
        obj_half_xy = self._target_dims[:2] / 2.0  # (2,)
        table_half = torch.tensor(
            [
                self.table_size[0] / 2.0 - self.placement_margin,
                self.table_size[1] / 2.0 - self.placement_margin,
            ],
            dtype=torch.float32,
        )
        xy_range = table_half - obj_half_xy  # (2,)

        # --- Compute Z from bbox ---
        wz = self.table_pos_t[2] + self._target_dims[2] / 2.0 - self._target_center[2]

        if n_envs is None:
            # Unbatched ------------------------------------------------
            target_xy = (torch.rand(2, generator=self.rng) * 2 - 1) * xy_range
            target_pos = torch.stack([
                self.table_pos_t[0] + target_xy[0],
                self.table_pos_t[1] + target_xy[1],
                wz,
            ])

            # Goal — different random XY on the table.
            goal_xy = (torch.rand(2, generator=self.rng) * 2 - 1) * xy_range
            self._goal_pos = torch.stack([
                self.table_pos_t[0] + goal_xy[0],
                self.table_pos_t[1] + goal_xy[1],
                wz,
            ])

            self.target_object.set_pos_3d(
                self._target_entity,
                target_pos,
                random_quaternion=False,
                rng=self.rng,
            )

            # Tell clutter manager to avoid the target area.
            centres = (target_pos[:2]).unsqueeze(0).unsqueeze(0)  # (1,1,2)
            radii = torch.tensor([[self._target_fp_r]])           # (1,1)
            self.clutter_manager.set_next_reset_avoid_spaces((centres, radii))

        else:
            # Batched --------------------------------------------------
            if mask is not None:
                mask_t = (
                    mask if isinstance(mask, torch.Tensor)
                    else torch.as_tensor(mask, dtype=torch.bool)
                )
                reset_idx = torch.where(mask_t)[0]
            else:
                reset_idx = torch.arange(n_envs)

            n_reset = reset_idx.numel()
            if n_reset == 0:
                return

            raw_target = torch.rand(n_reset, 2, generator=self.rng) * 2 - 1
            target_xy = raw_target * xy_range.unsqueeze(0)  # (n_reset, 2)
            target_pos = torch.zeros(n_reset, 3)
            target_pos[:, 0] = self.table_pos_t[0] + target_xy[:, 0]
            target_pos[:, 1] = self.table_pos_t[1] + target_xy[:, 1]
            target_pos[:, 2] = wz

            raw_goal = torch.rand(n_reset, 2, generator=self.rng) * 2 - 1
            goal_xy = raw_goal * xy_range.unsqueeze(0)
            goal_pos = torch.zeros(n_reset, 3)
            goal_pos[:, 0] = self.table_pos_t[0] + goal_xy[:, 0]
            goal_pos[:, 1] = self.table_pos_t[1] + goal_xy[:, 1]
            goal_pos[:, 2] = wz

            if self._goal_pos is None or mask is None:
                self._goal_pos = torch.zeros(n_envs, 3)
            self._goal_pos[reset_idx] = goal_pos

            envs_idx = reset_idx if mask is not None else None
            self.target_object.set_pos_3d(
                self._target_entity,
                target_pos,
                random_quaternion=False,
                rng=self.rng,
                envs_idx=envs_idx,
            )

            # Build avoid spaces for the full env set (clutter indexes
            # into it via reset_idx internally).
            all_centres = torch.zeros(n_envs, 1, 2)
            all_centres[reset_idx, 0, :] = target_xy
            # For non-reset envs, read current target position.
            if mask is not None:
                cur_pos = self._target_entity.get_pos().detach()  # (n_envs, 3)
                non_reset = torch.ones(n_envs, dtype=torch.bool)
                non_reset[reset_idx] = False
                all_centres[non_reset, 0, :] = (
                    cur_pos[non_reset, :2] - self.table_pos_t[:2].unsqueeze(0)
                )
            all_radii = torch.full((n_envs, 1), self._target_fp_r)
            self.clutter_manager.set_next_reset_avoid_spaces(
                (all_centres, all_radii),
            )

        self._reward = 0.0
        self._terminated = False

    # ------------------------------------------------------------------
    # Step signals
    # ------------------------------------------------------------------

    def post_environment_step(self, dt, *, priority=0):
        if priority != 0:
            return
        cur_pos = self._target_entity.get_pos().detach()  # (3,) or (n_envs,3)

        if self.world.batch_size is None:
            dist = float(torch.linalg.norm(cur_pos - self._goal_pos))
            self._reward = -dist
            self._terminated = dist < self.success_threshold
        else:
            dist = torch.linalg.norm(
                cur_pos - self._goal_pos, dim=-1,
            )  # (n_envs,)
            self._reward = -dist
            self._terminated = dist < self.success_threshold

    def get_observation(self):
        return {
            "target_position": self._target_entity.get_pos().detach(),
            "goal_position": self._goal_pos,
        }

    def get_reward(self):
        return self._reward

    def get_termination(self):
        return self._terminated


# ============================================================================
# Main
# ============================================================================


def main():
    world = GenesisWorld()

    ycb_parser = YCBParser()

    # -- Floor --
    plane = PlaneBuilder(world, name="floor", height=0.0)

    # -- Table --
    table = TableBuilder(
        world,
        name="table",
        pos=TABLE_POS,
        quat=(0.7071068, 0, 0, 0.7071068),
    )

    # -- Robot --
    tianji = TianjiMarvinWithRobotiq2F85(
        world,
        name="tianji",
        pos=(-0.7, 0, ROBOT_Z),
        quat=(1, 0, 0, 0),
    )

    # -- Distractor objects --
    distractor_pool = [
        # BoxObject(size=(0.05, 0.03, 0.06), name="green_box", color=(0, 0.8, 0.2, 1)),
        # SphereObject(radius=0.025, name="blue_ball", color=(0, 0.3, 1, 1)),
        SphereObject(radius=0.02, name="yellow_ball", color=(1, 1, 0, 1)),
        CylinderObject(
            radius=0.02, height=0.08, name="orange_can", color=(1, 0.5, 0, 1),
        ),
        CylinderObject(
            radius=0.015, height=0.06, name="purple_peg", color=(0.6, 0, 0.8, 1),
        ),
        ycb_parser.create(model_id='048_hammer'),
        ycb_parser.create(model_id='030_fork')
    ]

    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=distractor_pool,
        n_objects_range=(3, 5),
        table_size=TABLE_SIZE,
        table_pos=TABLE_POS,
        random_quaternion=True,
    )

    # -- Task: relocate the red target cube --
    target_cube = BoxObject(
        size=(0.04, 0.04, 0.04), name="target_cube", color=(1, 0, 0, 1),
    )
    task = CubeRelocateTask(
        world,
        name="task",
        clutter_manager=clutter,
        target_object=target_cube,
        table_size=TABLE_SIZE,
        table_pos=TABLE_POS,
        success_threshold=0.03,
    )

    # -- Camera -- facing the robot
    cam_pos = (1.0, 0.0, 1.4)
    cam_lookat = (0.0, 0.0, 0.6)
    camera = CameraSensorNode(
        world,
        name="camera",
        option=CameraSensorNodeOptions(
            pos=cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
            res=(640, 480),
            fov=60,
        ),
        render_rgb=True,
    )

    # -- Environment --
    # Node order: plane(100) > table(100) > robot(50) > task(26) > clutter(25) > camera(-50)
    env = EpisodeVideoWrapper(
        WorldEnv(
            world,
            [plane, table, tianji, task, clutter, camera],
            render_mode="auto",
        ),
        store_dir=VIDEO_DIR,
        format="mp4",
    )

    from xbarray.transformations.rotation_conversions.pytorch import (
        quaternion_to_matrix,
        matrix_to_euler_angles,
    )

    for episode in range(N_EPISODES):
        context, obs, info = env.reset(seed=episode, reload=True)

        # Read task observations.
        task_obs = obs["task"]
        target_pos = task_obs["target_position"]
        goal_pos = task_obs["goal_position"]
        print(
            f"\nEpisode {episode} | "
            f"target={target_pos.tolist()}  goal={goal_pos.tolist()}"
        )

        # Read home EEF state (left arm only — single-arm reach demo).
        left_obs = obs["tianji"]["left_arm"]
        right_obs = obs["tianji"]["right_arm"]

        left_home = left_obs["eef_position"].clone()
        left_euler = matrix_to_euler_angles(
            quaternion_to_matrix(left_obs["eef_quaternion"].clone()), "XYZ",
        )
        right_home = right_obs["eef_position"].clone()
        right_euler = matrix_to_euler_angles(
            quaternion_to_matrix(right_obs["eef_quaternion"].clone()), "XYZ",
        )

        # Left arm will reach toward the target cube.
        # Hover 10 cm above the target position.
        left_target = target_pos.clone()
        left_target[2] = TABLE_SURFACE_Z + 0.10

        # Gripper orientation: pointing down (-Z).
        left_euler_down = torch.tensor([0, math.pi / 2, 0], dtype=torch.float32)
        right_euler_fwd = torch.tensor([0, math.pi / 2, 0], dtype=torch.float32)

        # Right arm stays at a safe hover position.
        right_table = right_home.clone()
        right_table[0] = right_home[0] + 0.30
        right_table[1] = right_home[1] * 0.4
        right_table[2] = TABLE_SURFACE_Z + 0.10

        OPEN = torch.tensor([-1.0])
        CLOSE = torch.tensor([1.0])

        for step in range(TOTAL_STEPS):
            if step < 150:
                # Phase 0: left arm reaches toward cube, right arm to hover
                alpha = step / 150.0
                l_pos = (1 - alpha) * left_home + alpha * left_target
                l_euler = (1 - alpha) * left_euler + alpha * left_euler_down
                r_pos = (1 - alpha) * right_home + alpha * right_table
                r_euler = (1 - alpha) * right_euler + alpha * right_euler_fwd
                lg, rg = OPEN, OPEN
            elif step < 250:
                # Phase 1: hold above target, close left gripper
                l_pos = left_target
                l_euler = left_euler_down
                r_pos = right_table
                r_euler = right_euler_fwd
                lg, rg = CLOSE, OPEN
            else:
                # Phase 2: return home, open grippers
                alpha = (step - 250) / 100.0
                l_pos = (1 - alpha) * left_target + alpha * left_home
                l_euler = (1 - alpha) * left_euler_down + alpha * left_euler
                r_pos = (1 - alpha) * right_table + alpha * right_home
                r_euler = (1 - alpha) * right_euler_fwd + alpha * right_euler
                lg, rg = OPEN, OPEN

            action = {
                "left_arm": torch.cat([l_pos, l_euler]),
                "right_arm": torch.cat([r_pos, r_euler]),
                "left_gripper": lg,
                "right_gripper": rg,
            }
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 50 == 0:
                cur_target = obs["task"]["target_position"]
                dist_to_goal = torch.linalg.norm(
                    cur_target - goal_pos,
                ).item()
                print(
                    f"  Step {step:3d} | "
                    f"reward={reward:.4f}  dist_to_goal={dist_to_goal:.4f}"
                )

    env.close()

    # Report saved videos.
    for episode in range(N_EPISODES):
        step = episode * TOTAL_STEPS
        path = os.path.join(VIDEO_DIR, f"episode_{episode}_step_{step}.mp4")
        if os.path.exists(path):
            print(f"Video saved: {path}")
        else:
            print(f"Warning: expected video not found at {path}")


if __name__ == "__main__":
    main()
