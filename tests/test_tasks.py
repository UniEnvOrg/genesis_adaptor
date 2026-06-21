"""Tests for the tabletop tasks: ReachTargetTask.

Covers:
  - ReachTargetTask: target in workspace, reward = -distance(eef, target).
  - ReachTargetTask batched: per-env independent target sampling.
"""

import math
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.tasks.tabletop import (
    ReachTargetTask,
)
from unienv_interface.world import WorldEnv


# ------------------------------------------------------------------
# Mock EEF controller for ReachTargetTask
# ------------------------------------------------------------------

class _MockEEFController:
    """Minimal stand-in exposing get_eef_pose_absolute()."""

    def __init__(self, pos, quat=None):
        self._pos = torch.tensor(pos, dtype=torch.float32)
        self._quat = torch.tensor(
            quat if quat is not None else (1.0, 0.0, 0.0, 0.0),
            dtype=torch.float32,
        )

    def get_eef_pose_absolute(self):
        return self._pos, self._quat


# ------------------------------------------------------------------
# ReachTargetTask
# ------------------------------------------------------------------

def test_reach_unbatched():
    """Reach: target in workspace, reward = -distance(eef, target)."""
    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")

    eef_pos = (0.4, 0.0, 0.5)
    mock_eef = _MockEEFController(pos=eef_pos)

    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world,
        eef_controller=mock_eef,
        target_workspace=workspace,
        success_threshold=0.02,
    )
    env = WorldEnv(world, [plane, task], render_mode=None)
    env.reset(seed=0, reload=True)

    obs = task.get_observation()
    eef_obs = obs["eef_position"]
    target_obs = obs["target_position"]

    # EEF position matches mock.
    assert torch.allclose(
        eef_obs, torch.tensor(eef_pos, dtype=eef_obs.dtype), atol=1e-5
    )

    # Target within workspace bounds.
    low = torch.tensor(workspace[0])
    high = torch.tensor(workspace[1])
    assert torch.all(target_obs >= low - 1e-6), f"target {target_obs} < low {low}"
    assert torch.all(target_obs <= high + 1e-6), f"target {target_obs} > high {high}"

    # Reward = -distance.
    task.post_environment_step(dt=world.world_timestep, priority=0)
    reward = task.get_reward()
    dist = math.sqrt(
        sum((eef_obs[i].item() - target_obs[i].item()) ** 2 for i in range(3))
    )
    assert abs(reward - (-dist)) < 1e-4, f"reward {reward} != {-dist}"

    print(f"  PASS: reach unbatched (dist={dist:.3f}, reward={reward:.3f})")
    env.close()


def test_reach_batched():
    """Reach batched: per-env independent target sampling."""
    N_ENVS = 8
    world = GenesisWorld(batch_size=N_ENVS)
    plane = PlaneBuilder(world, name="floor")

    mock_eef = _MockEEFController(pos=(0.4, 0.0, 0.5))
    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world, eef_controller=mock_eef, target_workspace=workspace,
    )
    env = WorldEnv(world, [plane, task], render_mode=None)
    env.reset(seed=0, reload=True)

    obs = task.get_observation()
    target_obs = obs["target_position"]  # (N, 3)
    assert target_obs.shape == (N_ENVS, 3)

    # Targets should vary across envs.
    assert target_obs.unique(dim=0).shape[0] > 1, "All envs same target"

    task.post_environment_step(dt=world.world_timestep, priority=0)
    reward = task.get_reward()
    assert reward.shape == (N_ENVS,)

    print(f"  PASS: reach batched ({N_ENVS} envs)")
    env.close()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== test_reach_unbatched ===")
    test_reach_unbatched()
    print()
    print("=== test_reach_batched ===")
    test_reach_batched()
    print()
    print("All tests passed!")
