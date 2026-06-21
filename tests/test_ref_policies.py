"""Tests for GoToLambdaRefPolicy and PickPlaceRefPolicy.

Covers:
  - GoToLambdaRefPolicy: reads target from lambda at reset, produces action, detects success.
  - GoToLambdaRefPolicy: batched target tracking.
  - GoToLambdaRefPolicy: target_quat_fn=None uses EEF's current orientation.
  - GoToLambdaRefPolicy: seed reproducibility (same task state → same action).
  - PickPlaceRefPolicy: graph structure (nodes, edges, initial node).
  - PickPlaceRefPolicy: all motion nodes are GoToLambdaRefPolicy instances.
"""

import math
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.cached_table_builder import CachedTableBuilder
from unienv_genesis_collection.objects import BoxObject
from unienv_genesis_collection.tasks.tabletop import (
    ReachTargetTask,
    ObjectRelocateTask,
)
from unienv_genesis_refpolicy.implementations.goto_lambda_policy import GoToLambdaRefPolicy
from unienv_genesis_refpolicy.implementations.pickplace_policy import PickPlaceRefPolicy
from unienv_genesis_refpolicy.implementations.combined_policy import CombinedRefPolicy
from unienv_genesis_refpolicy.implementations.graph_policy import GraphRefPolicy
from unienv_genesis_refpolicy.implementations.grasp_policy import BoxGripperGraspPolicy
from unienv_genesis_refpolicy.implementations.repeat_action_policy import RepeatActionRefPolicy
from unienv_interface.space import DictSpace, BoxSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.world import WorldEnv


# ------------------------------------------------------------------
# Mock EEF controller
# ------------------------------------------------------------------

class _MockEEFController:
    """Complete mock EEF controller for policy testing."""

    has_eef = True

    def __init__(self, world, pos, quat=None, action_dim=7):
        self.world = world
        self._pos = torch.tensor(pos, dtype=torch.float32, device=world.device)
        self._quat = torch.tensor(
            quat if quat is not None else (1.0, 0.0, 0.0, 0.0),
            dtype=torch.float32, device=world.device,
        )
        self._action_dim = action_dim

    def get_eef_pose_absolute(self):
        return self._pos, self._quat

    def get_action_world(self, pos, quat):
        batch_size = self.world.batch_size
        if batch_size is not None:
            action = torch.zeros(batch_size, self._action_dim, dtype=torch.float32, device=self.world.device)
            action[:, :3] = pos
            if self._action_dim >= 7:
                action[:, 3:7] = quat
        else:
            action = torch.zeros(self._action_dim, dtype=torch.float32, device=self.world.device)
            action[:3] = pos
            if self._action_dim >= 7:
                action[3:7] = quat
        return action

    def get_action_dict(self, action, dt=None):
        return {"action": action}

    def get_action_dict_space(self):
        from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
        batch_size = self.world.batch_size
        return DictSpace(
            PyTorchComputeBackend,
            {"action": BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf, high=torch.inf,
                shape=get_batch_aware_shape(batch_size, self._action_dim),
                dtype=torch.float32,
                device=self.world.device,
            )},
            device=self.world.device,
        )


class _MockGripperController:
    has_eef = False
    gripper_open_action = -1.0
    gripper_close_action = 1.0
    gripper_width = 0.085
    gripper_depth = 0.035
    approach_direction = [0.0, 0.0, 1.0]
    open_direction = [0.0, 1.0, 0.0]

    def is_grasping(self, target_entity):
        return None

    def get_action_dict(self, action, dt=None):
        return {"action": action}

    def get_action_dict_space(self):
        return DictSpace(
            PyTorchComputeBackend,
            {"action": BoxSpace(
                PyTorchComputeBackend,
                low=-torch.inf, high=torch.inf,
                shape=(1,), dtype=torch.float32,
            )},
        )


# ------------------------------------------------------------------
# GoToLambdaRefPolicy
# ------------------------------------------------------------------

def test_goto_lambda_unbatched():
    """GoToLambdaRefPolicy: reads target from lambda, produces action, detects success."""
    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")

    eef_pos = (0.4, 0.0, 0.5)
    mock_eef = _MockEEFController(world, pos=eef_pos)

    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world, eef_controller=mock_eef, target_workspace=workspace,
        success_threshold=0.02,
    )
    env = WorldEnv(world, [plane, task], render_mode=None)
    env.reset(seed=0, reload=True)

    policy = GoToLambdaRefPolicy(
        world, controller=mock_eef,
        target_pos_fn=lambda: task._target_pos,
        position_threshold=0.02, stall_seconds=0,
    )
    policy.reset(seed=0)

    # Action should be produced.
    action = policy.get_action()
    assert action is not None
    assert action.shape[-1] == 7  # pos(3) + quat(4)

    # Target should match the task's target.
    target_pos = task._target_pos
    assert torch.allclose(action[:3], target_pos, atol=1e-5)

    # Step once — EEF hasn't moved (mock), so not reached yet.
    policy.step()
    assert policy.get_success() is False
    assert policy.get_failure() is False

    # Simulate the EEF reaching the target by moving the mock.
    mock_eef._pos = target_pos.clone()
    policy.step()
    assert policy.get_success() is True

    print(f"  PASS: goto_lambda unbatched (target={target_pos.tolist()})")
    env.close()


def test_goto_lambda_batched():
    """GoToLambdaRefPolicy batched: per-env target tracking and success."""
    N_ENVS = 4
    world = GenesisWorld(batch_size=N_ENVS)
    plane = PlaneBuilder(world, name="floor")

    mock_eef = _MockEEFController(world, pos=(0.4, 0.0, 0.5))
    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world, eef_controller=mock_eef, target_workspace=workspace,
    )
    env = WorldEnv(world, [plane, task], render_mode=None)
    env.reset(seed=0, reload=True)

    policy = GoToLambdaRefPolicy(
        world, controller=mock_eef,
        target_pos_fn=lambda: task._target_pos,
        stall_seconds=0,
    )
    policy.reset(seed=0)

    action = policy.get_action()
    assert action.shape == (N_ENVS, 7)

    # No env should be at target initially.
    policy.step()
    success = policy.get_success()
    assert isinstance(success, torch.Tensor)
    assert success.shape == (N_ENVS,)
    assert not success.any()

    # Move EEF to per-env targets.
    mock_eef._pos = task._target_pos.clone()
    policy.step()
    assert policy.get_success().all()

    print(f"  PASS: goto_lambda batched ({N_ENVS} envs)")
    env.close()


def test_goto_lambda_target_quat_fn_none():
    """GoToLambdaRefPolicy with target_quat_fn=None uses EEF's current orientation."""
    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")

    eef_pos = (0.4, 0.0, 0.5)
    eef_quat = (0.707, 0.0, 0.0, 0.707)  # 90° rotation around Z
    mock_eef = _MockEEFController(world, pos=eef_pos, quat=eef_quat)

    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world, eef_controller=mock_eef, target_workspace=workspace,
    )
    env = WorldEnv(world, [plane, task], render_mode=None)
    env.reset(seed=0, reload=True)

    policy = GoToLambdaRefPolicy(
        world, controller=mock_eef,
        target_pos_fn=lambda: task._target_pos,
        # target_quat_fn omitted → should use EEF's current orientation
    )
    policy.reset(seed=0)

    # The target_quat should match the EEF's current orientation.
    expected_quat = torch.tensor(eef_quat, dtype=torch.float32, device=world.device)
    assert torch.allclose(policy.target_quat, expected_quat, atol=1e-5), \
        f"target_quat {policy.target_quat} != EEF quat {expected_quat}"

    print("  PASS: goto_lambda target_quat_fn=None uses EEF orientation")
    env.close()


def test_goto_lambda_seed_reproducibility():
    """Same task state (same seed) → same action."""
    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")
    mock_eef = _MockEEFController(world, pos=(0.4, 0.0, 0.5))
    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(world, eef_controller=mock_eef, target_workspace=workspace)
    env = WorldEnv(world, [plane, task], render_mode=None)

    # Initial reload (scene build).
    env.reset(seed=0, reload=True)

    # Two resets with the same seed → same target → same action.
    env.reset(seed=42)
    policy = GoToLambdaRefPolicy(
        world, controller=mock_eef,
        target_pos_fn=lambda: task._target_pos,
        stall_seconds=0,
    )
    policy.reset(seed=42)
    action_a = policy.get_action().clone()

    env.reset(seed=42)
    policy.reset(seed=42)
    action_b = policy.get_action().clone()

    assert torch.allclose(action_a, action_b, atol=1e-5), "Same seed gave different actions"
    print("  PASS: goto_lambda seed reproducibility")
    env.close()


# ------------------------------------------------------------------
# PickPlaceRefPolicy — graph structure verification
# ------------------------------------------------------------------

def test_pickplace_policy_construction():
    """PickPlaceRefPolicy: full construction with a loaded entity (scene build).

    Verifies the internal graph has the correct node names, edge structure,
    and that all motion phases are GoToLambdaRefPolicy instances.
    """
    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=(0.0, 0.0, 0.65))
    target = BoxObject(size=(0.04, 0.04, 0.04), name="cube", color=(1, 0, 0, 1))
    task = ObjectRelocateTask(
        world, table_builder=table, clutter_manager=None,
        target_object=target,
    )

    env = WorldEnv(world, [plane, table, task], render_mode=None)
    env.reset(seed=0, reload=True)

    assert task.target_object_entity is not None
    assert task._goal_pos is not None

    mock_arm = _MockEEFController(world, pos=(0.0, 0.0, 0.8))
    mock_gripper = _MockGripperController()

    policy = PickPlaceRefPolicy(
        world=world,
        eef_controller=mock_arm,
        gripper_controller=mock_gripper,
        task=task,
        target_object=target,
        target_entity=task.target_object_entity,
    )

    # Inspect the internal graph.
    graph = policy._graph
    assert isinstance(graph, GraphRefPolicy)

    # Check node names.
    expected_nodes = {"grasp", "lift", "move_to_goal", "lower", "release", "retreat"}
    assert set(graph._node_names) == expected_nodes, \
        f"Nodes: {set(graph._node_names)} != {expected_nodes}"

    # Check initial node.
    assert graph._initial == "grasp"

    # Check success edges.
    expected_success = {
        ("grasp", "lift"),
        ("lift", "move_to_goal"),
        ("move_to_goal", "lower"),
        ("lower", "release"),
        ("release", "retreat"),
    }
    assert set(graph._success_map.items()) == expected_success, \
        f"Success edges: {set(graph._success_map.items())} != {expected_success}"

    # Check no fail edges.
    assert graph._fail_map == {}

    # Check node types.
    # grasp is a factory (resolved lazily on first reset, after env.reset loads the entity)
    assert graph._node_policies["grasp"] is None
    # Motion phases are CombinedRefPolicy wrapping GoToLambdaRefPolicy + RepeatActionRefPolicy
    for name in ["lift", "move_to_goal", "lower", "retreat"]:
        assert isinstance(graph._node_policies[name], CombinedRefPolicy), \
            f"Node '{name}' should be CombinedRefPolicy, got {type(graph._node_policies[name])}"
    assert isinstance(graph._node_policies["release"], CombinedRefPolicy)

    print("  PASS: pickplace policy graph structure correct")
    env.close()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== test_goto_lambda_unbatched ===")
    test_goto_lambda_unbatched()
    print()
    print("=== test_goto_lambda_batched ===")
    test_goto_lambda_batched()
    print()
    print("=== test_goto_lambda_target_quat_fn_none ===")
    test_goto_lambda_target_quat_fn_none()
    print()
    print("=== test_goto_lambda_seed_reproducibility ===")
    test_goto_lambda_seed_reproducibility()
    print()
    print("=== test_pickplace_policy_construction ===")
    test_pickplace_policy_construction()
    print()
    print("All tests passed!")
