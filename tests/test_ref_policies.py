"""Tests for GoToLambdaRefPolicy and PickPlaceRefPolicy.

Covers:
  - GoToLambdaRefPolicy: reads target from lambda at reset, produces action, detects success.
  - GoToLambdaRefPolicy: batched target tracking.
  - GoToLambdaRefPolicy: target_quat_fn=None uses EEF's current orientation.
  - GoToLambdaRefPolicy: seed reproducibility (same task state → same action).
  - PickPlaceRefPolicy: graph structure (nodes, edges, initial node, node types).
  - PickPlaceRefPolicy: phase-specific kwargs split (transport/release/retreat) over goto_kwargs.
  - PickPlaceRefPolicy: set_transport_slip_detection affects transport phases only.
  - PickPlaceRefPolicy: transport_slip_detection constructor flag (default enabled,
    disabled-at-init, and setter interaction).
  - MoveObjectPolicy: grasp-offset preservation, quaternion rotation, slip checks.
"""

import math
import torch
import genesis as gs
import pytest
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.cached_table_builder import CachedTableBuilder
from unienv_genesis_collection.objects import BoxObject
from unienv_genesis_collection.objects.base_object import GraspCandidates
from unienv_genesis_collection.tasks.tabletop import (
    ReachTargetTask,
    ObjectRelocateTask,
)
from unienv_genesis_refpolicy.implementations.goto_lambda_policy import GoToLambdaRefPolicy
from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy
from unienv_genesis_refpolicy.implementations.move_object_policy import MoveObjectPolicy
from unienv_genesis_refpolicy.implementations.move_object_lambda_policy import MoveObjectLambdaPolicy
from unienv_genesis_refpolicy.implementations.pickplace_policy import PickPlaceRefPolicy
from unienv_genesis_refpolicy.implementations.combined_policy import CombinedRefPolicy
from unienv_genesis_refpolicy.implementations.graph_policy import GraphRefPolicy
from unienv_genesis_refpolicy.implementations.grasp_policy import BoxGripperGraspPolicy
from unienv_genesis_refpolicy.implementations.repeat_action_policy import RepeatActionRefPolicy
from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider
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

    def __init__(self, grasping=True):
        self._grasping = grasping

    def is_grasping(self, target_entity):
        del target_entity
        return self._grasping

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


class _MockPoseEntity:
    def __init__(self, pos, quat, world):
        self._pos = torch.tensor(pos, dtype=torch.float32, device=world.device)
        self._quat = torch.tensor(quat, dtype=torch.float32, device=world.device)

    def get_pos(self):
        return self._pos

    def get_quat(self):
        return self._quat


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
    and arm-policy types for transport vs. release/retreat phases.
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
    # Motion phases are CombinedRefPolicy wrapping an arm policy + RepeatActionRefPolicy
    for name in ["lift", "move_to_goal", "lower", "retreat", "release"]:
        assert isinstance(graph._node_policies[name], CombinedRefPolicy), \
            f"Node '{name}' should be CombinedRefPolicy, got {type(graph._node_policies[name])}"

    # Transport phases (lift/move_to_goal/lower) wrap MoveObjectLambdaPolicy; the
    # release and retreat phases wrap GoToLambdaRefPolicy (EEF-centric, no slip).
    for name in ["lift", "move_to_goal", "lower"]:
        arm_policy = graph._node_policies[name]._policies["arm"]
        assert isinstance(arm_policy, MoveObjectLambdaPolicy), \
            f"Transport node '{name}' arm policy should be MoveObjectLambdaPolicy, got {type(arm_policy)}"
    for name in ["release", "retreat"]:
        arm_policy = graph._node_policies[name]._policies["arm"]
        assert isinstance(arm_policy, GoToLambdaRefPolicy), \
            f"Node '{name}' arm policy should be GoToLambdaRefPolicy, got {type(arm_policy)}"

    print("  PASS: pickplace policy graph structure correct")
    env.close()


def test_pickplace_transport_slip_detection_api():
    """set_transport_slip_detection affects transport phases only, not retreat/release."""
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

    graph = policy._graph

    def _transport_policies():
        return [
            graph._node_policies[name]._policies["arm"]
            for name in ("lift", "move_to_goal", "lower")
        ]

    release_arm = graph._node_policies["release"]._policies["arm"]
    retreat_arm = graph._node_policies["retreat"]._policies["arm"]

    # Construction-time defaults: slip thresholds are the MoveObject defaults.
    defaults = policy._transport_slip_defaults
    assert len(defaults) == 3
    for d in defaults:
        assert d["slip_position_threshold"] == 0.01
        assert d["slip_orientation_threshold"] == 0.05
        assert d["track_relative_pose"] is True

    # Disabling should zero thresholds and turn off relative-pose tracking.
    policy.set_transport_slip_detection(False)
    for p in _transport_policies():
        assert p.slip_position_threshold == 0.0
        assert p.slip_orientation_threshold == 0.0
        assert p.track_relative_pose is False

    # Enabling with no overrides restores the construction-time defaults.
    policy.set_transport_slip_detection(True)
    for p in _transport_policies():
        assert p.slip_position_threshold == 0.01
        assert p.slip_orientation_threshold == 0.05
        assert p.track_relative_pose is True

    # Disabling with an explicit override respects the override.
    policy.set_transport_slip_detection(
        False, position_threshold=0.02, track_relative_pose=True,
    )
    for p in _transport_policies():
        assert p.slip_position_threshold == 0.02
        assert p.slip_orientation_threshold == 0.0
        assert p.track_relative_pose is True

    # release/retreat GoToLambda policies have no slip attributes; confirm
    # the transport-only API did not add any.
    assert not hasattr(release_arm, "slip_position_threshold")
    assert not hasattr(release_arm, "track_relative_pose")
    assert not hasattr(retreat_arm, "slip_position_threshold")
    assert not hasattr(retreat_arm, "track_relative_pose")

    print("  PASS: set_transport_slip_detection affects transport phases only")
    env.close()


def test_pickplace_transport_slip_detection_init_default():
    """Default constructor (no transport_slip_detection arg) keeps slip detection enabled."""
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

    mock_arm = _MockEEFController(world, pos=(0.0, 0.0, 0.8))
    mock_gripper = _MockGripperController()

    # No transport_slip_detection argument -> defaults to enabled (backward compatible).
    policy = PickPlaceRefPolicy(
        world=world,
        eef_controller=mock_arm,
        gripper_controller=mock_gripper,
        task=task,
        target_object=target,
        target_entity=task.target_object_entity,
    )

    graph = policy._graph
    for name in ("lift", "move_to_goal", "lower"):
        p = graph._node_policies[name]._policies["arm"]
        assert p.slip_position_threshold == 0.01, f"{name} default slip_position_threshold"
        assert p.slip_orientation_threshold == 0.05, f"{name} default slip_orientation_threshold"
        assert p.track_relative_pose is True, f"{name} default track_relative_pose"

    # Explicit transport_slip_detection=True behaves identically to the default.
    policy_explicit = PickPlaceRefPolicy(
        world=world,
        eef_controller=mock_arm,
        gripper_controller=mock_gripper,
        task=task,
        target_object=target,
        target_entity=task.target_object_entity,
        transport_slip_detection=True,
    )
    for name in ("lift", "move_to_goal", "lower"):
        p = policy_explicit._graph._node_policies[name]._policies["arm"]
        assert p.slip_position_threshold == 0.01
        assert p.slip_orientation_threshold == 0.05
        assert p.track_relative_pose is True

    print("  PASS: transport slip detection enabled by default at init")
    env.close()


def test_pickplace_transport_slip_detection_disabled_at_init():
    """transport_slip_detection=False disables slip at init; setter can re-enable."""
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

    mock_arm = _MockEEFController(world, pos=(0.0, 0.0, 0.8))
    mock_gripper = _MockGripperController()

    policy = PickPlaceRefPolicy(
        world=world,
        eef_controller=mock_arm,
        gripper_controller=mock_gripper,
        task=task,
        target_object=target,
        target_entity=task.target_object_entity,
        # Custom transport threshold: the recorded construction-time defaults
        # must reflect this even when slip detection starts disabled.
        transport_kwargs={"slip_position_threshold": 0.03},
        transport_slip_detection=False,
    )

    graph = policy._graph

    def _transport_policies():
        return [
            graph._node_policies[name]._policies["arm"]
            for name in ("lift", "move_to_goal", "lower")
        ]

    release_arm = graph._node_policies["release"]._policies["arm"]
    retreat_arm = graph._node_policies["retreat"]._policies["arm"]

    # Slip detection starts disabled on the transport phases.
    for p in _transport_policies():
        assert p.slip_position_threshold == 0.0
        assert p.slip_orientation_threshold == 0.0
        assert p.track_relative_pose is False

    # The recorded construction-time defaults are preserved (incl. the custom
    # transport_kwargs threshold), so the runtime setter can restore them.
    defaults = policy._transport_slip_defaults
    assert len(defaults) == 3
    for d in defaults:
        assert d["slip_position_threshold"] == 0.03
        assert d["slip_orientation_threshold"] == 0.05
        assert d["track_relative_pose"] is True

    # release/retreat GoToLambda policies remain untouched (no slip attributes).
    assert not hasattr(release_arm, "slip_position_threshold")
    assert not hasattr(release_arm, "track_relative_pose")
    assert not hasattr(retreat_arm, "slip_position_threshold")
    assert not hasattr(retreat_arm, "track_relative_pose")

    # Constructor/API interaction: the runtime setter still works afterwards,
    # restoring the construction-time defaults (including the custom 0.03).
    policy.set_transport_slip_detection(True)
    for p in _transport_policies():
        assert p.slip_position_threshold == 0.03
        assert p.slip_orientation_threshold == 0.05
        assert p.track_relative_pose is True

    # And disabling again via the setter reproduces the init-time disabled state.
    policy.set_transport_slip_detection(False)
    for p in _transport_policies():
        assert p.slip_position_threshold == 0.0
        assert p.slip_orientation_threshold == 0.0
        assert p.track_relative_pose is False

    print("  PASS: transport_slip_detection=False at init; setter restores defaults")
    env.close()


def test_pickplace_phase_specific_kwargs_split():
    """Phase-specific kwargs override goto_kwargs on the right phases only."""
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

    mock_arm = _MockEEFController(world, pos=(0.0, 0.0, 0.8))
    mock_gripper = _MockGripperController()

    # Shared baseline applied everywhere; phase-specific values override it.
    policy = PickPlaceRefPolicy(
        world=world,
        eef_controller=mock_arm,
        gripper_controller=mock_gripper,
        task=task,
        target_object=target,
        target_entity=task.target_object_entity,
        goto_kwargs={"position_threshold": 0.05, "max_seconds": 10.0},
        transport_kwargs={"position_threshold": 0.02, "slip_position_threshold": 0.02},
        release_kwargs={"position_threshold": 0.08},
        retreat_kwargs={"position_threshold": 0.04},
    )

    graph = policy._graph
    lift = graph._node_policies["lift"]._policies["arm"]
    move_to_goal = graph._node_policies["move_to_goal"]._policies["arm"]
    lower = graph._node_policies["lower"]._policies["arm"]
    release = graph._node_policies["release"]._policies["arm"]
    retreat = graph._node_policies["retreat"]._policies["arm"]

    # Transport phases get transport_kwargs overriding goto_kwargs, plus the
    # shared max_seconds baseline.
    for name, p in (("lift", lift), ("move_to_goal", move_to_goal), ("lower", lower)):
        assert p.position_threshold == 0.02, f"{name} position_threshold"
        assert p.max_seconds == 10.0, f"{name} max_seconds baseline"
        assert p.slip_position_threshold == 0.02, f"{name} slip threshold (transport-only)"

    # Release uses release_kwargs merged over goto_kwargs; no slip attribute.
    assert release.position_threshold == 0.08
    assert release.max_seconds == 10.0
    assert not hasattr(release, "slip_position_threshold")

    # Retreat uses retreat_kwargs merged over goto_kwargs; no slip attribute.
    assert retreat.position_threshold == 0.04
    assert retreat.max_seconds == 10.0
    assert not hasattr(retreat, "slip_position_threshold")

    print("  PASS: phase-specific kwargs split and merge correct")
    env.close()


def test_move_object_policy_preserves_grasp_offset_position_only():
    world = GenesisWorld()
    mock_eef = _MockEEFController(
        world,
        pos=(0.30, 0.08, 0.46),
        quat=(1.0, 0.0, 0.0, 0.0),
    )
    entity = _MockPoseEntity(
        pos=(0.25, 0.10, 0.30),
        quat=(1.0, 0.0, 0.0, 0.0),
        world=world,
    )
    policy = MoveObjectPolicy(
        world=world,
        controller=mock_eef,
        object_entity=entity,
        target_pos=torch.tensor([0.40, -0.05, 0.50]),
        target_quat=None,
        stall_seconds=0,
    )
    policy.reset()

    expected_offset = torch.tensor([0.05, -0.02, 0.16], device=world.device)
    expected_pos = torch.tensor([0.45, -0.07, 0.66], device=world.device)
    assert torch.allclose(policy.target_pos, expected_pos, atol=1e-5)
    assert torch.allclose(policy.target_quat, torch.tensor([1.0, 0.0, 0.0, 0.0], device=world.device), atol=1e-5)
    assert torch.allclose(policy.target_pos - torch.tensor([0.40, -0.05, 0.50], device=world.device), expected_offset, atol=1e-5)


def test_move_object_policy_rotates_preserved_offset_and_quaternion():
    from xbarray.transformations.rotation_conversions.pytorch import (
        quaternion_apply,
        quaternion_multiply,
    )

    world = GenesisWorld()
    q_rel = torch.tensor([0.9238795, 0.0, 0.3826834, 0.0], dtype=torch.float32, device=world.device)
    q_target = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], dtype=torch.float32, device=world.device)
    obj_pos = torch.tensor([0.25, 0.10, 0.30], dtype=torch.float32, device=world.device)
    rel_pos = torch.tensor([0.05, -0.02, 0.16], dtype=torch.float32, device=world.device)
    eef_pos = obj_pos + rel_pos
    mock_eef = _MockEEFController(world, pos=tuple(float(v) for v in eef_pos.cpu()), quat=tuple(float(v) for v in q_rel.cpu()))
    entity = _MockPoseEntity(pos=tuple(float(v) for v in obj_pos.cpu()), quat=(1.0, 0.0, 0.0, 0.0), world=world)
    target_pos = torch.tensor([0.40, -0.05, 0.50], dtype=torch.float32, device=world.device)
    policy = MoveObjectPolicy(
        world=world,
        controller=mock_eef,
        object_entity=entity,
        target_pos=target_pos,
        target_quat=q_target,
        stall_seconds=0,
    )
    policy.reset()

    expected_pos = target_pos + quaternion_apply(q_target, rel_pos)
    expected_quat = quaternion_multiply(q_target, q_rel)
    assert torch.allclose(policy.target_pos, expected_pos, atol=1e-5)
    assert torch.allclose(policy.target_quat, expected_quat, atol=1e-5)


def test_move_object_policy_fails_on_position_slip():
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.30, 0.08, 0.46), quat=(1.0, 0.0, 0.0, 0.0))
    entity = _MockPoseEntity(pos=(0.25, 0.10, 0.30), quat=(1.0, 0.0, 0.0, 0.0), world=world)
    policy = MoveObjectPolicy(
        world=world,
        controller=mock_eef,
        object_entity=entity,
        target_pos=torch.tensor([0.40, -0.05, 0.50]),
        slip_position_threshold=0.01,
        stall_seconds=0,
    )
    policy.reset()
    mock_eef._pos = torch.tensor([0.35, 0.08, 0.46], dtype=torch.float32, device=world.device)
    policy.step()
    assert policy.get_failure() is True


def test_move_object_policy_fails_on_orientation_slip():
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.30, 0.08, 0.46), quat=(1.0, 0.0, 0.0, 0.0))
    entity = _MockPoseEntity(pos=(0.25, 0.10, 0.30), quat=(1.0, 0.0, 0.0, 0.0), world=world)
    policy = MoveObjectPolicy(
        world=world,
        controller=mock_eef,
        object_entity=entity,
        target_pos=torch.tensor([0.40, -0.05, 0.50]),
        slip_orientation_threshold=0.05,
        stall_seconds=0,
    )
    policy.reset()
    mock_eef._quat = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], dtype=torch.float32, device=world.device)
    policy.step()
    assert policy.get_failure() is True


def test_move_object_policy_optional_contact_loss_failure():
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.30, 0.08, 0.46), quat=(1.0, 0.0, 0.0, 0.0))
    entity = _MockPoseEntity(pos=(0.25, 0.10, 0.30), quat=(1.0, 0.0, 0.0, 0.0), world=world)

    gripper_a = _MockGripperController(grasping=True)
    policy_a = MoveObjectPolicy(
        world=world,
        controller=mock_eef,
        object_entity=entity,
        target_pos=torch.tensor([0.40, -0.05, 0.50]),
        gripper_controller=gripper_a,
        fail_on_gripper_contact_loss=True,
        stall_seconds=0,
    )
    policy_a.reset()
    gripper_a._grasping = False
    policy_a.step()
    assert policy_a.get_failure() is True

    gripper_b = _MockGripperController(grasping=True)
    policy_b = MoveObjectPolicy(
        world=world,
        controller=mock_eef,
        object_entity=entity,
        target_pos=torch.tensor([0.40, -0.05, 0.50]),
        gripper_controller=gripper_b,
        fail_on_gripper_contact_loss=False,
        stall_seconds=0,
    )
    policy_b.reset()
    gripper_b._grasping = False
    policy_b.step()
    assert policy_b.get_failure() is False


def test_move_object_policy_nonpositive_slip_thresholds_disable_checks():
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.30, 0.08, 0.46), quat=(1.0, 0.0, 0.0, 0.0))
    entity = _MockPoseEntity(pos=(0.25, 0.10, 0.30), quat=(1.0, 0.0, 0.0, 0.0), world=world)
    policy = MoveObjectPolicy(
        world=world,
        controller=mock_eef,
        object_entity=entity,
        target_pos=torch.tensor([0.40, -0.05, 0.50]),
        slip_position_threshold=0.0,
        slip_orientation_threshold=-1.0,
        stall_seconds=0,
    )
    policy.reset()
    mock_eef._pos = torch.tensor([0.35, 0.08, 0.46], dtype=torch.float32, device=world.device)
    mock_eef._quat = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], dtype=torch.float32, device=world.device)
    policy.step()
    assert policy.get_failure() is False


def test_box_gripper_grasp_policy_uses_candidate_provider(monkeypatch):
    world = GenesisWorld()
    mock_arm = _MockEEFController(world, pos=(0.0, 0.0, 0.8))
    mock_gripper = _MockGripperController()

    class _FailingObject:
        def list_box_gripper_grasping_candidates(self, *args, **kwargs):
            raise AssertionError("geometry candidate path should not be used")

    class _Provider(GraspCandidateProvider):
        def propose(self, *, mask=None, seed=None):
            del mask, seed
            return GraspCandidates(
                position=torch.tensor([[0.3, 0.1, 0.7]], dtype=torch.float32),
                approach_direction=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
                opening_direction=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
            )

    def _fake_reachable(_eef_ctrl, target_pos, target_quat, envs_idx=None):
        del _eef_ctrl, target_quat, envs_idx
        return torch.zeros(target_pos.shape[:-1] + (6,), dtype=torch.float32)

    monkeypatch.setattr(GoToRefPolicy, "check_reachable_error", staticmethod(_fake_reachable))

    policy = BoxGripperGraspPolicy(
        world=world,
        eef_controller=mock_arm,
        gripper_controller=mock_gripper,
        target_object=_FailingObject(),
        target_entity=object(),
        candidate_provider=_Provider(),
    )
    policy.reset()

    assert not policy.get_failure()
    assert torch.allclose(policy._target_grasp_pos, torch.tensor([0.3, 0.1, 0.7]))


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
    print("=== test_pickplace_transport_slip_detection_api ===")
    test_pickplace_transport_slip_detection_api()
    print()
    print("=== test_pickplace_transport_slip_detection_init_default ===")
    test_pickplace_transport_slip_detection_init_default()
    print()
    print("=== test_pickplace_transport_slip_detection_disabled_at_init ===")
    test_pickplace_transport_slip_detection_disabled_at_init()
    print()
    print("=== test_pickplace_phase_specific_kwargs_split ===")
    test_pickplace_phase_specific_kwargs_split()
    print()
    print("All tests passed!")
