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
  - Explicit dt: required keyword-only step argument; controls GoTo
    timeout/stall, grasp phase timing/timeout, and RepeatActionRefPolicy
    duration-based success; forwarded through CombinedRefPolicy and
    GraphRefPolicy; PickPlace release duration tracked in seconds.
  - Interface: step() without dt raises TypeError on base, composite, and
    graph policies.
  - Common-rate boundary timing: 1.0s durations/timeouts at 60 Hz and 100 Hz
    complete on the exact boundary step (float64 elapsed counters);
    MoveObjectPolicy timeout boundaries verified at 15 Hz and 30 Hz.
  - GoToRefPolicy: frame-dependent action recomputed on reset (full + masked)
    when the controller base frame changes.
"""

import math
import numpy as np
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
from unienv_genesis_refpolicy.implementations.grasp_policy import (
    BoxGripperGraspPolicy,
    GraspPhase,
    filter_fn_clean_top_down,
)
from unienv_genesis_refpolicy.implementations.repeat_action_policy import RepeatActionRefPolicy
from unienv_genesis_refpolicy.base.policy import BaseRefPolicy
from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider
from unienv_genesis.utils.batch_dim_utils import is_batched
from unienv_interface.space import DictSpace, BoxSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.world import WorldEnv


# ------------------------------------------------------------------
# Mock EEF controller
# ------------------------------------------------------------------

class _MockEEFController:
    """Complete mock EEF controller for policy testing."""

    has_eef = True

    def __init__(self, world, pos, quat=None, action_dim=7, base_pos=None):
        self.world = world
        self._pos = torch.tensor(pos, dtype=torch.float32, device=world.device)
        self._quat = torch.tensor(
            quat if quat is not None else (1.0, 0.0, 0.0, 0.0),
            dtype=torch.float32, device=world.device,
        )
        self._action_dim = action_dim
        # Optional mock base frame.  When set, get_action_world converts
        # world-frame targets into this (translated) base frame, mimicking
        # EndEffectorPoseController.convert_absolute_to_base.
        self._base_pos = (
            None if base_pos is None
            else torch.tensor(base_pos, dtype=torch.float32, device=world.device)
        )

    def get_eef_pose_absolute(self):
        return self._pos, self._quat

    def get_action_world(self, pos, quat):
        if self._base_pos is not None:
            pos = pos - self._base_pos
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
    policy.step(dt=0.05)
    assert policy.get_success() is False
    assert policy.get_failure() is False

    # Simulate the EEF reaching the target by moving the mock.
    mock_eef._pos = target_pos.clone()
    policy.step(dt=0.05)
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
    policy.step(dt=0.05)
    success = policy.get_success()
    assert isinstance(success, torch.Tensor)
    assert success.shape == (N_ENVS,)
    assert not success.any()

    # Move EEF to per-env targets.
    mock_eef._pos = task._target_pos.clone()
    policy.step(dt=0.05)
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
    policy.step(dt=0.05)
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
    policy.step(dt=0.05)
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
    policy_a.step(dt=0.05)
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
    policy_b.step(dt=0.05)
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
    policy.step(dt=0.05)
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


@pytest.mark.parametrize(
    "cost, expected_x",
    [
        (None, 0.30),               # lowest IK error wins
        ([0.5, 0.1, 0.0], 0.31),    # cheapest *reachable* candidate wins
    ],
)
def test_box_gripper_grasp_policy_cost_ranks_only_reachable_candidates(
    monkeypatch, cost, expected_x,
):
    world = GenesisWorld()

    class _Provider(GraspCandidateProvider):
        def propose(self, *, mask=None, seed=None):
            del mask, seed
            return GraspCandidates(
                position=torch.tensor(
                    [[0.30, 0.0, 0.7], [0.31, 0.0, 0.7], [0.50, 0.0, 0.7]],
                    dtype=torch.float32,
                ),
                approach_direction=torch.tensor([[0.0, 0.0, -1.0]] * 3),
                opening_direction=torch.tensor([[0.0, 1.0, 0.0]] * 3),
            )

    def _fake_reachable(_eef_ctrl, target_pos, target_quat, envs_idx=None):
        # IK error grows with x: 0, 0.01 and 0.2 (beyond the 0.05 threshold).
        del _eef_ctrl, target_quat, envs_idx
        errors = torch.zeros(target_pos.shape[:-1] + (6,), dtype=torch.float32)
        errors[..., 0] = target_pos[..., 0] - 0.30
        return errors

    monkeypatch.setattr(GoToRefPolicy, "check_reachable_error", staticmethod(_fake_reachable))

    policy = BoxGripperGraspPolicy(
        world=world,
        eef_controller=_MockEEFController(world, pos=(0.0, 0.0, 0.8)),
        gripper_controller=_MockGripperController(),
        target_object=object(),
        target_entity=object(),
        candidate_provider=_Provider(),
        candidate_cost_fn=None if cost is None else (lambda _c: torch.tensor(cost)),
    )
    policy.reset()

    assert not policy.get_failure()
    assert float(policy._target_grasp_pos[0]) == pytest.approx(expected_x)


# ------------------------------------------------------------------
# Restrictive top-down candidate filter
# ------------------------------------------------------------------

def _candidates(approach, opening, position=None):
    approach = torch.as_tensor(approach, dtype=torch.float32)
    opening = torch.as_tensor(opening, dtype=torch.float32)
    if position is None:
        position = torch.zeros(approach.shape)
    return GraspCandidates(
        position=torch.as_tensor(position, dtype=torch.float32),
        approach_direction=approach,
        opening_direction=opening,
    )


def test_filter_fn_clean_top_down_keeps_only_clean_top_downs():
    """Clean top-down kept; shallow, degenerate-frame, and side grasps rejected."""
    # (a) Clean top-down: approach straight down, opening orthogonal to it.
    keep = filter_fn_clean_top_down(
        _candidates([[0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0]])
    )
    assert keep.shape == (1,) and keep.dtype == torch.bool
    assert bool(keep[0]), "clean top-down candidate must be kept"

    # (b) Near-horizontal approach (approach_z = -0.04 > -0.5) → rejected.
    keep = filter_fn_clean_top_down(
        _candidates([[0.0, 0.9992, -0.04]], [[1.0, 0.0, 0.0]])
    )
    assert not bool(keep[0]), "near-horizontal candidate must be rejected"

    # (c) Degenerate frame: approach points down-ish (z = -0.8) but is almost
    #     collinear with the opening (|dot| ≈ 0.99 > 0.5) → rejected.
    approach = torch.tensor([[0.0, -0.6, -0.8]])
    opening = approach + 0.14 * torch.tensor([[1.0, 0.0, 0.0]])
    opening = opening / opening.norm(dim=-1, keepdim=True)
    assert abs(float((approach * opening).sum())) > 0.5  # sanity: degenerate
    keep = filter_fn_clean_top_down(
        _candidates(approach, opening)
    )
    assert not bool(keep[0]), "degenerate-frame candidate must be rejected"

    # (d) Side grasp (approach horizontal) → rejected.
    keep = filter_fn_clean_top_down(
        _candidates([[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    )
    assert not bool(keep[0]), "side grasp must be rejected"

    # Relaxed threshold sanity check: the shallow one passes with min_down_cos=0.
    keep = filter_fn_clean_top_down(
        _candidates([[0.0, 0.9992, -0.04]], [[1.0, 0.0, 0.0]]),
        min_down_cos=0.0,
    )
    assert bool(keep[0])


def test_filter_fn_clean_top_down_batched_per_row_fallback():
    """Batched rows with no clean candidate fall back to keeping all of them."""
    # Row 0: one clean top-down + one side grasp → keep only the clean one.
    # Row 1: only dirty candidates → keep all (per-row fallback).
    candidates = _candidates(
        approach=[
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.9992, -0.04]],
        ],
        opening=[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ],
    )
    keep = filter_fn_clean_top_down(candidates)
    assert keep.shape == (2, 2)
    assert keep[0].tolist() == [True, False]
    assert keep[1].tolist() == [True, True], "zero-clean row keeps all candidates"


# ------------------------------------------------------------------
# Explicit-dt timing semantics
# ------------------------------------------------------------------

class _RecordingPolicy(BaseRefPolicy):
    """Minimal policy that records the dt/mask passed to step()."""

    def __init__(self, world):
        super().__init__(world)
        self.step_dts = []
        self.step_masks = []

    def reset(self, *, mask=None, seed=None):
        pass

    def step(self, *, mask=None, dt: float):
        self.step_dts.append(dt)
        self.step_masks.append(mask)

    def get_action(self):
        if is_batched(self.world.batch_size):
            return torch.zeros(self.world.batch_size, 1, device=self.world.device)
        return torch.zeros(1, device=self.world.device)

    def _false(self):
        if is_batched(self.world.batch_size):
            return torch.zeros(
                self.world.batch_size, dtype=torch.bool, device=self.world.device,
            )
        return False

    def get_success(self):
        return self._false()

    def get_failure(self):
        return self._false()

    def get_timeout(self):
        return self._false()


def test_graph_policy_visited_nodes_unbatched():
    """``get_visited_node_names`` tracks only the nodes actually entered."""
    world = GenesisWorld()

    class _SucceedAfter(_RecordingPolicy):
        def __init__(self, world, steps):
            super().__init__(world)
            self._steps = steps
            self._count = 0

        def reset(self, *, mask=None, seed=None):
            self._count = 0

        def get_success(self):
            self._count += 1
            return self._count >= self._steps

    a = _SucceedAfter(world, steps=2)  # succeeds on the 2nd get_success call
    b = _RecordingPolicy(world)
    c = _RecordingPolicy(world)
    graph = GraphRefPolicy(
        world,
        nodes={"a": a, "b": b, "c": c},
        success_edges=[("a", "b"), ("b", "c")],
        fail_edges=[],
        initial="a",
    )
    # Initial node is visited from the start.
    assert graph.get_visited_node_names() == ["a"]

    graph.step(dt=0.1)  # a: count 1 (<2), no transition
    assert graph.get_visited_node_names() == ["a"]
    graph.step(dt=0.1)  # a: count 2 -> success -> transition to b
    assert set(graph.get_visited_node_names()) == {"a", "b"}
    # c is never entered.
    assert "c" not in graph.get_visited_node_names()

    # Reset returns to the initial node and clears the visit history.
    graph.reset()
    assert graph.get_visited_node_names() == ["a"]

    print("  PASS: graph policy visited nodes (unbatched)")


def test_graph_policy_visited_nodes_batched():
    """Per-env visit tracking: only the env that transitions visits the node."""
    world = GenesisWorld(batch_size=2)

    class _SelectiveSuccess(_RecordingPolicy):
        # Env 0 succeeds immediately; env 1 never does.
        def get_success(self):
            return torch.tensor([True, False], device=world.device)

    a = _SelectiveSuccess(world)
    b = _RecordingPolicy(world)
    graph = GraphRefPolicy(
        world,
        nodes={"a": a, "b": b},
        success_edges=[("a", "b")],
        fail_edges=[],
        initial="a",
    )
    visited = graph.get_visited_node_names()
    assert visited == [["a"], ["a"]]

    graph.step(dt=0.1)
    visited = graph.get_visited_node_names()
    assert set(visited[0]) == {"a", "b"}  # env 0 transitioned to b
    assert visited[1] == ["a"]            # env 1 never left a

    # A masked reset of env 0 clears only env 0's history.
    mask = torch.tensor([True, False], device=world.device)
    graph.reset(mask=mask)
    visited = graph.get_visited_node_names()
    assert visited[0] == ["a"]
    assert visited[1] == ["a"]

    print("  PASS: graph policy visited nodes (batched)")


def test_goto_explicit_dt_controls_timeout():
    """max_seconds must be measured against the explicit step dt."""
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.0, 0.0, 0.5))
    policy = GoToRefPolicy(
        world, controller=mock_eef,
        target_pos=torch.tensor([0.5, 0.0, 0.5]),
        target_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        stall_seconds=0,
        max_seconds=1.0,
    )
    policy.reset()

    # Explicit dt=0.25 — deliberately different from world.world_timestep.
    for _ in range(3):
        policy.step(dt=0.25)
    assert policy._elapsed_seconds.item() == pytest.approx(0.75)
    assert policy.get_timeout() is False
    policy.step(dt=0.25)  # total = 1.0 >= max_seconds
    assert policy._elapsed_seconds.item() == pytest.approx(1.0)
    assert policy.get_timeout() is True

    print("  PASS: goto explicit dt controls timeout")


def test_goto_explicit_dt_controls_stall():
    """stall_seconds must accumulate the explicit step dt."""
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.0, 0.0, 0.5))
    policy = GoToRefPolicy(
        world, controller=mock_eef,
        target_pos=torch.tensor([0.5, 0.0, 0.5]),
        target_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        stall_seconds=0.5,
        max_seconds=0,
    )
    policy.reset()

    # Static EEF → stalled every step; explicit dt=0.25 → stall after 2 steps.
    policy.step(dt=0.25)
    assert policy._stall_elapsed_seconds.item() == pytest.approx(0.25)
    assert policy.get_failure() is False
    policy.step(dt=0.25)
    assert policy._stall_elapsed_seconds.item() == pytest.approx(0.5)
    assert policy.get_failure() is True

    print("  PASS: goto explicit dt controls stall")


def _make_grasp_policy(monkeypatch, world, **overrides):
    """Build a BoxGripperGraspPolicy with a single canned candidate."""
    mock_arm = _MockEEFController(world, pos=(0.0, 0.0, 0.8))
    mock_gripper = _MockGripperController()

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

    kwargs = dict(
        world=world,
        eef_controller=mock_arm,
        gripper_controller=mock_gripper,
        target_object=object(),
        target_entity=object(),
        candidate_provider=_Provider(),
        open_gripper_seconds=0.5,
        close_gripper_seconds=0.5,
        phase_patience_seconds=10.0,
        max_seconds=100.0,
    )
    kwargs.update(overrides)
    return BoxGripperGraspPolicy(**kwargs)


def test_grasp_explicit_dt_controls_phase_timing(monkeypatch):
    """open_gripper_seconds phase timing must follow the explicit step dt."""
    world = GenesisWorld()
    policy = _make_grasp_policy(monkeypatch, world)
    policy.reset()

    # Explicit dt=0.25: 0.5s open-gripper phase completes after 2 steps.
    policy.step(dt=0.25)
    assert policy._phase.item() == GraspPhase.OPEN_GRIPPER
    policy.step(dt=0.25)
    assert policy._phase.item() == GraspPhase.PRE_GRASP
    assert policy._total_elapsed_seconds.item() == pytest.approx(0.5)

    print("  PASS: grasp explicit dt controls phase timing")


def test_grasp_explicit_dt_controls_timeout(monkeypatch):
    """max_seconds timeout must follow the explicit step dt."""
    world = GenesisWorld()
    policy = _make_grasp_policy(monkeypatch, world, max_seconds=0.5)
    policy.reset()

    policy.step(dt=0.25)
    assert policy.get_timeout() is False
    policy.step(dt=0.25)  # total = 0.5 >= max_seconds; phase != DONE, not failed
    assert policy.get_timeout() is True

    print("  PASS: grasp explicit dt controls timeout")


def test_repeat_action_duration_seconds_uses_explicit_dt():
    """duration_seconds success must accumulate the explicit step dt."""
    world = GenesisWorld()
    policy = RepeatActionRefPolicy(
        world, action=torch.tensor([-1.0]), duration_seconds=0.5,
    )
    policy.reset()

    policy.step(dt=0.25)
    assert policy.get_success() is False
    policy.step(dt=0.25)
    assert policy.get_success() is True

    print("  PASS: repeat-action duration_seconds uses explicit dt")


def test_repeat_action_duration_15hz_boundary_with_explicit_dt():
    """15 steps of dt=1/15 complete a 1.0s duration exactly.

    Guards the float-accumulation epsilon: 15 * (1/15) = 0.9999999999999999
    must still count as reaching 1.0s.
    """
    world = GenesisWorld()
    policy = RepeatActionRefPolicy(
        world, action=torch.tensor([-1.0]), duration_seconds=1.0,
    )
    policy.reset()
    for _ in range(14):
        policy.step(dt=1.0 / 15.0)
        assert policy.get_success() is False
    policy.step(dt=1.0 / 15.0)  # 15th step
    assert policy.get_success() is True

    print("  PASS: repeat-action 15 Hz duration boundary with explicit dt")


def test_repeat_action_indefinite_hold_never_succeeds():
    """duration_seconds=None holds indefinitely: success never fires."""
    world = GenesisWorld()
    policy = RepeatActionRefPolicy(world, action=torch.tensor([-1.0]))
    policy.reset()
    for _ in range(10):
        policy.step(dt=0.25)
        assert policy.get_success() is False

    print("  PASS: repeat-action indefinite hold never succeeds")


def test_repeat_action_rejects_non_positive_duration():
    """duration_seconds must be strictly positive when supplied."""
    world = GenesisWorld()
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError):
            RepeatActionRefPolicy(
                world, action=torch.tensor([-1.0]), duration_seconds=bad,
            )

    print("  PASS: repeat-action rejects non-positive duration")


def test_repeat_action_reset_and_mask_batched():
    """Masked step/reset accumulate and clear elapsed time per env."""
    world = GenesisWorld(batch_size=3)
    policy = RepeatActionRefPolicy(
        world, action=torch.tensor([-1.0]), duration_seconds=0.5,
    )
    policy.reset()

    mask = torch.tensor([True, False, True], device=world.device)
    policy.step(mask=mask, dt=0.25)
    policy.step(mask=mask, dt=0.25)
    success = policy.get_success()
    assert success.tolist() == [True, False, True]

    # Masked reset clears only the masked envs.
    reset_mask = torch.tensor([True, False, False], device=world.device)
    policy.reset(mask=reset_mask)
    assert policy._elapsed_seconds.tolist() == [0.0, 0.0, 0.5]

    # Full reset clears everything.
    policy.reset()
    assert policy._elapsed_seconds.tolist() == [0.0, 0.0, 0.0]
    assert policy.get_success().tolist() == [False, False, False]

    print("  PASS: repeat-action reset/mask behavior (batched)")


def test_combined_policy_forwards_dt():
    """CombinedRefPolicy must forward the explicit dt to all sub-policies."""
    world = GenesisWorld()
    a = _RecordingPolicy(world)
    b = _RecordingPolicy(world)
    combined = CombinedRefPolicy(world, {"a": a, "b": b})

    combined.step(dt=0.123)
    assert a.step_dts == [0.123]
    assert b.step_dts == [0.123]
    combined.step(dt=0.25)
    assert a.step_dts == [0.123, 0.25]
    assert b.step_dts == [0.123, 0.25]

    print("  PASS: combined policy forwards dt")


def test_graph_policy_forwards_dt_unbatched():
    """GraphRefPolicy must forward the explicit dt to the active node (unbatched)."""
    world = GenesisWorld()
    a = _RecordingPolicy(world)
    graph = GraphRefPolicy(
        world, nodes={"a": a}, success_edges=[], fail_edges=[], initial="a",
    )
    graph.step(dt=0.5)
    assert a.step_dts == [0.5]
    graph.step(dt=0.25)
    assert a.step_dts == [0.5, 0.25]

    print("  PASS: graph policy forwards dt (unbatched)")


def test_graph_policy_forwards_dt_batched():
    """GraphRefPolicy must forward dt to active nodes (batched, masked)."""
    world = GenesisWorld(batch_size=3)
    a = _RecordingPolicy(world)
    graph = GraphRefPolicy(
        world, nodes={"a": a}, success_edges=[], fail_edges=[], initial="a",
    )
    mask = torch.tensor([True, False, True], device=world.device)
    graph.step(mask=mask, dt=0.25)
    assert a.step_dts == [0.25]
    # The node mask must restrict stepping to alive, non-terminal envs.
    node_mask = a.step_masks[0]
    assert node_mask.shape == (3,)
    assert node_mask[0] and not node_mask[1] and node_mask[2]

    print("  PASS: graph policy forwards dt (batched)")


# ------------------------------------------------------------------
# Required-dt step interface
# ------------------------------------------------------------------

def test_step_requires_dt_keyword():
    """step() without a dt keyword must raise TypeError everywhere."""
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.0, 0.0, 0.5))
    goto = GoToRefPolicy(
        world, controller=mock_eef,
        target_pos=torch.tensor([0.5, 0.0, 0.5]),
        target_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        stall_seconds=0,
    )
    repeat = RepeatActionRefPolicy(world, action=torch.tensor([-1.0]))
    combined = CombinedRefPolicy(world, {"a": _RecordingPolicy(world)})
    graph = GraphRefPolicy(
        world, nodes={"a": _RecordingPolicy(world)},
        success_edges=[], fail_edges=[], initial="a",
    )
    for policy in (goto, repeat, combined, graph):
        with pytest.raises(TypeError):
            policy.step()

    print("  PASS: step requires dt keyword")


def test_recording_policy_signature_matches_base():
    """The test double honors the required keyword-only dt interface."""
    import inspect

    for cls in (BaseRefPolicy, _RecordingPolicy, CombinedRefPolicy, GraphRefPolicy):
        params = inspect.signature(cls.step).parameters
        assert "dt" in params
        assert params["dt"].default is inspect.Parameter.empty
        assert params["dt"].kind is inspect.Parameter.KEYWORD_ONLY

    print("  PASS: step dt is required keyword-only across policies")


# ------------------------------------------------------------------
# Common-rate (60 Hz / 100 Hz) boundary timing
# ------------------------------------------------------------------

@pytest.mark.parametrize("rate_hz", [60.0, 100.0])
def test_repeat_action_duration_boundary_common_rates(rate_hz):
    """A 1.0s duration at 60/100 Hz completes on the exact boundary step.

    Regression: float32 elapsed counters accumulate enough error over
    ``rate`` additions to land a hair below the threshold at the boundary
    step, delaying the transition by one full step.
    """
    world = GenesisWorld()
    dt = 1.0 / rate_hz
    policy = RepeatActionRefPolicy(
        world, action=torch.tensor([-1.0]), duration_seconds=1.0,
    )
    policy.reset()
    for _ in range(int(rate_hz) - 1):
        policy.step(dt=dt)
        assert policy.get_success() is False
    policy.step(dt=dt)  # exactly 1.0s elapsed
    assert policy.get_success() is True

    print(f"  PASS: repeat-action 1.0s boundary at {rate_hz:.0f} Hz")


@pytest.mark.parametrize("rate_hz", [60.0, 100.0])
def test_goto_timeout_boundary_common_rates(rate_hz):
    """GoTo max_seconds=1.0 at 60/100 Hz times out on the boundary step."""
    world = GenesisWorld()
    dt = 1.0 / rate_hz
    mock_eef = _MockEEFController(world, pos=(0.0, 0.0, 0.5))
    policy = GoToRefPolicy(
        world, controller=mock_eef,
        target_pos=torch.tensor([0.5, 0.0, 0.5]),
        target_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        stall_seconds=0,
        max_seconds=1.0,
    )
    policy.reset()
    for _ in range(int(rate_hz) - 1):
        policy.step(dt=dt)
        assert policy.get_timeout() is False
    policy.step(dt=dt)  # exactly 1.0s elapsed
    assert policy.get_timeout() is True

    print(f"  PASS: goto timeout 1.0s boundary at {rate_hz:.0f} Hz")


@pytest.mark.parametrize("rate_hz", [60.0, 100.0])
def test_goto_stall_boundary_common_rates(rate_hz):
    """GoTo stall_seconds=1.0 at 60/100 Hz fails on the boundary step."""
    world = GenesisWorld()
    dt = 1.0 / rate_hz
    mock_eef = _MockEEFController(world, pos=(0.0, 0.0, 0.5))
    policy = GoToRefPolicy(
        world, controller=mock_eef,
        target_pos=torch.tensor([0.5, 0.0, 0.5]),
        target_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        stall_seconds=1.0,
        max_seconds=0,
    )
    policy.reset()
    # Static EEF → stalled every step.
    for _ in range(int(rate_hz) - 1):
        policy.step(dt=dt)
        assert policy.get_failure() is False
    policy.step(dt=dt)  # exactly 1.0s stalled
    assert policy.get_failure() is True

    print(f"  PASS: goto stall 1.0s boundary at {rate_hz:.0f} Hz")


@pytest.mark.parametrize("rate_hz", [15.0, 30.0])
def test_move_object_timeout_boundary_common_rates(rate_hz):
    """MoveObjectPolicy max_seconds=1.0 at 15/30 Hz times out on the boundary step.

    Regression: the override compared against ``max_seconds`` without the
    shared epsilon, so ``15 * (1/15) = 0.9999999999999999 < 1.0`` in float64
    slipped the timeout by one full step.
    """
    world = GenesisWorld()
    dt = 1.0 / rate_hz
    mock_eef = _MockEEFController(
        world, pos=(0.30, 0.08, 0.46), quat=(1.0, 0.0, 0.0, 0.0),
    )
    entity = _MockPoseEntity(
        pos=(0.25, 0.10, 0.30), quat=(1.0, 0.0, 0.0, 0.0), world=world,
    )
    policy = MoveObjectPolicy(
        world=world,
        controller=mock_eef,
        object_entity=entity,
        target_pos=torch.tensor([0.40, -0.05, 0.50]),
        stall_seconds=0,  # isolate the timeout path
        max_seconds=1.0,
    )
    policy.reset()
    # Static EEF and object → never reached, never slipped.
    for _ in range(int(rate_hz) - 1):
        policy.step(dt=dt)
        assert policy.get_timeout() is False
    policy.step(dt=dt)  # exactly 1.0s elapsed
    assert policy.get_timeout() is True

    print(f"  PASS: move-object timeout 1.0s boundary at {rate_hz:.0f} Hz")


@pytest.mark.parametrize("rate_hz", [60.0, 100.0])
def test_grasp_phase_boundary_common_rates(monkeypatch, rate_hz):
    """open_gripper_seconds=1.0 at 60/100 Hz transitions on the boundary step."""
    world = GenesisWorld()
    dt = 1.0 / rate_hz
    policy = _make_grasp_policy(
        monkeypatch, world,
        open_gripper_seconds=1.0,
        phase_patience_seconds=100.0,
        max_seconds=0.0,
    )
    policy.reset()
    for _ in range(int(rate_hz) - 1):
        policy.step(dt=dt)
        assert policy._phase.item() == GraspPhase.OPEN_GRIPPER
    policy.step(dt=dt)  # exactly 1.0s in the open-gripper phase
    assert policy._phase.item() == GraspPhase.PRE_GRASP

    print(f"  PASS: grasp phase 1.0s boundary at {rate_hz:.0f} Hz")


@pytest.mark.parametrize("rate_hz", [60.0, 100.0])
def test_grasp_timeout_boundary_common_rates(monkeypatch, rate_hz):
    """Grasp max_seconds=1.0 at 60/100 Hz times out on the boundary step."""
    world = GenesisWorld()
    dt = 1.0 / rate_hz
    policy = _make_grasp_policy(
        monkeypatch, world,
        open_gripper_seconds=100.0,  # stay in OPEN_GRIPPER for the whole test
        phase_patience_seconds=100.0,
        max_seconds=1.0,
    )
    policy.reset()
    for _ in range(int(rate_hz) - 1):
        policy.step(dt=dt)
        assert policy.get_timeout() is False
    policy.step(dt=dt)  # exactly 1.0s elapsed
    assert policy.get_timeout() is True

    print(f"  PASS: grasp timeout 1.0s boundary at {rate_hz:.0f} Hz")


def test_pickplace_release_tracks_duration_in_seconds():
    """The release phase must track release_seconds via the explicit dt."""
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
        release_seconds=1.0,
    )

    # Structural: the release gripper policy uses duration-based tracking.
    release_node = policy._graph._node_policies["release"]
    release_gripper = release_node._policies["gripper"]
    assert isinstance(release_gripper, RepeatActionRefPolicy)
    assert release_gripper.duration_seconds == 1.0

    # Functional: stepping the release node (a CombinedRefPolicy) with an
    # explicit dt controls the release duration — 1.0s at dt=0.25 succeeds
    # after exactly 4 steps, proving dt propagates through the composite.
    release_node.reset()
    for _ in range(3):
        release_node.step(dt=0.25)
        assert release_node.get_success() is False
    release_node.step(dt=0.25)
    assert release_node.get_success() is True

    print("  PASS: pickplace release tracks duration in seconds via dt")
    env.close()


# ------------------------------------------------------------------
# GoTo action recomputation on reset (base-frame changes)
# ------------------------------------------------------------------

def test_goto_action_recomputed_on_reset_when_base_frame_changes():
    """A changed base frame must change the action after reset."""
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.0, 0.0, 0.5), base_pos=(0.0, 0.0, 0.0))
    target_pos = torch.tensor([0.5, 0.0, 0.5])
    target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    policy = GoToRefPolicy(
        world, controller=mock_eef,
        target_pos=target_pos, target_quat=target_quat,
        stall_seconds=0,
    )
    policy.reset()
    action_before = policy.get_action().clone()
    assert torch.allclose(action_before[:3], target_pos, atol=1e-6)

    # Move the base frame; the world-frame target is unchanged, so the
    # base-frame action must be recomputed on reset.
    mock_eef._base_pos = torch.tensor([0.1, 0.0, 0.0], device=world.device)
    policy.reset()
    action_after = policy.get_action().clone()
    assert not torch.allclose(action_before, action_after)
    assert torch.allclose(
        action_after[:3], target_pos - torch.tensor([0.1, 0.0, 0.0]), atol=1e-6,
    )

    print("  PASS: goto action recomputed on reset after base-frame change")


def test_goto_action_recomputed_selectively_for_mask():
    """Masked reset must recompute the action only for the masked envs."""
    world = GenesisWorld(batch_size=2)
    mock_eef = _MockEEFController(
        world,
        pos=[[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
        quat=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        base_pos=(0.0, 0.0, 0.0),
    )
    target_pos = torch.tensor([[0.5, 0.0, 0.5], [0.6, 0.0, 0.5]])
    target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    policy = GoToRefPolicy(
        world, controller=mock_eef,
        target_pos=target_pos, target_quat=target_quat,
        stall_seconds=0,
    )
    policy.reset()
    action_before = policy.get_action().clone()
    assert torch.allclose(action_before[:, :3], target_pos, atol=1e-6)

    # Move the base frame in BOTH envs, but reset only env 0: env 0's action
    # must reflect the new base frame while env 1 keeps its previous action.
    mock_eef._base_pos = torch.tensor(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], device=world.device,
    )
    mask = torch.tensor([True, False], device=world.device)
    policy.reset(mask=mask)
    action_after = policy.get_action().clone()

    assert torch.allclose(
        action_after[0, :3],
        target_pos[0] - torch.tensor([0.1, 0.0, 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(action_after[1], action_before[1], atol=1e-6)

    print("  PASS: goto action recomputed selectively for mask")


def test_goto_lambda_action_follows_base_frame_change():
    """GoToLambdaRefPolicy keeps a single, correct action recomputation."""
    world = GenesisWorld()
    mock_eef = _MockEEFController(world, pos=(0.0, 0.0, 0.5), base_pos=(0.0, 0.0, 0.0))
    fixed_target = torch.tensor([0.5, 0.0, 0.5], device=world.device)
    policy = GoToLambdaRefPolicy(
        world, controller=mock_eef,
        target_pos_fn=lambda: fixed_target,
        stall_seconds=0,
    )
    policy.reset()
    action_before = policy.get_action().clone()
    assert torch.allclose(action_before[:3], fixed_target, atol=1e-6)

    mock_eef._base_pos = torch.tensor([0.1, 0.0, 0.0], device=world.device)
    policy.reset()
    action_after = policy.get_action().clone()
    assert torch.allclose(action_after[:3], fixed_target - 0.1 * torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)

    print("  PASS: goto-lambda action follows base-frame change")


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
    print("=== test_goto_explicit_dt_controls_timeout ===")
    test_goto_explicit_dt_controls_timeout()
    print()
    print("=== test_goto_explicit_dt_controls_stall ===")
    test_goto_explicit_dt_controls_stall()
    print()
    print("=== test_repeat_action_duration_seconds_uses_explicit_dt ===")
    test_repeat_action_duration_seconds_uses_explicit_dt()
    print()
    print("=== test_repeat_action_duration_15hz_boundary_with_explicit_dt ===")
    test_repeat_action_duration_15hz_boundary_with_explicit_dt()
    print()
    print("=== test_repeat_action_indefinite_hold_never_succeeds ===")
    test_repeat_action_indefinite_hold_never_succeeds()
    print()
    print("=== test_repeat_action_rejects_non_positive_duration ===")
    test_repeat_action_rejects_non_positive_duration()
    print()
    print("=== test_repeat_action_reset_and_mask_batched ===")
    test_repeat_action_reset_and_mask_batched()
    print()
    print("=== test_combined_policy_forwards_dt ===")
    test_combined_policy_forwards_dt()
    print()
    print("=== test_graph_policy_forwards_dt_unbatched ===")
    test_graph_policy_forwards_dt_unbatched()
    print()
    print("=== test_graph_policy_forwards_dt_batched ===")
    test_graph_policy_forwards_dt_batched()
    print()
    print("=== test_step_requires_dt_keyword ===")
    test_step_requires_dt_keyword()
    print()
    print("=== test_recording_policy_signature_matches_base ===")
    test_recording_policy_signature_matches_base()
    print()
    for rate in (60.0, 100.0):
        print(f"=== test_repeat_action_duration_boundary_common_rates[{rate:.0f}] ===")
        test_repeat_action_duration_boundary_common_rates(rate)
        print()
        print(f"=== test_goto_timeout_boundary_common_rates[{rate:.0f}] ===")
        test_goto_timeout_boundary_common_rates(rate)
        print()
        print(f"=== test_goto_stall_boundary_common_rates[{rate:.0f}] ===")
        test_goto_stall_boundary_common_rates(rate)
        print()
    for rate in (15.0, 30.0):
        print(f"=== test_move_object_timeout_boundary_common_rates[{rate:.0f}] ===")
        test_move_object_timeout_boundary_common_rates(rate)
        print()
    print("=== test_pickplace_release_tracks_duration_in_seconds ===")
    test_pickplace_release_tracks_duration_in_seconds()
    print()
    print("=== test_goto_action_recomputed_on_reset_when_base_frame_changes ===")
    test_goto_action_recomputed_on_reset_when_base_frame_changes()
    print()
    print("=== test_goto_action_recomputed_selectively_for_mask ===")
    test_goto_action_recomputed_selectively_for_mask()
    print()
    print("=== test_goto_lambda_action_follows_base_frame_change ===")
    test_goto_lambda_action_follows_base_frame_change()
    print()
    print("All tests passed!")


# ------------------------------------------------------------------
# BboxGraspCandidateProvider
# ------------------------------------------------------------------


class _FakeBoxMesh:
    """Minimal trimesh stand-in reporting a volume equal to its bbox.

    ``extents``/``volume`` are numpy (as real trimesh returns them) so the
    provider's fill-ratio gate is exercised against the real dtypes.
    """

    def __init__(self, extents, volume):
        self.extents = np.asarray(extents, dtype=np.float64)
        self.volume = volume


class _FakeBboxObject:
    """Minimal object descriptor for BboxGraspCandidateProvider tests."""

    def __init__(self, low, high, fill_ratio):
        self._low = torch.as_tensor(low, dtype=torch.float32)
        self._high = torch.as_tensor(high, dtype=torch.float32)
        extents = (self._high - self._low)
        bbox_volume = float(torch.prod(extents).item())
        self._mesh = _FakeBoxMesh(extents.tolist(), volume=fill_ratio * bbox_volume)

    def _load_trimesh(self):
        return self._mesh

    def get_local_bounding_box(self):
        from unienv_genesis_collection.objects.base_object import make_bbox

        return make_bbox(self._low, self._high)


class _FakeEntity:
    """Entity returning a fixed identity world pose."""

    def get_pos(self):
        return torch.zeros(3)

    def get_quat(self):
        return torch.tensor([1.0, 0.0, 0.0, 0.0])


class _FakeBaseProvider:
    """Returns a fixed set of base candidates (mimics the antipodal provider)."""

    def __init__(self, n, device="cpu"):
        from unienv_genesis_collection.objects.base_object import GraspCandidates

        self._candidates = GraspCandidates(
            position=torch.zeros(n, 3, device=device),
            approach_direction=torch.zeros(n, 3, device=device),
            opening_direction=torch.zeros(n, 3, device=device),
        )

    def propose(self, *, mask=None, seed=None):
        return self._candidates


def test_bbox_provider_supplements_box_like_object():
    """A box-like object gets bbox-face grasps appended to the base candidates."""
    from unienv_genesis_refpolicy.grasp_providers.bbox import (
        BboxGraspCandidateProvider,
    )

    # Box: 0.05 (x) x 0.06 (y) x 0.11 (z), fully filled (fill ratio 1.0).
    obj = _FakeBboxObject([0.0, 0.0, 0.0], [0.05, 0.06, 0.11], fill_ratio=1.0)
    provider = BboxGraspCandidateProvider(
        target_object=obj,
        target_entity=_FakeEntity(),
        gripper_width=0.085,
        gripper_depth=0.01,
        base_provider=_FakeBaseProvider(n=7),
    )
    assert provider.is_box_like()

    out = provider.propose()
    n_base = 7
    # x (0.05) and y (0.06) fit the 0.085 opening (with its 1 cm slack); z
    # (0.11) does not -> 2
    # eligible axes x 8 candidates each (2 approach axes x 2 approach signs x
    # 2 opening signs) = 16.
    n_bbox = int(out.position.shape[-2]) - n_base
    assert n_bbox == 16

    bbox_apr = out.approach_direction[n_base:]
    bbox_opn = out.opening_direction[n_base:]
    # Some candidates are genuine top-down (approach straight down).
    n_top_down = int((bbox_apr[:, 2] < -0.9).sum())
    assert n_top_down == 4  # closing along x, approaching down (+-z)
    # approach ⟂ opening for every bbox candidate.
    dots = (bbox_apr * bbox_opn).sum(-1).abs()
    assert float(dots.max()) < 1e-4


def test_bbox_provider_silent_for_non_convex_object():
    """A non-convex object (low fill ratio) yields only the base candidates."""
    from unienv_genesis_refpolicy.grasp_providers.bbox import (
        BboxGraspCandidateProvider,
    )

    # Toy-plane-like: bbox is large but the mesh fills a small fraction.
    obj = _FakeBboxObject([0.0, 0.0, 0.0], [0.2, 0.2, 0.05], fill_ratio=0.15)
    provider = BboxGraspCandidateProvider(
        target_object=obj,
        target_entity=_FakeEntity(),
        gripper_width=0.085,
        gripper_depth=0.01,
        base_provider=_FakeBaseProvider(n=5),
    )
    assert not provider.is_box_like()
    out = provider.propose()
    assert int(out.position.shape[-2]) == 5  # unchanged (base only)


def test_bbox_provider_respects_gripper_width():
    """Axes wider than the gripper opening produce no grasp on that axis."""
    from unienv_genesis_refpolicy.grasp_providers.bbox import (
        BboxGraspCandidateProvider,
    )

    # Box with one axis (y=0.20) wider than the 0.085 gripper opening.
    obj = _FakeBboxObject([0.0, 0.0, 0.0], [0.05, 0.20, 0.06], fill_ratio=1.0)
    provider = BboxGraspCandidateProvider(
        target_object=obj,
        target_entity=_FakeEntity(),
        gripper_width=0.085,
        gripper_depth=0.01,
        base_provider=_FakeBaseProvider(n=0),
    )
    out = provider.propose()
    # No candidate may open along the too-wide y axis.
    opn = out.opening_direction
    assert int((opn[:, 1].abs() > 0.5).sum()) == 0
    # x (0.05) and z (0.06) fit; y (0.20) does not -> 2 eligible axes x 8 = 16.
    assert int(out.position.shape[-2]) == 16


@pytest.mark.parametrize("width_margin, n_axes", [(0.0, 3), (0.01, 2)])
def test_bbox_provider_leaves_width_slack(width_margin, n_axes):
    """An axis within ``width_margin`` of the gripper opening is not grasped across."""
    from unienv_genesis_refpolicy.grasp_providers.bbox import (
        BboxGraspCandidateProvider,
    )

    # y (0.08) fits the 0.085 opening but not with 1 cm of slack.
    obj = _FakeBboxObject([0.0, 0.0, 0.0], [0.05, 0.08, 0.06], fill_ratio=1.0)
    provider = BboxGraspCandidateProvider(
        target_object=obj,
        target_entity=_FakeEntity(),
        gripper_width=0.085,
        gripper_depth=0.01,
        base_provider=_FakeBaseProvider(n=0),
        width_margin=width_margin,
    )
    out = provider.propose()
    assert int(out.position.shape[-2]) == 8 * n_axes


def test_bbox_provider_topdown_grasps_high_on_tall_box():
    """Downward-approach bbox grasps sit near the TOP face, not the bottom."""
    from unienv_genesis_refpolicy.grasp_providers.bbox import (
        BboxGraspCandidateProvider,
    )

    # Tall box: 0.05 (x) x 0.05 (y) footprint, 0.27 tall (z).  x and y fit the
    # 0.085 gripper; the downward approach must grasp near the top face.
    obj = _FakeBboxObject([0.0, 0.0, 0.0], [0.05, 0.05, 0.27], fill_ratio=1.0)
    provider = BboxGraspCandidateProvider(
        target_object=obj,
        target_entity=_FakeEntity(),  # identity pose at world origin
        gripper_width=0.085,
        gripper_depth=0.035,
        base_provider=_FakeBaseProvider(n=0),
    )
    assert provider.is_box_like()
    out = provider.propose()
    # Downward-approach candidates (approach straight down).
    down = out.approach_direction[:, 2] < -0.9
    assert int(down.sum()) > 0
    down_z = out.position[down][:, 2]
    # All downward grasps sit a finger-depth below the top face (0.27 - 0.035),
    # NOT a finger-depth above the bottom (0.035).
    assert float(down_z.min()) > 0.20, f"downward grasps too low: {down_z}"
    assert float(down_z.max()) <= 0.27
