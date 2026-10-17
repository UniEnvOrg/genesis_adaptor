"""Tests for the tabletop tasks: ReachTargetTask, ObjectRelocateTask.

Covers:
  - ReachTargetTask: target in workspace, reward = -distance(eef, target).
  - ReachTargetTask batched: per-env independent target sampling.
  - ReachTargetTask batched: batch/device-aware observation space; fixed
    target orientation observation matches the batched declared shape.
  - ReachTargetTask batched masked reset: unmasked targets preserved.
  - ReachTargetTask: deterministic after_reload priority selection.
  - ObjectRelocateTask: explicit priority 0 honored; deterministic fallbacks
    (max clutter priority + 1) and real dispatch-order semantics.
  - ObjectRelocateTask batched masked reset: unmasked goals/poses preserved;
    boolean reset mask forwarded unchanged as the placement envs_idx;
    unmasked position/orientation/DOF velocity untouched by partial
    placement writes.
  - ObjectRelocateTask: both initial and goal footprints reserved as avoid
    spaces for the clutter manager, including the local bbox-centre offset
    for off-centre objects.
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
from unienv_genesis_collection.scenes.cluttered_table_manager import ClutteredTableManager
from unienv_genesis_collection.objects import BoxObject
from unienv_genesis_collection.objects.base_object import make_bbox
from unienv_genesis_collection.tasks.tabletop import (
    ReachTargetTask,
    ObjectRelocateTask,
)
from unienv_interface.world import WorldEnv
from unienv_interface.space.space_utils.metadata_utils import validate_metadata


# ------------------------------------------------------------------
# Mock EEF controller for ReachTargetTask
# ------------------------------------------------------------------

class _MockEEFController:
    """Minimal stand-in exposing get_eef_pose_absolute()."""

    def __init__(self, pos, quat=None, batch_size=None):
        self._pos = torch.tensor(pos, dtype=torch.float32)
        self._quat = torch.tensor(
            quat if quat is not None else (1.0, 0.0, 0.0, 0.0),
            dtype=torch.float32,
        )
        if batch_size is not None:
            self._pos = self._pos.unsqueeze(0).expand(batch_size, 3).contiguous()
            self._quat = self._quat.unsqueeze(0).expand(batch_size, 4).contiguous()

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
# ReachTargetTask — observation space & masked reset
# ------------------------------------------------------------------

def test_reach_observation_space_batched():
    """Reach batched: observation space is batch-aware and on the world device."""
    N_ENVS = 4
    world = GenesisWorld(batch_size=N_ENVS)
    plane = PlaneBuilder(world, name="floor")

    fixed_quat = (0.0, 1.0, 0.0, 0.0)
    mock_eef = _MockEEFController(pos=(0.4, 0.0, 0.5), batch_size=N_ENVS)
    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world,
        eef_controller=mock_eef,
        target_workspace=workspace,
        target_orientation=fixed_quat,
    )
    env = WorldEnv(world, [plane, task], render_mode=None)
    env.reset(seed=0, reload=True)

    # Declared space shapes include the batch dimension.
    assert task.observation_space["eef_position"].shape == (N_ENVS, 3)
    assert task.observation_space["target_position"].shape == (N_ENVS, 3)
    assert task.observation_space["target_orientation"].shape == (N_ENVS, 4)

    # Device-aware declaration.
    for key in ("eef_position", "target_position", "target_orientation"):
        assert task.observation_space[key].device == world.device, (
            f"{key} space on {task.observation_space[key].device}, expected {world.device}"
        )

    # Space metadata: world-frame task positions / orientation.
    validate_metadata(task.observation_space)
    for key in ("eef_position", "target_position"):
        md = task.observation_space[key].metadata.to_dict()
        assert (md["kind"], md["units"], md["frame"]) == ("task", "m", "world"), key
    md = task.observation_space["target_orientation"].metadata.to_dict()
    assert (md["kind"], md["rotation"], md["frame"]) == ("task", "quat_wxyz", "world")

    # Observed values match the declared batched shapes.
    obs = task.get_observation()
    assert obs["eef_position"].shape == (N_ENVS, 3)
    assert obs["target_position"].shape == (N_ENVS, 3)
    assert obs["target_orientation"].shape == (N_ENVS, 4), (
        f"Fixed target orientation obs shape {obs['target_orientation'].shape} "
        f"!= declared {(N_ENVS, 4)}"
    )

    # The fixed orientation is broadcast to every env.
    expected_quat = torch.tensor(fixed_quat, dtype=world.dtype, device=world.device)
    assert torch.allclose(
        obs["target_orientation"],
        expected_quat.unsqueeze(0).expand(N_ENVS, 4),
    )

    print("  PASS: reach batched observation space (shapes + device + fixed orientation)")
    env.close()


def test_reach_observation_space_unbatched():
    """Reach unbatched: observation space stays batch-free (regression)."""
    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")

    mock_eef = _MockEEFController(pos=(0.4, 0.0, 0.5))
    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world, eef_controller=mock_eef, target_workspace=workspace,
    )
    env = WorldEnv(world, [plane, task], render_mode=None)
    env.reset(seed=0, reload=True)

    assert task.observation_space["eef_position"].shape == (3,)
    assert task.observation_space["target_position"].shape == (3,)
    assert task.observation_space["eef_position"].device == world.device

    obs = task.get_observation()
    assert obs["eef_position"].shape == (3,)
    assert obs["target_position"].shape == (3,)

    print("  PASS: reach unbatched observation space")
    env.close()


def test_reach_masked_reset_isolation():
    """Reach batched masked reset: unmasked targets / reward / termination preserved."""
    N_ENVS = 8
    world = GenesisWorld(batch_size=N_ENVS)
    plane = PlaneBuilder(world, name="floor")

    mock_eef = _MockEEFController(pos=(0.4, 0.0, 0.5), batch_size=N_ENVS)
    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world, eef_controller=mock_eef, target_workspace=workspace,
    )
    env = WorldEnv(world, [plane, task], render_mode=None)
    env.reset(seed=0, reload=True)

    targets_before = task._target_pos.clone()

    # Plant sentinel reward/termination values to verify preservation.
    sentinel_reward = torch.arange(N_ENVS, dtype=world.dtype, device=world.device) + 1.0
    sentinel_term = torch.zeros(N_ENVS, dtype=torch.bool, device=world.device)
    sentinel_term[[1, 4]] = True
    task._reward = sentinel_reward.clone()
    task._terminated = sentinel_term.clone()

    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[0, 3, 7]] = True
    env.reset(mask=reset_mask)

    preserved = [1, 2, 4, 5, 6]
    # Unmasked targets unchanged.
    assert torch.allclose(task._target_pos[preserved], targets_before[preserved]), (
        "Masked reset overwrote unmasked targets"
    )
    # Masked targets re-sampled (continuous space → new values w.h.p.).
    assert not torch.allclose(task._target_pos[[0, 3, 7]], targets_before[[0, 3, 7]]), (
        "Masked reset did not re-sample masked targets"
    )
    # Masked targets still inside the workspace.
    low = torch.tensor(workspace[0], dtype=world.dtype, device=world.device)
    high = torch.tensor(workspace[1], dtype=world.dtype, device=world.device)
    assert torch.all(task._target_pos >= low - 1e-6)
    assert torch.all(task._target_pos <= high + 1e-6)

    # Reward/termination: zeroed for masked envs, preserved elsewhere.
    assert torch.all(task._reward[[0, 3, 7]] == 0.0)
    assert torch.all(task._reward[preserved] == sentinel_reward[preserved])
    assert torch.all(task._terminated[[0, 3, 7]] == False)
    assert torch.equal(task._terminated[preserved], sentinel_term[preserved])

    print("  PASS: reach masked reset isolation")
    env.close()


def test_reach_after_reload_priority_deterministic():
    """Reach: after_reload triggers reset at min(reset_priorities)."""
    world = GenesisWorld()
    mock_eef = _MockEEFController(pos=(0.4, 0.0, 0.5))
    workspace = ((0.2, -0.2, 0.3), (0.6, 0.2, 0.8))
    task = ReachTargetTask(
        world, eef_controller=mock_eef, target_workspace=workspace,
    )

    # Multi-element priority set: selection must be deterministic (min).
    task.reset_priorities = {25, 30}
    task.after_reload_priorities = {25}

    seen = []
    orig_reset = task.reset
    def reset_spy(*, priority=0, **kwargs):
        seen.append(priority)
        return orig_reset(priority=priority, **kwargs)
    task.reset = reset_spy

    task.after_reload(priority=25)
    assert seen == [25], f"after_reload used priorities {seen}, expected [min=25]"

    print("  PASS: reach after_reload priority deterministic")


# ------------------------------------------------------------------
# ObjectRelocateTask
# ------------------------------------------------------------------

_TABLE_POS = (0.0, 0.0, 0.65)
_TABLE_SIZE = (0.8, 0.5)


def _make_relocate_env(world, n_clutter=2, **task_kwargs):
    """Build a table + clutter + ObjectRelocateTask environment."""
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=[
            BoxObject(size=(0.04, 0.04, 0.04), name="cube_r", color=(1, 0, 0, 1)),
            BoxObject(size=(0.03, 0.03, 0.03), name="cube_w", color=(1, 1, 1, 1)),
        ],
        n_objects_range=(n_clutter, n_clutter),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    target = BoxObject(size=(0.05, 0.05, 0.05), name="target", color=(0, 0, 1, 1))
    task = ObjectRelocateTask(
        world,
        table_builder=table,
        clutter_manager=clutter,
        target_object=target,
        **task_kwargs,
    )
    env = WorldEnv(world, [plane, table, clutter, task], render_mode=None)
    return env, task, clutter, target


def test_pickplace_priorities_explicit_zero():
    """ObjectRelocateTask: explicit 0 priorities honored (not falsy-skipped)."""
    world = GenesisWorld()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world, name="clutter",
        object_pool=[BoxObject(size=(0.03, 0.03, 0.03), name="c", color=(1, 0, 0, 1))],
        n_objects_range=(1, 1),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    target = BoxObject(size=(0.05, 0.05, 0.05), name="target", color=(0, 0, 1, 1))
    task = ObjectRelocateTask(
        world,
        table_builder=table,
        clutter_manager=clutter,
        target_object=target,
        reload_priority=0,
        reset_priority=0,
        after_reload_priority=0,
    )
    assert task.reload_priorities == {0}, task.reload_priorities
    assert task.reset_priorities == {0}, task.reset_priorities
    assert task.after_reload_priorities == {0}, task.after_reload_priorities

    # Space metadata: world-frame object / goal positions.
    validate_metadata(task.observation_space)
    for key in ("target_position", "goal_position"):
        md = task.observation_space[key].metadata.to_dict()
        assert (md["kind"], md["units"], md["frame"]) == ("task", "m", "world"), key

    print("  PASS: pickplace explicit priority 0 honored")
    world.close()


def test_pickplace_priorities_deterministic_fallback():
    """ObjectRelocateTask: fallback priorities derived deterministically (max + 1)."""
    world = GenesisWorld()
    clutter = ClutteredTableManager(
        world, name="clutter",
        object_pool=[BoxObject(size=(0.03, 0.03, 0.03), name="c", color=(1, 0, 0, 1))],
        n_objects_range=(1, 1),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    # Multi-element priority sets: fallback must sit above the MAXIMUM so the
    # task runs before every clutter callback (descending dispatch).
    clutter.reload_priorities = {25, 40}
    clutter.after_reload_priorities = {25, 40}
    clutter.reset_priorities = {25, 40}

    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    target = BoxObject(size=(0.05, 0.05, 0.05), name="target", color=(0, 0, 1, 1))
    task = ObjectRelocateTask(
        world, table_builder=table, clutter_manager=clutter, target_object=target,
    )
    assert task.reload_priorities == {41}, task.reload_priorities
    assert task.after_reload_priorities == {41}, task.after_reload_priorities
    assert task.reset_priorities == {41}, task.reset_priorities

    # after_reload triggers reset at min(reset_priorities).
    task.reset_priorities = {41, 60}
    task.target_object_entity = object()  # bypass the loaded-entity assert
    seen = []
    orig_reset = task.reset
    def reset_spy(*, priority=0, **kwargs):
        seen.append(priority)
        return None  # do not run the real placement
    task.reset = reset_spy
    task.after_reload(priority=41)
    assert seen == [41], f"after_reload used priorities {seen}, expected [min=41]"
    task.reset = orig_reset

    print("  PASS: pickplace deterministic priority fallback")
    world.close()


def test_pickplace_priority_ordering_semantics():
    """Task reset runs before EVERY clutter reset callback in a real env.

    With a multi-priority clutter manager {25, 40}, the fallback task
    priority must be max+1 = 41 so descending dispatch invokes the task
    ahead of both clutter callbacks (a min+1 = 26 task would run AFTER the
    priority-40 clutter callback).
    """
    N_ENVS = 2
    world = GenesisWorld(batch_size=N_ENVS)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world, name="clutter",
        object_pool=[
            BoxObject(size=(0.04, 0.04, 0.04), name="cube_r", color=(1, 0, 0, 1)),
            BoxObject(size=(0.03, 0.03, 0.03), name="cube_w", color=(1, 1, 1, 1)),
        ],
        n_objects_range=(1, 1),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    clutter.reload_priorities = {25, 40}
    clutter.after_reload_priorities = {25, 40}
    clutter.reset_priorities = {25, 40}

    target = BoxObject(size=(0.05, 0.05, 0.05), name="target", color=(0, 0, 1, 1))
    task = ObjectRelocateTask(
        world, table_builder=table, clutter_manager=clutter, target_object=target,
    )
    assert task.reset_priorities == {41}, task.reset_priorities

    # Record every reset callback (node, priority) during a real env reset.
    calls = []
    orig_task_reset = task.reset
    orig_clutter_reset = clutter.reset

    def task_reset_spy(*, priority=0, **kwargs):
        calls.append(("task", priority))
        return orig_task_reset(priority=priority, **kwargs)

    def clutter_reset_spy(*, priority=0, **kwargs):
        calls.append(("clutter", priority))
        return orig_clutter_reset(priority=priority, **kwargs)

    task.reset = task_reset_spy
    clutter.reset = clutter_reset_spy

    env = WorldEnv(world, [plane, table, clutter, task], render_mode=None)
    env.reset(seed=0, reload=True)
    calls.clear()
    env.reset(seed=1)

    task.reset = orig_task_reset
    clutter.reset = orig_clutter_reset

    task_calls = [i for i, (node, _p) in enumerate(calls) if node == "task"]
    clutter_calls = [i for i, (node, _p) in enumerate(calls) if node == "clutter"]
    assert task_calls, f"task reset never dispatched: {calls}"
    # Both clutter priorities were dispatched.
    clutter_priorities = sorted(p for node, p in calls if node == "clutter")
    assert clutter_priorities == [25, 40], f"clutter callbacks: {calls}"
    # The task ran BEFORE every clutter callback.
    assert max(task_calls) < min(clutter_calls), (
        f"task reset did not precede all clutter callbacks: {calls}"
    )

    print("  PASS: pickplace priority ordering semantics")
    env.close()


def test_pickplace_masked_reset_isolation():
    """ObjectRelocateTask batched masked reset: unmasked goals/poses preserved."""
    N_ENVS = 6
    world = GenesisWorld(batch_size=N_ENVS)
    env, task, clutter, target = _make_relocate_env(world)
    env.reset(seed=0, reload=True)

    goals_before = task._goal_pos.clone()
    pos_before = task.target_object_entity.get_pos().clone()

    # Plant sentinel reward/termination values to verify preservation.
    sentinel_reward = torch.arange(N_ENVS, dtype=world.dtype, device=world.device) + 1.0
    task._reward = sentinel_reward.clone()
    task._terminated = torch.ones(N_ENVS, dtype=torch.bool, device=world.device)

    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[1, 4]] = True
    env.reset(mask=reset_mask)

    preserved = [0, 2, 3, 5]
    # Unmasked goals unchanged.
    assert torch.allclose(task._goal_pos[preserved], goals_before[preserved]), (
        "Masked reset overwrote unmasked goals"
    )
    # Unmasked object poses unchanged.
    pos_after = task.target_object_entity.get_pos()
    assert torch.allclose(pos_after[preserved], pos_before[preserved], atol=1e-5), (
        "Masked reset moved the target object in unmasked envs"
    )
    # Masked goals re-sampled (continuous space → new values w.h.p.).
    assert not torch.allclose(task._goal_pos[[1, 4]], goals_before[[1, 4]]), (
        "Masked reset did not re-sample masked goals"
    )
    # Masked object poses moved to the table surface.
    assert torch.all(pos_after[[1, 4], 2] > _TABLE_POS[2]), (
        "Masked envs' target object not placed on the table"
    )

    # Reward/termination: zeroed for masked envs, preserved elsewhere.
    assert torch.all(task._reward[[1, 4]] == 0.0)
    assert torch.all(task._reward[preserved] == sentinel_reward[preserved])
    assert torch.all(task._terminated[[1, 4]] == False)
    assert torch.all(task._terminated[preserved] == True)

    print("  PASS: pickplace masked reset isolation")
    env.close()


def test_pickplace_mask_forwarded_to_placement():
    """ObjectRelocateTask: the boolean reset mask is forwarded as envs_idx.

    Genesis accepts boolean masks directly for ``envs_idx`` — the task must
    use the reset mask consistently (task-state indexing, selected placement
    rows, Genesis env selector) instead of converting to integer indices.
    Integer-index masks are rejected by the strict mask contract.
    """
    N_ENVS = 4
    world = GenesisWorld(batch_size=N_ENVS)
    env, task, clutter, target = _make_relocate_env(world)
    env.reset(seed=0, reload=True)

    captured = {}
    orig_set_pos_3d = target.set_pos_3d
    def set_pos_3d_spy(entity, positions, random_quaternion=True, rng=None, envs_idx=None):
        captured["positions"] = positions
        captured["envs_idx"] = envs_idx
        return orig_set_pos_3d(
            entity, positions,
            random_quaternion=random_quaternion, rng=rng, envs_idx=envs_idx,
        )
    target.set_pos_3d = set_pos_3d_spy

    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[0, 2]] = True
    env.reset(mask=reset_mask)

    envs_idx = captured.get("envs_idx")
    assert envs_idx is not None, "set_pos_3d was not called with envs_idx"
    # The boolean mask is forwarded unchanged (not converted to indices).
    assert isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool, (
        f"envs_idx must be the boolean reset mask, got "
        f"{type(envs_idx)} dtype={getattr(envs_idx, 'dtype', None)}"
    )
    assert torch.equal(envs_idx, reset_mask), envs_idx
    # Only the selected placement rows are written.
    assert captured["positions"].shape == (2, 3), captured["positions"].shape
    target.set_pos_3d = orig_set_pos_3d

    # Integer-index masks are rejected by the strict mask contract.
    try:
        env.reset(mask=torch.tensor([1, 3], dtype=torch.long))
        raise RuntimeError("expected AssertionError for integer reset mask")
    except AssertionError:
        pass

    # Unmasked position / orientation isolation still holds.
    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[1, 3]] = True
    pos_before = task.target_object_entity.get_pos().clone()
    quat_before = task.target_object_entity.get_quat().clone()
    env.reset(mask=reset_mask)
    pos_after = task.target_object_entity.get_pos()
    quat_after = task.target_object_entity.get_quat()
    for env_j in (0, 2):
        assert torch.allclose(pos_after[env_j], pos_before[env_j], atol=1e-6), (
            f"Env {env_j}: position changed after masked reset"
        )
        assert torch.allclose(quat_after[env_j], quat_before[env_j], atol=1e-6), (
            f"Env {env_j}: orientation changed after masked reset"
        )

    print("  PASS: pickplace boolean mask forwarded to placement")
    env.close()


def test_pickplace_avoid_zones_initial_and_goal():
    """ObjectRelocateTask: both initial and goal footprints reserved for clutter."""
    N_ENVS = 4
    world = GenesisWorld(batch_size=N_ENVS)
    env, task, clutter, target = _make_relocate_env(world)
    env.reset(seed=0, reload=True)

    # Run the task reset directly so the queued avoid spaces are not consumed
    # by the clutter manager's reset.
    task.reset(priority=min(task.reset_priorities))
    pending = clutter._pending_avoid_spaces
    assert pending is not None, "Task did not queue avoid spaces for clutter"
    centres, radii = pending

    # Two zones per env: initial object footprint + goal footprint.
    assert centres.shape == (N_ENVS, 2, 2), centres.shape
    assert radii.shape == (N_ENVS, 2), radii.shape

    # Zone 0 = the placed object's footprint centre (entity xy for a centred box).
    obj_pos = task.target_object_entity.get_pos()
    assert torch.allclose(centres[:, 0], obj_pos[:, :2], atol=1e-5), (
        f"Initial footprint zone {centres[:, 0]} != object xy {obj_pos[:, :2]}"
    )
    # Zone 1 = the goal footprint.
    assert torch.allclose(centres[:, 1], task._goal_pos[:, :2], atol=1e-5), (
        f"Goal footprint zone {centres[:, 1]} != goal xy {task._goal_pos[:, :2]}"
    )
    # Both zones use the object's footprint radius.
    expected_r = target.get_footprint_radius()
    assert torch.allclose(radii, torch.full_like(radii, expected_r))

    print("  PASS: pickplace initial+goal avoid zones")
    env.close()


def test_pickplace_masked_reset_preserves_unmasked_state():
    """Masked reset must not perturb unmasked envs' pose OR velocities.

    Regression for ``ManipulatableObject.set_pos_3d`` partial updates: the
    old read-modify-write path issued full-batch ``entity.set_pos`` /
    ``set_quat`` calls whose Genesis default ``zero_velocity=True`` wiped
    the velocities of ALL envs, including unmasked ones.  Selective writes
    (positions + boolean ``envs_idx`` mask) leave unmasked envs fully
    untouched.
    """
    N_ENVS = 4
    world = GenesisWorld(batch_size=N_ENVS)
    env, task, clutter, target = _make_relocate_env(world)
    env.reset(seed=0, reload=True)

    entity = task.target_object_entity

    # Give the object a distinctive nonzero velocity in every env.
    n_dofs = entity.n_dofs
    velocity = (
        torch.arange(N_ENVS * n_dofs, dtype=world.dtype, device=world.device)
        .reshape(N_ENVS, n_dofs) * 0.01 + 0.05
    )
    entity.set_dofs_velocity(velocity)

    pos_before = entity.get_pos().clone()
    quat_before = entity.get_quat().clone()
    vel_before = entity.get_dofs_velocity().clone()
    assert torch.all(vel_before != 0.0), "test setup: velocities must be nonzero"

    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[1, 3]] = True
    env.reset(mask=reset_mask)

    preserved = [0, 2]
    pos_after = entity.get_pos()
    quat_after = entity.get_quat()
    vel_after = entity.get_dofs_velocity()

    for env_j in preserved:
        assert torch.allclose(pos_after[env_j], pos_before[env_j], atol=1e-6), (
            f"Env {env_j}: position changed after masked reset"
        )
        assert torch.allclose(quat_after[env_j], quat_before[env_j], atol=1e-6), (
            f"Env {env_j}: orientation changed after masked reset"
        )
        assert torch.allclose(vel_after[env_j], vel_before[env_j], atol=1e-7), (
            f"Env {env_j}: velocity {vel_after[env_j]} != {vel_before[env_j]} "
            f"after masked reset — selective write leaked into unmasked env"
        )

    # Masked envs were re-placed on the table (and teleports zero their velocity).
    assert torch.all(pos_after[[1, 3], 2] > _TABLE_POS[2])

    print("  PASS: pickplace masked reset preserves unmasked pose + velocities")
    env.close()


class _OffCenterBoxObject(BoxObject):
    """Box whose local bounding-box centre is offset from the entity origin.

    Mimics off-centre GSO/YCB meshes: the entity origin sits at one end of
    the box instead of its geometric centre.  The XY offset means the goal
    footprint centre must NOT be the raw goal (origin) position.
    """

    _XY_OFFSET = (0.06, -0.03)

    def get_local_bounding_box(self):
        half = torch.tensor(self.size) / 2.0
        offset = torch.tensor([self._XY_OFFSET[0], self._XY_OFFSET[1], 0.0])
        return make_bbox(-half + offset, half + offset)


def test_pickplace_avoid_zones_offcenter_object():
    """Goal avoid zone uses the local bbox-centre offset (off-centre object).

    Would fail with origin-only goal centres: for this object the goal
    footprint centre is ``goal_xy + (0.06, -0.03)``, not ``goal_xy``.
    """
    N_ENVS = 4
    world = GenesisWorld(batch_size=N_ENVS)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world, name="clutter",
        object_pool=[
            BoxObject(size=(0.04, 0.04, 0.04), name="cube_r", color=(1, 0, 0, 1)),
            BoxObject(size=(0.03, 0.03, 0.03), name="cube_w", color=(1, 1, 1, 1)),
        ],
        n_objects_range=(1, 1),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    target = _OffCenterBoxObject(
        size=(0.05, 0.05, 0.05), name="target", color=(0, 0, 1, 1),
    )
    task = ObjectRelocateTask(
        world, table_builder=table, clutter_manager=clutter, target_object=target,
    )
    env = WorldEnv(world, [plane, table, clutter, task], render_mode=None)
    env.reset(seed=0, reload=True)

    # Run the task reset directly so the queued avoid spaces are not consumed
    # by the clutter manager's reset.
    task.reset(priority=min(task.reset_priorities))
    centres, radii = clutter._pending_avoid_spaces
    assert centres.shape == (N_ENVS, 2, 2), centres.shape
    assert radii.shape == (N_ENVS, 2), radii.shape

    offset_xy = torch.tensor(
        _OffCenterBoxObject._XY_OFFSET, dtype=world.dtype, device=world.device,
    )

    # Zone 0 (initial): bbox centre at the placed pose = origin xy + offset
    # (identity placement orientation).
    obj_pos = task.target_object_entity.get_pos()
    assert torch.allclose(centres[:, 0], obj_pos[:, :2] + offset_xy, atol=1e-5), (
        f"Initial zone {centres[:, 0]} != object bbox centre "
        f"{obj_pos[:, :2] + offset_xy}"
    )
    # Zone 1 (goal): goal ORIGIN xy + the same local centre offset.
    assert torch.allclose(centres[:, 1], task._goal_pos[:, :2] + offset_xy, atol=1e-5), (
        f"Goal zone {centres[:, 1]} != goal bbox centre "
        f"{task._goal_pos[:, :2] + offset_xy} — local bbox-centre offset missing"
    )
    # Sanity: the goal zone must NOT be the bare origin position.
    assert not torch.allclose(centres[:, 1], task._goal_pos[:, :2], atol=1e-3), (
        "Goal zone equals the bare origin — offset not applied"
    )

    print("  PASS: pickplace off-centre goal avoid zone")
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
    print("=== test_reach_observation_space_batched ===")
    test_reach_observation_space_batched()
    print()
    print("=== test_reach_observation_space_unbatched ===")
    test_reach_observation_space_unbatched()
    print()
    print("=== test_reach_masked_reset_isolation ===")
    test_reach_masked_reset_isolation()
    print()
    print("=== test_reach_after_reload_priority_deterministic ===")
    test_reach_after_reload_priority_deterministic()
    print()
    print("=== test_pickplace_priorities_explicit_zero ===")
    test_pickplace_priorities_explicit_zero()
    print()
    print("=== test_pickplace_priorities_deterministic_fallback ===")
    test_pickplace_priorities_deterministic_fallback()
    print()
    print("=== test_pickplace_priority_ordering_semantics ===")
    test_pickplace_priority_ordering_semantics()
    print()
    print("=== test_pickplace_masked_reset_isolation ===")
    test_pickplace_masked_reset_isolation()
    print()
    print("=== test_pickplace_mask_forwarded_to_placement ===")
    test_pickplace_mask_forwarded_to_placement()
    print()
    print("=== test_pickplace_avoid_zones_initial_and_goal ===")
    test_pickplace_avoid_zones_initial_and_goal()
    print()
    print("=== test_pickplace_masked_reset_preserves_unmasked_state ===")
    test_pickplace_masked_reset_preserves_unmasked_state()
    print()
    print("=== test_pickplace_avoid_zones_offcenter_object ===")
    test_pickplace_avoid_zones_offcenter_object()
    print()
    print("All tests passed!")
