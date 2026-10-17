"""Tests for :class:`OpenArticulatedJointTask` (plan §4 / stage S6).

Two layers:

* **Offline (no Genesis scene)** — the same fake-world / fake-built-entity
  pattern :mod:`tests.test_room_builder` uses, extended to *batched* entities so
  masked resets are exercised without a simulator.  Covers construction and
  validation, priority derivation, declared observation-space shapes, reset
  sampling (goal range + initial-state placement), masked-reset isolation,
  reward / termination arithmetic, ``get_info`` and the handle grasp hook.
* **Real scenes** — the packaged RoboCasa ``drawer`` fixture (already shipped
  under ``unienv_genesis_collection/assets/objects/robocasa/``) loaded through
  :class:`RoboCasaFixture` into a minimal one-piece :class:`RoomBuilder` room
  inside a real ``WorldEnv``, on the CPU backend plus a CUDA-marked variant.
  This is the path that proves joint / link addressing really is resolved *by
  name* against the built entity (``slidejoint`` → its child link
  ``inner_box``, handle ``drawer_door_handle``).

Device isolation: ``gs.init`` installs the compute device as torch's *ambient*
default, so every tensor these tests build is pinned to the world's own
``device``/``dtype`` (or to :data:`_CPU` for the offline fakes) instead of
inheriting it — the tests therefore pass standalone *and* after a CUDA-marked
module ran first.
"""

from __future__ import annotations

import math

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

import trimesh

from unienv_genesis import GenesisWorld
from unienv_interface.world import WorldEnv
from unienv_interface.space.space_utils.metadata_utils import validate_metadata

from unienv_genesis_collection.objects.articulated_object import (
    ArticulatedObject,
    ArticulationError,
)
from unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture import (
    RoboCasaFixture,
)
from unienv_genesis_collection.objects.base_object import GraspCandidates
from unienv_genesis_collection.scenes.room_builder import (
    FurniturePlacement,
    RoomBuilder,
    RoomLayoutSpec,
)
from unienv_genesis_collection.tasks import OpenArticulatedJointTask


# ------------------------------------------------------------------
# Offline fakes (no Genesis scene build)
# ------------------------------------------------------------------

# The offline tests model a **CPU** world.  ``gs.init`` installs the compute
# device as torch's *ambient* default (``torch.set_default_device``), so a bare
# ``torch.tensor(...)`` lands wherever the last Genesis initialisation put it —
# which is why every tensor the fakes create is pinned here instead of
# inheriting that global state (see the module docstring's "device isolation"
# note).
_CPU = torch.device("cpu")


def _host(value, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a host tensor from *value*, rejecting foreign-device tensors."""
    if isinstance(value, torch.Tensor) and value.device != _CPU:
        raise AssertionError(
            f"offline fake expected a host tensor, got device={value.device} — "
            "pin ``device=`` explicitly (torch's ambient default device leaks "
            "across test modules once a GPU backend has been initialised)"
        )
    return torch.as_tensor(value, dtype=dtype, device=_CPU)


class _FakeJoint:
    def __init__(self, name: str, dofs_idx_local, link: "_FakeLink"):
        self.name = name
        self.dofs_idx_local = list(dofs_idx_local)
        self.link = link


class _FakeLink:
    def __init__(self, name: str, idx_local: int):
        self.name = name
        self.idx_local = idx_local


class _FakeRigidEntity:
    """Offline stand-in for a built Genesis ``RigidEntity``.

    Exposes exactly the signatures the ``ArticulatedObject`` runtime API needs
    (joint/link name resolution, DoF state and limits, link poses, base
    placement) and records writes so tests can assert on them.  *Batched*
    worlds get per-env DoF state and a **shared** base pose (the default
    layout, ``batch_fixed_verts`` or not); ``set_dofs_position`` re-evaluates
    the fake's rigid kinematics so link poses track the joint like a real
    entity.  All state lives on :data:`_CPU` (matching :attr:`_FakeWorld.device`).
    """

    #: in-entity (base-frame) offset of each link origin at q = 0, and the
    #: per-DoF translation axis of the drawer joint (mimics the fixture).
    _DRAWER_ORIGIN = (0.0, 0.0, 0.625)
    _HANDLE_ORIGIN = (0.05, 0.0, 0.625)
    _SLIDE_AXIS = 0  # x

    def __init__(self, name, joints, links, q_min, q_max, n_envs=None,
                 pos=(1.0, 1.0, 0.0)):
        self.name = name
        self.joints = joints
        self.links = links
        self.batch_size = n_envs
        self.q_min = _host(q_min)
        self.q_max = _host(q_max)
        n_dofs = self.q_min.numel()
        shape = (n_envs, n_dofs) if n_envs is not None else (n_dofs,)
        self.qpos = torch.zeros(shape, dtype=torch.float32, device=_CPU)
        self._pos = _host(pos)
        self._quat = _host((1.0, 0.0, 0.0, 0.0))
        self.set_dofs_position_calls = []
        self._update_links()

    # -- base placement ------------------------------------------------
    def get_pos(self):
        if self.batch_size is not None:
            return self._pos.unsqueeze(0).expand(self.batch_size, 3).clone()
        return self._pos.clone()

    def get_quat(self):
        if self.batch_size is not None:
            return self._quat.unsqueeze(0).expand(self.batch_size, 4).clone()
        return self._quat.clone()

    # -- DoF state ------------------------------------------------------
    @staticmethod
    def _dofs_idx(dofs_idx_local, n):
        return list(range(n)) if dofs_idx_local is None else list(dofs_idx_local)

    def _select(self, value, envs_idx):
        if envs_idx is None:
            return value
        return value[envs_idx]

    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None):
        assert envs_idx is None, "non-batched DOF info does not accept envs_idx"
        idx = self._dofs_idx(dofs_idx_local, self.q_min.numel())
        return self.q_min[idx].clone(), self.q_max[idx].clone()

    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None):
        idx = self._dofs_idx(dofs_idx_local, self.qpos.shape[-1])
        return self._select(self.qpos[..., idx], envs_idx).clone()

    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None,
                          zero_velocity=True):
        idx = self._dofs_idx(dofs_idx_local, self.qpos.shape[-1])
        written = _host(position)
        self.set_dofs_position_calls.append(
            (written.clone(), list(idx), None if envs_idx is None else envs_idx.clone())
        )
        if envs_idx is None:
            self.qpos[..., idx] = written.reshape(self.qpos[..., idx].shape)
        else:
            # Boolean-mask row selection must go through index_put-style
            # advanced indexing (``qpos[envs_idx]`` would return a detached
            # copy and silently drop the write).
            rows = torch.nonzero(envs_idx, as_tuple=True)[0]
            self.qpos[rows.unsqueeze(-1), torch.as_tensor(idx, device=_CPU)] = (
                written.reshape(rows.numel(), len(idx))
            )
        self._update_links()

    # -- link poses (rigid kinematics of the fake fixture) ---------------
    def _update_links(self):
        q = self.qpos[..., 0]
        n = self.batch_size  # None → unbatched world
        base = self._pos
        origins = torch.tensor(
            [(0.0, 0.0, 0.0), self._DRAWER_ORIGIN, self._HANDLE_ORIGIN],
            dtype=torch.float32, device=_CPU,
        ).unsqueeze(0).repeat(1 if n is None else n, 1, 1)
        origins[..., self._SLIDE_AXIS] += q.reshape(-1, 1)
        self.link_pos = base + origins
        self.link_quat = torch.zeros(
            (1 if n is None else n, len(self.links), 4),
            dtype=torch.float32, device=_CPU,
        )
        self.link_quat[..., 0] = 1.0
        if n is None:
            self.link_pos = self.link_pos[0]
            self.link_quat = self.link_quat[0]

    def get_links_pos(self, links_idx_local=None, envs_idx=None, relative=True):
        assert relative, "the descriptor reads the user-frame (relative) pose"
        idx = list(range(len(self.links))) if links_idx_local is None else list(links_idx_local)
        return self._select(self.link_pos[..., idx, :], envs_idx).clone()

    def get_links_quat(self, links_idx_local=None, envs_idx=None, relative=True):
        assert relative, "the user-frame (relative) pose is queried"
        idx = list(range(len(self.links))) if links_idx_local is None else list(links_idx_local)
        return self._select(self.link_quat[..., idx, :], envs_idx).clone()


class _FakeScene:
    """Recording stand-in for ``gs.Scene`` (nothing is ever built)."""

    def __init__(self):
        self.add_entity_calls = []

    def add_entity(self, morph, name=None, **kwargs):
        self.add_entity_calls.append((morph, name, kwargs))
        return type("_ShellEntityStub", (), {"name": name})()


class _FakeWorld:
    """Offline stand-in for a :class:`GenesisWorld` (never built).

    ``device``/``dtype`` are pinned to host ``cpu``/``float32`` so the task's
    declared spaces and sampled tensors stay host-side no matter which Genesis
    backend happens to be initialised in this pytest process.
    """

    world_timestep = 0.02

    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.device = _CPU
        self.dtype = torch.float32
        self.scene = _FakeScene()


class _Counter(ArticulatedObject):
    """Synthetic articulated counter with a sliding drawer and a handle link.

    ``joint_children`` maps the joint name to its child link (the fake
    entity's joint/DOF assignment); ``q_limits`` gives the fake entity's
    position limits.  All link meshes are boxes at the **reference**
    configuration (q = 0), matching what the offline FK would produce.
    """

    def __init__(self, name: str = "counter"):
        super().__init__(name=name, fixed=True)
        self.load_calls = []

    def _load_link_meshes_at_reference(self):
        meshes = {
            "body": trimesh.creation.box(extents=(1.2, 0.6, 0.9)).apply_translation(
                (0.0, 0.0, 0.45)
            ),
            "drawer": trimesh.creation.box(extents=(1.0, 0.56, 0.25)).apply_translation(
                (0.0, 0.0, 0.625)
            ),
            # A small bar handle rigidly attached in front of the drawer: its
            # bbox centre (0.05, 0, 0.70) is NOT the link origin, so the task's
            # cached local handle offset is exercised.
            "handle": trimesh.creation.box(extents=(0.10, 0.06, 0.10)).apply_translation(
                (0.05, 0.0, 0.70)
            ),
        }
        return meshes

    def _load_link_local_meshes_at_reference(self):
        # The fake entity offsets each link origin from the base (see
        # _FakeRigidEntity._DRAWER_ORIGIN / _HANDLE_ORIGIN), so the
        # link-local geometry is the base-frame mesh translated back by
        # that link's origin.
        origins = {
            "body": (0.0, 0.0, 0.0),
            "drawer": _FakeRigidEntity._DRAWER_ORIGIN,
            "handle": _FakeRigidEntity._HANDLE_ORIGIN,
        }
        return {
            name: mesh.copy().apply_translation([-c for c in origins[name]])
            for name, mesh in self._load_link_meshes_at_reference().items()
        }

    def load_into_scene(self, world, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                        name_override=None, batch_fixed_verts=False):
        self.load_calls.append({
            "world": world,
            "pos": tuple(pos),
            "quat": tuple(quat),
            "name_override": name_override,
            "batch_fixed_verts": bool(batch_fixed_verts),
        })
        links = [
            _FakeLink("body", 0),
            _FakeLink("drawer", 1),
            _FakeLink("handle", 2),
        ]
        joints = [_FakeJoint("drawer_slide", [0], links[1])]
        return _FakeRigidEntity(
            name=name_override or self.name,
            joints=joints,
            links=links,
            q_min=[0.0],
            q_max=[0.6],
            n_envs=world.batch_size,
            pos=pos,
        )


def _loaded_room(batch_size=None, name="room"):
    """A room with one explicit-layout counter, reloaded at priority 110."""
    descriptor = _Counter()
    world = _FakeWorld(batch_size=batch_size)
    room = RoomBuilder(
        world,
        name=name,
        layout=RoomLayoutSpec(
            shell_id=None,
            room_size=(4.0, 3.0, 2.6),
            furniture=(FurniturePlacement(
                object_id="counter", source="procedural",
                pos=(1.0, 1.0, 0.0), yaw=0.0,
            ),),
        ),
        furniture_pool=[descriptor],
    )
    room.reload(priority=110)
    return room, descriptor, world


def _bound_task(batch_size=None, **task_kwargs):
    """Room + task bound through the real lifecycle (reload → after_reload)."""
    room, descriptor, world = _loaded_room(batch_size=batch_size)
    task = OpenArticulatedJointTask(world, room, "counter", "drawer_slide", **task_kwargs)
    room.after_reload(priority=110)  # applies the (empty) initial_open
    task.reload(priority=111)
    task.after_reload(priority=111)  # binds + runs the first full reset
    return task, room, descriptor, world


# ------------------------------------------------------------------
# Construction / validation
# ------------------------------------------------------------------


def test_construction_validates_parameters():
    world = _FakeWorld()
    room = RoomBuilder(world)
    with pytest.raises(ValueError, match="target_fraction_range"):
        OpenArticulatedJointTask(world, room, "counter", "drawer_slide",
                                 target_fraction_range=(0.2,))
    with pytest.raises(ValueError, match="target_fraction_range"):
        OpenArticulatedJointTask(world, room, "counter", "drawer_slide",
                                 target_fraction_range=(0.8, 0.2))
    with pytest.raises(ValueError, match="target_fraction_range"):
        OpenArticulatedJointTask(world, room, "counter", "drawer_slide",
                                 target_fraction_range=(-0.1, 0.5))
    with pytest.raises(ValueError, match="target_fraction_range"):
        OpenArticulatedJointTask(world, room, "counter", "drawer_slide",
                                 target_fraction_range=(0.2, 1.5))
    with pytest.raises(ValueError, match="success_threshold"):
        OpenArticulatedJointTask(world, room, "counter", "drawer_slide",
                                 success_threshold=0.0)
    with pytest.raises(ValueError, match="priority"):
        OpenArticulatedJointTask(world, room, "counter", "drawer_slide",
                                 bogus_priority=3)

    # A room whose layout is already resolved fails fast on an unknown key.
    resolved_room, _, _ = _loaded_room()
    with pytest.raises(KeyError, match="Unknown furniture key"):
        OpenArticulatedJointTask(_FakeWorld(), resolved_room, "ghost", "drawer_slide")

    # The in-range/default parameters are accepted and stored.
    room, _, world = _loaded_room()
    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide",
        target_fraction_range=(0.3, 0.9), success_threshold=0.02,
    )
    assert task.target_fraction_range == (0.3, 0.9)
    assert task.success_threshold == 0.02
    assert task.has_reward and task.has_termination_signal
    assert not task.has_truncation_signal


def test_priorities_default_above_room_and_explicit_zero():
    # RoomBuilder declares {110} per phase → the task defaults sit at 111.
    room, _, world = _loaded_room()
    task = OpenArticulatedJointTask(world, room, "counter", "drawer_slide")
    assert task.reload_priorities == {111}
    assert task.after_reload_priorities == {111}
    assert task.reset_priorities == {111}
    assert task.post_environment_step_priorities == {0}

    # Explicit values (including 0) always win.
    room, _, world = _loaded_room()
    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide",
        reload_priority=0, reset_priority=0, after_reload_priority=0,
        post_environment_step_priority=7,
    )
    assert task.reload_priorities == {0}
    assert task.reset_priorities == {0}
    assert task.after_reload_priorities == {0}
    assert task.post_environment_step_priorities == {7}


def test_observation_space_batched_and_unbatched():
    for batch_size, expected in ((4, (4,)), (None, ())):
        room, _, world = _loaded_room(batch_size=batch_size)
        task = OpenArticulatedJointTask(world, room, "counter", "drawer_slide")
        spaces = task.observation_space.spaces
        assert set(spaces) == {"joint_position", "target_fraction", "handle_position"}
        assert spaces["joint_position"].shape == expected
        assert spaces["target_fraction"].shape == expected
        assert spaces["handle_position"].shape == (*expected, 3)
        for key, space in spaces.items():
            assert space.device == world.device, key
            assert space.dtype == world.dtype, key

        # Space metadata: the "joint_position" is an open fraction, not radians.
        validate_metadata(task.observation_space)
        for key in ("joint_position", "target_fraction"):
            md = spaces[key].metadata.to_dict()
            assert md["kind"] == "task" and md["units"] == "normalized", key
            assert md["normalized"] is True, key
            assert md["joint"] == "drawer_slide" and md["furniture"] == "counter", key
        handle = spaces["handle_position"].metadata.to_dict()
        assert (handle["kind"], handle["units"], handle["frame"]) == ("task", "m", "world")


def test_after_reload_binds_by_name():
    room, descriptor, world = _loaded_room()
    task = OpenArticulatedJointTask(world, room, "counter", "drawer_slide")

    # reload() at its own priority only invalidates; nothing is bound yet.
    task.reload(priority=111)
    with pytest.raises(RuntimeError, match="not bound"):
        task.reset(priority=min(task.reset_priorities))
    assert task.descriptor is None and task.entity is None

    task.after_reload(priority=111)
    assert task.descriptor is descriptor
    assert task.entity is room.furniture_entities["counter"]
    assert task._dofs_idx == [0]
    # The default handle link is the joint's CHILD link, resolved off the
    # built entity by name (no registry).
    assert task.get_handle_link_name() == "drawer"

    # The after_reload hook already ran the first full reset (goal sampled).
    target = task.get_target_fraction()
    assert target is not None and target.shape == ()
    # reload() at a foreign priority keeps the binding.
    entity_before = task.entity
    task.reload(priority=999)
    assert task.entity is entity_before


def test_binding_errors_fail_loud():
    # Unknown joint: ArticulationError listing the entity's joints.
    room, _, world = _loaded_room()
    task = OpenArticulatedJointTask(world, room, "counter", "ghost_joint")
    with pytest.raises(ArticulationError) as excinfo:
        task.after_reload(priority=111)
    message = str(excinfo.value)
    assert "ghost_joint" in message
    assert "drawer_slide" in message  # available joints are listed

    # Unknown explicit handle link: ArticulationError listing the links.
    room, _, world = _loaded_room()
    task = OpenArticulatedJointTask(world, room, "counter", "drawer_slide",
                                    grasp_link_name="no_such_handle")
    with pytest.raises(ArticulationError) as excinfo:
        task.after_reload(priority=111)
    assert "no_such_handle" in str(excinfo.value)

    # Unknown furniture key in after_reload (room resolved later).
    world = _FakeWorld()
    room = RoomBuilder(
        world,
        layout=RoomLayoutSpec(shell_id=None, room_size=(4.0, 3.0, 2.6), furniture=()),
        furniture_pool=[_Counter()],
    )
    task = OpenArticulatedJointTask(world, room, "ghost", "drawer_slide")
    room.reload(priority=110)
    with pytest.raises(KeyError, match="Unknown furniture key"):
        task.after_reload(priority=111)

    # Unbound reads fail loudly.
    room, _, world = _loaded_room()
    task = OpenArticulatedJointTask(world, room, "counter", "drawer_slide")
    with pytest.raises(RuntimeError, match="not bound"):
        task.get_observation()
    with pytest.raises(RuntimeError, match="not bound"):
        task.get_handle_grasp_candidates()


def test_reset_unbatched_samples_and_places_state():
    room, descriptor, world = _loaded_room(batch_size=None)
    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide",
        target_fraction_range=(0.6, 1.0), success_threshold=0.05, seed=7,
    )
    room.after_reload(priority=110)
    task.reload(priority=111)
    task.after_reload(priority=111)

    entity = room.furniture_entities["counter"]
    target = task.get_target_fraction()
    fraction = task.get_observation()["joint_position"]

    # Goal inside the declared range.
    assert 0.6 <= float(target) <= 1.0
    # The initial state is placed strictly below the success band.
    assert float(fraction) < float(target) - task.success_threshold
    # The joint qpos was actually written through set_open_fraction
    # (fraction → q = q_min + fraction * (q_max - q_min)).
    assert float(entity.qpos[0]) == pytest.approx(float(fraction) * 0.6, abs=1e-5)
    position, dofs_idx, envs_idx = entity.set_dofs_position_calls[-1]
    assert list(dofs_idx) == [0]
    assert envs_idx is None
    assert float(position[0]) == pytest.approx(float(fraction) * 0.6, abs=1e-5)

    # Reward / termination start neutral (scalars in an unbatched world).
    assert task.get_reward() == 0.0
    assert task.get_termination() is False

    # Seeded determinism: a fresh task with the same seed samples the same
    # goal / initial state.
    room2, _, world2 = _loaded_room(batch_size=None, name="room2")
    task2 = OpenArticulatedJointTask(
        world2, room2, "counter", "drawer_slide",
        target_fraction_range=(0.6, 1.0), success_threshold=0.05, seed=7,
    )
    room2.after_reload(priority=110)
    task2.reload(priority=111)
    task2.after_reload(priority=111)
    assert float(task2.get_target_fraction()) == pytest.approx(float(target))
    assert torch.allclose(
        task2.get_observation()["joint_position"], fraction, atol=1e-6
    )


def test_sample_goal_and_initial_headroom_clamp():
    """``initial`` is drawn from [0, target - threshold], 0 when clamped."""
    room, _, world = _loaded_room(batch_size=None)
    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide",
        target_fraction_range=(0.0, 0.02), success_threshold=0.05, seed=3,
    )
    target, initial = task.sample_goal_and_initial()
    # Every goal sits inside the (already sub-threshold) range, so the
    # headroom clamp maps the initial draw to exactly 0.
    assert torch.all(target >= 0.0) and torch.all(target <= 0.02)
    assert torch.all(initial == 0.0)

    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide",
        target_fraction_range=(0.6, 1.0), success_threshold=0.05, seed=3,
    )
    target, initial = task.sample_goal_and_initial()
    assert torch.all(initial < target - task.success_threshold)
    assert torch.all(initial >= 0.0)


def test_reset_batched_masked_isolation():
    """Masked reset: unmasked envs keep goal, joint state, reward, termination."""
    task, room, descriptor, world = _bound_task(batch_size=4)
    entity = room.furniture_entities["counter"]
    targets_before = task.get_target_fraction().clone()
    fractions_before = entity.qpos.clone()

    mask = torch.zeros(4, dtype=torch.bool, device=_CPU)
    mask[[0, 2]] = True
    sentinel_reward = torch.arange(4, dtype=world.dtype, device=_CPU) + 5.0
    task._reward = sentinel_reward.clone()
    task._terminated = torch.ones(4, dtype=torch.bool, device=_CPU)

    task.reset(priority=min(task.reset_priorities), mask=mask)

    preserved = [1, 3]
    # Unmasked goals / joint state unchanged.
    assert torch.allclose(task.get_target_fraction()[preserved], targets_before[preserved])
    assert torch.allclose(entity.qpos[preserved], fractions_before[preserved])
    # Masked envs re-sampled below their own new goal.
    masked_target = task.get_target_fraction()[[0, 2]]
    masked_fraction = entity.qpos[[0, 2], 0]
    assert torch.all(masked_fraction < masked_target - task.success_threshold)
    # The boolean reset mask is forwarded unchanged as the Genesis envs_idx,
    # and one fraction per selected env is written (broadcast to the DOFs).
    position, dofs_idx, envs_idx = entity.set_dofs_position_calls[-1]
    assert isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool
    assert torch.equal(envs_idx, mask)
    assert position.shape == (2,)
    # Reward / termination: zeroed for masked envs, preserved elsewhere.
    assert torch.all(task._reward[[0, 2]] == 0.0)
    assert torch.allclose(task._reward[preserved], sentinel_reward[preserved])
    assert torch.all(~task._terminated[[0, 2]])

    # RNG consumption is identical for masked and unmasked resets: reseeding
    # to the same seed makes a masked reset reproduce exactly the goals a full
    # reset with that seed would sample for the selected envs.
    task.reset(priority=min(task.reset_priorities), seed=1, mask=None)
    first_goal = task.get_target_fraction().clone()
    mask2 = torch.zeros(4, dtype=torch.bool, device=_CPU)
    mask2[[1]] = True
    task.reset(priority=min(task.reset_priorities), seed=11, mask=mask2)
    # The unmasked envs kept the FIRST episode's goal...
    assert torch.allclose(task.get_target_fraction()[[0, 2, 3]], first_goal[[0, 2, 3]])
    masked_goal = task.get_target_fraction()[mask2].clone()
    # ...while a full reset with the same seed yields the same value for the
    # masked env (the masked reset consumed the same RNG stream).
    task.reset(priority=min(task.reset_priorities), seed=11, mask=None)
    full_goal = task.get_target_fraction().clone()
    assert torch.allclose(masked_goal, full_goal[mask2])


def test_reset_mask_validation():
    task, room, descriptor, world = _bound_task(batch_size=4)
    device_mask = torch.zeros(4, dtype=torch.bool, device=_CPU)
    device_mask[[1]] = True

    # Non-boolean masks are rejected (index tensors are not valid selectors).
    with pytest.raises(AssertionError, match="torch.bool"):
        task.reset(priority=min(task.reset_priorities),
                   mask=torch.tensor([0, 1, 0, 0], dtype=torch.long, device=_CPU))
    # Wrong shape.
    with pytest.raises(AssertionError, match=r"shape \(4,\)"):
        task.reset(priority=min(task.reset_priorities),
                   mask=torch.zeros(3, dtype=torch.bool, device=_CPU))
    # Foreign device (torch's ambient default is never trusted).
    with pytest.raises(AssertionError, match="world device"):
        task.reset(priority=min(task.reset_priorities),
                   mask=torch.zeros(4, dtype=torch.bool, device="meta"))

    # A valid mask still resets (the strict contract accepted it above).
    task.reset(priority=min(task.reset_priorities), mask=device_mask)


def test_reward_and_termination_batched():
    task, room, descriptor, world = _bound_task(batch_size=4)
    entity = room.furniture_entities["counter"]
    target = task.get_target_fraction()

    task.post_environment_step(dt=world.world_timestep, priority=0)
    reward = task.get_reward()
    terminated = task.get_termination()
    assert reward.shape == (4,) and terminated.shape == (4,)
    assert reward.dtype == world.dtype and terminated.dtype == torch.bool
    # Dense reward -|current fraction - target|; nothing terminated yet (the
    # reset guarantees the initial state sits outside the success band).
    fraction = task.get_observation()["joint_position"]
    assert torch.allclose(reward, -(target - fraction), atol=1e-5)
    assert not torch.any(terminated)

    # Drive every env exactly onto its own goal: reward ≈ 0, all terminated.
    descriptor.set_open_fraction(entity, "drawer_slide", target)
    task.post_environment_step(dt=world.world_timestep, priority=0)
    assert float(task.get_reward().abs().max()) < 1e-5
    assert torch.all(task.get_termination())

    # Overshooting to full-open reopens the |current - target| error.
    descriptor.set_open_fraction(
        entity, "drawer_slide",
        torch.ones(4, dtype=world.dtype, device=world.device),
    )
    task.post_environment_step(dt=world.world_timestep, priority=0)
    error = torch.abs(task.get_observation()["joint_position"] - target)
    assert torch.allclose(task.get_reward(), -error, atol=1e-5)


def test_reward_and_termination_unbatched():
    room, descriptor, world = _loaded_room(batch_size=None)
    task = OpenArticulatedJointTask(world, room, "counter", "drawer_slide", seed=5)
    room.after_reload(priority=110)
    task.reload(priority=111)
    task.after_reload(priority=111)
    entity = room.furniture_entities["counter"]

    task.post_environment_step(dt=world.world_timestep, priority=0)
    # Unbatched worlds produce plain Python scalars.
    reward = task.get_reward()
    assert isinstance(reward, float)
    fraction = task.get_observation()["joint_position"]
    assert reward == pytest.approx(
        -(float(task.get_target_fraction()) - float(fraction))
    )
    assert task.get_termination() is False

    descriptor.set_open_fraction(
        entity, "drawer_slide", task.get_target_fraction()
    )
    task.post_environment_step(dt=world.world_timestep, priority=0)
    assert task.get_reward() == pytest.approx(0.0, abs=1e-6)
    assert task.get_termination() is True


def test_observation_contract_and_handle_tracking():
    """Obs keys/shapes/device + the handle rides rigidly with the joint."""
    room, descriptor, world = _loaded_room(batch_size=4)
    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide", grasp_link_name="handle",
    )
    room.after_reload(priority=110)
    task.reload(priority=111)
    task.after_reload(priority=111)
    entity = room.furniture_entities["counter"]

    obs = task.get_observation()
    assert set(obs) == {"joint_position", "target_fraction", "handle_position"}
    assert obs["joint_position"].shape == (4,)
    assert obs["target_fraction"].shape == (4,)
    assert obs["handle_position"].shape == (4, 3)
    assert obs["handle_position"].device == world.device
    assert obs["handle_position"].dtype == world.dtype

    # The joint observation is the entity's real open fraction.
    direct = descriptor.get_open_fraction(entity, "drawer_slide")
    assert torch.allclose(obs["joint_position"], direct.reshape(-1), atol=1e-6)

    # Drive the drawer to fraction 0.5 (q = 0.3 m along +x).  The handle bbox
    # centre sits at in-entity (q + 0.05, 0, 0.70): its cached link-frame
    # offset must carry the +0.075 m z-offset along the slide.
    descriptor.set_open_fraction(
        entity, "drawer_slide",
        torch.full((4,), 0.5, dtype=world.dtype, device=world.device),
    )
    obs = task.get_observation()
    assert torch.allclose(
        obs["joint_position"],
        torch.full((4,), 0.5, dtype=world.dtype, device=world.device),
        atol=1e-5,
    )
    assert torch.allclose(
        obs["handle_position"],
        torch.tensor([1.35, 1.0, 0.70], dtype=world.dtype, device=world.device).expand(4, 3),
        atol=1e-5,
    )


def test_get_info_contract():
    room, descriptor, world = _loaded_room(batch_size=4)
    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide",
        target_fraction_range=(0.4, 0.8), success_threshold=0.03,
    )
    room.after_reload(priority=110)
    task.reload(priority=111)
    task.after_reload(priority=111)

    info = task.get_info()
    assert set(info) == {
        "furniture_key", "furniture_object_id", "furniture_name",
        "joint_name", "handle_link_name",
        "furniture_annotation", "furniture_annotations",
        "target_fraction_range", "success_threshold",
    }
    assert info["furniture_key"] == "counter"
    assert info["furniture_object_id"] == "counter"  # room placement's object id
    assert info["furniture_name"] == "counter"
    assert info["joint_name"] == "drawer_slide"
    assert info["handle_link_name"] == "drawer"
    assert info["furniture_annotation"] is None  # synthetic descriptor: none
    assert info["furniture_annotations"] == ()
    assert info["target_fraction_range"] == (0.4, 0.8)
    assert info["success_threshold"] == 0.03


def test_get_handle_grasp_candidates():
    room, descriptor, world = _loaded_room(batch_size=4)
    task = OpenArticulatedJointTask(world, room, "counter", "drawer_slide")
    room.after_reload(priority=110)
    task.reload(priority=111)
    task.after_reload(priority=111)
    entity = room.furniture_entities["counter"]

    # Default handle link = the joint's child link ("drawer").  Its bbox
    # (1.0 × 0.56 × 0.25 m) has no axis narrower than the 0.08 m gripper
    # opening, so the enumeration is legitimately empty (batched candidates
    # keep the per-env leading axis; ``len`` counts that axis).
    candidates = task.get_handle_grasp_candidates()
    assert isinstance(candidates, GraspCandidates)
    assert candidates.position.shape == (4, 0, 3)
    assert candidates.batch_shape == candidates.approach_direction.shape[:-1]

    # A wide gripper fits the drawer box (axis < width → 8 candidates).
    wide = task.get_handle_grasp_candidates(gripper_width=0.5)
    assert wide.position.shape == (4, 8, 3)
    assert wide.position.device == world.device

    # An explicit handle link (small bar) yields candidates at the default
    # width, and the mask selects only the requested environments.
    room, descriptor, world = _loaded_room(batch_size=4)
    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide", grasp_link_name="handle",
    )
    room.after_reload(priority=110)
    task.reload(priority=111)
    task.after_reload(priority=111)
    entity = room.furniture_entities["counter"]
    assert task.get_handle_grasp_candidates().position.shape == (4, 8, 3)

    mask = torch.zeros(4, dtype=torch.bool, device=_CPU)
    mask[[1, 3]] = True
    all_envs = task.get_handle_grasp_candidates()
    masked = task.get_handle_grasp_candidates(envs_idx=mask)
    assert masked.position.shape == (2, 8, 3)
    # The masked rows correspond to the selected envs' full-batch candidates.
    assert torch.allclose(masked.position, all_envs.position[mask])

    # The candidates track the joint: opening the slide to q = 0.3 m
    # translates every candidate by exactly (0.3 - q_before) along the slide
    # axis (rigid link attachment).
    before = task.get_handle_grasp_candidates().position.clone()
    q_before = entity.qpos[:, 0].clone()
    descriptor.set_open_fraction(
        entity, "drawer_slide",
        torch.full((4,), 0.5, dtype=world.dtype, device=world.device),
    )
    after = task.get_handle_grasp_candidates().position
    delta = after - before
    expected = torch.zeros(4, 1, 3, dtype=world.dtype, device=world.device)
    expected[..., 0] = (0.3 - q_before).unsqueeze(-1)
    assert torch.allclose(delta, expected.expand(4, 8, 3), atol=1e-5)

    # Unbatched world: (N, 3) world-frame candidates.
    room, descriptor, world = _loaded_room(batch_size=None)
    task = OpenArticulatedJointTask(
        world, room, "counter", "drawer_slide", grasp_link_name="handle",
    )
    room.after_reload(priority=110)
    task.reload(priority=111)
    task.after_reload(priority=111)
    unbatched = task.get_handle_grasp_candidates()
    assert unbatched.position.shape == (8, 3)
    assert unbatched.batch_shape == (8,)


# ------------------------------------------------------------------
# Real scenes: packaged RoboCasa drawer fixture (plan §4, stage S6)
# ------------------------------------------------------------------

# Genesis only allows one initialisation per process; track the active backend
# so tests can switch (and restore) deterministically.
_CURRENT_BACKEND = {"name": "cpu" if not gs._initialized else "unknown"}


def _ensure_genesis_backend(name: str) -> None:
    """(Re)initialise Genesis with the requested backend if needed."""
    if _CURRENT_BACKEND["name"] == name and gs._initialized:
        return
    if gs._initialized:
        gs.destroy()
    gs.init(backend=gs_backend.gpu if name == "gpu" else gs_backend.cpu)
    _CURRENT_BACKEND["name"] = name


def _real_drawer_env(batch_size=None):
    """Packaged RoboCasa ``drawer`` in a minimal room, with the task bound."""
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=batch_size, scene_init_kwargs={"show_viewer": False})
    drawer = RoboCasaFixture("drawer")
    room = RoomBuilder(
        world,
        name="room",
        layout=RoomLayoutSpec(
            shell_id=None,
            room_size=(4.0, 3.0, 2.6),
            furniture=(FurniturePlacement(
                object_id="drawer", source="robocasa",
                pos=(0.5, -0.5, 0.0), yaw=math.pi,
            ),),
        ),
        furniture_pool=[drawer],
    )
    task = OpenArticulatedJointTask(
        world, room, "drawer", "slidejoint",
        grasp_link_name="drawer_door_handle",
        target_fraction_range=(0.6, 1.0), success_threshold=0.05, seed=0,
    )
    env = WorldEnv(world, [room, task], render_mode=None)
    return env, world, room, drawer, task


def test_real_scene_unbatched_cpu():
    """Packaged drawer: addressing by name really resolves on a built entity."""
    env, world, room, drawer, task = _real_drawer_env()
    device, dtype = world.device, world.dtype
    assert device.type == "cpu"

    env.reset(seed=0, reload=True)

    entity = room.furniture_entities["drawer"]
    # The runtime resolvers found the MJCF's names on the built entity.
    assert [joint.name for joint in entity.joints] == ["slidejoint"]
    assert "inner_box" in [link.name for link in entity.links]
    assert task.descriptor is drawer
    assert task.get_handle_link_name() == "drawer_door_handle"

    obs = task.get_observation()
    fraction, target, handle = (
        obs["joint_position"], obs["target_fraction"], obs["handle_position"],
    )
    assert fraction.shape == () and handle.shape == (3,)
    assert fraction.device == device and handle.device == device
    assert handle.dtype == dtype
    # The reset guarantees an opening motion is needed.
    assert float(fraction) < float(target) - task.success_threshold
    # The observation fraction equals the entity's runtime fraction.
    direct = drawer.get_open_fraction(entity, "slidejoint")
    assert float(direct.reshape(())) == pytest.approx(float(fraction), abs=1e-6)

    task.post_environment_step(dt=world.world_timestep, priority=0)
    assert task.get_reward() == pytest.approx(
        -(float(target) - float(fraction)), abs=1e-5
    )
    assert task.get_termination() is False

    # Handle tracking: driving the slide (fraction 1 = q at the joint's lower
    # limit) moves the handle link's bbox centre in the world.
    handle_before = handle.clone()
    drawer.set_open_fraction(
        entity, "slidejoint",
        torch.tensor(1.0, dtype=dtype, device=device),
    )
    obs = task.get_observation()
    assert float(obs["joint_position"]) == pytest.approx(1.0, abs=1e-5)
    moved = float((obs["handle_position"] - handle_before).abs().max())
    assert moved > 0.05, f"handle did not track the drawer (moved {moved:.4f} m)"

    # Reference-policy hook: world-frame grasps on the real handle link.
    candidates = task.get_handle_grasp_candidates()
    assert isinstance(candidates, GraspCandidates)
    assert len(candidates) > 0
    assert candidates.position.shape == (len(candidates), 3)
    assert candidates.position.device == device

    info = task.get_info()
    assert info["furniture_key"] == "drawer"
    assert info["furniture_object_id"] == "drawer"
    assert info["joint_name"] == "slidejoint"
    assert info["handle_link_name"] == "drawer_door_handle"
    assert info["furniture_name"] == "drawer"
    assert len(info["furniture_annotations"]) > 0
    env.close()


def test_real_scene_batched_masked_reset_cpu():
    """Batched CPU world: per-env goals, masked reset isolation on real DoFs."""
    n_envs = 4
    env, world, room, drawer, task = _real_drawer_env(batch_size=n_envs)
    device, dtype = world.device, world.dtype

    env.reset(seed=0, reload=True)
    obs = task.get_observation()
    fraction, target = obs["joint_position"], obs["target_fraction"]
    assert fraction.shape == (n_envs,) and target.shape == (n_envs,)
    assert fraction.device == device and target.device == device
    assert (fraction < target - task.success_threshold).all()
    assert (target >= 0.6).all() and (target <= 1.0).all()

    task.post_environment_step(dt=world.world_timestep, priority=0)
    assert torch.allclose(task.get_reward(), -(target - fraction), atol=1e-5)

    # Masked reset: unmasked envs keep goal + real DoF state.
    mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    mask[[0, 2]] = True
    targets_before = target.clone()
    fractions_before = drawer.get_dofs_position(room.furniture_entities["drawer"],
                                               "slidejoint").clone()
    task._reward = torch.arange(n_envs, dtype=dtype, device=device) + 5.0
    task._terminated = torch.ones(n_envs, dtype=torch.bool, device=device)

    env.reset(mask=mask)

    entity = room.furniture_entities["drawer"]
    fraction_after = drawer.get_dofs_position(entity, "slidejoint")
    preserved = [1, 3]
    assert torch.allclose(task.get_target_fraction()[preserved], targets_before[preserved])
    assert torch.allclose(
        drawer.get_dofs_position(entity, "slidejoint")[preserved],
        fractions_before[preserved], atol=1e-5,
    )
    # Masked envs re-placed strictly below their own new goal.
    masked_fraction = drawer.get_open_fraction(entity, "slidejoint", envs_idx=mask)
    assert (
        task.get_target_fraction()[mask] - task.success_threshold > masked_fraction.reshape(-1)
    ).all()
    assert torch.all(task._reward[mask] == 0.0)
    assert torch.all(task._reward[preserved] == torch.tensor([6.0, 8.0], device=device))
    assert torch.all(~task._terminated[mask])
    env.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the GPU device-placement variant",
)
def test_real_scene_cuda_device_placement():
    """CUDA variant: batched real drawer + device-pinned state and masks."""
    n_envs = 4
    try:
        _ensure_genesis_backend("gpu")
        world = GenesisWorld(batch_size=n_envs, scene_init_kwargs={"show_viewer": False})
        assert world.device.type == "cuda"
        drawer = RoboCasaFixture("drawer")
        room = RoomBuilder(
            world,
            name="room",
            layout=RoomLayoutSpec(
                shell_id=None,
                room_size=(4.0, 3.0, 2.6),
                furniture=(FurniturePlacement(
                    object_id="drawer", source="robocasa",
                    pos=(0.5, -0.5, 0.0), yaw=math.pi,
                ),),
            ),
            furniture_pool=[drawer],
        )
        task = OpenArticulatedJointTask(
            world, room, "drawer", "slidejoint",
            grasp_link_name="drawer_door_handle",
            target_fraction_range=(0.6, 1.0), success_threshold=0.05, seed=0,
        )
        env = WorldEnv(world, [room, task], render_mode=None)
        env.reset(seed=0, reload=True)

        device, dtype = world.device, world.dtype
        obs = task.get_observation()
        assert obs["joint_position"].shape == (n_envs,)
        assert obs["handle_position"].shape == (n_envs, 3)
        for key in ("joint_position", "target_fraction", "handle_position"):
            assert obs[key].device == device, key
        assert (obs["joint_position"] < obs["target_fraction"] - task.success_threshold).all()

        task.post_environment_step(dt=world.world_timestep, priority=0)
        assert task.get_reward().device == device
        assert torch.allclose(
            task.get_reward(), -(obs["target_fraction"] - obs["joint_position"]), atol=1e-5
        )

        # A host-pinned mask violates the task's mask contract (torch's
        # ambient default device is never trusted for reset selectors).
        host_mask = torch.zeros(n_envs, dtype=torch.bool, device="cpu")
        host_mask[[0]] = True
        with pytest.raises(AssertionError, match="world device"):
            task.reset(priority=min(task.reset_priorities), mask=host_mask)

        # Device masks flow through a real masked reset.
        mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
        mask[[0, 2]] = True
        targets_before = task.get_target_fraction().clone()
        env.reset(mask=mask)
        entity = room.furniture_entities["drawer"]
        assert torch.allclose(task.get_target_fraction()[[1, 3]], targets_before[[1, 3]])
        assert (
            task.get_target_fraction()[mask] - task.success_threshold
            > drawer.get_open_fraction(entity, "slidejoint", envs_idx=mask).reshape(-1)
        ).all()
        masked_candidates = task.get_handle_grasp_candidates(envs_idx=mask)
        assert masked_candidates.position.shape[0] == 2
        assert masked_candidates.position.shape[-1] == 3
        assert masked_candidates.position.device == device
        env.close()
    finally:
        # Leave the CPU backend active for the rest of the suite.
        _ensure_genesis_backend("cpu")
