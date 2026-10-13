"""Tests for :mod:`unienv_genesis_collection.scenes.room_builder` (RoomBuilder).

Mostly offline: layout-spec resolution (explicit + baked JSON), procedural
sampling determinism, wall-following placement with footprint-overlap
rejection, the room-frame pose helpers, the task-facing receptacle /
interaction-point API and ``sample_free_floor_positions`` run against a
**fake world** (a recording scene + a fake built ``RigidEntity`` exposing
the exact call signatures the descriptors use) and synthetic
``ArticulatedObject`` subclasses with known box meshes — no real
RoboCasa/ReplicaCAD assets, no Genesis scene builds.

Minimal CPU Genesis scenes close the loop on the scene-facing path:
a procedural shell reload (primitive floor + four walls), a baked shell
asset loaded as ``gs.morphs.Mesh(fixed=True)`` and an explicit-layout
URDF drawer really built into the scene with its ``initial_open`` fraction
applied by ``after_reload``.

Covers:
  - construction: mutually exclusive modes, parameter validation,
    batch_fixed_verts defaulting from world.batch_size
  - spec resolution: explicit spec, duplicate furniture keys, descriptor
    resolution via furniture_pool / procedural KeyError / unknown source
  - baked-spec JSON: round-trip, ``arrangement=`` / ``layout_yaml=`` loading
    from a repointed ASSETS_DIR, FileNotFoundError for absent baked specs
  - procedural sampling: seeded determinism, seed variation, room bounds,
    pairwise footprint-overlap rejection, crowded-room ValueError,
    interior-grid fallback (walls rejected), wall-flush + inward-yaw math,
    off-centre footprint compensation, oversized-furniture rejection
  - xy_to_world / placement_world_pose math (unbatched + batched shapes,
    including nonzero pos/yaw scene offsets)
  - scene offset: shell / baked-shell / furniture morphs carry the world
    pose at load (converted exactly once), get_receptacles interiors and
    sample_free_floor_positions map to the world frame under an offset
  - get_receptacles shape (top slab + shrunk link interiors, world frame)
  - get_interaction_points shape + link aliases + gripper width forwarding
  - sample_free_floor_positions: bounds, footprint avoidance, determinism,
    crowded-room error
  - reset: initial_open application and unknown-joint ArticulationError
  - set_furniture_pose_3d guards (batch_fixed_verts path)
  - yaw_to_quat parity with TableBuilder

Device isolation: ``gs.init()`` installs the compute device as torch's
*ambient* default (``torch.set_default_device``), and it stays there for the
rest of the process even after a GPU smoke test in another module destroys
Genesis again.  The offline fakes therefore model a **host (CPU) world**
(:data:`_CPU`) and every tensor they build or compare against is pinned to
that device, so these tests pass standalone *and* after a CUDA-marked module
(``test_robocasa_fixture.py`` / ``test_replicacad_furniture.py``) ran first.
Where the API under test is world-device (``get_receptacles``,
``sample_free_floor_positions``) the tests read the results through
``.detach().cpu()`` / ``float()`` and assert the returned device matches the
world's, instead of assuming host tensors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_interface.space import BoxSpace

import unienv_genesis_collection.asset_path as asset_path
import unienv_genesis_collection.scenes.room_builder as rb
from unienv_genesis_collection.objects.articulated_object import (
    ArticulatedObject,
    ArticulationError,
    urdf_link_meshes_at_config,
)
from unienv_genesis_collection.objects.base_object import (
    GraspCandidates,
    bbox_size,
)
from unienv_genesis_collection.scenes.room_builder import (
    DEFAULT_N_FURNITURE,
    DEFAULT_WALL_CLEARANCE,
    FurniturePlacement,
    PLACEMENT_PADDING,
    RoomBuilder,
    RoomLayoutSpec,
)
from unienv_interface.world import WorldEnv


# ------------------------------------------------------------------
# Offline fakes (no Genesis scene build)
# ------------------------------------------------------------------

# The offline tests model a **CPU** world.  ``gs.init`` installs the compute
# device as torch's *ambient* default (``torch.set_default_device``), so a
# bare ``torch.tensor(...)``/``torch.as_tensor(...)`` lands wherever the last
# Genesis initialisation put it — which is why every tensor the fakes create
# is pinned here instead of inheriting that global state (see the module
# docstring's "device isolation" note).
_CPU = torch.device("cpu")


def _corners(region: BoxSpace) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host copies of a region's ``low``/``high`` corners.

    :meth:`RoomBuilder.get_receptacles` returns *world-device* tensors (the
    codebase convention for world-frame queries), while these assertions
    compare against Python floats through ``pytest.approx`` — which goes via
    NumPy and therefore needs host memory.  Reading the regions through this
    helper keeps the tests valid for a CPU world and a CUDA world alike.
    """
    return region.low.detach().cpu(), region.high.detach().cpu()


def _host(value, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a host tensor from *value*, rejecting foreign-device tensors.

    A real CPU Genesis entity raises when it is handed CUDA state; the fakes
    do the same so a device mix-up fails loudly instead of silently
    producing host/device-inconsistent tensors.
    """
    if isinstance(value, torch.Tensor) and value.device != _CPU:
        raise AssertionError(
            f"offline fake expected a host tensor, got device={value.device} — "
            "pin ``device=`` explicitly (torch's ambient default device leaks "
            "across test modules once a GPU backend has been initialised)"
        )
    return torch.as_tensor(value, dtype=dtype, device=_CPU)


class _FakeJoint:
    def __init__(self, name: str, dofs_idx_local, link):
        self.name = name
        self.dofs_idx_local = list(dofs_idx_local)
        self.link = link


class _FakeLink:
    def __init__(self, name: str, idx_local: int):
        self.name = name
        self.idx_local = idx_local


class _FakeRigidEntity:
    """Offline stand-in for a built Genesis ``RigidEntity`` (unbatched).

    Exposes exactly the signatures the ``ArticulatedObject`` runtime API
    needs (joint/link name resolution, DoF state and limits, link poses,
    base placement) and records writes so tests can assert on them.

    All state tensors live on :data:`_CPU` — the device that matches
    :attr:`_FakeWorld.device` — so this fake never inherits torch's *ambient*
    default device (``gs.init`` sets it to the compute device, and a GPU
    smoke test in another module would otherwise move this entity's state to
    CUDA while the fake world still claims CPU).
    """

    def __init__(self, name, joints, links, q_min, q_max, pos=(0.0, 0.0, 0.0),
                 quat=(1.0, 0.0, 0.0, 0.0)):
        self.name = name
        self.joints = joints
        self.links = links
        self.q_min = _host(q_min)
        self.q_max = _host(q_max)
        self.qpos = torch.zeros(self.q_min.numel(), dtype=torch.float32, device=_CPU)
        self._pos = _host(pos)
        self._quat = _host(quat)
        self.link_pos = torch.zeros(len(links), 3, dtype=torch.float32, device=_CPU)
        self.link_quat = torch.zeros(len(links), 4, dtype=torch.float32, device=_CPU)
        self.link_quat[:, 0] = 1.0
        self.set_dofs_position_calls = []

    # -- base placement -----------------------------------------------
    def get_pos(self):
        return self._pos.clone()

    def get_quat(self):
        return self._quat.clone()

    def set_pos(self, position, envs_idx=None):
        assert envs_idx is None
        self._pos = _host(position).clone()

    def set_quat(self, quat, envs_idx=None):
        assert envs_idx is None
        self._quat = _host(quat).clone()

    # -- DoF state ----------------------------------------------------
    @staticmethod
    def _dofs_idx(dofs_idx_local, n):
        return list(range(n)) if dofs_idx_local is None else list(dofs_idx_local)

    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None):
        assert envs_idx is None, "non-batched DOF info does not accept envs_idx"
        idx = self._dofs_idx(dofs_idx_local, self.q_min.numel())
        return self.q_min[idx].clone(), self.q_max[idx].clone()

    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None):
        assert envs_idx is None
        idx = self._dofs_idx(dofs_idx_local, self.qpos.numel())
        return self.qpos[idx].clone()

    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None,
                          zero_velocity=True):
        assert envs_idx is None
        idx = self._dofs_idx(dofs_idx_local, self.qpos.numel())
        written = _host(position).clone()
        self.set_dofs_position_calls.append((written, list(idx)))
        self.qpos[_host(idx, dtype=torch.long)] = written.reshape(-1)

    # -- link poses ---------------------------------------------------
    def get_links_pos(self, links_idx_local=None, envs_idx=None, relative=True):
        assert relative and envs_idx is None
        idx = links_idx_local if links_idx_local is not None else list(range(len(self.links)))
        return self.link_pos[idx].clone()

    def get_links_quat(self, links_idx_local=None, envs_idx=None, relative=True):
        assert relative and envs_idx is None
        idx = links_idx_local if links_idx_local is not None else list(range(len(self.links)))
        return self.link_quat[idx].clone()


class _FakeScene:
    """Recording stand-in for ``gs.Scene`` (nothing is ever built)."""

    def __init__(self):
        self.add_entity_calls = []

    def add_entity(self, morph, name=None, **kwargs):
        self.add_entity_calls.append((morph, name, kwargs))
        return type("_ShellEntityStub", (), {"name": name})()


class _FakeWorld:
    """Offline stand-in for a :class:`GenesisWorld` (never built).

    ``device``/``dtype`` are pinned to host ``cpu``/``float32`` so the room's
    task-facing API returns host tensors here no matter which Genesis backend
    happens to be initialised in this pytest process; room_builder reads them
    (``world.device``/``world.dtype``) rather than assuming a default.
    """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.device = _CPU
        self.dtype = torch.float32
        self.scene = _FakeScene()


class _BoxFurniture(ArticulatedObject):
    """Synthetic articulated furniture with known box meshes — no assets.

    Args:
        link_boxes: link name → (min corner, max corner) in the base-link
            frame (drives the offline union bbox via trimesh boxes).
        joint_children: joint name → child link name (drives the fake
            entity's joints / DOF assignment).
        q_limits: joint name → (q_min, q_max) for the fake entity.
    """

    def __init__(
        self,
        name: str,
        link_boxes: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
        joint_children: Optional[Dict[str, str]] = None,
        q_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        fixed: bool = True,
    ):
        super().__init__(name=name, fixed=fixed)
        self.link_boxes = dict(link_boxes)
        self.joint_children = dict(joint_children or {})
        self.q_limits = dict(q_limits or {})
        self.load_calls = []

    def _load_link_meshes_at_reference(self):
        import trimesh

        meshes = {}
        for link_name, (lo, hi) in self.link_boxes.items():
            lo_t = _host(lo, dtype=torch.float64)
            hi_t = _host(hi, dtype=torch.float64)
            mesh = trimesh.creation.box(extents=(hi_t - lo_t).tolist())
            mesh.apply_translation(((lo_t + hi_t) / 2.0).tolist())
            meshes[link_name] = mesh
        return meshes

    def _load_link_local_meshes_at_reference(self):
        # The fake entity's link frames all coincide with the base link
        # (zero origins), so the base-frame boxes are also the link-local
        # geometry.
        return self._load_link_meshes_at_reference()

    def load_into_scene(self, world, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                        name_override=None, batch_fixed_verts=False):
        self.load_calls.append({
            "world": world,
            "pos": tuple(pos),
            "quat": tuple(quat),
            "name_override": name_override,
            "batch_fixed_verts": bool(batch_fixed_verts),
        })
        links = [_FakeLink(name, idx) for idx, name in enumerate(self.link_boxes)]
        link_by_name = {link.name: link for link in links}
        joints, q_min, q_max = [], [], []
        for joint_name, child_name in self.joint_children.items():
            lo, hi = self.q_limits.get(joint_name, (0.0, 1.0))
            joints.append(_FakeJoint(joint_name, [len(q_min)], link_by_name[child_name]))
            q_min.append(lo)
            q_max.append(hi)
        return _FakeRigidEntity(
            name=name_override or self.name,
            joints=joints,
            links=links,
            q_min=q_min,
            q_max=q_max,
            pos=pos,
            quat=quat,
        )


def _chair() -> _BoxFurniture:
    """A 0.5 × 0.5 × 0.9 chair with a hinged seat."""
    return _BoxFurniture(
        name="chair",
        link_boxes={
            "body": ((-0.25, -0.25, 0.0), (0.25, 0.25, 0.9)),
            "seat": ((-0.22, -0.22, 0.35), (0.22, 0.22, 0.5)),
        },
        joint_children={"seat_hinge": "seat"},
        q_limits={"seat_hinge": (0.0, 1.2)},
    )


def _counter() -> _BoxFurniture:
    """A 1.2 × 0.6 × 0.9 counter with a sliding drawer (centred footprint)."""
    return _BoxFurniture(
        name="counter",
        link_boxes={
            "body": ((-0.6, -0.3, 0.0), (0.6, 0.3, 0.9)),
            "drawer": ((-0.5, -0.28, 0.5), (0.5, 0.28, 0.75)),
        },
        joint_children={"drawer_slide": "drawer"},
        q_limits={"drawer_slide": (0.0, 0.6)},
    )


def _pool():
    """A pool of descriptors with distinct footprints for procedural sampling."""
    return [_chair(), _counter(), _BoxFurniture(
        name="shelf",
        link_boxes={"body": ((-0.4, -0.18, 0.0), (0.4, 0.18, 1.6))},
    )]


def _procedural_room(seed, room_size=(6.0, 5.0, 2.7), n_furniture=DEFAULT_N_FURNITURE,
                     pool=None, name="room"):
    return RoomBuilder(
        _FakeWorld(),
        name=name,
        procedural=True,
        room_size=room_size,
        furniture_pool=_pool() if pool is None else pool,
        n_furniture=n_furniture,
        seed=seed,
    )


def _loaded_room(placement_pos=(1.0, 2.0, 0.0), yaw=0.0, initial_open=None,
                 pool=None, name="room"):
    """A room with one explicit-layout piece, reloaded at priority 110.

    When *pool* is given its first descriptor is the placed one; otherwise a
    fresh counter is created and registered.
    """
    if pool is not None:
        descriptor = pool[0]
        furniture_pool = list(pool)
    else:
        descriptor = _counter()
        furniture_pool = [descriptor]
    placement = FurniturePlacement(
        object_id=descriptor.name,
        source="procedural",
        pos=placement_pos,
        yaw=yaw,
        initial_open=initial_open or {},
    )
    world = _FakeWorld()
    room = RoomBuilder(
        world,
        name=name,
        layout=RoomLayoutSpec(
            shell_id=None,
            room_size=(6.0, 5.0, 2.7),
            furniture=(placement,),
        ),
        furniture_pool=furniture_pool,
    )
    room.reload(priority=110)
    return room, descriptor, world


# ------------------------------------------------------------------
# Construction / mode validation
# ------------------------------------------------------------------


def test_modes_are_mutually_exclusive():
    world = _FakeWorld()
    spec = RoomLayoutSpec(room_size=(4.0, 4.0, 2.7), furniture=())
    with pytest.raises(ValueError, match="mutually exclusive"):
        RoomBuilder(world, layout=spec, procedural=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        RoomBuilder(world, arrangement="replicacad/apt_0", procedural=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        RoomBuilder(world, layout=spec, layout_yaml="robocasa/u_shape")
    with pytest.raises(ValueError, match="mutually exclusive"):
        RoomBuilder(
            world,
            layout=spec,
            arrangement="replicacad/apt_0",
            layout_yaml="robocasa/u_shape",
            procedural=True,
        )


def test_parameter_validation():
    world = _FakeWorld()
    with pytest.raises(ValueError, match="room_size"):
        RoomBuilder(world, room_size=(0.0, 5.0, 2.7))
    with pytest.raises(ValueError, match="room_size"):
        RoomBuilder(world, room_size=(6.0, 5.0))
    with pytest.raises(ValueError, match="room_size"):
        RoomBuilder(world, room_size=(6.0, 5.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="n_furniture"):
        RoomBuilder(world, procedural=True, furniture_pool=_pool(), n_furniture=(5, 2))
    with pytest.raises(ValueError, match="n_furniture"):
        RoomBuilder(world, procedural=True, furniture_pool=_pool(), n_furniture=(-1, 2))
    with pytest.raises(ValueError, match="wall_clearance"):
        RoomBuilder(world, wall_clearance=-0.01)
    with pytest.raises(ValueError, match="furniture_pool"):
        RoomBuilder(world, procedural=True, furniture_pool=[])


def test_pos_yaw_validation_and_readonly_properties():
    world = _FakeWorld()
    # Defaults keep the historical behaviour (room frame == world frame).
    default = RoomBuilder(world)
    assert default.pos == (0.0, 0.0, 0.0)
    assert default.yaw == 0.0

    room = RoomBuilder(world, pos=(1.0, -2.0, 0.75), yaw=0.5)
    assert room.pos == (1.0, -2.0, 0.75)
    assert room.yaw == 0.5

    # Captured at construction; read-only properties (no mid-run mutation).
    with pytest.raises(AttributeError):
        room.pos = (9.0, 9.0, 9.0)
    with pytest.raises(AttributeError):
        room.yaw = 1.0

    for bad in ((1.0, 2.0), (1.0, 2.0, 3.0, 4.0), ("a", 0.0, 0.0), (None, 0.0, 0.0)):
        with pytest.raises(ValueError, match="pos"):
            RoomBuilder(world, pos=bad)

    # Mode mutual-exclusion still fires with pos/yaw present.
    spec = RoomLayoutSpec(room_size=(4.0, 4.0, 2.7), furniture=())
    with pytest.raises(ValueError, match="mutually exclusive"):
        RoomBuilder(world, layout=spec, procedural=True,
                    pos=(1.0, 0.0, 0.0), yaw=0.25)


def test_batch_fixed_verts_defaults_from_world_batch_size():
    assert RoomBuilder(_FakeWorld()).batch_fixed_verts is False
    assert RoomBuilder(_FakeWorld(batch_size=4)).batch_fixed_verts is True
    forced = RoomBuilder(_FakeWorld(batch_size=4), batch_fixed_verts=False)
    assert forced.batch_fixed_verts is False


def test_yaw_to_quat_matches_table_builder():
    from unienv_genesis_collection.scenes.table_builder import yaw_to_quat as table_yaw_to_quat

    for yaw in (0.0, 0.5, -1.0, math.pi, 2 * math.pi):
        assert rb.yaw_to_quat(yaw) == table_yaw_to_quat(yaw)


# ------------------------------------------------------------------
# Spec resolution
# ------------------------------------------------------------------


def test_shell_only_mode_resolves_empty_spec():
    world = _FakeWorld()
    room = RoomBuilder(world, room_size=(7.0, 3.0, 2.5))
    assert room._mode == "shell_only"
    spec = room.resolve_layout_spec()
    assert spec.shell_id is None
    assert spec.room_size == (7.0, 3.0, 2.5)
    assert spec.furniture == ()
    assert room.room_size == (7.0, 3.0, 2.5)


def test_explicit_layout_resolution_and_duplicate_keys():
    world = _FakeWorld()
    chair = _chair()
    placement = FurniturePlacement(
        object_id="chair", source="procedural", pos=(0.5, 0.5, 0.0), yaw=0.3,
        initial_open={"seat_hinge": 0.5},
    )
    spec = RoomLayoutSpec(
        shell_id=None,
        room_size=(4.0, 4.0, 2.7),
        furniture=(placement, placement),
    )
    room = RoomBuilder(world, name="kitchen", layout=spec, furniture_pool=[chair])
    room.reload(priority=110)

    # Duplicate object ids get deterministic suffixed keys.
    assert set(room.furniture) == {"chair", "chair_1"}
    assert set(room.furniture_entities) == {"chair", "chair_1"}
    assert set(room.placements) == {"chair", "chair_1"}
    assert room.layout is spec

    # Procedural shell: floor + four walls, named after the builder.
    assert len(world.scene.add_entity_calls) == 5
    assert [name for _, name, _ in world.scene.add_entity_calls] == [
        "kitchen_floor",
        "kitchen_wall_north",
        "kitchen_wall_south",
        "kitchen_wall_east",
        "kitchen_wall_west",
    ]

    # Descriptor load: world pose (the default room offset maps room-frame
    # placements to themselves, to morph float precision), yaw quaternion,
    # namespaced name and batch_fixed_verts defaulted False (fake world is
    # unbatched).
    assert chair.load_calls[0]["world"] is world
    assert chair.load_calls[0]["pos"] == pytest.approx((0.5, 0.5, 0.0))
    assert chair.load_calls[0]["quat"] == pytest.approx(rb.yaw_to_quat(0.3), abs=1e-6)
    assert chair.load_calls[0]["name_override"] == "kitchen_chair"
    assert chair.load_calls[0]["batch_fixed_verts"] is False
    assert room.furniture_entities["chair"].name == "kitchen_chair"


def test_descriptor_resolution_via_pool_registry_and_errors():
    world = _FakeWorld()
    chair = _chair()
    # The pool doubles as an id registry and shadows hub-catalog resolution
    # (a "robocasa"-sourced placement resolves from the pool offline).
    placement = FurniturePlacement(
        object_id="chair", source="robocasa", pos=(0.0, 0.0, 0.0), yaw=0.0,
    )
    room = RoomBuilder(world, layout=RoomLayoutSpec(furniture=(placement,)),
                       furniture_pool=[chair])
    room.reload(priority=110)
    assert room.furniture["chair"] is chair

    # Unknown procedural id → KeyError naming the pool.
    ghost = FurniturePlacement(object_id="ghost", source="procedural",
                               pos=(0.0, 0.0, 0.0), yaw=0.0)
    room = RoomBuilder(_FakeWorld(), layout=RoomLayoutSpec(furniture=(ghost,)),
                       furniture_pool=[chair])
    with pytest.raises(KeyError, match="not in furniture_pool"):
        room.reload(priority=110)

    # Unknown source (and id not in the pool) → ValueError.
    alien = FurniturePlacement(object_id="alien_chair", source="minecraft",
                               pos=(0.0, 0.0, 0.0), yaw=0.0)
    room = RoomBuilder(_FakeWorld(), layout=RoomLayoutSpec(furniture=(alien,)),
                       furniture_pool=[chair])
    with pytest.raises(ValueError, match="Unknown furniture source"):
        room.reload(priority=110)


def test_reload_at_other_priority_is_a_noop():
    room, _, world = _loaded_room()
    calls_before = len(world.scene.add_entity_calls)
    room.reload(priority=999)
    assert len(world.scene.add_entity_calls) == calls_before
    room.reset(priority=999)  # also a noop


# ------------------------------------------------------------------
# Baked layout specs (plan §3.4)
# ------------------------------------------------------------------


_BAKED_SPEC = {
    "shell_id": "replicacad/stage_greyscale",
    "room_size": [5.0, 4.0, 2.8],
    "furniture": [
        {
            "object_id": "fridge",
            "source": "replicacad",
            "pos": [1.5, -1.0, 0.0],
            "yaw": math.pi / 2,
            "initial_open": {"top_door": 0.5},
        },
        {
            "object_id": "counter_hinge",
            "source": "robocasa",
            "pos": [-1.5, 1.0, 0.0],
            "yaw": math.pi,
        },
    ],
}


def _write_baked_spec(root: Path, relative: Tuple[str, ...], data) -> Path:
    path = root.joinpath(*relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_baked_arrangement_spec_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    _write_baked_spec(
        tmp_path,
        ("scenes", "rooms", "replicacad", "arrangements", "apt_0.json"),
        _BAKED_SPEC,
    )
    room = RoomBuilder(_FakeWorld(), arrangement="replicacad/apt_0")
    spec = room.resolve_layout_spec()
    assert spec.shell_id == "replicacad/stage_greyscale"
    assert spec.room_size == (5.0, 4.0, 2.8)
    assert [p.object_id for p in spec.furniture] == ["fridge", "counter_hinge"]
    assert [p.source for p in spec.furniture] == ["replicacad", "robocasa"]
    assert spec.furniture[0].pos == (1.5, -1.0, 0.0)
    assert spec.furniture[0].yaw == pytest.approx(math.pi / 2)
    assert spec.furniture[0].initial_open == {"top_door": 0.5}
    assert spec.furniture[1].initial_open == {}


def test_baked_layout_yaml_spec_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    _write_baked_spec(
        tmp_path,
        ("scenes", "rooms", "robocasa", "layouts", "u_shape.json"),
        _BAKED_SPEC,
    )
    # The (unbaked) YAML extension is accepted and stripped.
    room = RoomBuilder(_FakeWorld(), layout_yaml="robocasa/u_shape.yaml")
    spec = room.resolve_layout_spec()
    assert spec.room_size == (5.0, 4.0, 2.8)
    assert spec.furniture[0].object_id == "fridge"


def test_baked_spec_json_roundtrip(tmp_path, monkeypatch):
    placement = FurniturePlacement(
        object_id="counter_hinge_0",
        source="robocasa",
        pos=(1.0, -2.0, 0.3),
        yaw=-0.5,
        initial_open={"top_drawer": 1.0},
    )
    spec = RoomLayoutSpec(shell_id=None, room_size=(6.0, 5.0, 2.7),
                          furniture=(placement,))
    data = rb.room_layout_spec_to_json_dict(spec)
    assert rb._spec_from_mapping(data) == spec

    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    path = _write_baked_spec(tmp_path, ("spec.json",), data)
    assert rb.load_room_layout_spec_json(path) == spec


def test_missing_baked_spec_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)  # empty assets dir

    def _fail_get_asset_path(*args, **kwargs):
        raise FileNotFoundError(f"Asset '{'/'.join(args)}' is not available locally.")

    # Stub the resolver so the test never touches the Hub (baked specs are
    # produced by the parallel S5 asset-preparation stage and not yet
    # uploaded); the loader's FileNotFoundError mapping is still exercised.
    monkeypatch.setattr(rb, "get_asset_path", _fail_get_asset_path)
    room = RoomBuilder(_FakeWorld(), arrangement="replicacad/apt_0")
    with pytest.raises(
        FileNotFoundError, match="Baked room layout spec 'replicacad/apt_0'"
    ):
        room.resolve_layout_spec()

    yaml_room = RoomBuilder(_FakeWorld(), layout_yaml="robocasa/u_shape")
    with pytest.raises(FileNotFoundError, match="robocasa/layouts/u_shape.json"):
        yaml_room.resolve_layout_spec()


def test_baked_spec_id_validation():
    with pytest.raises(ValueError, match="source>/<name>"):
        rb.baked_spec_relative_path("apt_0", kind="arrangement")
    with pytest.raises(ValueError, match="kind"):
        rb.baked_spec_relative_path("a/b", kind="bogus")
    assert rb.baked_spec_relative_path("replicacad/apt_0.json", kind="arrangement") == (
        "scenes", "rooms", "replicacad", "arrangements", "apt_0.json",
    )
    assert rb.baked_spec_relative_path("robocasa/u_shape.yaml", kind="layout") == (
        "scenes", "rooms", "robocasa", "layouts", "u_shape.json",
    )


# ------------------------------------------------------------------
# Procedural sampling (plan §3.3)
# ------------------------------------------------------------------


def test_procedural_sampling_is_deterministic_and_in_range():
    spec_a = _procedural_room(seed=17).resolve_layout_spec()
    spec_b = _procedural_room(seed=17).resolve_layout_spec()
    assert spec_a == spec_b  # same seed → identical layout

    n = len(spec_a.furniture)
    assert DEFAULT_N_FURNITURE[0] <= n <= DEFAULT_N_FURNITURE[1]
    assert [p.source for p in spec_a.furniture] == ["procedural"] * n
    pool_names = {d.name for d in _pool()}
    assert {p.object_id for p in spec_a.furniture} <= pool_names


def test_procedural_sampling_varies_with_seed():
    layouts = set()
    for seed in range(6):
        spec = _procedural_room(seed=seed).resolve_layout_spec()
        layouts.add(tuple(
            (p.object_id, round(p.pos[0], 4), round(p.pos[1], 4), round(p.yaw, 4))
            for p in spec.furniture
        ))
    assert len(layouts) > 1, "different seeds produced identical layouts"


def test_procedural_placements_respect_room_bounds_and_floor():
    room_size = (4.0, 4.0, 2.7)
    room = _procedural_room(seed=5, room_size=room_size)
    spec = room.resolve_layout_spec()
    descriptors = {d.name: d for d in _pool()}
    width, depth, height = room_size
    for placement in spec.furniture:
        descriptor = descriptors[placement.object_id]
        center_xy, _, (z_lo, z_hi) = room._rotated_aabb_of(
            descriptor, placement.pos, placement.yaw
        )
        assert -width / 2.0 <= center_xy[0] <= width / 2.0
        assert -depth / 2.0 <= center_xy[1] <= depth / 2.0
        local = descriptor.get_local_bounding_box()
        assert z_lo == pytest.approx(float(local.low[2]) + placement.pos[2], abs=1e-5)
        assert z_lo >= -1e-6  # bbox bottom rests on (or above) the floor
        assert z_hi <= height + 1e-6


def test_procedural_placements_never_overlap():
    room_size = (4.0, 4.0, 2.7)
    room = _procedural_room(seed=5, room_size=room_size)
    spec = room.resolve_layout_spec()
    descriptors = {d.name: d for d in _pool()}
    footprints = []
    for placement in spec.furniture:
        descriptor = descriptors[placement.object_id]
        center_xy, _, _ = room._rotated_aabb_of(descriptor, placement.pos, placement.yaw)
        footprints.append((center_xy, descriptor.get_footprint_radius()))
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            (ci, ri), (cj, rj) = footprints[i], footprints[j]
            dist = math.hypot(ci[0] - cj[0], ci[1] - cj[1])
            assert dist > ri + rj + PLACEMENT_PADDING - 1e-6, (
                f"footprints {i} and {j} overlap: dist={dist:.3f} <= "
                f"r_i+r_j+pad={ri + rj + PLACEMENT_PADDING:.3f}"
            )


def test_footprint_free_circular_check():
    room = _procedural_room(seed=0)
    occupied = [(0.0, 0.0, 0.5)]
    # Touching-within-padding → rejected; just beyond → accepted.
    assert not room._footprint_free((1.0, 0.0), 0.5, occupied)
    assert room._footprint_free((1.3, 0.0), 0.5, occupied)
    assert room._footprint_free((0.0, 0.0), 0.5, [])


def test_wall_candidate_is_flush_and_faces_inward():
    room = _procedural_room(seed=0, room_size=(4.0, 4.0, 2.7))
    descriptor = _counter()  # 1.2 × 0.6 footprint, centred on the origin
    local = descriptor.get_local_bounding_box()
    size = bbox_size(local)
    center = rb.bbox_center(local)
    depth = 4.0

    # North wall (+y): flush at y = depth/2 - clearance, yaw facing -y.
    pos, yaw = room._wall_candidate(0, size, center, room._rng)
    assert yaw == pytest.approx(-math.pi / 2)
    _, (half_x, half_y), _ = rb._rotated_aabb(size, center, yaw)
    assert pos[1] + half_y == pytest.approx(depth / 2.0 - DEFAULT_WALL_CLEARANCE)
    assert -2.0 <= pos[0] <= 2.0

    # South wall (-y): flush at y = -(depth/2 - clearance), yaw facing +y.
    pos, yaw = room._wall_candidate(1, size, center, room._rng)
    assert yaw == pytest.approx(math.pi / 2)
    _, (half_x, half_y), _ = rb._rotated_aabb(size, center, yaw)
    assert pos[1] - half_y == pytest.approx(-depth / 2.0 + DEFAULT_WALL_CLEARANCE)
    assert -2.0 <= pos[0] <= 2.0

    # East wall (+x): flush at x = width/2 - clearance, yaw facing -x.
    pos, yaw = room._wall_candidate(2, size, center, room._rng)
    assert yaw == pytest.approx(math.pi)
    _, (half_x, half_y), _ = rb._rotated_aabb(size, center, yaw)
    assert pos[0] + half_x == pytest.approx(2.0 - DEFAULT_WALL_CLEARANCE)
    assert -2.0 <= pos[1] <= 2.0

    # West wall (-x): flush at x = -(width/2 - clearance), yaw facing +x.
    pos, yaw = room._wall_candidate(3, size, center, room._rng)
    assert yaw == pytest.approx(0.0)
    _, (half_x, half_y), _ = rb._rotated_aabb(size, center, yaw)
    assert pos[0] - half_x == pytest.approx(-2.0 + DEFAULT_WALL_CLEARANCE)
    assert -2.0 <= pos[1] <= 2.0


def test_wall_candidate_compensates_offcentre_footprint():
    room = _procedural_room(seed=0, room_size=(4.0, 4.0, 2.7))
    # Entity origin 0.3 m behind the footprint centre along +x: the flush
    # distance must be computed for the *rotated* footprint centre.
    descriptor = _BoxFurniture(
        name="offset_counter",
        link_boxes={"body": ((0.1, -0.3, 0.0), (1.3, 0.3, 0.9))},
    )
    local = descriptor.get_local_bounding_box()
    # Local-frame bboxes are host-side descriptor data (the base_object
    # device convention), so they never follow torch's ambient default device.
    assert local.low.device == _CPU and local.high.device == _CPU
    size = bbox_size(local)
    center = rb.bbox_center(local)
    assert float(center[0]) == pytest.approx(0.7)

    pos, yaw = room._wall_candidate(0, size, center, room._rng)
    assert yaw == pytest.approx(-math.pi / 2)
    _, (half_x, half_y), _ = rb._rotated_aabb(size, center, yaw)
    rotated_cy = float(center[0]) * math.sin(yaw) + float(center[1]) * math.cos(yaw)  # -0.3
    world_max_y = pos[1] + rotated_cy + half_y
    assert world_max_y == pytest.approx(2.0 - DEFAULT_WALL_CLEARANCE)
    # Slide range honours the rotated half-extents, not the raw size.
    assert -2.0 + half_x + DEFAULT_WALL_CLEARANCE <= pos[0] <= (
        2.0 - half_x - DEFAULT_WALL_CLEARANCE
    )


def test_wall_candidate_rejects_oversized_furniture():
    room = _procedural_room(seed=0, room_size=(0.5, 0.5, 2.7))
    descriptor = _counter()  # 1.2 × 0.6 footprint > room width/depth
    size = bbox_size(descriptor.get_local_bounding_box())
    center = rb.bbox_center(descriptor.get_local_bounding_box())
    for wall in range(4):
        assert room._wall_candidate(wall, size, center, room._rng) is None


def test_procedural_crowded_room_raises():
    # A 1.2 × 1.2 room cannot hold two 1.4 × 1.4 boxes without overlap.
    big = _BoxFurniture(
        name="big", link_boxes={"body": ((-0.7, -0.7, 0.0), (0.7, 0.7, 1.0))},
    )
    room = _procedural_room(seed=0, room_size=(1.2, 1.2, 2.7), pool=[big],
                            n_furniture=(2, 2))
    with pytest.raises(ValueError, match="without overlap"):
        room.resolve_layout_spec()


def test_procedural_interior_grid_fallback(monkeypatch):
    room = _procedural_room(seed=1, room_size=(4.0, 4.0, 2.7),
                            n_furniture=(1, 1), pool=[_chair()])
    # Force every wall candidate to be rejected: the interior grid must
    # still find a non-overlapping spot.
    monkeypatch.setattr(room, "_wall_candidate", lambda *args, **kwargs: None)
    spec = room.resolve_layout_spec()
    assert len(spec.furniture) == 1
    placement = spec.furniture[0]
    assert -2.0 <= placement.pos[0] <= 2.0
    assert -2.0 <= placement.pos[1] <= 2.0
    assert placement.source == "procedural"


# ------------------------------------------------------------------
# Room-frame pose helpers
# ------------------------------------------------------------------


def test_xy_to_world_identity_and_shapes():
    world = _FakeWorld()
    room = RoomBuilder(world)
    xy = _host([1.0, 2.0])
    assert torch.allclose(room.xy_to_world(xy), xy)
    batch = _host([[1.0, 2.0], [3.0, -4.0]])
    assert torch.allclose(room.xy_to_world(batch), batch)
    # A plain sequence has no device of its own: it lands on the world's.
    from_seq = room.xy_to_world([0.5, 0.5])
    assert from_seq.device == world.device
    assert torch.allclose(from_seq, _host([0.5, 0.5]))
    with pytest.raises(ValueError, match="last dimension 2"):
        room.xy_to_world(_host([0.0, 0.0, 0.0]))


def test_placement_world_pose_math():
    world = _FakeWorld()
    room = RoomBuilder(world)

    pos, quat = room.placement_world_pose([1.0, 2.0, 3.0])
    assert pos.device == world.device  # sequence input → world device
    assert torch.allclose(pos, _host([1.0, 2.0, 3.0]))
    # No local quaternion → the room's own orientation (identity).
    assert torch.allclose(quat, _host([1.0, 0.0, 0.0, 0.0]))

    # 2-D input → floor height (room origin z = 0.0).
    pos2, _ = room.placement_world_pose([0.2, -0.4])
    assert torch.allclose(pos2, _host([0.2, -0.4, 0.0]))

    # A local quaternion composes with the room orientation (identity here).
    local_quat = _host([0.7071068, 0.0, 0.0, 0.7071068])
    _, quat2 = room.placement_world_pose([0.0, 0.0, 0.0], local_quat)
    assert torch.allclose(quat2, local_quat, atol=1e-6)

    # Batched inputs keep their batch shapes.
    batch = _host([[1.0, 2.0, 0.5], [-1.0, 0.0, 0.25]])
    posb, quatb = room.placement_world_pose(batch, local_quat)
    assert posb.shape == (2, 3) and quatb.shape == (2, 4)
    assert torch.allclose(posb, batch)
    assert torch.allclose(quatb, local_quat.expand(2, 4), atol=1e-6)

    with pytest.raises(ValueError, match="last dimension 2 or 3"):
        room.placement_world_pose(_host([0.0, 0.0, 0.0, 0.0]))


# ------------------------------------------------------------------
# Scene offset (pos + yaw): frame math (plan: convert-once at morph load)
# ------------------------------------------------------------------

_OFFSET_POS = (10.0, -4.0, 1.5)
_OFFSET_YAW = 0.7


def _rotate_xy(x: float, y: float, yaw: float):
    """Hand-rolled R(yaw) @ (x, y) — independent of the builder's math."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (x * c - y * s, x * s + y * c)


def _hamilton(q1, q2):
    """Hand-rolled Hamilton product ``q1 ⊗ q2`` (w, x, y, z quaternions)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _offset_room(**kwargs):
    return RoomBuilder(_FakeWorld(), pos=_OFFSET_POS, yaw=_OFFSET_YAW, **kwargs)


def test_xy_to_world_offset_and_yaw():
    # Translation only.
    moved = RoomBuilder(_FakeWorld(), pos=(3.0, -5.0, 2.0))
    assert torch.allclose(moved.xy_to_world(_host([4.0, 6.0])), _host([7.0, 1.0]))

    # Yaw only: R(90°) maps local +x → world +y.
    turned = RoomBuilder(_FakeWorld(), yaw=math.pi / 2)
    assert torch.allclose(
        turned.xy_to_world(_host([1.0, 0.0])), _host([0.0, 1.0]), atol=1e-6
    )

    # Offset + yaw: world = pos[:2] + R(yaw) @ local (hand-computed).
    room = _offset_room()
    local = _host([[1.0, 2.0], [-0.5, 0.25]])
    got = room.xy_to_world(local)
    expected = _host([
        [_OFFSET_POS[0] + _rotate_xy(1.0, 2.0, _OFFSET_YAW)[0],
         _OFFSET_POS[1] + _rotate_xy(1.0, 2.0, _OFFSET_YAW)[1]],
        [_OFFSET_POS[0] + _rotate_xy(-0.5, 0.25, _OFFSET_YAW)[0],
         _OFFSET_POS[1] + _rotate_xy(-0.5, 0.25, _OFFSET_YAW)[1]],
    ])
    assert got.shape == (2, 2)
    assert torch.allclose(got, expected, atol=1e-6)

    # Sequence input lands on the world device (default offset keeps identity).
    assert torch.allclose(RoomBuilder(_FakeWorld()).xy_to_world([0.5, 0.5]), _host([0.5, 0.5]))


def test_placement_world_pose_offset_and_yaw():
    room = _offset_room()
    c, s = math.cos(_OFFSET_YAW), math.sin(_OFFSET_YAW)

    # 3-D room-frame position (hand-computed rotation + translation).
    got, quat = room.placement_world_pose([1.0, 2.0, 3.0])
    expected = _host(
        [_OFFSET_POS[0] + (1.0 * c - 2.0 * s),
         _OFFSET_POS[1] + (1.0 * s + 2.0 * c),
         _OFFSET_POS[2] + 3.0]
    )
    assert torch.allclose(got, expected, atol=1e-6)
    # No local quaternion → the room's own orientation.
    room_quat = rb.yaw_to_quat(_OFFSET_YAW)
    assert torch.allclose(quat, _host(room_quat), atol=1e-6)

    # 2-D input gets z of the room origin (pos[2]), not the world floor 0.
    pos2, _ = room.placement_world_pose([0.0, 0.0])
    assert torch.allclose(pos2, _host(_OFFSET_POS), atol=1e-6)

    # A local quaternion composes as room_quat ⊗ local_quat (hand-computed
    # Hamilton product — no library call on the expected side).
    local_quat = (0.5, 0.5, 0.5, 0.5)
    _, quat2 = room.placement_world_pose([0.0, 0.0, 0.0], local_quat)
    expected_quat = _hamilton(rb.yaw_to_quat(_OFFSET_YAW), local_quat)
    assert torch.allclose(quat2, _host(expected_quat), atol=1e-6)

    # Batched inputs keep their batch shapes.
    batch = _host([[1.0, 2.0, 0.5], [-1.0, 0.0, 0.25]])
    posb, quatb = room.placement_world_pose(batch, local_quat)
    assert posb.shape == (2, 3) and quatb.shape == (2, 4)
    assert torch.allclose(quatb, _host(expected_quat).expand(2, 4), atol=1e-6)


# ------------------------------------------------------------------
# Scene offset (pos + yaw): scene-level load sites (offline fakes)
# ------------------------------------------------------------------


def test_offset_procedural_shell_morph_world_poses():
    world = _FakeWorld()
    room = RoomBuilder(
        world, pos=_OFFSET_POS, yaw=_OFFSET_YAW, room_size=(4.0, 3.0, 2.6)
    )
    room.reload(priority=110)
    assert room._mode == "shell_only"

    calls = world.scene.add_entity_calls
    assert [name for _, name, _ in calls] == [
        "room_floor",
        "room_wall_north",
        "room_wall_south",
        "room_wall_east",
        "room_wall_west",
    ]

    # Local (room-frame) morph placements, exactly the builder's values.
    thickness = rb.DEFAULT_WALL_THICKNESS
    width, depth, height = 4.0, 3.0, 2.6
    locals_ = {
        "room_floor": (0.0, 0.0, -thickness / 2.0),
        "room_wall_north": (0.0, depth / 2.0 + thickness / 2.0, height / 2.0),
        "room_wall_south": (0.0, -(depth / 2.0 + thickness / 2.0), height / 2.0),
        "room_wall_east": (width / 2.0 + thickness / 2.0, 0.0, height / 2.0),
        "room_wall_west": (-(width / 2.0 + thickness / 2.0), 0.0, height / 2.0),
    }
    for morph, name, _ in calls:
        local = locals_[name]
        # World pose = room offset applied to the local morph pose.  The
        # morph itself carries the offset (no post-build set_pose), so a
        # batched build stays correct.
        rx, ry = _rotate_xy(local[0], local[1], _OFFSET_YAW)
        assert morph.pos == pytest.approx(
            (_OFFSET_POS[0] + rx, _OFFSET_POS[1] + ry, _OFFSET_POS[2] + local[2]),
            abs=1e-5,
        )
        assert morph.quat == pytest.approx(rb.yaw_to_quat(_OFFSET_YAW), abs=1e-6)


def test_offset_baked_shell_morph_pose(tmp_path, monkeypatch):
    import trimesh

    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    shell_obj = tmp_path / "scenes" / "rooms" / "tests" / "shell.obj"
    shell_obj.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(3.0, 3.0, 2.5)).export(shell_obj)

    world = _FakeWorld()
    room = RoomBuilder(
        world,
        pos=_OFFSET_POS,
        yaw=_OFFSET_YAW,
        layout=RoomLayoutSpec(
            shell_id="tests/shell.obj", room_size=(3.0, 3.0, 2.5), furniture=()
        ),
    )
    room.reload(priority=110)

    assert len(world.scene.add_entity_calls) == 1
    morph, name, _ = world.scene.add_entity_calls[0]
    assert name == "room_shell"
    assert isinstance(morph, gs.morphs.Mesh)
    # The baked shell's asset origin is the room-frame origin; its morph pose
    # is the room pose.  Baked specs never carry world offsets themselves.
    assert morph.pos == pytest.approx(_OFFSET_POS, abs=1e-5)
    assert morph.quat == pytest.approx(rb.yaw_to_quat(_OFFSET_YAW), abs=1e-6)


def test_offset_furniture_load_world_pose_compose_order():
    descriptor = _counter()
    standing = (0.5, 0.5, 0.5, 0.5)  # non-identity upright orientation
    descriptor.standing_quaternion = standing
    placement = FurniturePlacement(
        object_id="counter", source="procedural", pos=(1.0, 2.0, 0.0), yaw=0.4,
    )
    world = _FakeWorld()
    room = RoomBuilder(
        world,
        pos=_OFFSET_POS,
        yaw=_OFFSET_YAW,
        layout=RoomLayoutSpec(
            shell_id=None, room_size=(6.0, 5.0, 2.7), furniture=(placement,)
        ),
        furniture_pool=[descriptor],
    )
    room.reload(priority=110)

    call = descriptor.load_calls[0]
    rx, ry = _rotate_xy(placement.pos[0], placement.pos[1], _OFFSET_YAW)
    assert call["pos"] == pytest.approx(
        (_OFFSET_POS[0] + rx, _OFFSET_POS[1] + ry, _OFFSET_POS[2]), abs=1e-5
    )
    # Orientation: room_quat ⊗ (placement_yaw ⊗ standing) — the room yaw is
    # the outermost composition (hand-computed Hamilton product).
    expected = _hamilton(
        rb.yaw_to_quat(_OFFSET_YAW), rb.placement_load_quat(placement.yaw, standing)
    )
    assert call["quat"] == pytest.approx(expected, abs=1e-5)
    # Internal state stays room-frame (the spec is frame-invariant): the
    # occupied footprint is inside the room rect, not shifted by the offset.
    assert room.placements["counter"] == placement
    ox, oy, _ = room._occupied[0]
    assert -3.0 <= ox <= 3.0 and -2.5 <= oy <= 2.5


def test_offset_get_receptacles_world_frame():
    # Translation-only offset: the link interior/top corners are the local
    # corners offset by pos — deliberately far from the world origin.
    world = _FakeWorld()
    counter = _counter()
    placement = FurniturePlacement(
        object_id="counter", source="procedural", pos=(1.0, 2.0, 0.0), yaw=0.0,
    )
    room = RoomBuilder(
        world,
        pos=_OFFSET_POS,
        yaw=0.0,
        layout=RoomLayoutSpec(
            shell_id=None, room_size=(6.0, 5.0, 2.7), furniture=(placement,)
        ),
        furniture_pool=[counter],
    )
    room.reload(priority=110)
    regions = room.get_receptacles()["counter"]
    margin = rb.DEFAULT_INTERIOR_MARGIN

    low, high = _corners(regions["drawer"])
    assert low[0] == pytest.approx(_OFFSET_POS[0] + 1.0 + (-0.5 + margin))
    assert high[0] == pytest.approx(_OFFSET_POS[0] + 1.0 + (0.5 - margin))
    assert low[1] == pytest.approx(_OFFSET_POS[1] + 2.0 + (-0.28 + margin))
    assert high[1] == pytest.approx(_OFFSET_POS[1] + 2.0 + (0.28 - margin))
    assert low[2] == pytest.approx(_OFFSET_POS[2] + 0.5 + margin)
    assert high[2] == pytest.approx(_OFFSET_POS[2] + 0.75 - margin)
    # The region lives near the offset furniture, not around the world origin.
    assert math.hypot(float(low[0]), float(low[1])) > 5.0

    top_low, top_high = _corners(regions["top"])
    assert top_low[0] == pytest.approx(_OFFSET_POS[0] + 1.0 - 0.6)
    assert top_high[0] == pytest.approx(_OFFSET_POS[0] + 1.0 + 0.6)
    assert top_high[1] == pytest.approx(_OFFSET_POS[1] + 2.0 + 0.3)
    assert top_high[2] == pytest.approx(_OFFSET_POS[2] + 0.9)

    # Yaw 90° room + yaw-0 placement: the interior's world AABB axes swap.
    room_yaw = RoomBuilder(
        world,
        pos=_OFFSET_POS,
        yaw=math.pi / 2,
        layout=RoomLayoutSpec(
            shell_id=None, room_size=(6.0, 5.0, 2.7), furniture=(placement,)
        ),
        furniture_pool=[_counter()],
    )
    room_yaw.reload(priority=110)
    low, high = _corners(room_yaw.get_receptacles()["counter"]["drawer"])
    centre = (
        _OFFSET_POS[0] - 2.0,
        _OFFSET_POS[1] + 1.0,
        _OFFSET_POS[2] + 0.625,
    )
    half = (
        (0.56 - 2.0 * margin) / 2.0,
        (1.0 - 2.0 * margin) / 2.0,
        (0.25 - 2.0 * margin) / 2.0,
    )
    for axis in range(3):
        assert low[axis] == pytest.approx(centre[axis] - half[axis], abs=1e-5)
        assert high[axis] == pytest.approx(centre[axis] + half[axis], abs=1e-5)


def test_offset_sample_free_floor_positions_world_frame():
    world = _FakeWorld()
    counter = _counter()
    placement = FurniturePlacement(
        object_id="counter", source="procedural", pos=(1.0, 2.0, 0.0), yaw=0.5,
    )
    room = RoomBuilder(
        world,
        pos=_OFFSET_POS,
        yaw=_OFFSET_YAW,
        layout=RoomLayoutSpec(
            shell_id=None, room_size=(6.0, 5.0, 2.7), furniture=(placement,)
        ),
        furniture_pool=[counter],
    )
    room.reload(priority=110)
    samples = room.sample_free_floor_positions(
        64, torch.Generator().manual_seed(3)
    )
    assert samples.shape == (64, 3)
    assert samples.device == world.device

    # z is the room origin's floor height, not the world origin.
    assert torch.allclose(
        samples[:, 2],
        torch.full_like(samples[:, 2], _OFFSET_POS[2]),
    )
    # Samples are far from the world origin: the room really moved.
    assert float(samples[:, :2].norm(dim=-1).min()) > 5.0

    # Map world → room frame (inverse rotation) and check the translated,
    # rotated room rect bounds (room_size/2 - margin).
    c, s = math.cos(_OFFSET_YAW), math.sin(_OFFSET_YAW)
    rel = samples[:, :2] - torch.tensor(
        _OFFSET_POS[:2], dtype=samples.dtype, device=samples.device
    )
    back = torch.stack(
        (rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c), dim=-1
    )
    assert (back[:, 0].abs() <= 3.0 - rb.DEFAULT_WALL_CLEARANCE + 1e-5).all()
    assert (back[:, 1].abs() <= 2.5 - rb.DEFAULT_WALL_CLEARANCE + 1e-5).all()

    # Furniture avoidance still holds in the world frame (footprint-overlap
    # rejection itself stays room-frame).
    (cx, cy), _, _ = room._rotated_aabb_of(counter, placement.pos, placement.yaw)
    wx, wy = _rotate_xy(cx, cy, _OFFSET_YAW)
    radius = counter.get_footprint_radius() + rb.PLACEMENT_PADDING
    dist = torch.hypot(
        samples[:, 0] - (_OFFSET_POS[0] + wx), samples[:, 1] - (_OFFSET_POS[1] + wy)
    )
    assert (dist > radius - 1e-5).all()


# ------------------------------------------------------------------
# Task-facing API
# ------------------------------------------------------------------


def test_task_api_requires_reload():
    room = RoomBuilder(_FakeWorld())  # nothing resolved yet
    with pytest.raises(RuntimeError, match="reload"):
        room.get_receptacles()
    with pytest.raises(RuntimeError, match="reload"):
        room.get_interaction_points()
    with pytest.raises(RuntimeError, match="reload"):
        room.sample_free_floor_positions(1, None)
    with pytest.raises(RuntimeError, match="reload"):
        room.set_furniture_pose_3d("x", _host([0.0, 0.0, 0.0]))


def test_get_receptacles_top_slab_and_link_interiors():
    room, descriptor, world = _loaded_room(placement_pos=(1.0, 2.0, 0.0), yaw=0.0)
    receptacles = room.get_receptacles()
    assert set(receptacles) == {"counter"}
    regions = receptacles["counter"]

    # "top" = upper slab of the union bbox (body ∪ drawer), translated by
    # the placement pos (yaw 0 → no rotation).
    top = regions["top"]
    assert isinstance(top, BoxSpace)
    # World-frame regions are returned on the *world's* device/dtype (they are
    # derived from the built entity's runtime pose), never on whatever torch's
    # ambient default device happens to be.
    assert top.low.device == world.device and top.high.device == world.device
    assert top.low.dtype == world.dtype
    top_low, top_high = _corners(top)
    assert top_low[0] == pytest.approx(1.0 - 0.6)
    assert top_high[0] == pytest.approx(1.0 + 0.6)
    assert top_high[1] == pytest.approx(2.0 + 0.3)
    assert top_high[2] == pytest.approx(0.9)
    assert top_low[2] == pytest.approx(0.9 - rb.DEFAULT_RECEPTACLE_SLAB_THICKNESS)
    assert (top.low <= top.high).all()

    # Drawer interior: reference bbox shrunk by the interior margin on all
    # axes, then placed (yaw 0 → pure translation).
    interior = regions["drawer"]
    assert interior.low.device == world.device and interior.high.device == world.device
    assert interior.low.dtype == world.dtype
    margin = rb.DEFAULT_INTERIOR_MARGIN
    interior_low, interior_high = _corners(interior)
    assert interior_low[0] == pytest.approx(1.0 + (-0.5 + margin))
    assert interior_high[0] == pytest.approx(1.0 + (0.5 - margin))
    assert interior_low[1] == pytest.approx(2.0 + (-0.28 + margin))
    assert interior_high[1] == pytest.approx(2.0 + (0.28 - margin))
    assert interior_low[2] == pytest.approx(0.5 + margin)
    assert interior_high[2] == pytest.approx(0.75 - margin)
    assert (interior.low <= interior.high).all()

    # The static body link is not a receptacle: no movable joint points at
    # it and the slab region is keyed "top".
    assert "body" not in regions


def test_get_receptacles_world_frame_under_yaw():
    room, _, _ = _loaded_room(placement_pos=(1.0, 2.0, 0.0), yaw=math.pi / 2)
    interior = room.get_receptacles()["counter"]["drawer"]
    # yaw 90° maps local +x → +y: the world AABB half extents swap.
    size = bbox_size(interior).detach().cpu()
    margin2 = 2 * rb.DEFAULT_INTERIOR_MARGIN
    assert size[0] == pytest.approx(0.56 - margin2)
    assert size[1] == pytest.approx(1.0 - margin2)
    assert size[2] == pytest.approx(0.25 - margin2)
    assert (interior.low <= interior.high).all()


def test_get_receptacles_custom_region_params():
    room, _, _ = _loaded_room()
    regions = room.get_receptacles(slab_thickness=0.2, interior_margin=0.05)
    top_low, _ = _corners(regions["counter"]["top"])
    assert top_low[2] == pytest.approx(0.9 - 0.2)
    interior_low, _ = _corners(regions["counter"]["drawer"])
    assert interior_low[2] == pytest.approx(0.5 + 0.05)


def test_get_interaction_points_shape_and_aliases():
    room, _, _ = _loaded_room()
    interaction = room.get_interaction_points()
    assert set(interaction) == {"counter"}
    parts = interaction["counter"]
    assert set(parts) == {"body", "drawer"}  # one entry per built-entity link
    for candidates in parts.values():
        assert isinstance(candidates, GraspCandidates)
        assert candidates.position.shape[-1] == 3
        assert candidates.batch_shape == candidates.approach_direction.shape[:-1]

    # Aliases are passed through: the alias resolves to its real link and
    # the candidates are keyed by the alias name.
    aliased = room.get_interaction_points({"handle": "drawer"})["counter"]
    assert "handle" in aliased
    assert torch.allclose(aliased["handle"].position, parts["drawer"].position)

    # Custom gripper width is forwarded (a wide gripper finds more
    # candidates on the drawer's thin axes).
    wide = room.get_interaction_points(gripper_width=0.5)["counter"]
    assert len(wide["drawer"]) > 0


# ------------------------------------------------------------------
# sample_free_floor_positions
# ------------------------------------------------------------------


def test_sample_free_floor_positions_bounds_and_avoidance():
    room, descriptor, world = _loaded_room(placement_pos=(0.0, 0.0, 0.0))
    samples = room.sample_free_floor_positions(64, torch.Generator().manual_seed(3))
    assert samples.shape == (64, 3)
    # The samples are world-frame spawn positions → world device/dtype.
    assert samples.device == world.device and samples.dtype == world.dtype
    width, depth, _ = (6.0, 5.0, 2.7)
    bound = torch.tensor([width / 2.0, depth / 2.0], dtype=samples.dtype,
                         device=samples.device)
    assert (samples[:, :2].abs() <= bound).all()
    assert (samples[:, 2] == 0.0).all()
    # No sample falls inside the furniture's padded footprint circle.
    radius = descriptor.get_footprint_radius() + PLACEMENT_PADDING
    dist = torch.linalg.vector_norm(samples[:, :2], dim=-1)
    assert (dist > radius).all()


def test_sample_free_floor_positions_is_deterministic():
    room, _, _ = _loaded_room(placement_pos=(0.0, 0.0, 0.0))
    # A host generator matches this (CPU) fake world; the room samples on
    # ``world.device``, so the two are consistent whatever the ambient
    # default device is.
    a = room.sample_free_floor_positions(16, torch.Generator().manual_seed(11))
    assert a.device == _CPU
    b = room.sample_free_floor_positions(16, torch.Generator().manual_seed(11))
    assert torch.allclose(a, b)
    c = room.sample_free_floor_positions(16, torch.Generator().manual_seed(12))
    assert not torch.allclose(a, c)


def test_sample_free_floor_positions_crowded_room_raises():
    # A 6 × 6 m footprint (radius 4.24 + padding) exceeds every corner of
    # the free rect (max distance sqrt(2.95² + 2.45²) ≈ 3.84) — nothing fits.
    big = _BoxFurniture(
        name="big", link_boxes={"body": ((-3.0, -3.0, 0.0), (3.0, 3.0, 1.0))},
    )
    room = _loaded_room(pool=[big], placement_pos=(0.0, 0.0, 0.0))[0]
    with pytest.raises(RuntimeError, match="too crowded"):
        room.sample_free_floor_positions(4, torch.Generator().manual_seed(0))


# ------------------------------------------------------------------
# reset / initial_open
# ------------------------------------------------------------------


def test_reset_applies_initial_open_fractions():
    room, _, _ = _loaded_room(initial_open={"drawer_slide": 0.5})
    entity = room.furniture_entities["counter"]
    # reload() alone does not touch joint state — after_reload() applies it
    # (mirroring pickplace: the reload phase does not run reset()).
    assert float(entity.qpos[0]) == pytest.approx(0.0)
    room.after_reload(priority=110)
    assert float(entity.qpos[0]) == pytest.approx(0.3)
    # An explicit reset re-applies the same fraction deterministically.
    room.reload(priority=110)
    room.after_reload(priority=110)
    assert float(entity.qpos[0]) == pytest.approx(0.3)
    position, dofs_idx = entity.set_dofs_position_calls[-1]
    assert list(dofs_idx) == [0]
    assert float(position[0]) == pytest.approx(0.0 + 0.5 * (0.6 - 0.0))


def test_reset_clamps_and_skips_empty_initial_open():
    # Fractions outside [0, 1] are clamped by set_open_fraction (5.0 → the
    # joint's upper limit 0.6), applied by after_reload.
    room, _, _ = _loaded_room(initial_open={"drawer_slide": 5.0})
    entity = room.furniture_entities["counter"]
    room.after_reload(priority=110)
    assert float(entity.qpos[0]) == pytest.approx(0.6)

    room, _, _ = _loaded_room()  # no initial_open at all
    entity = room.furniture_entities["counter"]
    assert entity.qpos.numel() == 1
    assert float(entity.qpos[0]) == pytest.approx(0.0)


def test_reset_unknown_joint_raises_listing_available():
    room, _, _ = _loaded_room(initial_open={"ghost_joint": 0.5})
    with pytest.raises(ArticulationError) as excinfo:
        room.reset(priority=110)
    message = str(excinfo.value)
    assert "ghost_joint" in message
    assert "drawer_slide" in message  # available joints are listed
    assert "RoomBuilder furniture 'counter'" in message


# ------------------------------------------------------------------
# set_furniture_pose_3d (batch_fixed_verts path, plan §3.2 / §7 risk #2)
# ------------------------------------------------------------------


def test_set_furniture_pose_3d_guards():
    world = _FakeWorld(batch_size=4)
    room = RoomBuilder(
        world,
        layout=RoomLayoutSpec(
            shell_id=None,
            room_size=(6.0, 5.0, 2.7),
            furniture=(FurniturePlacement(
                object_id="counter", source="procedural",
                pos=(1.0, 1.0, 0.0), yaw=0.0,
            ),),
        ),
        furniture_pool=[_counter()],
        batch_fixed_verts=False,  # explicit off (per-env layout disabled)
    )
    room.reload(priority=110)
    with pytest.raises(RuntimeError, match="batch_fixed_verts"):
        room.set_furniture_pose_3d("counter", _host([[0.0] * 3] * 4))
    with pytest.raises(KeyError, match="Unknown furniture key"):
        room.set_furniture_pose_3d("ghost", _host([0.0, 0.0, 0.0]))
    # A single shared pose is always allowed.  Poses are supplied *by the
    # caller* on the world device (this fake world is CPU); the fake entity
    # rejects anything else, mirroring a real CPU scene.
    entity = room.furniture_entities["counter"]
    room.set_furniture_pose_3d("counter", _host([2.0, 2.0, 0.0]))
    assert entity._pos.device == world.device
    assert torch.allclose(entity._pos, _host([2.0, 2.0, 0.0]))
    # The orientation is kept (no random quaternion).
    assert torch.allclose(entity._quat, _host([1.0, 0.0, 0.0, 0.0]))


# ------------------------------------------------------------------
# Minimal real Genesis scenes (CPU)
# ------------------------------------------------------------------


def _real_world():
    from unienv_genesis import GenesisWorld

    return GenesisWorld()


def test_procedural_shell_reload_minimal_scene():
    world = _real_world()
    room = RoomBuilder(
        world,
        name="room",
        procedural=True,
        room_size=(4.0, 3.0, 2.6),
        furniture_pool=[_chair()],
        n_furniture=(1, 2),
        seed=11,
    )
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    assert room.layout is not None and room.layout.shell_id is None
    assert room.room_size == (4.0, 3.0, 2.6)
    # Primitive floor + four walls built into the real scene.
    assert len(room.shell_entities) == 5
    names = [entity.name for entity in room.shell_entities]
    assert names[0] == "room_floor"
    assert set(names[1:]) == {
        "room_wall_north", "room_wall_south", "room_wall_east", "room_wall_west",
    }
    # Sampled furniture descriptors were asked to load with the room pose.
    assert 1 <= len(room.furniture) <= 2
    # The same descriptor instance may back several keys when sampled twice;
    # dedupe by identity before reading its recorded load calls.
    unique_descriptors = {id(d): d for d in room.furniture.values()}
    all_load_calls = [call for d in unique_descriptors.values() for call in d.load_calls]
    assert all(call["world"] is world and call["batch_fixed_verts"] is False
               for call in all_load_calls)
    # Every placement key must have produced exactly one namespaced load call.
    name_overrides = [call["name_override"] for call in all_load_calls]
    assert sorted(name_overrides) == sorted(f"room_{key}" for key in room.furniture)
    # Each placement was loaded at its world pose (the default room offset
    # maps the sampled room-frame pose to itself, to morph float precision).
    for placement in room.placements.values():
        matches = [
            call for call in all_load_calls
            if call["pos"] == pytest.approx(placement.pos, abs=1e-6)
            and call["quat"] == pytest.approx(rb.yaw_to_quat(placement.yaw), abs=1e-6)
        ]
        assert matches, f"no load call matches placement {placement}"
    env.close()

def test_asset_shell_reload_minimal_scene(tmp_path, monkeypatch):
    import trimesh

    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    # A tiny 3 × 3 × 2.5 box "shell" exported as OBJ.
    shell_mesh = trimesh.creation.box(extents=(3.0, 3.0, 2.5))
    shell_obj = tmp_path / "scenes" / "rooms" / "tests" / "shell.obj"
    shell_obj.parent.mkdir(parents=True, exist_ok=True)
    shell_mesh.export(shell_obj)

    world = _real_world()
    room = RoomBuilder(
        world,
        name="room",
        layout=RoomLayoutSpec(
            shell_id="tests/shell.obj",
            room_size=(3.0, 3.0, 2.5),
            furniture=(),
        ),
    )
    # Wrap the world's create_scene so the (rebuilt) scene's add_entity is
    # recorded while the shell is still really built.
    original_create_scene = world.create_scene
    add_entity_calls = []

    def _recording_create_scene(disable_viewer: bool = False):
        scene = original_create_scene(disable_viewer=disable_viewer)
        original_add_entity = scene.add_entity

        def _recording_add_entity(morph, *args, **kwargs):
            add_entity_calls.append((morph, kwargs.get("name")))
            return original_add_entity(morph, *args, **kwargs)

        scene.add_entity = _recording_add_entity
        return scene

    world.create_scene = _recording_create_scene
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    assert len(room.shell_entities) == 1
    morph, name = add_entity_calls[0]
    assert name == "room_shell"
    assert isinstance(morph, gs.morphs.Mesh)
    assert bool(morph.fixed) is True
    assert str(morph.file) == str(shell_obj)
    env.close()


# ------------------------------------------------------------------
# Explicit-layout furniture really loads into a minimal CPU scene
# ------------------------------------------------------------------

_URDF = """<?xml version='1.0' encoding='utf-8'?>
<robot name="unit_drawer">
    <link name="base">
        <visual>
            <geometry><mesh filename="base.obj" scale="1 1 1" /></geometry>
        </visual>
    </link>
    <link name="drawer">
        <visual>
            <geometry><mesh filename="drawer.obj" scale="1 1 1" /></geometry>
            <origin xyz="0.4 0 0" rpy="0 0 0" />
        </visual>
    </link>
    <joint name="drawer_slide" type="prismatic">
        <parent link="base" />
        <child link="drawer" />
        <axis xyz="1 0 0" />
        <limit lower="0" upper="0.6" />
    </joint>
</robot>
"""


class _URDFDrawer(ArticulatedObject):
    """Synthetic URDF-backed descriptor loaded through ``gs.morphs.URDF``."""

    def __init__(self, urdf_file, name="unit_drawer"):
        super().__init__(name=name, fixed=True)
        self.urdf_file = str(urdf_file)

    def _load_link_meshes_at_reference(self):
        return urdf_link_meshes_at_config(self.urdf_file, self.reference_qpos)

    def _load_link_local_meshes_at_reference(self):
        return urdf_link_meshes_at_config(self.urdf_file, self.reference_qpos, link_local=True)

    def load_into_scene(self, world, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                        name_override=None, batch_fixed_verts=False):
        return world.scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_file,
                pos=pos,
                quat=quat,
                align=False,
                fixed=self.fixed,
                merge_fixed_links=False,
                batch_fixed_verts=batch_fixed_verts,
            ),
            name=name_override or self.name,
        )


def test_explicit_layout_furniture_loads_into_minimal_scene(tmp_path):
    import trimesh

    def _write_box(path: Path, size, offset=(0.0, 0.0, 0.0)):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(offset)
        mesh.export(path)

    _write_box(tmp_path / "base.obj", (0.8, 0.6, 0.9))
    _write_box(tmp_path / "drawer.obj", (0.5, 0.5, 0.25), offset=(0.4, 0.0, 0.6))
    urdf_file = tmp_path / "unit_drawer.urdf"
    urdf_file.write_text(_URDF, encoding="utf-8")

    world = _real_world()
    drawer = _URDFDrawer(urdf_file)
    room = RoomBuilder(
        world,
        name="room",
        layout=RoomLayoutSpec(
            shell_id=None,
            room_size=(4.0, 3.0, 2.6),
            furniture=(FurniturePlacement(
                object_id="unit_drawer", source="procedural",
                pos=(1.0, -0.5, 0.0), yaw=math.pi / 2,
                initial_open={"drawer_slide": 0.5},
            ),),
        ),
        furniture_pool=[drawer],
    )
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    # A real entity was built with the placement pose carried by the morph.
    entity = room.furniture_entities["unit_drawer"]
    assert entity is not None
    pos = torch.as_tensor(entity.get_pos())
    assert pos.shape == (3,)
    # Expected values follow the entity's device/dtype (the backend active in
    # this process may be CPU or CUDA depending on test order).
    assert torch.allclose(
        pos, torch.tensor([1.0, -0.5, 0.0], dtype=pos.dtype, device=pos.device),
        atol=1e-4,
    )
    quat = torch.as_tensor(entity.get_quat())
    assert quat.shape == (4,)

    # The drawer DoF sits at the requested open fraction (after_reload ran
    # the room's reset, which applied the placement's initial_open).
    fraction = drawer.get_open_fraction(entity, "drawer_slide")
    assert torch.allclose(
        fraction, torch.tensor([0.5], dtype=fraction.dtype, device=fraction.device),
        atol=1e-5,
    )
    env.close()
