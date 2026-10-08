"""Batched-world tests for :mod:`unienv_genesis_collection.scenes.room_builder`.

Real Genesis scenes (synthetic URDF assets — no RoboCasa/ReplicaCAD
downloads) exercising the batched behaviour plan §3.2 / §7 risk #2 pin:

  - ``batch_fixed_verts`` defaults from ``world.batch_size is not None``
    (the exact check ``pickplace.py:132`` performs) and the layout is
    **shared** across envs by default (per-env base poses are opt-in).
  - Per-env ``initial_open`` articulation works **without**
    ``batch_fixed_verts`` (DoF state is per-env even for fixed entities),
    including masked resets that leave unmasked envs untouched.
  - ``batch_fixed_verts=True`` enables the per-env base-pose path via
    :meth:`RoomBuilder.set_furniture_pose_3d` (full batch + masked subset).
  - Scene offset (``pos``/``yaw``) is applied once at morph load: batched
    envs share the offset world pose, masked ``initial_open`` resets are
    unaffected and ``set_furniture_pose_3d`` stays world-frame.

The CPU backend batched path runs wherever Genesis supports it; the
GPU-only checks (device placement) are skipped when CUDA is unavailable.
Genesis is re-initialised per backend inside the tests and restored to CPU
afterwards so the rest of the suite is unaffected.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

import trimesh

from unienv_genesis import GenesisWorld
from unienv_interface.world import WorldEnv

from unienv_genesis_collection.objects.articulated_object import (
    ArticulatedObject,
    urdf_link_meshes_at_config,
)
from unienv_genesis_collection.scenes.room_builder import (
    FurniturePlacement,
    RoomBuilder,
    RoomLayoutSpec,
)

# Genesis only allows one initialisation per process; track the active
# backend so tests can switch (and restore) deterministically.
_CURRENT_BACKEND = {"name": "cpu" if not gs._initialized else "unknown"}


def _ensure_genesis_backend(name: str) -> None:
    """(Re)initialise Genesis with the requested backend if needed."""
    if _CURRENT_BACKEND["name"] == name and gs._initialized:
        return
    if gs._initialized:
        gs.destroy()
    gs.init(backend=gs_backend.gpu if name == "gpu" else gs_backend.cpu)
    _CURRENT_BACKEND["name"] = name


_BATCH_SIZE = 4


# ------------------------------------------------------------------
# Synthetic URDF asset (no RoboCasa / ReplicaCAD downloads)
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


def _write_box(path: Path, size, offset=(0.0, 0.0, 0.0)) -> None:
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(offset)
    mesh.export(path)


class _URDFDrawer(ArticulatedObject):
    """Synthetic URDF-backed fixed furniture loaded through ``gs.morphs.URDF``."""

    def __init__(self, urdf_file: Path, name: str = "unit_drawer"):
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


@pytest.fixture(scope="module")
def drawer_urdf(tmp_path_factory) -> Path:
    """A synthetic one-drawer URDF with its box meshes (module-scoped)."""
    assets = tmp_path_factory.mktemp("drawer_assets")

    base = trimesh.creation.box(extents=(0.8, 0.6, 0.9))
    base.export(assets / "base.obj")
    drawer = trimesh.creation.box(extents=(0.5, 0.5, 0.25))
    drawer.apply_translation((0.4, 0.0, 0.6))
    drawer.export(assets / "drawer.obj")

    urdf_file = assets / "unit_drawer.urdf"
    urdf_file.write_text(_URDF, encoding="utf-8")
    return urdf_file


def _room(world, drawer, *, batch_fixed_verts, initial_open=0.5,
          room_size=(4.0, 3.0, 2.6), pos=(1.0, -0.5, 0.0), yaw=math.pi / 2,
          room_pos=(0.0, 0.0, 0.0), room_yaw=0.0):
    """A one-drawer RoomBuilder wired through a WorldEnv (not yet reset).

    ``pos``/``yaw`` are the *placement* (room frame); ``room_pos``/
    ``room_yaw`` are the scene offset of the room itself.
    """
    room = RoomBuilder(
        world,
        name="room",
        layout=RoomLayoutSpec(
            shell_id=None,
            room_size=room_size,
            furniture=(FurniturePlacement(
                object_id="unit_drawer",
                source="procedural",
                pos=pos,
                yaw=yaw,
                initial_open={"drawer_slide": initial_open} if initial_open is not None else {},
            ),),
        ),
        furniture_pool=[drawer],
        batch_fixed_verts=batch_fixed_verts,
        pos=room_pos,
        yaw=room_yaw,
    )
    return room


# ------------------------------------------------------------------
# Shared layout by default (batch_fixed_verts defaulting, plan §3.2)
# ------------------------------------------------------------------


def test_batched_world_defaults_batch_fixed_verts_and_shares_layout(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    drawer = _URDFDrawer(drawer_urdf)
    room = _room(world, drawer, batch_fixed_verts=None)  # default from world
    assert room.batch_fixed_verts is True  # world.batch_size is not None

    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    entity = room.furniture_entities["unit_drawer"]
    pos = torch.as_tensor(entity.get_pos())
    # Batched state: (n_envs, 3), one shared layout → identical rows.
    assert pos.shape == (_BATCH_SIZE, 3)
    assert torch.allclose(pos, pos[:1].expand(_BATCH_SIZE, 3))
    assert torch.allclose(pos[0], torch.tensor([1.0, -0.5, 0.0], dtype=pos.dtype), atol=1e-4)

    # The joint initial_open applies to every env (shared fraction).
    fraction = drawer.get_open_fraction(entity, "drawer_slide")
    assert torch.allclose(fraction, torch.full_like(fraction, 0.5), atol=1e-5)

    # Receptacles are batch-aware: (n_envs, 3) corners, identical across envs.
    regions = room.get_receptacles()["unit_drawer"]
    top = regions["top"]
    assert top.low.shape == (_BATCH_SIZE, 3)
    assert torch.allclose(top.low, top.low[:1].expand(_BATCH_SIZE, 3))
    drawer_interior = regions["drawer"]
    assert drawer_interior.high.shape == (_BATCH_SIZE, 3)
    assert (drawer_interior.low <= drawer_interior.high).all()
    env.close()


def test_batched_procedural_layout_is_shared(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    room = RoomBuilder(
        world,
        name="room",
        procedural=True,
        room_size=(4.0, 3.5, 2.6),
        furniture_pool=[_URDFDrawer(drawer_urdf), _URDFDrawer(drawer_urdf, name="unit_drawer_b")],
        n_furniture=(2, 2),
        seed=3,
    )
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=1, reload=True)

    spec = room.layout
    assert len(spec.furniture) == 2
    assert room.room_size == (4.0, 3.5, 2.6)
    for key, entity in room.furniture_entities.items():
        pos = torch.as_tensor(entity.get_pos())
        assert pos.shape == (_BATCH_SIZE, 3)
        # Shared layout: every env has the same furniture pose.
        assert torch.allclose(pos, pos[:1].expand(_BATCH_SIZE, 3))
        # The pose matches the sampled placement (entity origin).
        placement = room.placements[key]
        assert torch.allclose(pos[0], torch.tensor(placement.pos, dtype=pos.dtype), atol=1e-4)
    env.close()


# ------------------------------------------------------------------
# Per-env articulation WITHOUT batch_fixed_verts (plan §3.2)
# ------------------------------------------------------------------


def test_batched_per_env_initial_open_without_batch_fixed_verts(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    drawer = _URDFDrawer(drawer_urdf)
    room = _room(world, drawer, batch_fixed_verts=False)
    assert room.batch_fixed_verts is False

    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    entity = room.furniture_entities["unit_drawer"]
    # The whole batch sits at the layout's initial_open (masked-free path).
    fraction = drawer.get_open_fraction(entity, "drawer_slide")
    assert torch.allclose(fraction, torch.full_like(fraction, 0.5))

    # Per-env articulation needs no batch_fixed_verts: push envs 0 and 3
    # fully open.
    open_mask = torch.tensor([True, False, False, True])
    drawer.set_open_fraction(entity, "drawer_slide", 1.0, envs_idx=open_mask)
    pushed = drawer.get_open_fraction(entity, "drawer_slide")
    assert pushed.shape == (_BATCH_SIZE, 1)
    assert torch.allclose(pushed[:, 0], torch.tensor([1.0, 0.5, 0.5, 1.0]), atol=1e-5)

    # A masked room reset restores ONLY the selected envs to the layout's
    # initial_open; unmasked envs keep their (randomised) joint state.
    room.reset(priority=min(room.reset_priorities),
               mask=torch.tensor([True, False, False, False]))
    restored = drawer.get_open_fraction(entity, "drawer_slide")
    assert torch.allclose(restored[:, 0], torch.tensor([0.5, 0.5, 0.5, 1.0]), atol=1e-5)
    env.close()


def test_batched_masked_reset_preserves_unmasked_envs(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    drawer = _URDFDrawer(drawer_urdf)
    room = _room(world, drawer, batch_fixed_verts=False)
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=2, reload=True)
    entity = room.furniture_entities["unit_drawer"]

    # Randomise every env individually, then reset only env 2.
    custom = torch.tensor([1.0, 0.0, 0.75, 0.25])
    for env_idx, frac in enumerate(custom.tolist()):
        env_mask = torch.zeros(_BATCH_SIZE, dtype=torch.bool)
        env_mask[env_idx] = True
        drawer.set_open_fraction(entity, "drawer_slide", frac, envs_idx=env_mask)
    room.reset(priority=min(room.reset_priorities),
               mask=torch.tensor([False, False, True, False]))

    fractions = drawer.get_open_fraction(entity, "drawer_slide")
    expected = torch.tensor([1.0, 0.0, 0.5, 0.25])
    assert torch.allclose(fractions[:, 0], expected, atol=1e-5)
    env.close()


# ------------------------------------------------------------------
# batch_fixed_verts=True: per-env base-pose path (plan §3.2 / §7 risk #2)
# ------------------------------------------------------------------


def test_batched_fixed_verts_per_env_layout(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    drawer = _URDFDrawer(drawer_urdf)
    room = _room(world, drawer, batch_fixed_verts=True)
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    entity = room.furniture_entities["unit_drawer"]
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [3.0, 4.0, 0.0]],
        dtype=world.dtype,
        device=world.device,
    )
    # Per-env layout diversity (opt-in): each env gets its own base pose,
    # orientation kept (random_quaternion=False).
    room.set_furniture_pose_3d("unit_drawer", positions)
    pos = torch.as_tensor(entity.get_pos())
    assert pos.shape == (_BATCH_SIZE, 3)
    assert torch.allclose(pos, positions, atol=1e-4)

    # Masked subset update: only the selected env moves.
    env_mask = torch.tensor([True, False, False, False])
    room.set_furniture_pose_3d(
        "unit_drawer", torch.tensor([[9.0, 9.0, 0.0]], dtype=world.dtype, device=world.device),
        envs_idx=env_mask,
    )
    pos = torch.as_tensor(entity.get_pos())
    assert torch.allclose(pos[0], torch.tensor([9.0, 9.0, 0.0]), atol=1e-4)
    assert torch.allclose(pos[1], positions[1], atol=1e-4)
    env.close()


def test_batched_fixed_verts_false_blocks_per_env_poses(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    drawer = _URDFDrawer(drawer_urdf)
    room = _room(world, drawer, batch_fixed_verts=False)
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)
    positions = torch.zeros(_BATCH_SIZE, 3, dtype=world.dtype)
    with pytest.raises(RuntimeError, match="batch_fixed_verts"):
        room.set_furniture_pose_3d("unit_drawer", positions)
    env.close()


# ------------------------------------------------------------------
# Scene offset (pos + yaw) in batched worlds
# ------------------------------------------------------------------

_OFFSET_ROOM_POS = (2.0, -1.0, 0.5)
_OFFSET_ROOM_YAW = math.pi / 2


def test_batched_offset_shared_layout_base_pose_identical(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    drawer = _URDFDrawer(drawer_urdf)
    room = _room(
        world,
        drawer,
        batch_fixed_verts=None,
        room_pos=_OFFSET_ROOM_POS,
        room_yaw=_OFFSET_ROOM_YAW,
    )
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    entity = room.furniture_entities["unit_drawer"]
    pos = torch.as_tensor(entity.get_pos())
    assert pos.shape == (_BATCH_SIZE, 3)
    # Shared layout in an offset room: every env sits at the same world pose.
    assert torch.allclose(pos, pos[:1].expand(_BATCH_SIZE, 3))
    # World pose = room offset ∘ placement: R(90°) · (1.0, -0.5) = (0.5, 1.0).
    expected = torch.tensor(
        [_OFFSET_ROOM_POS[0] + 0.5, _OFFSET_ROOM_POS[1] + 1.0, _OFFSET_ROOM_POS[2]],
        dtype=pos.dtype,
    )
    assert torch.allclose(pos[0], expected, atol=1e-4)

    # Orientation composes room yaw ⊗ placement yaw: 90° + 90° = 180° about Z
    # (q and -q are the same rotation, so compare magnitudes).
    quat = torch.as_tensor(entity.get_quat())
    assert torch.allclose(
        quat.abs(), torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=quat.dtype), atol=1e-4
    )

    # The joint initial_open still applies to every env.
    fraction = drawer.get_open_fraction(entity, "drawer_slide")
    assert torch.allclose(fraction, torch.full_like(fraction, 0.5), atol=1e-5)
    env.close()


def test_batched_offset_masked_reset_preserves_unmasked_envs(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    drawer = _URDFDrawer(drawer_urdf)
    room = _room(
        world,
        drawer,
        batch_fixed_verts=False,
        room_pos=_OFFSET_ROOM_POS,
        room_yaw=_OFFSET_ROOM_YAW,
    )
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=2, reload=True)
    entity = room.furniture_entities["unit_drawer"]

    # Randomise every env individually, then reset only env 2.
    custom = torch.tensor([1.0, 0.0, 0.75, 0.25])
    for env_idx, frac in enumerate(custom.tolist()):
        env_mask = torch.zeros(_BATCH_SIZE, dtype=torch.bool)
        env_mask[env_idx] = True
        drawer.set_open_fraction(entity, "drawer_slide", frac, envs_idx=env_mask)
    room.reset(priority=min(room.reset_priorities),
               mask=torch.tensor([False, False, True, False]))

    fractions = drawer.get_open_fraction(entity, "drawer_slide")
    expected = torch.tensor([1.0, 0.0, 0.5, 0.25])
    assert torch.allclose(fractions[:, 0], expected, atol=1e-5)
    env.close()


def test_batched_offset_set_furniture_pose_3d_stays_world_frame(drawer_urdf):
    _ensure_genesis_backend("cpu")
    world = GenesisWorld(batch_size=_BATCH_SIZE)
    drawer = _URDFDrawer(drawer_urdf)
    room = _room(
        world,
        drawer,
        batch_fixed_verts=True,
        room_pos=_OFFSET_ROOM_POS,
        room_yaw=_OFFSET_ROOM_YAW,
    )
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    entity = room.furniture_entities["unit_drawer"]
    # The per-env override contract is world-frame: the positions are applied
    # verbatim (the room offset was already baked into the load-time morph).
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [3.0, 4.0, 0.0]],
        dtype=world.dtype,
        device=world.device,
    )
    room.set_furniture_pose_3d("unit_drawer", positions)
    pos = torch.as_tensor(entity.get_pos())
    assert pos.shape == (_BATCH_SIZE, 3)
    assert torch.allclose(pos, positions, atol=1e-4)

    # Masked subset update stays world-frame too.
    env_mask = torch.tensor([True, False, False, False])
    room.set_furniture_pose_3d(
        "unit_drawer",
        torch.tensor([[9.0, 9.0, 0.0]], dtype=world.dtype, device=world.device),
        envs_idx=env_mask,
    )
    pos = torch.as_tensor(entity.get_pos())
    assert torch.allclose(pos[0], torch.tensor([9.0, 9.0, 0.0]), atol=1e-4)
    assert torch.allclose(pos[1], positions[1], atol=1e-4)
    env.close()


# ------------------------------------------------------------------
# GPU backend (RTX 4090) — device-aware batched path
# ------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_fixed_verts_per_env_layout_gpu(drawer_urdf):
    _ensure_genesis_backend("gpu")
    try:
        world = GenesisWorld(batch_size=_BATCH_SIZE)
        drawer = _URDFDrawer(drawer_urdf)
        room = _room(world, drawer, batch_fixed_verts=True)
        assert world.device.type == "cuda"

        env = WorldEnv(world, [room], render_mode=None)
        env.reset(seed=0, reload=True)

        entity = room.furniture_entities["unit_drawer"]
        pos = torch.as_tensor(entity.get_pos())
        assert pos.device.type == "cuda"
        assert pos.shape == (_BATCH_SIZE, 3)
        # Shared layout by default.
        assert torch.allclose(pos, pos[:1].expand(_BATCH_SIZE, 3))

        # batch_fixed_verts path: per-env poses, tensors on the world device.
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [3.0, 4.0, 0.0]],
            dtype=world.dtype,
            device=world.device,
        )
        room.set_furniture_pose_3d("unit_drawer", positions)
        pos = torch.as_tensor(entity.get_pos())
        assert pos.device.type == "cuda"
        assert torch.allclose(pos, positions, atol=1e-3)

        # Per-env articulation without batch_fixed_verts needs nothing extra.
        room2_drawer = drawer
        fraction = room2_drawer.get_open_fraction(entity, "drawer_slide")
        assert fraction.device.type == "cuda"
        assert torch.allclose(fraction, torch.full_like(fraction, 0.5), atol=1e-5)
        env.close()
    finally:
        # Restore the CPU backend for the rest of the suite.
        _ensure_genesis_backend("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_initial_open_gpu_matches_cpu_behaviour(drawer_urdf):
    _ensure_genesis_backend("gpu")
    try:
        world = GenesisWorld(batch_size=_BATCH_SIZE)
        drawer = _URDFDrawer(drawer_urdf)
        room = _room(world, drawer, batch_fixed_verts=False)
        env = WorldEnv(world, [room], render_mode=None)
        env.reset(seed=0, reload=True)

        entity = room.furniture_entities["unit_drawer"]
        fraction = drawer.get_open_fraction(entity, "drawer_slide")
        assert fraction.device.type == "cuda"
        assert torch.allclose(fraction, torch.full_like(fraction, 0.5), atol=1e-5)
        env.close()
    finally:
        _ensure_genesis_backend("cpu")
