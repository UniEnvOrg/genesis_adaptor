"""Tests for FloatingVisualPropsManager — pure + CPU-integration.

Pure / mocking tests cover points that do not need an actual Genesis
world:

  - Validation in :class:`FloatingVisualPropsManager.__init__`.
  - The ``gs.morphs.Mesh`` morph arguments built by
    ``_load_visual_entity``: ``fixed=True``, ``collision=False``,
    ``visualization=True``, ``decimate=False``, plus the surface
    binding (explicit textured ``gs.surfaces.Plastic`` for textured
    GSO/YCB descriptors, generic ``surface=None`` fallback otherwise).
  - Batched / masked reset behaviour, deterministic-seed reproducibility,
    bounds, uniqueness of parked XY, deterministic seed behaviour of the
    Shoemake quaternion helper.
  - The 3D sphere overlap rejection honours ``min_separation``.

CPU-integration tests use ``gs.init(cpu)`` (matching the upstream
``test_cluttered_table.py`` harness) with :class:`MeshObject` descriptors
backed by the bundled ``scenes/table/box_with_uv.obj`` asset — no Hub
downloads are required.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import pytest

import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.cached_table_builder import (
    CachedTableBuilder,
)
from unienv_genesis_collection.scenes.floating_visual_props_manager import (
    FloatingVisualPropsManager,
    _PARK_POS,
    _PARK_QUAT,
    _shoemake_quaternions,
)
from unienv_genesis_collection.objects.mesh_object import MeshObject
from unienv_interface.world import WorldEnv

# Visual-texture tests reuse the GSO/YCB fake-asset fixtures; pull the
# boilerplate in here rather than depending on cross-file pytest
# fixture imports.
import importlib.util
import json
import shutil

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_GSO_FUEL_METADATA = _SCRIPTS_DIR / "data" / "gso_fuel_metadata.json"
_GSO_SAMPLE_ID = "3D_Dollhouse_Lamp"
_YCB_CATALOG_SOURCE = _SCRIPTS_DIR / "data" / "ycb_catalog.json"

_GSO_MJCF_TEMPLATE = """\
<mujoco model="model">
  <asset>
    <texture type="2d" name="texture" file="texture.png"/>
    <material name="material_0" texture="texture"/>
    <mesh name="model" file="model.obj"/>
    <mesh name="model_collision_0" file="model_collision_0.obj"/>
    <mesh name="model_collision_1" file="model_collision_1.obj"/>
  </asset>
  <worldbody>
    <body name="model">
      <geom material="material_0" mesh="model" type="mesh" contype="0" conaffinity="0" group="2"/>
      <geom mesh="model_collision_0" type="mesh" group="3"/>
      <geom mesh="model_collision_1" type="mesh" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def _load_prepare_gso():
    spec = importlib.util.spec_from_file_location(
        "prepare_gso_assets", _SCRIPTS_DIR / "prepare_gso_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_gso_object_dir(assets_dir: Path, object_id: str) -> Path:
    from unienv_genesis_collection.objects.hub_objects.gso_object import (
        GSO_ASSETS_ROOT,
    )
    script = _load_prepare_gso()
    obj_dir = assets_dir / GSO_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.xml").write_text(_GSO_MJCF_TEMPLATE)
    (obj_dir / "model_freejoint.xml").write_text(
        script.inject_freejoint(_GSO_MJCF_TEMPLATE)
    )
    for name in (
        "model.obj", "texture.png",
        "model_collision_0.obj", "model_collision_1.obj",
    ):
        (obj_dir / name).write_text("placeholder")
    return obj_dir


_YCB_URDF_TEMPLATE = """\
<robot name="{oid}">
    <link name="base_link">
        <visual><geometry><mesh filename="textured.obj" scale="1 1 1"/></geometry></visual>
        <collision><geometry><mesh filename="textured_coacd_0.stl" scale="1 1 1"/></geometry></collision>
        <collision><geometry><mesh filename="textured_coacd_1.stl" scale="1 1 1"/></geometry></collision>
        <inertial><mass value="0.1"/><inertia ixx="1e-3" ixy="0.0" ixz="0.0" iyy="1e-3" iyz="0.0" izz="1e-3"/></inertial>
    </link>
</robot>
"""


def _make_fake_ycb_object_dir(assets_dir: Path, object_id: str) -> Path:
    from unienv_genesis_collection.objects.hub_objects.ycb_object import (
        YCB_ASSETS_ROOT,
    )
    obj_dir = assets_dir / YCB_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    (obj_dir / f"{object_id}.urdf").write_text(_YCB_URDF_TEMPLATE.format(oid=object_id))
    for name in (
        "textured.obj", "textured.mtl", "texture_map.png",
        "textured_coacd_0.stl", "textured_coacd_1.stl",
    ):
        (obj_dir / name).write_text("placeholder")
    return obj_dir


@pytest.fixture
def gso_assets_dir(tmp_path, monkeypatch):
    """Temp assets dir holding the real GSO catalog and one fake object."""
    from unienv_genesis_collection import asset_path
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    script = _load_prepare_gso()
    metadata = json.loads(_GSO_FUEL_METADATA.read_text())
    out_root = tmp_path / "objects" / "gso"
    out_root.mkdir(parents=True)
    script.write_catalog(out_root, list(metadata), metadata)
    return tmp_path


@pytest.fixture
def gso_fake_object(gso_assets_dir):
    return _make_fake_gso_object_dir(gso_assets_dir, _GSO_SAMPLE_ID)


@pytest.fixture
def ycb_assets_dir(tmp_path, monkeypatch):
    from unienv_genesis_collection import asset_path
    from unienv_genesis_collection.objects.hub_objects.ycb_object import (
        YCB_ASSETS_ROOT,
    )
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    catalog_file = tmp_path / YCB_ASSETS_ROOT / "catalog.json"
    catalog_file.parent.mkdir(parents=True)
    shutil.copy2(_YCB_CATALOG_SOURCE, catalog_file)
    return tmp_path


@pytest.fixture
def ycb_fake_banana(ycb_assets_dir):
    return _make_fake_ycb_object_dir(ycb_assets_dir, "banana")


@pytest.fixture
def no_download(monkeypatch):
    from unienv_genesis_collection import asset_path
    calls = []

    def _fail(**kwargs):
        calls.append(kwargs)
        raise AssertionError(f"Unexpected download attempt: {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail)
    return calls


_TABLE_POS = (0.0, 0.0, 0.65)
_TABLE_SIZE = (0.8, 0.5)
_PARK_Z = _PARK_POS[2]
_BOX_OBJ = str(
    Path(__file__).resolve().parents[1]
    / "unienv_genesis_collection" / "assets" / "scenes" / "table" / "box_with_uv.obj"
)


def _make_pool(n=4, scale=0.25):
    """A small pool of MeshObject descriptors backed by box_with_uv.obj."""
    return [
        MeshObject(
            mesh_file=_BOX_OBJ,
            scale=scale,
            name=f"box_{i}",
        )
        for i in range(n)
    ]


# ============================================================
# Pure / unit tests — Shoemake quaternion helper
# ============================================================

def test_shoemake_quaternions_unit_norm_uniform():
    g = torch.Generator().manual_seed(0)
    qs = _shoemake_quaternions(g, 256, torch.float32, "cpu")
    assert qs.shape == (256, 4)
    norms = qs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Uniform distribution on SO(3): real parts should be roughly
    # uniformly distributed in [-1, 1], not biased to identity.
    assert qs[:, 0].abs().mean().item() < 0.75  # roughly centred


def test_shoemake_quaternions_determinism():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    q1 = _shoemake_quaternions(g1, 16, torch.float32, "cpu")
    q2 = _shoemake_quaternions(g2, 16, torch.float32, "cpu")
    assert torch.equal(q1, q2)


def test_shoemake_empty_returns_empty():
    g = torch.Generator().manual_seed(0)
    qs = _shoemake_quaternions(g, 0, torch.float32, "cpu")
    assert qs.shape == (0, 4)


# ============================================================
# Pure / unit tests — constructor validation
# ============================================================

def _fresh_world():
    return GenesisWorld()


def test_pool_size_validation():
    world = _fresh_world()
    small = _make_pool(1)
    with pytest.raises(ValueError) as e:
        FloatingVisualPropsManager(
            world, object_pool=small, n_objects_range=(0, 5),
        )
    assert "5" in str(e.value) and "1" in str(e.value)
    # Equal is accepted.
    mgr = FloatingVisualPropsManager(
        world, object_pool=small, n_objects_range=(0, 1),
    )
    assert mgr is not None


def test_bad_park_clearance_rejected():
    world = _fresh_world()
    pool = _make_pool(3)
    for bad in (-0.01, float("nan")):
        with pytest.raises(ValueError, match="park_clearance"):
            FloatingVisualPropsManager(world, object_pool=pool, park_clearance=bad)


def test_bad_xy_margin_rejected():
    world = _fresh_world()
    with pytest.raises(ValueError, match="xy_margin"):
        FloatingVisualPropsManager(world, object_pool=_make_pool(), xy_margin=-0.05)


def test_bad_min_separation_rejected():
    world = _fresh_world()
    with pytest.raises(ValueError, match="min_separation"):
        FloatingVisualPropsManager(world, object_pool=_make_pool(), min_separation=-1.0)


def test_bad_z_offset_range_rejected():
    world = _fresh_world()
    for bad in ((0.5, 0.1), (float("nan"), 1.0)):
        with pytest.raises(ValueError, match="z_offset_range"):
            FloatingVisualPropsManager(
                world, object_pool=_make_pool(), z_offset_range=bad,
            )


# ============================================================
# Pure / unit tests — morph arguments (via a fake world)
# ============================================================

class _FakeScene:
    def __init__(self):
        self.added = []

    def add_entity(self, morph, surface=None, name=None):
        self.added.append(SimpleNamespace(morph=morph, surface=surface, name=name))
        return SimpleNamespace(
            morph=morph, surface=surface, name=name,
            # Entity object placeholder; the morph args are what is
            # tested.  The reset / apply paths below use the real CPU
            # world instead of this fake.
        )


# The morph-args test builds its own manager via a SyntheticNamespace
# world + a stubbed ``gs.morphs.Mesh``; no shared fake-world helper is
# needed.


def test_morph_args_fixed_no_collision_no_decimate_surface_none():
    """The morph built inside ``_load_visual_entity`` must set
    ``fixed=True``, ``collision=False``, ``visualization=True``,
    ``decimate=False``, ``scale`` from the descriptor, and the surface
    passed to ``world.scene.add_entity`` must be ``None``.

    This exercises the morph construction through a fake world — no
    Genesis scene exists; we monkeypatch ``gs.morphs.Mesh`` to capture
    kwargs.
    """
    import genesis as gs

    captured = []

    class _MeshSpy:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.kwargs = kwargs

    fake_world = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=None,
        scene=_FakeScene(),
    )
    pool = _make_pool(1, scale=0.42)
    mgr = FloatingVisualPropsManager(
        fake_world, object_pool=pool, n_objects_range=(0, 1),
    )

    with mock.patch.object(gs, "morphs", SimpleNamespace(Mesh=_MeshSpy)):
        entity = mgr._load_visual_entity(
            pool[0],
            pos=(0.1, 0.2, 0.3),
            quat=(1.0, 0.0, 0.0, 0.0),
            name_override="probe",
            batch_fixed_verts=False,
        )

    # Exactly one morph captured.
    assert len(captured) == 1, captured
    kw = captured[0]
    assert kw["fixed"] is True, kw
    assert kw["collision"] is False, kw
    assert kw["visualization"] is True, kw
    assert kw["decimate"] is False, kw
    assert kw["file"] == str(pool[0].mesh_file), kw
    # Scale forwarded verbatim (MeshObject normalises to per-axis tuple).
    assert kw["scale"] == pool[0].scale, kw
    assert kw["pos"] == (0.1, 0.2, 0.3)
    assert kw["name_override"] if "name_override" in kw else True
    # The surface forwarded to add_entity must be None.
    assert entity.surface is None
    # Name forwarded.
    assert entity.name == "probe"


# ============================================================
# CPU integration tests — full placement via real GenesisWorld
# ============================================================

def _build_cpu_world(unbatched=True, n_envs=4):
    if unbatched:
        return GenesisWorld()
    return GenesisWorld(batch_size=n_envs)


def test_unbatched_placement_bounds_and_parking():
    pool = _make_pool(4, scale=0.2)
    world = _build_cpu_world(unbatched=True)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    mgr = FloatingVisualPropsManager(
        world,
        name="float",
        object_pool=pool,
        n_objects_range=(2, 3),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
        z_offset_range=(0.15, 0.65),
        xy_margin=0.05,
        min_separation=0.10,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=0, reload=True)

    n_pool = len(pool)
    n_active = len(mgr.active_objects)
    n_active_e = len(mgr.active_entities)
    assert 2 <= n_active <= 3
    assert n_active == n_active_e
    assert mgr.active_mask.shape == (n_pool,)
    assert int(mgr.active_mask.sum()) == n_active

    # Active props: above the table (z > table_z + 0.15 - bbox_z)
    # but their world z >= table_pos[2] + 0.15 - max_bbox_radius.
    for i, entity in enumerate(mgr._loaded_entities):
        pos = entity.get_pos()
        if bool(mgr.active_mask[i]):
            # World-frame centre minus table surface should be
            # approximately in the requested z-offset range.
            z_above = float(pos[2].item()) - _TABLE_POS[2]
            # Entity origin offsets the bbox centre, so this is loose;
            # require it is comfortably above the surface and below
            # max + bbox extent.
            assert z_above > 0.0, pos
            # XY within table footprint plus margin tolerance.
            dx = abs(float(pos[0].item()) - _TABLE_POS[0])
            dy = abs(float(pos[1].item()) - _TABLE_POS[1])
            assert dx <= _TABLE_SIZE[0] / 2.0 + 0.05
            assert dy <= _TABLE_SIZE[1] / 2.0 + 0.05
        else:
            # Parked far below the table at the unique per-cell altitude.
            assert abs(float(pos[2].item()) - _PARK_Z) < 1e-6
    print(f"  PASS: unbatched n_active={n_active} bounds+parking OK")
    env.close()


def test_inactive_unique_parked_xy():
    pool = _make_pool(5, scale=0.2)
    world = _build_cpu_world()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    mgr = FloatingVisualPropsManager(
        world, object_pool=pool, n_objects_range=(0, 0),  # all parked
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=0, reload=True)

    xs = []
    ys = []
    for entity in mgr._loaded_entities:
        pos = entity.get_pos()
        xs.append(float(pos[0].item()))
        ys.append(float(pos[1].item()))
        assert abs(float(pos[2].item()) - _PARK_Z) < 1e-6
    # Primitives are centred ⇒ origin = lattice target ⇒ unique XY.
    assert len(set(zip(xs, ys))) == len(pool)
    # Parked quaternion is identity for all.
    park_q = torch.tensor(_PARK_QUAT, dtype=mgr._loaded_entities[0].get_quat().dtype)
    for entity in mgr._loaded_entities:
        assert torch.allclose(entity.get_quat(), park_q)
    env.close()


def test_min_separation_respected():
    pool = _make_pool(6, scale=0.15)
    world = _build_cpu_world()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    sep = 0.30  # generous to guarantee at least some separation
    mgr = FloatingVisualPropsManager(
        world, object_pool=pool, n_objects_range=(4, 4),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
        z_offset_range=(0.2, 0.5), xy_margin=0.05,
        min_separation=sep,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=1, reload=True)

    active_centres = []
    for i, entity in enumerate(mgr._loaded_entities):
        if bool(mgr.active_mask[i]):
            pos = entity.get_pos()
            # Use the per-object local bbox centre to recover the world
            # bbox centre (compensates the entity-origin offset back to
            # the placement target).
            lc = mgr._centers_t[i]
            centre = pos + lc
            active_centres.append(centre)

    # Every active pair should have centre-to-centre distance >= sep
    # (allowing a tiny numerical slack).  The bbox radii are additive
    # in the placement; we assert against ``sep`` itself since the
    # rejection test enforces that directly on the *target centres*.
    for i in range(len(active_centres)):
        for j in range(i + 1, len(active_centres)):
            d = float(
                torch.linalg.norm(active_centres[i] - active_centres[j]).item()
            )
            assert d + 1e-4 >= sep, (i, j, d, sep)
    env.close()


def test_seed_reproducibility():
    pool = _make_pool(4, scale=0.2)
    world = _build_cpu_world()
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    mgr = FloatingVisualPropsManager(
        world, object_pool=pool, n_objects_range=(2, 3),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=0, reload=True)

    env.reset(seed=42)
    pos_a = [e.get_pos().clone() for e in mgr._loaded_entities]
    mask_a = mgr.active_mask.clone()
    quat_a = [e.get_quat().clone() for e in mgr._loaded_entities]

    env.reset(seed=42)
    pos_b = [e.get_pos().clone() for e in mgr._loaded_entities]
    mask_b = mgr.active_mask.clone()
    quat_b = [e.get_quat().clone() for e in mgr._loaded_entities]

    assert torch.equal(mask_a, mask_b)
    for i in range(len(pool)):
        assert torch.allclose(pos_a[i], pos_b[i])
        assert torch.allclose(quat_a[i], quat_b[i])
    env.close()


# --- Batched ---------------------------------------------------------


def test_batched_placement_per_env():
    N_ENVS = 8
    pool = _make_pool(5, scale=0.2)
    world = _build_cpu_world(unbatched=False, n_envs=N_ENVS)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    mgr = FloatingVisualPropsManager(
        world, object_pool=pool, n_objects_range=(0, 3),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=0, reload=True)

    n_pool = len(pool)
    assert mgr.active_mask.shape == (n_pool, N_ENVS)

    # Per-env counts in [0, 3].
    for j in range(N_ENVS):
        cnt = int(mgr.active_mask[:, j].sum().item())
        assert 0 <= cnt <= 3, (j, cnt)

    # Likely some diversity of active masks across envs.
    masks = [tuple(mgr.active_mask[:, j].tolist()) for j in range(N_ENVS)]
    assert len(set(masks)) > 1

    # Active vs parked z check per env.
    for i, entity in enumerate(mgr._loaded_entities):
        pos = entity.get_pos()  # (N_ENVS, 3)
        for j in range(N_ENVS):
            if bool(mgr.active_mask[i, j]):
                assert float(pos[j, 2].item()) > _TABLE_POS[2] - 0.05
            else:
                assert abs(float(pos[j, 2].item()) - _PARK_Z) < 1e-6
    env.close()


def test_batched_masked_reset_preservation():
    N_ENVS = 8
    pool = _make_pool(5, scale=0.2)
    world = _build_cpu_world(unbatched=False, n_envs=N_ENVS)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    mgr = FloatingVisualPropsManager(
        world, object_pool=pool, n_objects_range=(1, 3),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=0, reload=True)

    before_pos = [e.get_pos().clone() for e in mgr._loaded_entities]
    before_quat = [e.get_quat().clone() for e in mgr._loaded_entities]
    mask_before = mgr.active_mask.clone()

    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[0, 4, 7]] = True
    env.reset(seed=99, mask=reset_mask)

    preserved = [j for j in range(N_ENVS) if not bool(reset_mask[j])]
    for i, entity in enumerate(mgr._loaded_entities):
        pos = entity.get_pos()
        quat = entity.get_quat()
        for j in preserved:
            assert torch.allclose(pos[j], before_pos[i][j], atol=1e-5)
            assert torch.allclose(quat[j], before_quat[i][j], atol=1e-5)
    for j in preserved:
        assert torch.equal(mgr.active_mask[:, j], mask_before[:, j])

    # Reset envs: their quats for active objects should be different in
    # at least one env in the high-probability case (we drew fresh
    # random SO(3) quaternions); on inactive objects, identity.
    changed = False
    for j in [0, 4, 7]:
        for i in range(len(pool)):
            before = before_pos[i][j]
            after = mgr._loaded_entities[i].get_pos()[j]
            if not torch.allclose(before, after, atol=1e-5):
                changed = True
                break
        if changed:
            break
    assert changed, "Masked reset envs did not change positions"
    env.close()


def test_batched_inactive_unique_parked_xy():
    N_ENVS = 6
    pool = _make_pool(5, scale=0.2)
    world = _build_cpu_world(unbatched=False, n_envs=N_ENVS)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    mgr = FloatingVisualPropsManager(
        world, object_pool=pool, n_objects_range=(1, 2),
        table_size=_TABLE_SIZE, table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=0, reload=True)

    for j in range(N_ENVS):
        xy = set()
        for i, entity in enumerate(mgr._loaded_entities):
            if not bool(mgr.active_mask[i, j]):
                p = entity.get_pos()[j]
                xy.add((round(float(p[0].item()), 6), round(float(p[1].item()), 6)))
                assert abs(float(p[2].item()) - _PARK_Z) < 1e-6
        # Should have at least two distinct parked XY cells when there is
        # more than one inactive object.
        assert len(xy) > 1, (j, xy)
    env.close()


# --- Nonzero table pose (position + yaw) transform ---------------


def _forward_rotate(mgr, local_xy):
    """Apply the manager's table_yaw + table_pos XY to *local_xy* ``(n, 2)``."""
    c_y, s_y = mgr._cos_yaw, mgr._sin_yaw
    wx = mgr.table_pos[0] + local_xy[:, 0] * c_y - local_xy[:, 1] * s_y
    wy = mgr.table_pos[1] + local_xy[:, 0] * s_y + local_xy[:, 1] * c_y
    return torch.stack([wx, wy], dim=-1)


def _inverse_rotate(mgr, world_xy):
    """Inverse of :func:`_forward_rotate` — world→table-local XY ``(n, 2)``."""
    c_y, s_y = mgr._cos_yaw, mgr._sin_yaw
    ox = world_xy[:, 0] - mgr.table_pos[0]
    oy = world_xy[:, 1] - mgr.table_pos[1]
    lx = ox * c_y + oy * s_y
    ly = -ox * s_y + oy * c_y
    return torch.stack([lx, ly], dim=-1)


def _world_bbox_centres(mgr, env_idx=None):
    """Return world-frame bbox centres of active props for a single env.

    Uses ``quaternion_apply`` to undo the bbox-centre compensation baked
    into each entity's origin so the resulting point is the *sampled*
    bbox-centre target — independent of the random-quat orientation
    policy.  When *env_idx* is None (unbatched) the entities' poses are
    ``(3,)`` / ``(4,)``; otherwise ``(n_envs, 3)`` / ``(n_envs, 4)``.
    """
    from xbarray.transformations.rotation_conversions.pytorch import (
        quaternion_apply,
    )
    centres = []
    for i, entity in enumerate(mgr._loaded_entities):
        if not bool(mgr.active_mask[i] if env_idx is None
                    else mgr.active_mask[i, env_idx]):
            continue
        pos = entity.get_pos()
        quat = entity.get_quat()
        if env_idx is not None:
            pos = pos[env_idx]
            quat = quat[env_idx]
        local_c = mgr._centers_t[i].to(pos.dtype).to(pos.device)
        centre = pos + quaternion_apply(
            quat.unsqueeze(0).to(pos.dtype), local_c.unsqueeze(0),
        ).squeeze(0)
        centres.append(centre)
    return torch.stack(centres, dim=0) if centres else torch.empty(0, 3)


def test_unbatched_nonzero_table_pos_and_yaw_transform():
    """Active-comp bbox centre must land inside the *rotated* footprint
    around ``table_pos`` for a nonzero table position **and** nonzero yaw.

    Prior to the fix, ``cand_xy`` was treated as world XY directly — so a
    nonzero ``table_pos`` offset or a ``table_yaw`` rotation was silently
    dropped and props clustered at the world origin / un-rotated box.
    """
    table_pos = (1.5, -0.7, 0.65)
    table_yaw = math.pi / 4.0
    table_size = (0.8, 0.5)
    xy_margin = 0.05
    zmin, zmax = 0.20, 0.40

    pool = _make_pool(4, scale=0.15)
    world = _build_cpu_world(unbatched=True)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(
        world, name="table", pos=table_pos, yaw=table_yaw,
    )
    mgr = FloatingVisualPropsManager(
        world, object_pool=pool, n_objects_range=(3, 4),
        table_size=table_size, table_pos=table_pos, table_yaw=table_yaw,
        z_offset_range=(zmin, zmax), xy_margin=xy_margin,
        min_separation=0.05, random_quaternion=False,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=0, reload=True)

    n_active = len(mgr.active_objects)
    assert n_active >= 1, n_active

    half_x = table_size[0] / 2.0 - xy_margin
    half_y = table_size[1] / 2.0 - xy_margin
    centres = _world_bbox_centres(mgr)  # (n_active, 3)
    assert centres.shape[0] == n_active

    # Every active bbox centre must invert-rotate back *into* the
    # margin-shrunk local footprint and sit within the requested z band
    # above the table surface.
    local = _inverse_rotate(mgr, centres[:, :2])
    assert torch.all(local[:, 0].abs() <= half_x + 1e-4), local
    assert torch.all(local[:, 1].abs() <= half_y + 1e-4), local
    z_above = centres[:, 2] - table_pos[2]
    assert torch.all(z_above >= zmin - 1e-4), z_above
    assert torch.all(z_above <= zmax + 1e-4), z_above

    # Nonzero pos/yaw must shift the bbox centres well away from the
    # origin — guards against the old "treat local as world" regression.
    assert torch.all(centres[:, 0].abs() > 0.5), centres
    assert torch.all(centres[:, 1].abs() > 0.5), centres
    env.close()


def test_batched_nonzero_table_pos_and_yaw_transform():
    """Same as the unbatched test but across a batched world."""
    N_ENVS = 6
    table_pos = (-0.9, 1.2, 0.65)
    table_yaw = math.pi / 3.0
    table_size = (0.7, 0.4)
    xy_margin = 0.05
    zmin, zmax = 0.15, 0.35

    pool = _make_pool(5, scale=0.12)
    world = _build_cpu_world(unbatched=False, n_envs=N_ENVS)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(
        world, name="table", pos=table_pos, yaw=table_yaw,
    )
    mgr = FloatingVisualPropsManager(
        world, object_pool=pool, n_objects_range=(2, 3),
        table_size=table_size, table_pos=table_pos, table_yaw=table_yaw,
        z_offset_range=(zmin, zmax), xy_margin=xy_margin,
        min_separation=0.05, random_quaternion=False,
    )
    env = WorldEnv(world, [plane, table, mgr], render_mode=None)
    env.reset(seed=3, reload=True)

    half_x = table_size[0] / 2.0 - xy_margin
    half_y = table_size[1] / 2.0 - xy_margin

    total_checked = 0
    for j in range(N_ENVS):
        n_active_j = int(mgr.active_mask[:, j].sum().item())
        if n_active_j == 0:
            continue
        centres = _world_bbox_centres(mgr, env_idx=j)
        assert centres.shape[0] == n_active_j
        local = _inverse_rotate(mgr, centres[:, :2])
        assert torch.all(local[:, 0].abs() <= half_x + 1e-4), (j, local)
        assert torch.all(local[:, 1].abs() <= half_y + 1e-4), (j, local)
        z_above = centres[:, 2] - table_pos[2]
        assert torch.all(z_above >= zmin - 1e-4), (j, z_above)
        assert torch.all(z_above <= zmax + 1e-4), (j, z_above)
        # Off-origin guard for every env.
        assert torch.all(centres[:, 0].abs() > 0.5), (j, centres)
        assert torch.all(centres[:, 1].abs() > 0.5), (j, centres)
        total_checked += n_active_j
    assert total_checked > 0, "No active props across the whole batch"
    env.close()


# --- Geometry cache ------------------------------------------------

def test_geometry_cache_bounding_radii():
    pool = _make_pool(4, scale=0.2)
    world = _build_cpu_world()
    mgr = FloatingVisualPropsManager(world, object_pool=pool, n_objects_range=(0, 2))
    mgr._ensure_geometry_cache()
    # bound_r == half-diagonal of local AABB.
    expected = (
        torch.linalg.norm(mgr._dims_t, dim=-1) / 2.0
    )
    assert torch.allclose(mgr._bound_r_t, expected)
    # max bound radius == max over pool.
    assert abs(mgr._max_bound_radius - float(mgr._bound_r_t.max().item())) < 1e-6
    # Park quats all _PARK_QUAT.
    assert mgr._park_quats_t.shape == (4, 4)
    assert torch.allclose(
        mgr._park_quats_t,
        torch.tensor([_PARK_QUAT] * 4, dtype=mgr._park_quats_t.dtype,
                     device=mgr._park_quats_t.device),
    )


def test_geometry_cache_reuse_signature():
    pool = _make_pool(3, scale=0.2)
    world = _build_cpu_world()
    mgr = FloatingVisualPropsManager(world, object_pool=pool, n_objects_range=(0, 2))
    mgr._ensure_geometry_cache()
    ref = mgr._park_positions_t
    sig = mgr._pool_signature_cached
    # Second ensure with identical pool ⇒ reuse without rebuild.
    mgr._ensure_geometry_cache()
    assert mgr._pool_signature_cached == sig
    assert mgr._park_positions_t is ref
    # Replace an entry ⇒ invalidation.
    mgr.object_pool[1] = _make_pool(1, scale=0.2)[0]
    mgr._ensure_geometry_cache()
    assert mgr._pool_signature_cached != sig


# ============================================================
# Visual texture binding — GSO / YCB / generic fallback
# ============================================================
#
# ``FloatingVisualPropsManager._load_visual_entity`` must build a
# ``gs.surfaces.Plastic(diffuse_texture=gs.textures.ImageTexture(...))``
# surface when the descriptor exposes an existing
# ``visual_texture_path`` (GSO ``texture.png`` / YCB ``texture_map.png``)
# so the asset's intrinsic texture survives the loss of its native
# MJCF/URDF material binding, while a generic :class:`MeshObject`
# without a texture path falls back to ``surface=None``.  Genesis is
# mocked end-to-end so no scene is built; only the surface-construction
# call graph is asserted.

class _TextureSpy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _PlasticSpy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.diffuse_texture = kwargs.get("diffuse_texture")


def _make_textured_desc(mesh_file, texture_path, name="textured_obj",
                       scale=(1.0, 1.0, 1.0), texture_exists=True):
    """A minimal :class:`MeshObject`-shaped stub with a texture path."""
    desc = SimpleNamespace(
        mesh_file=str(mesh_file),
        scale=scale,
        visual_texture_path=str(texture_path) if texture_path is not None else None,
        name=name,
    )
    return desc, (texture_path if texture_exists else None)


def _setup_gs_mocks(monkeypatch):
    """Replace ``gs.morphs.Mesh`` / ``gs.textures.ImageTexture`` /
    ``gs.surfaces.Plastic`` with spies; returns the three capture lists.

    Each capture list holds the **constructed instance** (its ``.kwargs``
    attribute exposes the construction kwargs) so callers can assert
    identity against what was forwarded to ``world.scene.add_entity``.
    """
    import genesis as gs

    mesh_captured = []
    texture_captured = []
    plastic_captured = []

    class _Mesh:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            mesh_captured.append(self)

    class _ImageTexture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            texture_captured.append(self)

    class _Plastic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.diffuse_texture = kwargs.get("diffuse_texture")
            plastic_captured.append(self)

    monkeypatch.setattr(gs, "morphs", SimpleNamespace(Mesh=_Mesh))
    monkeypatch.setattr(gs, "textures", SimpleNamespace(ImageTexture=_ImageTexture))
    monkeypatch.setattr(gs, "surfaces", SimpleNamespace(Plastic=_Plastic))
    return mesh_captured, texture_captured, plastic_captured


def test_textured_descriptor_builds_image_texture_surface(tmp_path, monkeypatch):
    """A descriptor whose ``visual_texture_path`` points at a real
    file yields a ``Plastic(diffuse_texture=ImageTexture(image_path=...))``
    surface — the GSO/YCB visual-texture preservation path."""
    tex_file = tmp_path / "texture.png"
    tex_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # tiny placeholder
    obj = tmp_path / "model.obj"
    obj.write_text("# stub mesh\n")

    mesh_captured, texture_captured, plastic_captured = _setup_gs_mocks(monkeypatch)

    fake_world = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=None,
        scene=_FakeScene(),
    )
    mgr = FloatingVisualPropsManager(
        fake_world, object_pool=[], n_objects_range=(0, 0),
    )

    desc, _ = _make_textured_desc(obj, tex_file, texture_exists=True)
    entity = mgr._load_visual_entity(desc, pos=(0, 0, 0), name_override="probe")

    # One Mesh morph, one ImageTexture, one Plastic surface.
    assert len(mesh_captured) == 1
    assert len(texture_captured) == 1
    assert len(plastic_captured) == 1
    assert texture_captured[0].kwargs["image_path"] == str(tex_file)
    assert (
        plastic_captured[0].diffuse_texture is texture_captured[0]
    )
    assert plastic_captured[0].kwargs["diffuse_texture"] is texture_captured[0]
    # The surface forwarded to add_entity is the Plastic spy (not None).
    assert entity.surface is not None
    assert entity.surface is plastic_captured[0]
    # Morph args still satisfy the visual-only contract.
    kw = mesh_captured[0].kwargs
    assert kw["fixed"] is True
    assert kw["collision"] is False
    assert kw["visualization"] is True
    assert kw["decimate"] is False
    assert kw["file"] == str(obj)


def test_generic_mesh_without_texture_falls_back_to_surface_none(monkeypatch):
    """A generic :class:`MeshObject`-shaped descriptor with
    ``visual_texture_path=None`` must keep ``surface=None`` (Genesis
    default surface) — no Plastic / ImageTexture construction."""
    obj = _BOX_OBJ
    mesh_captured, texture_captured, plastic_captured = _setup_gs_mocks(monkeypatch)

    fake_world = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=None,
        scene=_FakeScene(),
    )
    mgr = FloatingVisualPropsManager(
        fake_world, object_pool=[], n_objects_range=(0, 0),
    )

    # Use a real MeshObject — visual_texture_path defaults to None.
    desc = MeshObject(mesh_file=obj, scale=0.25, name="plain_box")
    assert desc.visual_texture_path is None
    entity = mgr._load_visual_entity(desc, name_override="probe")

    assert len(mesh_captured) == 1
    assert texture_captured == []
    assert plastic_captured == []
    assert entity.surface is None


def test_textured_descriptor_missing_file_falls_back_to_none(tmp_path, monkeypatch):
    """If the texture file is absent (e.g. partial asset download that
    the manager did not get a chance to repair), gracefully fall back to
    ``surface=None`` instead of raising — preserves the prior
    visual-only contract for partial-download states."""
    obj = tmp_path / "model.obj"
    obj.write_text("# stub\n")
    phantom_tex = tmp_path / "missing.png"

    mesh_captured, texture_captured, plastic_captured = _setup_gs_mocks(monkeypatch)

    fake_world = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=None,
        scene=_FakeScene(),
    )
    mgr = FloatingVisualPropsManager(
        fake_world, object_pool=[], n_objects_range=(0, 0),
    )
    desc, _ = _make_textured_desc(obj, phantom_tex, texture_exists=False)
    desc.visual_texture_path = str(phantom_tex)  # path string is set, but file absent
    entity = mgr._load_visual_entity(desc, name_override="probe")

    assert mesh_captured
    assert texture_captured == []
    assert plastic_captured == []
    assert entity.surface is None


# --- GSO / YCB descriptors expose the right texture path ------------

def test_gso_object_exposes_texture_png(gso_assets_dir, gso_fake_object, no_download):
    """A :class:`GSOObject`'s ``visual_texture_path`` points at
    ``asset_dir/texture.png`` so floating-props rendering can recover the
    GSO texture the MJCF would normally bind."""
    from unienv_genesis_collection.objects.hub_objects.gso_object import GSOObject

    obj = GSOObject(_GSO_SAMPLE_ID)
    assert obj.asset_dir == gso_assets_dir / "objects" / "gso" / _GSO_SAMPLE_ID
    assert obj.visual_texture_path == str(obj.asset_dir / "texture.png")
    # And the texture file is actually one of the required files.
    assert (obj.asset_dir / "texture.png") in obj._required_files()


def test_ycb_object_exposes_texture_map_png(ycb_assets_dir, ycb_fake_banana, no_download):
    """A :class:`YCBObject`'s ``visual_texture_path`` points at
    ``asset_dir/texture_map.png`` so floating-props rendering can recover
    the YCB texture the URDF/MTL would normally bind."""
    from unienv_genesis_collection.objects.hub_objects.ycb_object import YCBObject

    obj = YCBObject("banana")
    assert obj.asset_dir == ycb_assets_dir / "objects" / "ycb" / "banana"
    assert obj.visual_texture_path == str(obj.asset_dir / "texture_map.png")
    assert (obj.asset_dir / "texture_map.png") in obj._required_files()


def test_meshobject_default_visual_texture_path_is_none():
    """Plain :class:`MeshObject` (no Hub asset) defaults to
    ``visual_texture_path=None`` so the floating-props manager keeps
    ``surface=None`` when loading it."""
    plain = MeshObject(mesh_file=_BOX_OBJ, scale=0.25, name="plain")
    assert plain.visual_texture_path is None


# --- End-to-end (mocked Genesis): GSO floating load uses the texture --

def test_floating_load_of_gso_builds_textured_surface(
    gso_assets_dir, gso_fake_object, no_download, monkeypatch,
):
    """Calling ``FloatingVisualPropsManager._load_visual_entity`` on a
    real :class:`GSOObject` (asset dir materialised on disk) produces a
    ``Plastic(diffuse_texture=ImageTexture(image_path=<asset_dir>/texture.png))``
    surface — i.e. the manager re-attaches GSO's ``texture.png`` to the
    bare ``model.obj`` so the floating prop keeps its visual texture."""
    from unienv_genesis_collection.objects.hub_objects.gso_object import GSOObject

    gso = GSOObject(_GSO_SAMPLE_ID)
    gso.ensure_assets()
    assert (gso.asset_dir / "texture.png").is_file()

    mesh_captured, texture_captured, plastic_captured = _setup_gs_mocks(monkeypatch)

    fake_world = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=None,
        scene=_FakeScene(),
    )
    mgr = FloatingVisualPropsManager(
        fake_world, object_pool=[], n_objects_range=(0, 0),
    )
    entity = mgr._load_visual_entity(gso, name_override="gso_probe")

    # ImageTexture-bound Plastic surface was constructed with the
    # asset_dir/texture.png path.
    assert texture_captured and texture_captured[0].kwargs["image_path"] == str(
        gso.asset_dir / "texture.png"
    )
    assert plastic_captured and plastic_captured[0].diffuse_texture is not None
    # The morph still reads the raw visual OBJ.
    assert mesh_captured[0].kwargs["file"] == str(gso.asset_dir / "model.obj")
    assert mesh_captured[0].kwargs["fixed"] is True
    assert mesh_captured[0].kwargs["collision"] is False
    assert mesh_captured[0].kwargs["visualization"] is True
    assert mesh_captured[0].kwargs["decimate"] is False
    # The Plastic surface (not None) was forwarded to add_entity.
    assert entity.surface is plastic_captured[0]


if __name__ == "__main__":
    import sys
    funcs = [
        ("shoemake unit", test_shoemake_quaternions_unit_norm_uniform),
        ("shoemake determinism", test_shoemake_quaternions_determinism),
        ("shoemake empty", test_shoemake_quaternions_empty_returns_empty),
        ("pool validation", test_pool_size_validation),
        ("park clearance", test_bad_park_clearance_rejected),
        ("xy margin", test_bad_xy_margin_rejected),
        ("min sep", test_bad_min_separation_rejected),
        ("z offset", test_bad_z_offset_range_rejected),
        ("morph args", test_morph_args_fixed_no_collision_no_decimate_surface_none),
        ("unbatched bounds", test_unbatched_placement_bounds_and_parking),
        ("inactive unique XY", test_inactive_unique_parked_xy),
        ("min separation", test_min_separation_respected),
        ("seed repro", test_seed_reproducibility),
        ("batched", test_batched_placement_per_env),
        ("masked reset", test_batched_masked_reset_preservation),
        ("batched inactive unique", test_batched_inactive_unique_parked_xy),
        ("cache radii", test_geometry_cache_bounding_radii),
        ("cache reuse", test_geometry_cache_reuse_signature),
        ("unbatched nonzero pos/yaw", test_unbatched_nonzero_table_pos_and_yaw_transform),
        ("batched nonzero pos/yaw", test_batched_nonzero_table_pos_and_yaw_transform),
    ]
    failures = 0
    for name, fn in funcs:
        print(f"=== {name} ===")
        try:
            fn()
            print("  OK")
        except Exception as e:
            failures += 1
            print(f"  FAIL: {e!r}")
    sys.exit(1 if failures else 0)