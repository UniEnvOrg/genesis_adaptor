"""Offline tests for the Hub-backed articulated object base classes.

Fully offline: synthetic per-object asset directories are built in
``tmp_path`` and ``asset_path.ASSETS_DIR`` is repointed at it (the hub
classes read ``ASSETS_DIR`` via the module at construction, like
``HubAssetMeshObject``); ``asset_path.download_assets`` is stubbed so no Hub
access happens.  No Genesis scene is built — ``load_into_scene`` is
exercised by the downstream S2/S4 smoke tests on GPU.

Covers:
  - RoboCasaFixture / ReplicaCADFurniture construction (offline-safe, paths)
  - _required_files: description file + referenced meshes/textures +
    MJCF <include> chain (recursive, with compiler meshdir / texturedir)
  - _ensure_assets: no-op when complete, FileNotFoundError listing missing
    files when incomplete, allow_patterns of the interrupted re-download
  - _load_link_meshes_at_reference: per-link meshes + union bbox sanity
    (offline FK) for both MJCF and URDF synthetic assets
  - freejoint/fixed load-time warning helper (no Genesis needed)
  - Catalog helpers: list_robocasa_objects / list_replicacad_objects tag
    filtering and make_*_pool behaviour
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture import (
    ROBOCASA_ASSETS_ROOT,
    RoboCasaFixture,
    get_robocasa_catalog,
    list_robocasa_objects,
    make_robocasa_pool,
)
from unienv_genesis_collection.objects.hub_objects.articulated.replicacad_furniture import (
    REPLICACAD_ASSETS_ROOT,
    ReplicaCADFurniture,
    get_replicacad_catalog,
    list_replicacad_objects,
    make_replicacad_pool,
)
from unienv_genesis_collection.objects.hub_objects.articulated.hub_articulated_object import (
    HubAssetArticulatedObject,
    _mjcf_required_files,
)

# ------------------------------------------------------------------
# Synthetic assets
# ------------------------------------------------------------------

# RoboCasa fixture with a compiler meshdir, a texture asset and a top-level
# <include> (its own mesh asset, referenced by a geom) — exercises the
# recursive include handling and the meshdir/texturedir split.
_CABINET_MJCF = """\
<mujoco model="unit_cabinet">
    <compiler meshdir="meshes" />
    <include file="shared_assets.xml" />
    <asset>
        <texture type="2d" name="cab_tex" file="texture.png" />
        <material name="cab_mat" texture="cab_tex" />
        <mesh name="frame_mesh" file="frame.obj" />
        <mesh name="door_mesh" file="door.obj" />
    </asset>
    <worldbody>
        <body name="frame" pos="0 0 0">
            <geom name="frame_geom" type="mesh" mesh="frame_mesh" material="cab_mat" />
            <body name="door" pos="0.2 0 0.1">
                <joint name="door_hinge" type="hinge" axis="0 0 1" />
                <geom name="door_geom" type="mesh" mesh="door_mesh" />
                <geom name="shelf_geom" type="mesh" mesh="shelf_mesh" />
            </body>
        </body>
    </worldbody>
</mujoco>
"""

_CABINET_INCLUDE = """\
<mujocoinclude model="shared_assets">
    <asset>
        <mesh name="shelf_mesh" file="shelf.obj" />
    </asset>
</mujocoinclude>
"""

# Flat single-file MJCF fixture used for the offline-FK check: a welded frame
# with a drawer sliding along +x.
_DRAWER_MJCF = """\
<mujoco model="unit_drawer">
    <asset>
        <mesh name="frame_mesh" file="frame.obj" />
        <mesh name="box_mesh" file="drawer.obj" />
    </asset>
    <worldbody>
        <body name="frame" pos="0 0 0">
            <geom name="frame_geom" type="mesh" mesh="frame_mesh" />
            <body name="drawer_box" pos="0.2 0 0.05">
                <joint name="drawer_slide" type="slide" axis="1 0 0" />
                <geom name="drawer_geom" type="mesh" mesh="box_mesh" />
            </body>
        </body>
    </worldbody>
</mujoco>
"""

# ReplicaCAD furniture URDF: a welded frame with a door hinged about +z.
_FRIDGE_URDF = """\
<robot name="fridge">
    <link name="frame">
        <visual>
            <geometry><mesh filename="frame.obj" scale="1 1 1" /></geometry>
        </visual>
    </link>
    <link name="door">
        <visual>
            <geometry><mesh filename="door.obj" scale="1 1 1" /></geometry>
            <origin xyz="0.2 0 0.05" rpy="0 0 0" />
        </visual>
    </link>
    <joint name="door_hinge" type="revolute">
        <parent link="frame" />
        <child link="door" />
        <axis xyz="0 0 1" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <limit lower="0" upper="1.5707963" />
    </joint>
</robot>
"""

# URDF used only for the required-file enumeration (visual + collision refs).
_CABINET_URDF = """\
<robot name="base_cabinet">
    <link name="frame">
        <visual>
            <geometry><mesh filename="frame.obj" scale="1 1 1" /></geometry>
        </visual>
        <collision>
            <geometry><mesh filename="collision_0.stl" scale="1 1 1" /></geometry>
        </collision>
        <collision>
            <geometry><mesh filename="collision_1.stl" scale="1 1 1" /></geometry>
        </collision>
    </link>
</robot>
"""

# MJCF declaring a <freejoint/> root (a movable variant) for the warning check.
_FREEJOINT_MJCF = """\
<mujoco model="movable_variant">
    <asset><mesh name="frame_mesh" file="frame.obj" /></asset>
    <worldbody>
        <body name="frame" pos="0 0 0">
            <freejoint />
            <geom name="frame_geom" type="mesh" mesh="frame_mesh" />
        </body>
    </worldbody>
</mujoco>
"""

_ROBOCASA_CATALOG = {
    "unit_cabinet": {
        "language_annotations": ["furniture", "a kitchen cabinet"],
        "tags": ["furniture", "articulated", "kitchen"],
    },
    "unit_drawer": {
        "language_annotations": ["furniture", "a kitchen drawer"],
        "tags": ["furniture", "articulated", "kitchen", "drawer"],
    },
    "counter": {
        "language_annotations": ["furniture", "a kitchen counter"],
        "tags": ["furniture", "kitchen"],
    },
    "movable_variant": {
        "language_annotations": ["furniture", "a movable cabinet"],
        "tags": ["furniture", "articulated"],
    },
}

_REPLICACAD_CATALOG = {
    "fridge": {
        "language_annotations": ["furniture", "a fridge"],
        "tags": ["furniture", "articulated", "kitchen"],
    },
    "base_cabinet": {
        "language_annotations": ["furniture", "a cabinet"],
        "tags": ["furniture", "articulated"],
    },
    "table": {
        "language_annotations": ["furniture", "a table"],
        "tags": ["furniture"],
    },
}


def _write_box_mesh(path: Path, size: float = 0.1) -> None:
    """Write a centered axis-aligned box mesh (OBJ) to *path*."""
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(size, size, size)).export(path)


def _make_robocasa_cabinet(assets_dir: Path) -> Path:
    """Complete 'unit_cabinet' directory (MJCF + include + meshes + texture)."""
    obj_dir = assets_dir / ROBOCASA_ASSETS_ROOT / "unit_cabinet"
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.xml").write_text(_CABINET_MJCF)
    (obj_dir / "shared_assets.xml").write_text(_CABINET_INCLUDE)
    (obj_dir / "texture.png").write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")
    _write_box_mesh(obj_dir / "meshes" / "frame.obj", size=0.2)
    _write_box_mesh(obj_dir / "meshes" / "door.obj")
    _write_box_mesh(obj_dir / "meshes" / "shelf.obj")
    return obj_dir


def _make_robocasa_drawer(assets_dir: Path, object_id: str = "unit_drawer") -> Path:
    """Complete flat 'unit_drawer' directory (frame.obj + drawer.obj)."""
    obj_dir = assets_dir / ROBOCASA_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.xml").write_text(_DRAWER_MJCF)
    _write_box_mesh(obj_dir / "frame.obj", size=0.2)
    _write_box_mesh(obj_dir / "drawer.obj")
    return obj_dir


def _make_replicacad_fridge(assets_dir: Path) -> Path:
    """Complete 'fridge' directory (URDF + box meshes)."""
    obj_dir = assets_dir / REPLICACAD_ASSETS_ROOT / "fridge"
    obj_dir.mkdir(parents=True)
    (obj_dir / "fridge.urdf").write_text(_FRIDGE_URDF)
    _write_box_mesh(obj_dir / "frame.obj", size=0.2)
    _write_box_mesh(obj_dir / "door.obj")
    return obj_dir


def _make_replicacad_cabinet(assets_dir: Path) -> Path:
    """Complete 'base_cabinet' directory (URDF + placeholder meshes)."""
    obj_dir = assets_dir / REPLICACAD_ASSETS_ROOT / "base_cabinet"
    obj_dir.mkdir(parents=True)
    (obj_dir / "base_cabinet.urdf").write_text(_CABINET_URDF)
    for name in ("frame.obj", "collision_0.stl", "collision_1.stl"):
        (obj_dir / name).write_text("placeholder")
    return obj_dir


def _make_robocasa_movable(assets_dir: Path) -> Path:
    """'movable_variant': MJCF whose root declares a <freejoint/>."""
    obj_dir = assets_dir / ROBOCASA_ASSETS_ROOT / "movable_variant"
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.xml").write_text(_FREEJOINT_MJCF)
    _write_box_mesh(obj_dir / "frame.obj", size=0.2)
    return obj_dir


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------

@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir holding the two synthetic catalogs."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    for dataset, entries in (
        (ROBOCASA_ASSETS_ROOT, _ROBOCASA_CATALOG),
        (REPLICACAD_ASSETS_ROOT, _REPLICACAD_CATALOG),
    ):
        catalog_file = tmp_path / dataset / "catalog.json"
        catalog_file.parent.mkdir(parents=True)
        catalog_file.write_text(json.dumps(entries))
    return tmp_path


@pytest.fixture
def no_download(monkeypatch):
    """Fail the test if a Hub download is attempted; records calls."""
    calls = []

    def _fail(**kwargs):
        calls.append(kwargs)
        raise AssertionError(f"Unexpected download attempt: {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail)
    return calls


# ------------------------------------------------------------------
# Construction (offline-safe)
# ------------------------------------------------------------------

def test_robocasa_construction_offline_safe(assets_dir, no_download):
    fixture = RoboCasaFixture("unit_cabinet")
    expected_dir = assets_dir / ROBOCASA_ASSETS_ROOT / "unit_cabinet"
    assert fixture.asset_relative_dir == f"{ROBOCASA_ASSETS_ROOT}/unit_cabinet"
    assert fixture.asset_dir == expected_dir
    assert fixture.description_file == expected_dir / "model.xml"
    assert fixture.object_id == "unit_cabinet"
    assert fixture.name == "unit_cabinet"
    assert fixture.fixed is True  # furniture default
    assert fixture.reference_qpos == {}
    assert fixture.language_annotations == ("furniture", "a kitchen cabinet")
    assert fixture.scale == 1.0
    assert fixture._assets_ready is False
    assert not no_download  # no Hub access at construction


def test_replicacad_construction_offline_safe(assets_dir, no_download):
    item = ReplicaCADFurniture("fridge")
    expected_dir = assets_dir / REPLICACAD_ASSETS_ROOT / "fridge"
    assert item.asset_dir == expected_dir
    assert item.description_file == expected_dir / "fridge.urdf"
    assert item.fixed is True
    assert item.reference_qpos == {}
    assert item.language_annotations == ("furniture", "a fridge")
    assert not no_download


def test_construction_overrides_and_scale(assets_dir, no_download):
    fixture = RoboCasaFixture(
        "unit_drawer", name="my_drawer", language_annotations=["the drawer"],
        scale=0.5, reference_qpos={"drawer_slide": 0.1},
    )
    assert fixture.name == "my_drawer"
    assert fixture.language_annotations == ("the drawer",)
    assert fixture.scale == 0.5
    assert fixture._morph_scale == 0.5
    assert fixture.reference_qpos == {"drawer_slide": 0.1}

    item = ReplicaCADFurniture("fridge", scale=(2.0, 2.0, 2.0))
    assert item.scale == (2.0, 2.0, 2.0)
    assert item._morph_scale == (2.0, 2.0, 2.0)  # per-axis tuple reaches the morph


def test_unknown_ids_raise(assets_dir):
    with pytest.raises(KeyError, match="not_a_fixture"):
        RoboCasaFixture("not_a_fixture")
    with pytest.raises(KeyError, match="not_an_item"):
        ReplicaCADFurniture("not_an_item")


def test_robocasa_rejects_anisotropic_scale(assets_dir):
    # MJCF morphs are isotropic only — same constraint GSOObject encodes.
    with pytest.raises(ValueError, match="uniform"):
        RoboCasaFixture("unit_drawer", scale=(1.0, 2.0, 3.0))


def test_required_files_unknown_description_format_raises(assets_dir, no_download):
    # The base is abstract on purpose (format-specific load_into_scene);
    # a minimal concrete stub isolates the shared machinery.
    class _StubFixture(HubAssetArticulatedObject):
        def load_into_scene(self, *args, **kwargs):
            raise NotImplementedError

    base = _StubFixture(
        asset_relative_dir="objects/robocasa/unit_drawer",
        description_file_name="model.txt",
    )
    (base.asset_dir).mkdir(parents=True, exist_ok=True)
    (base.description_file).write_text("not a description file")
    with pytest.raises(ValueError, match="Unsupported description file format"):
        base._required_files()


# ------------------------------------------------------------------
# _required_files
# ------------------------------------------------------------------

def test_mjcf_required_files_includes_all_references(assets_dir):
    _make_robocasa_cabinet(assets_dir)
    fixture = RoboCasaFixture("unit_cabinet")
    required = fixture._required_files()
    assert set(required) == {
        fixture.description_file,
        # <texture file=…> resolves against the (absolutized) texturedir,
        # i.e. the object directory itself — NOT compiler meshdir.
        fixture.asset_dir / "texture.png",
        # <mesh file=…> resolves against compiler meshdir.
        fixture.asset_dir / "meshes" / "frame.obj",
        fixture.asset_dir / "meshes" / "door.obj",
        # <include file=…> — spliced in by Genesis at load time.
        fixture.asset_dir / "shared_assets.xml",
        # …and the mesh referenced *inside* the included file (its
        # <mesh file=…> is rewritten to be relative to the include's dir).
        fixture.asset_dir / "meshes" / "shelf.obj",
    }
    assert required[0] == fixture.description_file
    assert len(required) == len(set(required))  # de-duplicated


def test_urdf_required_files_includes_all_references(assets_dir):
    _make_replicacad_cabinet(assets_dir)
    item = ReplicaCADFurniture("base_cabinet")
    required = item._required_files()
    assert set(required) == {
        item.description_file,
        item.asset_dir / "frame.obj",
        item.asset_dir / "collision_0.stl",
        item.asset_dir / "collision_1.stl",
    }


def test_required_files_without_description_file(assets_dir, no_download):
    # Until the description file is downloaded the referenced set cannot be
    # known — only the description file itself is required.
    fixture = RoboCasaFixture("unit_cabinet")
    assert fixture._required_files() == [fixture.description_file]
    assert fixture._missing_files() == [fixture.description_file]


def test_mjcf_required_files_nested_include_accumulates_parent_path(tmp_path):
    # Mirrors genesis.utils.mjcf.build_model: an <include> nested inside an
    # already-included file resolves against that parent include's *own path*.
    (tmp_path / "main.xml").write_text(
        '<mujoco model="main"><include file="shared.xml" /></mujoco>'
    )
    (tmp_path / "shared.xml").write_text(
        '<mujocoinclude model="shared"><include file="nested.xml" /></mujocoinclude>'
    )
    files = _mjcf_required_files(tmp_path / "main.xml")
    assert tmp_path / "shared.xml" in files
    assert tmp_path / "shared.xml" / "nested.xml" in files


# ------------------------------------------------------------------
# _ensure_assets
# ------------------------------------------------------------------

def test_ensure_assets_complete_is_noop(assets_dir, no_download):
    _make_robocasa_cabinet(assets_dir)
    _make_replicacad_fridge(assets_dir)
    fixture = RoboCasaFixture("unit_cabinet")
    item = ReplicaCADFurniture("fridge")
    fixture._ensure_assets()
    item._ensure_assets()
    assert fixture._assets_ready and item._assets_ready
    # Second call short-circuits.
    fixture._ensure_assets()
    assert not no_download


def test_ensure_assets_missing_raises_listing_files(assets_dir, monkeypatch):
    obj_dir = _make_robocasa_cabinet(assets_dir)
    (obj_dir / "texture.png").unlink()
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    fixture = RoboCasaFixture("unit_cabinet")
    with pytest.raises(FileNotFoundError, match="texture.png") as excinfo:
        fixture._ensure_assets()
    assert "robocasa" in str(excinfo.value) and "unit_cabinet" in str(excinfo.value)
    assert fixture._assets_ready is False


def test_ensure_assets_urdf_missing_mesh_raises(assets_dir, monkeypatch):
    obj_dir = _make_replicacad_fridge(assets_dir)
    (obj_dir / "door.obj").unlink()
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    item = ReplicaCADFurniture("fridge")
    with pytest.raises(FileNotFoundError, match="door.obj"):
        item._ensure_assets()
    assert item._assets_ready is False


def test_ensure_assets_missing_description_file_raises(assets_dir, monkeypatch):
    # Nothing downloaded at all: the description file itself is missing.
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    fixture = RoboCasaFixture("unit_cabinet")
    with pytest.raises(FileNotFoundError, match="model.xml"):
        fixture._ensure_assets()


def test_ensure_assets_partial_dir_triggers_download(assets_dir, monkeypatch):
    obj_dir = _make_robocasa_cabinet(assets_dir)
    (obj_dir / "meshes" / "door.obj").unlink()
    calls = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        _write_box_mesh(obj_dir / "meshes" / "door.obj")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    RoboCasaFixture("unit_cabinet")._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == f"{ROBOCASA_ASSETS_ROOT}/unit_cabinet/**"


def test_ensure_assets_rediscovers_included_files_after_download(assets_dir, monkeypatch):
    # An include file missing at first access is fetched; only after it
    # appears can its own references be enumerated (and be required too).
    obj_dir = _make_robocasa_cabinet(assets_dir)
    (obj_dir / "shared_assets.xml").unlink()
    (obj_dir / "meshes" / "shelf.obj").unlink()
    calls = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        (obj_dir / "shared_assets.xml").write_text(_CABINET_INCLUDE)

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    fixture = RoboCasaFixture("unit_cabinet")
    with pytest.raises(FileNotFoundError, match="shelf.obj"):
        fixture._ensure_assets()  # include present now -> its refs are required

    def _fake_download_full(**kwargs):
        calls.append(kwargs)
        _write_box_mesh(obj_dir / "meshes" / "shelf.obj")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download_full)
    fixture._ensure_assets()
    assert len(calls) == 2 and fixture._assets_ready


def test_ensure_assets_public_facade(assets_dir, no_download):
    _make_robocasa_drawer(assets_dir)
    fixture = RoboCasaFixture("unit_drawer")
    fixture.ensure_assets()
    assert fixture._assets_ready
    assert not no_download


# ------------------------------------------------------------------
# _load_link_meshes_at_reference (offline FK, union bbox sanity)
# ------------------------------------------------------------------

def _union_bounds(meshes):
    import numpy as np

    lows = [mesh.bounds[0] for mesh in meshes.values()]
    highs = [mesh.bounds[1] for mesh in meshes.values()]
    return (
        np.min(np.stack(lows), axis=0),
        np.max(np.stack(highs), axis=0),
    )


def test_robocasa_link_meshes_at_reference(assets_dir):
    _make_robocasa_drawer(assets_dir)
    fixture = RoboCasaFixture("unit_drawer")

    meshes = fixture._load_link_meshes_at_reference()
    assert set(meshes) == {"frame", "drawer_box"}
    low, high = _union_bounds(meshes)
    # frame: 0.2 box at the origin; drawer: 0.1 box at (0.2, 0, 0.05).
    assert low == pytest.approx((-0.1, -0.1, -0.1), abs=1e-6)
    assert high == pytest.approx((0.25, 0.1, 0.1), abs=1e-6)

    # Same meshes at a reference configuration: the drawer slides +0.3 along
    # x, so the union bbox extends accordingly.
    open_fixture = RoboCasaFixture(
        "unit_drawer", reference_qpos={"drawer_slide": 0.3}
    )
    open_meshes = open_fixture._load_link_meshes_at_reference()
    _, open_high = _union_bounds(open_meshes)
    assert open_high[0] == pytest.approx(0.55, abs=1e-6)

    # The descriptor's cached union bbox agrees with the per-link meshes.
    bbox = fixture.get_local_bounding_box()
    assert bbox.low.tolist() == pytest.approx(list(low), abs=1e-5)
    assert bbox.high.tolist() == pytest.approx(list(high), abs=1e-5)


def test_replicacad_link_meshes_at_reference(assets_dir):
    _make_replicacad_fridge(assets_dir)
    item = ReplicaCADFurniture("fridge")

    meshes = item._load_link_meshes_at_reference()
    assert set(meshes) == {"frame", "door"}
    low, high = _union_bounds(meshes)
    # frame: 0.2 box at the origin; door: 0.1 box offset (0.2, 0, 0.05).
    assert low == pytest.approx((-0.1, -0.1, -0.1), abs=1e-6)
    assert high == pytest.approx((0.25, 0.1, 0.1), abs=1e-6)

    # Door hinged about +z by 90°: the door slab swings to +y.
    open_item = ReplicaCADFurniture(
        "fridge", reference_qpos={"door_hinge": 1.5707963267948966}
    )
    _, open_high = _union_bounds(open_item._load_link_meshes_at_reference())
    assert open_high[1] == pytest.approx(0.25, abs=1e-6)
    assert open_high[0] == pytest.approx(0.1, abs=1e-6)


def test_link_meshes_require_assets(assets_dir, monkeypatch):
    # The FK query goes through the verified asset machinery: with an
    # incomplete directory it fails loud instead of parsing a partial asset.
    _make_robocasa_drawer(assets_dir)
    (assets_dir / ROBOCASA_ASSETS_ROOT / "unit_drawer" / "drawer.obj").unlink()
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    fixture = RoboCasaFixture("unit_drawer")
    with pytest.raises(FileNotFoundError, match="drawer.obj"):
        fixture._load_link_meshes_at_reference()


# ------------------------------------------------------------------
# Load-time freejoint check (offline — no Genesis scene needed)
# ------------------------------------------------------------------

def test_freejoint_mismatch_warns(assets_dir, caplog):
    _make_robocasa_movable(assets_dir)
    fixture = RoboCasaFixture("movable_variant")  # fixed=True (default)
    with caplog.at_level(logging.WARNING, logger="unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture"):
        fixture._warn_on_freejoint_mismatch()
    assert "freejoint" in caplog.text


def test_freejoint_consistent_no_warning(assets_dir, caplog):
    _make_robocasa_drawer(assets_dir)
    with caplog.at_level(logging.WARNING, logger="unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture"):
        # fixed=True on a jointless MJCF: consistent, no warning.
        RoboCasaFixture("unit_drawer")._warn_on_freejoint_mismatch()
    assert "freejoint" not in caplog.text

    # fixed=False on a jointless MJCF: mismatch -> warning (robot_node pattern).
    with caplog.at_level(logging.WARNING, logger="unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture"):
        RoboCasaFixture("unit_drawer", fixed=False)._warn_on_freejoint_mismatch()
    assert "freejoint" in caplog.text


# ------------------------------------------------------------------
# Catalog helpers
# ------------------------------------------------------------------

def test_robocasa_catalog(assets_dir):
    catalog = get_robocasa_catalog()
    assert set(catalog) == {"unit_cabinet", "unit_drawer", "counter", "movable_variant"}
    assert catalog["unit_cabinet"].object_id == "unit_cabinet"
    assert "articulated" in catalog["unit_cabinet"].tags
    assert catalog["counter"].standing_quaternion is None
    assert get_robocasa_catalog() is get_robocasa_catalog()  # cached


def test_list_robocasa_objects_filters(assets_dir):
    assert list_robocasa_objects() == sorted(get_robocasa_catalog())
    kitchen = list_robocasa_objects(tags="kitchen")
    assert "unit_cabinet" in kitchen and "counter" in kitchen
    # All tags must be carried.
    drawers = list_robocasa_objects(tags={"kitchen", "drawer"})
    assert drawers == ["unit_drawer"]
    assert set(drawers) <= set(kitchen)
    # 'counter' is furniture but not articulated.
    assert "counter" not in list_robocasa_objects(tags="articulated")
    assert "counter" in list_robocasa_objects(tags="furniture")
    # Substring filter, combinable with tags.
    assert list_robocasa_objects(contains="CABINET") == ["unit_cabinet"]
    assert list_robocasa_objects(tags="kitchen", contains="counter") == ["counter"]
    assert list_robocasa_objects(tags="no_such_tag") == []


def test_replicacad_catalog_and_filters(assets_dir):
    catalog = get_replicacad_catalog()
    assert set(catalog) == {"fridge", "base_cabinet", "table"}
    assert catalog["fridge"].object_id == "fridge"
    assert list_replicacad_objects() == sorted(catalog)
    assert list_replicacad_objects(tags="articulated") == ["base_cabinet", "fridge"]
    assert list_replicacad_objects(tags={"articulated", "kitchen"}) == ["fridge"]
    assert "table" not in list_replicacad_objects(tags="articulated")
    assert list_replicacad_objects(tags="no_such_tag") == []


# ------------------------------------------------------------------
# Pool helpers
# ------------------------------------------------------------------

def test_robocasa_pool(assets_dir, no_download):
    pool = make_robocasa_pool(["unit_drawer", "unit_cabinet"], scale=0.5)
    assert [o.object_id for o in pool] == ["unit_drawer", "unit_cabinet"]
    assert all(isinstance(o, RoboCasaFixture) for o in pool)
    assert pool[0].scale == 0.5
    assert not no_download  # pool construction downloads nothing


def test_robocasa_pool_validation(assets_dir):
    with pytest.raises(ValueError, match="not both"):
        make_robocasa_pool(["unit_drawer"], tags="kitchen")
    with pytest.raises(ValueError, match="not both"):
        make_robocasa_pool(["unit_drawer"], contains="drawer")
    with pytest.raises(ValueError, match="only 2"):
        make_robocasa_pool(["unit_drawer", "unit_cabinet"], n=3)


def test_replicacad_pool_sampling_reproducible(assets_dir):
    pool_a = make_replicacad_pool(
        tags="articulated", n=2, rng=torch.Generator().manual_seed(7)
    )
    pool_b = make_replicacad_pool(
        tags="articulated", n=2, rng=torch.Generator().manual_seed(7)
    )
    ids_a = [o.object_id for o in pool_a]
    assert ids_a == [o.object_id for o in pool_b]
    assert set(ids_a) == {"base_cabinet", "fridge"}
    assert all(isinstance(o, ReplicaCADFurniture) for o in pool_a)
