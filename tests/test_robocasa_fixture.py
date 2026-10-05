"""Tests for the RoboCasa kitchen fixture pipeline (S4).

Mirrors ``tests/test_gso_object.py``: the prepare script is loaded by path and
exercised against synthetic per-fixture asset trees under ``tmp_path`` —
``asset_path.ASSETS_DIR`` is repointed at it and
``asset_path.download_assets`` is stubbed, so everything except the marked GPU
smoke test runs fully offline.

Covers:
  - Fixture registry integrity + the ``--fixtures`` selection resolver
  - Generated catalog integrity (annotations, tags incl. furniture /
    articulated / kitchen, nothing beyond the catalog schema)
  - list_robocasa_objects tag / substring filtering (+ make_robocasa_pool)
  - Repackaging: referenced meshes/textures copied + rewritten
    directory-local, <include> chain resolved recursively (matching what
    ``HubAssetArticulatedObject._mjcf_required_files`` enumerates — the
    loader that will open these files)
  - freejoint stripping (welded-root convention; nested free joints reject)
  - Primitive baking: box/cylinder geoms → exact-dimension mesh geoms
    (watertight meshes, region markers untouched)
  - MuJoCo compile gate with the degenerate-mesh ``inertia="shell"`` patch
  - Offline FK bbox sanity (per-link meshes + union bbox at reference qpos)
  - GPU smoke of ``RoboCasaFixture.load_into_scene`` + articulation
    (``skipif`` — needs CUDA *and* the packaged assets)
"""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.hub_objects.articulated.hub_articulated_object import (
    _mjcf_required_files,
)
from unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture import (
    ROBOCASA_ASSETS_ROOT,
    RoboCasaFixture,
    get_robocasa_catalog,
    list_robocasa_objects,
    make_robocasa_pool,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

# Composed-fixture registry entries whose Genesis-gate joint counts the catalog
# tags derive from (unit-test subset mirrors scripts/prepare_robocasa_assets.py).
_ARTICULATED_IDS = ("cabinet_hinge", "cabinet_single", "drawer")
_RIGID_IDS = ("panel", "counter", "dishwasher_basic_stainless")


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_robocasa_assets", _SCRIPTS_DIR / "prepare_robocasa_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_prepare_script()

# A minimal valid PNG (MuJoCo decodes texture assets at compile time).
def _make_png_bytes(width: int = 4, height: int = 4) -> bytes:
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x40\x20" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


_PNG_BYTES = _make_png_bytes()


# ---------------------------------------------------------------------------


def _write_box_obj(path: Path, half_extents=(0.05, 0.05, 0.05)) -> None:
    script._write_box_obj(path, half_extents)


# A RoboCasa-shaped fixture: compiler meshdir, a texture outside the fixture's
# own directory, a local mesh, a top-level <include> contributing another mesh
# referenced by a geom (exercises recursive include + reference resolution).
_UNIT_CABINET_MJCF = """\
<mujoco model="unit_cabinet">
    <compiler angle="radian" meshdir="meshes" />
    <include file="shared_assets.xml" />
    <asset>
        <texture type="2d" name="tex" file="../shared/wood_grain_1.png" />
        <material name="mat" texture="tex" />
        <mesh name="frame_mesh" file="frame.obj" />
    </asset>
    <worldbody>
        <body name="frame" pos="0 0 0">
            <geom name="frame_geom" type="mesh" mesh="frame_mesh" material="mat" />
            <body name="door" pos="0.2 0 0.1">
                <joint name="door_hinge" type="hinge" axis="0 0 1" range="0 1.57" />
                <geom name="door_geom" type="mesh" mesh="door_mesh" material="mat" />
                <geom name="shelf_geom" type="mesh" mesh="shelf_mesh" material="mat" />
            </body>
        </body>
    </worldbody>
</mujoco>
"""

_UNIT_CABINET_INCLUDE = """\
<mujocoinclude model="shared_assets">
    <asset>
        <mesh name="door_mesh" file="door.obj" />
        <mesh name="shelf_mesh" file="shelf.obj" />
    </asset>
</mujocoinclude>
"""


def _write_source_tree(tmp: Path) -> Path:
    """Synthetic RoboCasa assets tree (fixtures/ + textures/) under *tmp*."""
    source = tmp / "rc_source"
    cabinets = source / "fixtures" / "cabinets"
    cabinets.mkdir(parents=True, exist_ok=True)
    (cabinets / "unit_cab.xml").write_text(_UNIT_CABINET_MJCF)
    (cabinets / "shared_assets.xml").write_text(_UNIT_CABINET_INCLUDE)
    meshes = cabinets / "meshes"
    meshes.mkdir(exist_ok=True)
    _write_box_obj(meshes / "frame.obj", (0.2, 0.2, 0.2))
    _write_box_obj(cabinets / "door.obj", (0.1, 0.1, 0.1))
    _write_box_obj(cabinets / "shelf.obj", (0.05, 0.05, 0.05))
    shared = source / "fixtures" / "shared"
    shared.mkdir(exist_ok=True)
    (shared / "wood_grain_1.png").write_bytes(_PNG_BYTES)
    (source / "textures" / "wood").mkdir(parents=True, exist_ok=True)
    (source / "textures" / "wood" / "wood_grain_1.png").write_bytes(_PNG_BYTES)
    (source / "LICENSE").write_text("CC BY 4.0\n")
    return source


def _catalog_records() -> list:
    return (
        [{"fixture_id": fid, "n_joints": 2} for fid in _ARTICULATED_IDS]
        + [{"fixture_id": fid} for fid in _RIGID_IDS]
    )


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir with a generated catalog + one packaged fixture."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    source = _write_source_tree(tmp_path)
    out_root = tmp_path / ROBOCASA_ASSETS_ROOT
    out_root.mkdir(parents=True)
    records = (
        [{"fixture_id": "unit_cabinet", "n_joints": 1,
          "annotations": ["furniture", "kitchen cabinet", "unit kitchen cabinet"],
          "tags": ["cabinet"]}]
        + [{"fixture_id": fid, "n_joints": 2} for fid in _ARTICULATED_IDS]
        + [{"fixture_id": fid} for fid in _RIGID_IDS]
    )
    script.write_catalog(out_root, records)
    repkg = script.Repackager(source, out_root)
    repkg.repackage(source / "fixtures" / "cabinets" / "unit_cab.xml", "unit_cabinet")
    # clear any catalog cached by an earlier (repo-backed) construction
    import unienv_genesis_collection.objects.hub_objects.catalog as catalog_module

    catalog_module._CACHE.clear()
    return tmp_path


@pytest.fixture
def packaged_fixture(assets_dir):
    """The packaged synthetic fixture directory (unit_cabinet)."""
    return assets_dir / ROBOCASA_ASSETS_ROOT / "unit_cabinet"


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
# Fixture registry + --fixtures selection
# ------------------------------------------------------------------

def test_registry_ids_unique_and_complete():
    ids = [spec["fixture_id"] for spec in script.FIXTURE_REGISTRY.items()] \
        if isinstance(script.FIXTURE_REGISTRY, list) else list(script.FIXTURE_REGISTRY)
    assert len(ids) == len(set(ids))
    assert tuple(sorted(ids)) == tuple(sorted(script.ALL_FIXTURES))
    for fixture_id, spec in script.FIXTURE_REGISTRY.items():
        assert spec["kind"] in ("composed", "static")
        assert spec["family"]
        assert spec["annotations"] and all(a == a.lower() for a in spec["annotations"])
        assert "expect_joints" in spec
        if spec["kind"] == "composed":
            assert spec["template"].endswith(".xml")
        else:
            assert spec["source"]

def test_registry_covers_requested_families():
    families = {spec["family"] for spec in script.FIXTURE_REGISTRY.values()}
    assert families >= {
        "cabinet", "drawer", "handle", "dishwasher", "oven", "fridge",
        "microwave", "stove", "sink", "counter",
    }
    # fridge ×3 as the plan's registry subset requires
    assert sum(1 for s in script.FIXTURE_REGISTRY.values() if s["family"] == "fridge") == 3
    # plan risk #1: cabinet_hinge + drawer are the cheapest first signal
    assert script.CHEAPEST_SIGNAL == ("cabinet_hinge", "drawer")

def test_resolve_fixture_selection_groups_and_ids():
    assert script.resolve_fixture_selection(["cabinets"]) == [
        "cabinet_hinge", "cabinet_single", "drawer", "panel",
    ]
    assert script.resolve_fixture_selection(["handles", "counter"]) == [
        "handle_bar", "handle_boxed", "handle_knob", "counter",
    ]
    assert script.resolve_fixture_selection(["drawer", "drawer"]) == ["drawer"]
    assert script.resolve_fixture_selection(["all"]) == list(script.ALL_FIXTURES)
    assert script.resolve_fixture_selection([]) == []

def test_resolve_fixture_selection_unknown_group_raises():
    import pytest

    with pytest.raises(ValueError, match="no_such_fixture"):
        script.resolve_fixture_selection(["no_such_fixture"])


# ------------------------------------------------------------------
# Generated catalog
# ------------------------------------------------------------------

def test_catalog_integrity(assets_dir, no_download):
    catalog = get_robocasa_catalog()
    assert len(catalog) == 7
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert object_id and "/" not in object_id and " " not in object_id
        assert info.language_annotations and all(info.language_annotations)
        for annotation in info.language_annotations:
            assert annotation == annotation.lower()
        for tag in info.tags:
            assert tag == tag.lower() and " " not in tag
        # every fixture carries the pipeline's base tags
        assert {"furniture", "kitchen"} <= info.tags

def test_catalog_tags_derive_from_gate_observed_joints(assets_dir, no_download):
    catalog = get_robocasa_catalog()
    for fixture_id in _ARTICULATED_IDS:
        assert "articulated" in catalog[fixture_id].tags
    for fixture_id in _RIGID_IDS:
        assert "articulated" not in catalog[fixture_id].tags
    # family tags separate from the base tags
    assert "cabinet" in catalog["cabinet_hinge"].tags
    assert "counter" in catalog["counter"].tags
    # language annotations stay broad → detailed and separate from tags
    assert "kitchen cabinet" in catalog["cabinet_hinge"].language_annotations

def test_catalog_only_carries_language_annotations_and_tags(assets_dir, no_download):
    raw = json.loads((assets_dir / ROBOCASA_ASSETS_ROOT / "catalog.json").read_text())
    for entry in raw.values():
        assert set(entry) == {"language_annotations", "tags"}


# ------------------------------------------------------------------
# list_robocasa_objects / make_robocasa_pool
# ------------------------------------------------------------------

def test_list_all_objects_sorted(assets_dir, no_download):
    ids = list_robocasa_objects()
    assert ids == sorted(ids)
    assert len(ids) == 7

def test_list_filter_by_tag(assets_dir, no_download):
    articulated = list_robocasa_objects(tags="articulated")
    assert set(articulated) == set(_ARTICULATED_IDS) | {"unit_cabinet"}
    kitchen = list_robocasa_objects(tags="kitchen")
    assert set(kitchen) == set(list_robocasa_objects())

def test_list_multiple_tags_require_all(assets_dir, no_download):
    expected = sorted(set(_ARTICULATED_IDS) | {"unit_cabinet"})
    assert list_robocasa_objects(tags=["furniture", "articulated"]) == expected
    assert list_robocasa_objects(tags=["articulated", "counter"]) == []

def test_list_contains_filter_case_insensitive(assets_dir, no_download):
    assert "cabinet_hinge" in list_robocasa_objects(contains="HINGE")
    assert set(list_robocasa_objects(contains="cabinet")) <= set(list_robocasa_objects())

def test_list_no_match(assets_dir, no_download):
    assert list_robocasa_objects(contains="zzz_no_such_fixture") == []
    assert list_robocasa_objects(tags="zzz_no_such_tag") == []

def test_pool_explicit_ids_and_sampling(assets_dir, no_download):
    pool = make_robocasa_pool(["cabinet_hinge", "drawer", "panel"][:2])
    assert [o.object_id for o in pool] == ["cabinet_hinge", "drawer"]
    sampled = make_robocasa_pool(tags="articulated", n=2,
                                 rng=torch.Generator().manual_seed(0))
    assert len({o.object_id for o in sampled}) == 2
    with pytest.raises(ValueError):
        make_robocasa_pool(["cabinet_hinge"], tags="kitchen")
    with pytest.raises(ValueError):
        make_robocasa_pool(["cabinet_hinge"], n=2)


# ------------------------------------------------------------------
# Repackaging: copy + rewrite refs (incl. recursive <include>)
# ------------------------------------------------------------------

def test_repackaged_fixture_is_self_contained(packaged_fixture):
    root = ET.parse(packaged_fixture / "model.xml").getroot()
    assert root.tag == "mujoco"
    missing = [
        (element.tag, element.get("file"))
        for element in root.iter()
        if element.get("file") and not (packaged_fixture / element.get("file")).is_file()
    ]
    assert missing == []

def test_repackaged_refs_rewritten_directory_local(packaged_fixture):
    text = (packaged_fixture / "model.xml").read_text()
    # the upstream (source-relative) refs are gone; refs are directory-local
    assert '../shared/wood_grain_1.png' not in text
    # include + referenced assets live inside the fixture directory
    assert "shared_assets.xml" in text
    assert (packaged_fixture / "shared_assets.xml").is_file()
    assert (packaged_fixture / "meshes" / "frame.obj").is_file()
    assert (packaged_fixture / "door.obj").is_file()
    assert (packaged_fixture / "shelf.obj").is_file()
    # texture copied under the source-tree-relative subpath
    assert (packaged_fixture / "fixtures" / "shared" / "wood_grain_1.png").is_file()
    assert 'file="fixtures/shared/wood_grain_1.png"' in text

def test_required_files_match_packaged_tree(packaged_fixture, no_download):
    """The descriptor's enumeration names exactly what the packager copied."""
    obj = RoboCasaFixture("unit_cabinet")
    required = {p.relative_to(obj.asset_dir) for p in obj._required_files()}
    present = {p.relative_to(obj.asset_dir) for p in packaged_fixture.rglob("*") if p.is_file()}
    assert present <= required  # every copied file is required by the descriptor
    assert obj._missing_files() == []
    assert not no_download  # nothing needed fetching

def test_required_files_include_recursion_matches_hub_helper(packaged_fixture, no_download):
    """The packager's reference semantics equal ``_mjcf_required_files``.

    The Hub descriptor enumerates the include chain the same way the packaging
    resolved it — so packaged assets are complete for the real loader.
    """
    obj = RoboCasaFixture("unit_cabinet")
    enumerated = [p.relative_to(obj.asset_dir) for p in _mjcf_required_files(
        packaged_fixture / "model.xml"
    )]
    present = {p.relative_to(obj.asset_dir).as_posix() for p in packaged_fixture.rglob("*") if p.is_file()}
    for relative in enumerated:
        assert relative.as_posix() in present, relative

def test_ensure_assets_complete_no_download(packaged_fixture, no_download):
    obj = RoboCasaFixture("unit_cabinet")
    obj._ensure_assets()
    assert obj._assets_ready
    assert not no_download

def test_missing_reference_fails_packaging(assets_dir):
    source = _write_source_tree(assets_dir)
    (source / "fixtures" / "cabinets" / "shelf.obj").unlink()
    repkg = script.Repackager(source, assets_dir / ROBOCASA_ASSETS_ROOT)
    import pytest

    with pytest.raises(FileNotFoundError, match="shelf.obj"):
        repkg.repackage(source / "fixtures" / "cabinets" / "unit_cab.xml", "broken_fixture")


# ------------------------------------------------------------------
# freejoint stripping (welded-root convention)
# ------------------------------------------------------------------

def test_strip_freejoint_removes_root_freejoint():
    text = """\
<mujoco model="movable">
  <worldbody>
    <body name="root">
      <freejoint name="floating"/>
      <geom name="g" type="box" size="0.1 0.1 0.1"/>
      <body name="door">
        <joint name="door_hinge" type="hinge" axis="0 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    root = ET.fromstring(text)
    removed = script.strip_freejoint(root)
    assert removed == ["floating"]
    root_body = root.find("./worldbody/body")
    assert root_body.find("freejoint") is None
    assert root_body.find("body/joint") is not None  # interior joints kept

def test_strip_freejoint_rejects_nested_freejoint():
    text = """\
<mujoco model="odd">
  <worldbody>
    <body name="root">
      <body name="child">
        <freejoint/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    import pytest

    with pytest.raises(ValueError, match="nested body"):
        script.strip_freejoint(ET.fromstring(text))


# ------------------------------------------------------------------
# Primitive baking (composed fixtures)
# ------------------------------------------------------------------

def test_bake_primitive_geoms_replaces_boxes_with_meshes(tmp_path):
    fixture_dir = tmp_path / "unit_cab"
    fixture_dir.mkdir()
    root = ET.fromstring("""\
<mujoco model="unit_cab">
  <compiler angle="radian"/>
  <asset>
    <texture file="textures/wood/wood_grain_1.png" type="2d" name="tex"/>
    <material name="mat" texture="tex"/>
  </asset>
  <worldbody>
    <body name="frame">
      <geom name="top" type="box" size="0.2 0.15 0.01" pos="0 0 0.14"/>
      <geom name="reg_main" type="box" size="0.0001 0.0001 0.0001" class="region"/>
    </body>
  </worldbody>
</mujoco>
""")
    baked = script.bake_primitive_geoms(root, fixture_dir)
    assert baked == ["meshes/top_baked.obj"]
    # the baked mesh asset + rewritten geom, the region marker untouched
    mesh_asset = root.find("./asset/mesh")
    assert mesh_asset.get("name") == "top_baked" and mesh_asset.get("file") == "meshes/top_baked.obj"
    geoms = list(root.iter("geom"))
    top = next(g for g in geoms if g.get("name") == "top")
    assert top.get("type") == "mesh" and top.get("mesh") == "top_baked"
    assert "size" not in top.attrib and top.get("pos") == "0 0 0.14"
    region = next(g for g in geoms if g.get("name") == "reg_main")
    assert region.get("type") == "box" and region.get("mesh") is None
    # the mesh is a watertight box with the authored half-extents
    import trimesh

    mesh = trimesh.load(fixture_dir / "meshes" / "top_baked.obj", force="mesh")
    assert mesh.is_watertight
    assert torch.allclose(
        torch.tensor(mesh.bounds, dtype=torch.float32),
        torch.tensor([[-0.2, -0.15, -0.01], [0.2, 0.15, 0.01]], dtype=torch.float32),
        atol=1e-4,
    )

def test_baked_meshes_keep_fk_geometry_consistent(tmp_path):
    """A baked fixture's offline FK bbox equals the authored box extents."""
    fixture_dir = tmp_path / "unit_drawer"
    fixture_dir.mkdir()
    fixture_dir.mkdir(exist_ok=True)
    root = ET.fromstring("""\
<mujoco model="unit_drawer">
  <compiler angle="radian"/>
  <asset>
    <texture file="textures/wood/wood_grain_1.png" type="2d" name="tex"/>
    <material name="mat" texture="tex"/>
  </asset>
  <worldbody>
    <body name="frame">
      <geom name="top" type="box" size="0.2 0.3 0.02" pos="0 0 0.28"/>
      <body name="inner_box" pos="0 0 0">
        <joint name="slidejoint" type="slide" axis="0 1 0" range="-0.6 0"/>
        <geom name="inner_bottom" type="box" size="0.18 0.3 0.02" pos="0 0 -0.28"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    script.bake_primitive_geoms(root, fixture_dir)
    ET.indent(root)
    (fixture_dir / "model.xml").write_text(ET.tostring(root, encoding="unicode"))
    from unienv_genesis_collection.objects.articulated_object import (
        mjcf_link_meshes_at_config,
    )

    meshes = mjcf_link_meshes_at_config(fixture_dir / "model.xml")
    assert set(meshes) == {"frame", "inner_box"}
    import numpy as np

    union_low = np.min([m.bounds[0] for m in meshes.values()], axis=0)
    union_high = np.max([m.bounds[1] for m in meshes.values()], axis=0)
    assert np.allclose(union_low, [-0.2, -0.3, -0.30], atol=1e-6)
    assert np.allclose(union_high, [0.2, 0.3, 0.30], atol=1e-6)


# ------------------------------------------------------------------
# MuJoCo compile gate + degenerate mesh patch
# ------------------------------------------------------------------

def test_compile_check_patches_degenerate_mesh(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "unit_fixture"
    fixture_dir.mkdir()
    (fixture_dir / "model.xml").write_text("""\
<mujoco model="unit_fixture">
  <asset>
    <mesh name="degenerate_mesh" file="visuals/mesh.obj"/>
  </asset>
  <worldbody>
    <body name="object">
      <geom name="g" type="mesh" mesh="degenerate_mesh"/>
    </body>
  </worldbody>
</mujoco>
""")
    (fixture_dir / "visuals").mkdir()
    _write_box_obj(fixture_dir / "visuals" / "mesh.obj")

    import pytest
    import sys

    mujoco_module = pytest.importorskip("mujoco")
    calls = {"n": 0}
    original = mujoco_module.MjModel.from_xml_path

    def flaky(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(
                "Error: mesh volume is too small: degenerate_mesh . "
                "Try setting inertia to shell"
            )
        return original(path, **kwargs)

    monkeypatch.setattr(mujoco_module.MjModel, "from_xml_path", staticmethod(flaky))
    patched = script.compile_check(fixture_dir / "model.xml")
    assert patched == ["degenerate_mesh"]
    mesh = ET.parse(fixture_dir / "model.xml").getroot().find("./asset/mesh")
    assert mesh.get("inertia") == "shell"
    assert calls["n"] == 2


# ------------------------------------------------------------------
# Composed fixtures (robocasa ProcGen port)
# ------------------------------------------------------------------

def _write_cabinet_source(tmp: Path) -> Path:
    """Minimal robocasa template tree for ``compose_fixture``."""
    source = tmp / "rc_source"
    cabinets = source / "fixtures" / "cabinets"
    handles = source / "fixtures" / "handles"
    cabinets.mkdir(parents=True, exist_ok=True)
    handles.mkdir(parents=True, exist_ok=True)
    (cabinets / "cabinet_single.xml").write_text("""\
<mujoco model="hinge_cab">
  <compiler angle="radian"/>
  <asset>
    <texture file="visuals/wood1.png" type="cube" name="tex"/>
    <material name="mat" shininess="0.1" texture="tex"/>
  </asset>
  <default>
    <default class="cab">
      <geom group="0" density="10"/>
    </default>
    <default class="region">
      <geom group="1" conaffinity="0" contype="0" rgba="0 1 0 0"/>
    </default>
  </default>
  <worldbody>
    <body>
      <body name="object">
        <geom pos="0.0 0 0.28" size="0.18 0.3 0.02" type="box" material="mat" class="cab" name="top"/>
        <geom pos="0.0 0 -0.28" size="0.18 0.3 0.02" type="box" material="mat" class="cab" name="bottom"/>
        <geom pos="0.0 0.28 0" size="0.18 0.02 0.26" type="box" material="mat" class="cab" name="back"/>
        <geom pos="0.20 0 0" size="0.02 0.3 0.3" type="box" material="mat" class="cab" name="right"/>
        <geom pos="-0.20 0 0" size="0.02 0.3 0.3" type="box" material="mat" class="cab" name="left"/>
        <body name="hingedoor" pos="0.20 -0.32 0">
          <joint axis="0 0 1" name="doorhinge" limited="true" range="0 3.00" damping="2"/>
          <site rgba="1 0 0 0" size="0.005" pos="0.0 -0.20 0" name="door_p1"/>
        </body>
        <geom class="region" name="reg_main" type="box" pos="0 0 0" size="0.0001 0.0001 0.0001"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    (source / "textures" / "wood").mkdir(parents=True, exist_ok=True)
    (source / "textures" / "wood" / "wood_grain_1.png").write_bytes(_PNG_BYTES)
    (source / "textures" / "metals").mkdir(parents=True, exist_ok=True)
    (source / "textures" / "metals" / "bright_metal.png").write_bytes(_PNG_BYTES)
    (source / "LICENSE").write_text("CC BY 4.0\n")
    return source


def test_compose_fixture_structure(tmp_path):
    source = _write_cabinet_source(tmp_path)
    spec = dict(script.FIXTURE_REGISTRY["cabinet_single"])
    root = script.compose_fixture("cabinet_single", spec, source)
    # robocasa's cabinet geometry: housing resized to [0.65, 0.6, 0.84]
    top = next(g for g in root.iter("geom") if g.get("name") == "top")
    assert [float(v) for v in top.get("size").split()] == [0.325, 0.285, 0.015]
    assert [float(v) for v in top.get("pos").split()] == [0.0, 0.015, 0.405]
    # the door body keeps its joint; the composed panel + handle hang off it
    door_body = next(b for b in root.iter("body") if b.get("name") == "hingedoor")
    assert door_body.find("joint").get("name") == "doorhinge"
    panel = next(b for b in door_body.iter("body") if b.get("name") == "cabinet_single_door")
    assert panel.get("pos") == "0 -0.285 0"  # [0, -y + th, 0]
    door_geom = next(g for g in panel.iter("geom") if g.get("name").endswith("_door_door"))
    assert door_geom.get("euler") == "1.5708 0 0"  # robocasa's rotation trick
    handle_body = next(b for b in panel.iter("body") if b.get("name").endswith("_handle"))
    assert [float(v) for v in handle_body.get("pos").split()] != [0.0, 0.0, 0.0]
    assert any(g.get("name", "").endswith("handle_handle") for g in handle_body.iter("geom"))
    # fixed furniture: jointless root
    assert root.find("./worldbody/body/freejoint") is None


def test_composed_fixture_offline_fk_bbox(tmp_path, assets_dir, no_download):
    """A composed fixture packaged end-to-end yields a correct offline bbox."""
    source = _write_cabinet_source(assets_dir)
    spec = dict(script.FIXTURE_REGISTRY["cabinet_single"])
    fixture_dir = assets_dir / ROBOCASA_ASSETS_ROOT / "cabinet_single"
    fixture_dir.mkdir(exist_ok=True)
    root = script.compose_fixture("cabinet_single", spec, source)
    script.copy_asset_refs(root, source, (source / "fixtures" / "cabinets"), fixture_dir)
    script.bake_primitive_geoms(root, fixture_dir)
    script.strip_freejoint(root)
    ET.indent(root)
    (fixture_dir / "model.xml").write_text(ET.tostring(root, encoding="unicode"))

    obj = RoboCasaFixture("cabinet_single")
    bbox = obj.get_local_bounding_box()
    low = bbox.low.numpy()
    high = bbox.high.numpy()
    # size [0.65, 0.6, 0.84]: robocasa centres the housing at its origin
    # (z ±0.42, y −0.27…+0.3); the front door panel + handle reach y ≈ −0.348.
    # Both sides pinned to the host device: after a CUDA-marked module runs
    # first (test_replicacad_furniture.py's GPU smoke), the ambient
    # ``torch.set_default_device`` makes a bare ``torch.tensor(...)`` land on
    # CUDA while this offline bbox is host-built — see the device-isolation
    # note in tests/test_room_builder.py.
    expected_low = torch.tensor([-0.325, -0.348, -0.42], dtype=bbox.low.dtype,
                                device=bbox.low.device)
    expected_high = torch.tensor([0.325, 0.3, 0.42], dtype=bbox.high.dtype,
                                 device=bbox.high.device)
    assert torch.allclose(bbox.low, expected_low, atol=2e-3)
    assert torch.allclose(bbox.high, expected_high, atol=2e-3)
    assert (high[2] - low[2]) == pytest.approx(0.84, abs=1e-3)

def test_composed_fixture_required_files_and_fk_links(tmp_path, assets_dir, no_download):
    source = _write_cabinet_source(assets_dir)
    spec = dict(script.FIXTURE_REGISTRY["cabinet_single"])
    fixture_dir = assets_dir / ROBOCASA_ASSETS_ROOT / "cabinet_single"
    fixture_dir.mkdir(exist_ok=True)
    root = script.compose_fixture("cabinet_single", spec, source)
    script.copy_asset_refs(root, source, (source / "fixtures" / "cabinets"), fixture_dir)
    script.bake_primitive_geoms(root, fixture_dir)
    ET.indent(root)
    (fixture_dir / "model.xml").write_text(ET.tostring(root, encoding="unicode"))
    script.compile_check(fixture_dir / "model.xml")

    obj = RoboCasaFixture("cabinet_single")
    obj._ensure_assets()
    assert obj._missing_files() == []
    meshes = obj._load_link_meshes_at_reference()
    # housing + door-panel links carry visual meshes (region markers and the
    # joint-only door body don't — the panel body carries the door geometry)
    assert {"object", "cabinet_single_door"} <= set(meshes)
    assert "hingedoor" not in meshes
    link_bbox = obj.get_link_local_bounding_box("cabinet_single_door")
    assert torch.all(link_bbox.high > link_bbox.low)


# ------------------------------------------------------------------
# Packaging log / provenance artifacts
# ------------------------------------------------------------------

def test_write_catalog_annotations_and_packaged_files(assets_dir, packaged_fixture, no_download):
    catalog = get_robocasa_catalog()
    info = catalog["unit_cabinet"]
    assert info.language_annotations[-1] == "unit kitchen cabinet"
    assert "furniture" in info.tags and "kitchen" in info.tags
    # the pipeline generates the provenance artifacts next to the assets
    assert "CC BY 4.0" in script.PROVENANCE_README
    assert "RoboCasaFixture.load_into_scene" in script.PROVENANCE_README
    assert "prepare_robocasa_assets" in script.SHELL_README
    assert script.CC_BY_40_TEXT


# ------------------------------------------------------------------
# GPU smoke (real packaged assets + CUDA)
# ------------------------------------------------------------------

def _assets_available() -> bool:
    catalog = Path(__file__).parent.parent / "unienv_genesis_collection" / "assets" / (
        "objects/robocasa/catalog.json"
    )
    return catalog.is_file()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _assets_available(),
    reason="CUDA and the packaged RoboCasa assets are required",
)
def test_load_and_articulate_packaged_fixture():
    """Real packaged fixture: Genesis load, joints, bbox cross-check, motion."""
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)
    # NOTE: deliberately NOT destroy()+re-init here.  In this Genesis/usd-core
    # combination a destroy()+init() *after any pxr stage has been opened*
    # corrupts the process-global Tf notice registry (pxr aborts with
    # "TfNotice type 'TfDebugSymbolsChangedNotice' has NO base types"), taking
    # every later test down.  Genesis re-init after a USD scene is therefore
    # avoided; a stale CUDA default device is handled per-assertion instead
    # (the device-pinning convention of tests/test_room_builder.py).

    from unienv_genesis import GenesisWorld
    import unienv_genesis_collection.asset_path as asset_path_module

    asset_path_module.ASSETS_DIR = (
        Path(__file__).parent.parent / "unienv_genesis_collection" / "assets"
    )
    import unienv_genesis_collection.objects.hub_objects.catalog as catalog_module

    catalog_module._CACHE.clear()

    world = GenesisWorld(scene_init_kwargs={"show_viewer": False})
    fixture = RoboCasaFixture("cabinet_hinge")
    entity = fixture.load_into_scene(world, pos=(0.0, 0.0, 0.0))
    world.build_scene(world.scene)

    # non-empty joint set + welded root (fixed furniture)
    joint_names = [joint.name for joint in entity.joints]
    assert len(joint_names) >= 1
    assert entity.get_dofs_position(dofs_idx_local=list(entity.joints[0].dofs_idx_local)) is not None

    # offline FK bbox vs runtime vAABB at q = 0 (within the pipeline tolerance)
    local_bbox = fixture.get_local_bounding_box()
    v_aabb = entity.get_vAABB()
    low_v = v_aabb[0].detach().cpu().numpy()
    high_v = v_aabb[1].detach().cpu().numpy()
    low_l = local_bbox.low.detach().cpu().numpy()
    high_l = local_bbox.high.detach().cpu().numpy()
    assert np_allclose(low_v, low_l, atol=0.03)
    assert np_allclose(high_v, high_l, atol=0.03)

    # articulation: driving the door hinge moves the child link
    hinge = entity.get_joint(name=joint_names[0])
    dofs = list(hinge.dofs_idx_local)
    link_idx = hinge.link.idx_local
    p0 = entity.get_links_pos(links_idx_local=[link_idx], relative=True).detach().cpu()
    fixture.set_open_fraction(entity, joint_names[0], torch.tensor([0.0]))
    p1 = entity.get_links_pos(links_idx_local=[link_idx], relative=True).detach().cpu()
    assert float((p1 - p0).abs().max()) > 1e-4

    world.scene.destroy()
    # NOTE: no destroy()+re-init here either — see the note at the top of
    # this test (a re-init after USD stages have been opened corrupts the
    # pxr Tf notice registry process-globally).  The CUDA ambient default
    # this leaves behind is what the device-pinning assertions below and
    # the device-isolation convention (tests/test_room_builder.py) absorb.


def np_allclose(a, b, atol=1e-6) -> bool:
    import numpy as np

    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=atol))
