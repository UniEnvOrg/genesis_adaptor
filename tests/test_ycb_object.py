"""Tests for YCB object integration (no asset downloads, no scene builds).

The catalog ships with the assets as ``objects/ycb/catalog.json``; tests
install the curated source (``scripts/data/ycb_catalog.json``) into a
temporary assets dir, so catalog content checks run against the real data.

Covers:
  - Catalog integrity (multi-annotations broad → detailed, separate tags,
    box_graspable)
  - list_ycb_objects tag filtering
  - make_ycb_pool selection / sampling / validation
  - YCBObject construction and asset path resolution
  - Lazy asset completeness check (_ensure_assets) incl. partial-download repair
  - URDF/MJCF mesh-path rewriting in scripts/prepare_ycb_assets.py
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.hub_objects.ycb_object import (
    YCB_ASSETS_ROOT,
    YCBObject,
    get_ycb_catalog,
    list_ycb_objects,
    make_ycb_pool,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_CATALOG_SOURCE = _SCRIPTS_DIR / "data" / "ycb_catalog.json"


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------

_URDF_TEMPLATE = """\
<robot name="{oid}">
    <link name="base_link">
        <visual>
            <geometry><mesh filename="textured.obj" scale="1 1 1" /></geometry>
        </visual>
        <collision>
            <geometry><mesh filename="textured_coacd_0.stl" scale="1 1 1" /></geometry>
        </collision>
        <collision>
            <geometry><mesh filename="textured_coacd_1.stl" scale="1 1 1" /></geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="1e-3" ixy="0.0" ixz="0.0" iyy="1e-3" iyz="0.0" izz="1e-3"/>
        </inertial>
    </link>
</robot>
"""

_OBJECT_FILES = (
    "textured.obj",
    "textured.mtl",
    "texture_map.png",
    "textured_coacd_0.stl",
    "textured_coacd_1.stl",
)


def _make_fake_object_dir(assets_dir: Path, object_id: str) -> Path:
    """Create a complete fake YCB object directory under *assets_dir*."""
    obj_dir = assets_dir / YCB_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    (obj_dir / f"{object_id}.urdf").write_text(_URDF_TEMPLATE.format(oid=object_id))
    for name in _OBJECT_FILES:
        (obj_dir / name).write_text("placeholder")
    return obj_dir


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir holding the real curated catalog.json."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    catalog_file = tmp_path / YCB_ASSETS_ROOT / "catalog.json"
    catalog_file.parent.mkdir(parents=True)
    shutil.copy2(_CATALOG_SOURCE, catalog_file)
    return tmp_path


@pytest.fixture
def fake_banana(assets_dir):
    """A complete fake 'banana' object directory."""
    return _make_fake_object_dir(assets_dir, "banana")


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
# Catalog
# ------------------------------------------------------------------

def test_catalog_integrity(assets_dir):
    catalog = get_ycb_catalog()
    assert len(catalog) == 78
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert object_id == object_id.lower()
        assert " " not in object_id
        # Multiple annotations, broad -> detailed, separate from tags.
        assert len(info.language_annotations) >= 2
        assert len(info.language_annotations[0]) <= len(info.language_annotations[-1])
        for annotation in info.language_annotations:
            assert annotation == annotation.lower()
            assert not annotation.startswith(("a ", "an "))
            assert "(" not in annotation and ")" not in annotation
        assert info.tags
        for tag in info.tags:
            assert tag == tag.lower() and " " not in tag

def test_catalog_spot_checks(assets_dir):
    banana = get_ycb_catalog()["banana"]
    assert banana.language_annotations == ("fruit", "banana", "yellow banana")
    assert "food" in banana.tags and "box_graspable" in banana.tags
    # box_graspable is curated: large items don't carry it.
    assert "box_graspable" not in get_ycb_catalog()["wood_block"].tags
    assert len(list_ycb_objects(tags="box_graspable")) == 67
    # Variant families are fully present.
    assert sum(1 for i in get_ycb_catalog() if i.endswith("_cups")) == 10
    assert sum(1 for i in get_ycb_catalog() if i.endswith("_lego_duplo")) == 7

def test_catalog_is_cached(assets_dir, no_download):
    assert get_ycb_catalog() is get_ycb_catalog()


# ------------------------------------------------------------------
# list_ycb_objects
# ------------------------------------------------------------------

def test_list_all_objects_sorted(assets_dir):
    ids = list_ycb_objects()
    assert len(ids) == 78
    assert ids == sorted(ids)

def test_list_filter_by_string_tag(assets_dir):
    food = list_ycb_objects(tags="food")
    assert "banana" in food and "hammer" not in food

def test_list_filter_requires_all_tags(assets_dir):
    canned_food = list_ycb_objects(tags={"food", "can"})
    assert "tomato_soup_can" in canned_food
    assert "banana" not in canned_food  # food but not a can
    assert set(canned_food) <= set(list_ycb_objects(tags="food"))

def test_list_unknown_tag_empty(assets_dir):
    assert list_ycb_objects(tags="no_such_tag") == []


# ------------------------------------------------------------------
# make_ycb_pool
# ------------------------------------------------------------------

def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    pool = make_ycb_pool(["mug", "banana", "apple"])
    assert [o.object_id for o in pool] == ["mug", "banana", "apple"]

def test_pool_ids_and_tags_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError):
        make_ycb_pool(["banana"], tags="food")

def test_pool_sampling_reproducible(assets_dir, no_download):
    pool_a = make_ycb_pool(tags="box_graspable", n=5,
                           rng=torch.Generator().manual_seed(42))
    pool_b = make_ycb_pool(tags="box_graspable", n=5,
                           rng=torch.Generator().manual_seed(42))
    ids_a = [o.object_id for o in pool_a]
    assert ids_a == [o.object_id for o in pool_b]
    assert len(set(ids_a)) == 5  # distinct (sampling without replacement)

def test_pool_n_too_large(assets_dir):
    with pytest.raises(ValueError):
        make_ycb_pool(["banana", "mug"], n=3)

def test_pool_kwargs_forwarded(assets_dir, no_download):
    pool = make_ycb_pool(["banana"], scale=0.5, fixed=True)
    assert pool[0].scale == (0.5, 0.5, 0.5)
    assert pool[0].fixed is True


# ------------------------------------------------------------------
# YCBObject construction
# ------------------------------------------------------------------

def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="not_an_object"):
        YCBObject("not_an_object")

def test_construction_downloads_no_object_assets(assets_dir, no_download):
    obj = YCBObject("banana")
    assert obj.object_id == "banana"
    assert obj.name == "banana"
    assert obj.language_annotations == ("fruit", "banana", "yellow banana")
    assert not no_download  # only the (already local) catalog was touched

def test_paths_under_assets_dir(assets_dir):
    obj = YCBObject("mug")
    expected_dir = assets_dir / YCB_ASSETS_ROOT / "mug"
    assert obj.asset_dir == expected_dir
    assert obj.urdf_file == expected_dir / "mug.urdf"
    assert Path(obj.mesh_file) == expected_dir / "textured.obj"

def test_overrides(assets_dir):
    obj = YCBObject("banana", name="my_banana",
                    language_annotations=["the fruit"])
    assert obj.name == "my_banana"
    assert obj.language_annotations == ("the fruit",)


# ------------------------------------------------------------------
# Asset completeness (_ensure_assets)
# ------------------------------------------------------------------

def test_ensure_assets_complete_no_download(fake_banana, no_download):
    obj = YCBObject("banana")
    obj._ensure_assets()  # complete dir -> no Hub access
    assert obj._assets_ready

def test_ensure_assets_partial_dir_triggers_redownload(fake_banana, monkeypatch):
    # Simulate an interrupted download: texture missing, dir exists.
    (fake_banana / "texture_map.png").unlink()
    calls = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        (fake_banana / "texture_map.png").write_text("placeholder")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    obj = YCBObject("banana")
    obj._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == f"{YCB_ASSETS_ROOT}/banana/**"
    assert obj._assets_ready

def test_ensure_assets_urdf_collision_meshes_checked(fake_banana, monkeypatch):
    # A coacd STL referenced by the URDF is missing and the (no-op)
    # re-download does not produce it -> hard failure, not a silent
    # untextured/broken object.
    (fake_banana / "textured_coacd_1.stl").unlink()
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    obj = YCBObject("banana")
    with pytest.raises(FileNotFoundError, match="textured_coacd_1.stl"):
        obj._ensure_assets()
    assert not obj._assets_ready


# ------------------------------------------------------------------
# prepare_ycb_assets.py path rewriting
# ------------------------------------------------------------------

def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_ycb_assets", _SCRIPTS_DIR / "prepare_ycb_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_rewrite_mesh_paths_urdf():
    script = _load_prepare_script()
    text = '<mesh filename="banana/textured.obj" scale="1 1 1" />'
    assert script.rewrite_mesh_paths(text, "banana") == (
        '<mesh filename="textured.obj" scale="1 1 1" />'
    )

def test_rewrite_mesh_paths_mjcf():
    script = _load_prepare_script()
    text = (
        '<mesh name="banana_visual_mesh" file="banana/textured.obj" />\n'
        '<texture name="banana_texture" type="2d" file="banana/texture_map.png" />'
    )
    out = script.rewrite_mesh_paths(text, "banana")
    assert 'file="textured.obj"' in out
    assert 'file="texture_map.png"' in out
    assert "banana/" not in out

def test_rewrite_mesh_paths_leaves_other_ids_alone():
    script = _load_prepare_script()
    text = '<mesh filename="banana/textured.obj" /> name="banana_geom"'
    out = script.rewrite_mesh_paths(text, "apple")
    assert out == text

def test_write_catalog_copies_source(tmp_path):
    script = _load_prepare_script()
    object_ids = list(json.loads(_CATALOG_SOURCE.read_text()))
    script._write_catalog(tmp_path, object_ids)
    written = json.loads((tmp_path / "catalog.json").read_text())
    assert written == json.loads(_CATALOG_SOURCE.read_text())
