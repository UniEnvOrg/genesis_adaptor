"""Tests for Amazon Berkeley Objects integration (no downloads; no scenes,
except the end-to-end CPU Genesis scene-load test).

The catalog ships with the assets as ``objects/abo/catalog.json``; tests
generate it from fabricated listings records through the prepare script's own
builder, so the catalog content checks run against the real generation code.
The GLB → Z-up OBJ conversion and the texture rewriting are exercised against
small synthetic GLB/OBJ fixtures built in-test; the same synthetic GLB is
packaged and loaded through a real headless Genesis scene to pin the placed
entity's Z-up frame.

Covers:
  - Generated catalog integrity (annotations broad → detailed, separate tags)
  - list_abo_objects tag / substring filtering
  - make_abo_pool selection / sampling / validation
  - ABOObject construction, path resolution, lazy asset completeness
  - prepare_abo_assets.py metadata join / cleaning / catalog + graspable bake
  - prepare_abo_assets.py GLB → Z-up OBJ conversion and texture rewriting
  - Genesis scene load (CPU): packed OBJ → ABOObject.load_into_scene →
    entity AABB is Z-up and standing on the ground
"""

import gzip
import importlib.util
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.hub_objects.abo_object import (
    ABO_ASSETS_ROOT,
    ABOObject,
    get_abo_catalog,
    list_abo_objects,
    make_abo_pool,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_GRASPABLE_CACHE = _SCRIPTS_DIR / "data" / "abo_graspable.json"

_GRASPABLE_ID = "B000000001"
_EN_GB_ID = "B000000002"
_UNLISTED_ID = "B000000003"

# Fabricated listings records (the shape the ABO listings shards use).
_LISTINGS = {
    _GRASPABLE_ID: {
        "3dmodel_id": _GRASPABLE_ID,
        "product_type": [{"value": "HOME_FURNITURE_AND_DECOR"}],
        "node": [{"node_id": 1, "node_name": "/Products/Home/Furniture/Footstools & Ottomans"}],
        "item_name": [
            {"language_tag": "zh_CN", "value": "中文名称"},
            {"language_tag": "en_US", "value": 'AmazonBasics Velvet Storage Ottoman, 23"W, Grey'},
        ],
        "bullet_point": [
            {"language_tag": "en_US", "value": '23.62"W x 23.62"D x 19.69"H'},
            {"language_tag": "en_US", "value": "Soft velvet cover with lift-off lid"},
            {"language_tag": "zh_CN", "value": "中文要点"},
        ],
    },
    _EN_GB_ID: {
        "3dmodel_id": _EN_GB_ID,
        "product_type": [{"value": "WALL_ART"}],
        "node": [{"node_id": 2, "node_name": "/Products/Home Decor/Wall Art"}],
        "item_name": [{"language_tag": "en_GB", "value": "Amazon Brand Framed Print"}],
        "bullet_point": [{"language_tag": "en_GB", "value": "Ready to hang"}],
    },
}

_MATERIAL_MTL = """\
newmtl material_0
Ka 0.40000000 0.40000000 0.40000000
Kd 1.00000000 1.00000000 1.00000000
Ns 1.00000000
map_Kd texture.png
"""

_OBJ_TEMPLATE = """\
mtllib material.mtl
o model
usemtl material_0
v {x0} {y0} {z0}
v {x1} {y0} {z0}
v {x0} {y1} {z0}
v {x0} {y0} {z1}
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""

_OBJECT_FILES = ("model.obj", "material.mtl", "texture.png")


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_abo_assets", _SCRIPTS_DIR / "prepare_abo_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the packaging worker processes pickle
    # `_package_task` by reference, which requires the module to live under
    # its own name in sys.modules (spec loading alone skips that).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_shard(path: Path, records) -> Path:
    """Write a gzip-compressed listings shard (one JSON record per line)."""
    with gzip.open(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _make_fake_object_dir(assets_dir: Path, object_id: str) -> Path:
    """Create a complete fake ABO object directory under *assets_dir*."""
    obj_dir = assets_dir / ABO_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.obj").write_text(
        _OBJ_TEMPLATE.format(x0=0, y0=0, z0=0, x1=1, y1=1, z1=1)
    )
    (obj_dir / "material.mtl").write_text(_MATERIAL_MTL)
    (obj_dir / "texture.png").write_text("placeholder")
    return obj_dir


def _make_glb(path: Path, *, textured: bool = True, extents=(0.2, 0.4, 0.1)) -> Path:
    """Export a synthetic GLB (optionally textured, with a quoted material)."""
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    mesh = trimesh.creation.box(extents=extents)
    if textured:
        image = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8))
        mesh.visual = TextureVisuals(
            uv=mesh.vertices[:, :2].copy(),
            material=PBRMaterial(
                name='"ABC"', baseColorTexture=image,
                baseColorFactor=[255, 255, 255, 255],
            ),
        )
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="abo", geom_name="model")
    path.write_bytes(scene.export(file_type="glb"))
    return path


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir with catalog.json generated from fabricated listings."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    script = _load_prepare_script()
    out_root = tmp_path / ABO_ASSETS_ROOT
    out_root.mkdir(parents=True)
    records = script.build_catalog_records(
        list(_LISTINGS) + [_UNLISTED_ID], _LISTINGS, {_GRASPABLE_ID},
    )
    script.write_catalog(out_root, records)
    return tmp_path


@pytest.fixture
def fake_object(assets_dir):
    """A complete fake sample object directory."""
    return _make_fake_object_dir(assets_dir, _GRASPABLE_ID)


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
# Generated catalog
# ------------------------------------------------------------------

def test_catalog_integrity(assets_dir):
    catalog = get_abo_catalog()
    assert set(catalog) == {_GRASPABLE_ID, _EN_GB_ID, _UNLISTED_ID}
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert info.language_annotations and all(info.language_annotations)
        # Annotations are lowercase and free of the dimension strings the
        # listings carry ("23.62\"W x ...").
        for annotation in info.language_annotations:
            assert annotation == annotation.lower()
            assert '"' not in annotation
        for tag in info.tags:
            assert tag == tag.lower() and " " not in tag


def test_catalog_spot_checks(assets_dir):
    ottoman = get_abo_catalog()[_GRASPABLE_ID]
    # Broad (product type → catalog node) → detailed (item name → bullets).
    assert ottoman.language_annotations == (
        "home furniture and decor",
        "footstools & ottomans",
        "amazonbasics velvet storage ottoman, grey",
        "soft velvet cover with lift-off lid",
    )
    assert ottoman.tags == frozenset(
        {"home_furniture_and_decor", "footstools_ottomans", "box_gripper_graspable"}
    )
    assert ottoman.standing_quaternion is None
    # The en_US item name wins over the (first) zh_CN one; an object without
    # any listing still gets a generic, id-derived annotation.
    assert list_abo_objects(tags="box_gripper_graspable") == [_GRASPABLE_ID]
    assert get_abo_catalog()[_EN_GB_ID].language_annotations[0] == "wall art"
    assert get_abo_catalog()[_UNLISTED_ID].language_annotations == (
        "object", f"product {_UNLISTED_ID.lower()}",
    )
    assert get_abo_catalog()[_UNLISTED_ID].tags == frozenset()


def test_graspable_cache_is_checkable_data():
    """The checked-in graspable list is a sorted list of object ids."""
    ids = json.loads(_GRASPABLE_CACHE.read_text())
    assert isinstance(ids, list) and all(isinstance(i, str) for i in ids)
    assert ids == sorted(ids)


def test_gripper_width_matches_compute_graspable():
    """The packager's inline bake uses the shared gripper width / tag."""
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        import compute_graspable  # noqa: PLC0415
    finally:
        sys.path.remove(str(_SCRIPTS_DIR))
    script = _load_prepare_script()
    assert script.GRIPPER_WIDTH == compute_graspable.GRIPPER_WIDTH
    assert script.GRASPABLE_TAG == compute_graspable.TAG
    assert compute_graspable._MESH_NAME["abo"] == "model.obj"


# ------------------------------------------------------------------
# list_abo_objects
# ------------------------------------------------------------------

def test_list_all_objects(assets_dir):
    assert list_abo_objects() == sorted(
        [_GRASPABLE_ID, _EN_GB_ID, _UNLISTED_ID]
    )


def test_list_filter_by_tag(assets_dir):
    wall_art = list_abo_objects(tags="wall_art")
    assert wall_art == [_EN_GB_ID]
    assert all("wall_art" in get_abo_catalog()[oid].tags for oid in wall_art)


def test_list_contains_filter_case_insensitive(assets_dir):
    matches = list_abo_objects(contains="00000000")
    assert matches == sorted([_GRASPABLE_ID, _EN_GB_ID, _UNLISTED_ID])
    assert list_abo_objects(contains="b000000002") == [_EN_GB_ID]


def test_list_tags_and_contains_combine(assets_dir):
    assert list_abo_objects(tags="wall_art", contains="b000") == [_EN_GB_ID]
    assert list_abo_objects(tags="wall_art", contains="nope") == []


def test_list_no_match(assets_dir):
    assert list_abo_objects(contains="zzz_no_such_object") == []
    assert list_abo_objects(tags="zzz_no_such_tag") == []


# ------------------------------------------------------------------
# make_abo_pool
# ------------------------------------------------------------------

def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    ids = [_EN_GB_ID, _GRASPABLE_ID]
    pool = make_abo_pool(ids)
    assert [o.object_id for o in pool] == ids


def test_pool_ids_and_filters_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError):
        make_abo_pool([_GRASPABLE_ID], contains="0000")
    with pytest.raises(ValueError):
        make_abo_pool([_GRASPABLE_ID], tags="wall_art")


def test_pool_tags_filter(assets_dir, no_download):
    pool = make_abo_pool(tags="wall_art", n=1,
                         rng=torch.Generator().manual_seed(1))
    assert [o.object_id for o in pool] == [_EN_GB_ID]


def test_pool_sampling_reproducible(assets_dir, no_download):
    pool_a = make_abo_pool(n=2, rng=torch.Generator().manual_seed(7))
    pool_b = make_abo_pool(n=2, rng=torch.Generator().manual_seed(7))
    ids_a = [o.object_id for o in pool_a]
    assert ids_a == [o.object_id for o in pool_b]
    assert len(set(ids_a)) == 2


def test_pool_n_too_large(assets_dir):
    with pytest.raises(ValueError):
        make_abo_pool([_GRASPABLE_ID], n=2)


# ------------------------------------------------------------------
# ABOObject construction
# ------------------------------------------------------------------

def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="not_an_object"):
        ABOObject("not_an_object")


def test_construction_downloads_no_object_assets(assets_dir, no_download):
    obj = ABOObject(_GRASPABLE_ID)
    assert obj.object_id == _GRASPABLE_ID
    assert obj.name == _GRASPABLE_ID
    assert obj.language_annotations == (
        "home furniture and decor",
        "footstools & ottomans",
        "amazonbasics velvet storage ottoman, grey",
        "soft velvet cover with lift-off lid",
    )
    assert not no_download  # only the (already local) catalog was touched


def test_paths_and_scaling(assets_dir):
    expected_dir = assets_dir / ABO_ASSETS_ROOT / _GRASPABLE_ID
    obj = ABOObject(_GRASPABLE_ID)
    assert obj.asset_dir == expected_dir
    assert Path(obj.mesh_file) == expected_dir / "model.obj"
    assert Path(obj.visual_texture_path) == expected_dir / "texture.png"
    assert obj._morph_scale == 1.0
    # gs.morphs.Mesh supports per-axis scaling.
    anisotropic = ABOObject(_GRASPABLE_ID, scale=(1.0, 2.0, 1.0))
    assert anisotropic._morph_scale == (1.0, 2.0, 1.0)


def test_overrides(assets_dir):
    obj = ABOObject(_GRASPABLE_ID, name="ottoman", language_annotations=["tiny box"])
    assert obj.name == "ottoman"
    assert obj.language_annotations == ("tiny box",)


def test_local_bounding_box_uses_converted_mesh(fake_object, no_download):
    (fake_object / "model.obj").write_text(
        _OBJ_TEMPLATE.format(x0=-1, y0=-2, z0=-3, x1=4, y1=5, z1=6)
    )
    obj = ABOObject(_GRASPABLE_ID)
    bbox = obj.get_local_bounding_box()
    assert torch.allclose(
        bbox.low, torch.tensor([-1.0, -2.0, -3.0], dtype=bbox.low.dtype),
    )
    assert torch.allclose(
        bbox.high, torch.tensor([4.0, 5.0, 6.0], dtype=bbox.high.dtype),
    )


# ------------------------------------------------------------------
# Asset completeness (_ensure_assets)
# ------------------------------------------------------------------

def test_ensure_assets_complete_no_download(fake_object, no_download):
    obj = ABOObject(_GRASPABLE_ID)
    obj._ensure_assets()
    assert obj._assets_ready


def test_ensure_assets_partial_dir_triggers_redownload(fake_object, monkeypatch):
    (fake_object / "texture.png").unlink()
    calls = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        (fake_object / "texture.png").write_text("placeholder")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    obj = ABOObject(_GRASPABLE_ID)
    obj._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == (
        f"{ABO_ASSETS_ROOT}/{_GRASPABLE_ID}/**"
    )
    assert obj._assets_ready


def test_ensure_assets_missing_material_raises(fake_object, monkeypatch):
    (fake_object / "material.mtl").unlink()
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    obj = ABOObject(_GRASPABLE_ID)
    with pytest.raises(FileNotFoundError, match="material.mtl"):
        obj._ensure_assets()
    assert not obj._assets_ready


# ------------------------------------------------------------------
# prepare_abo_assets.py: annotation / catalog helpers
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ('87"W x 37.75\'\'D Kitchen Cart', "kitchen cart"),
        ('Stone & Beam Dalton Ottoman, 23"W, Charcoal',
         "stone & beam dalton ottoman, charcoal"),
        ("AmazonBasics Adjustable Speaker Stand - 3.8 to 6-Foot, Steel",
         "amazonbasics adjustable speaker stand, steel"),
        ('39.5\'\' Diameter x 17.25\'\'H', "diameter"),
        ("Product <b>with</b> HTML &amp; entity", "product with html & entity"),
        ("2-Shelf Bookcase, 1500W Heater, Set of 2",
         "2-shelf bookcase, heater, set of 2"),
        ("Lamp (small)", "lamp"),
        ('23.62"W x 23.62"D x 19.69"H', ""),
        ("", ""),
    ],
)
def test_clean_annotation(raw, expected):
    script = _load_prepare_script()
    assert script.clean_annotation(raw) == expected


def test_clean_annotation_truncates_long_text():
    script = _load_prepare_script()
    cleaned = script.clean_annotation("word " * 100)
    assert len(cleaned) <= script._MAX_ANNOTATION_LENGTH
    assert cleaned and not cleaned.endswith(" ")


def test_normalize_tag_and_node_leaves():
    script = _load_prepare_script()
    assert script.normalize_tag("Footstools & Ottomans") == "footstools_ottomans"
    assert script.normalize_tag("OFFICE_FURNITURE_&_LIGHTING") == (
        "office_furniture_lighting"
    )
    assert script.node_leaf_names(
        [{"node_id": 1, "node_name": "/Products/Home/Furniture/Stools/"}]
    ) == ["Stools"]


def test_localized_value_prefers_en_us():
    script = _load_prepare_script()
    values = [
        {"language_tag": "de_DE", "value": "deutsch"},
        {"language_tag": "en_US", "value": "english"},
        {"language_tag": "en_GB", "value": "british"},
    ]
    assert script.localized_value(values) == "english"
    assert script.localized_value([{"language_tag": "en_GB", "value": "british"}]) == "british"
    # No English at all: first occurrence (deterministic), never empty.
    assert script.localized_value([{"language_tag": "de_DE", "value": "deutsch"}]) == "deutsch"
    assert script.localized_value(None) is None


def test_build_catalog_entry_falls_back_without_listing():
    script = _load_prepare_script()
    entry = script.build_catalog_entry("B000000009", None)
    assert entry["language_annotations"] == ["object", "product b000000009"]
    assert entry["tags"] == []
    # A graspable verdict is a tag only, never an annotation.
    graspable = script.build_catalog_entry("B000000009", None, graspable=True)
    assert graspable["tags"] == ["box_gripper_graspable"]
    assert "box_gripper_graspable" not in graspable["language_annotations"]


def test_build_catalog_entry_orders_broad_to_detailed():
    script = _load_prepare_script()
    entry = script.build_catalog_entry(_GRASPABLE_ID, _LISTINGS[_GRASPABLE_ID])
    assert entry["language_annotations"] == [
        "home furniture and decor",
        "footstools & ottomans",
        "amazonbasics velvet storage ottoman, grey",
        "soft velvet cover with lift-off lid",
    ]
    # The dimension-only bullet is dropped, the zh_CN ones are ignored.
    assert not any("23.62" in a for a in entry["language_annotations"])
    assert entry["tags"] == [
        "footstools_ottomans", "home_furniture_and_decor",
    ]


def test_write_catalog_merges_incremental_runs(tmp_path):
    script = _load_prepare_script()
    out_root = tmp_path / "objects" / "abo"
    out_root.mkdir(parents=True)
    first = script.build_catalog_records([_GRASPABLE_ID], _LISTINGS, {_GRASPABLE_ID})
    script.write_catalog(out_root, first)
    second = script.build_catalog_records([_EN_GB_ID], _LISTINGS, set())
    script.write_catalog(out_root, second)
    catalog = json.loads((out_root / "catalog.json").read_text())
    assert set(catalog) == {_GRASPABLE_ID, _EN_GB_ID}
    assert "box_gripper_graspable" in catalog[_GRASPABLE_ID]["tags"]


def test_merge_graspable_accumulates(tmp_path):
    script = _load_prepare_script()
    path = tmp_path / "abo_graspable.json"
    assert script.merge_graspable(["B1", "B2"], path) == {"B1", "B2"}
    assert script.merge_graspable(["B2", "B0"], path) == {"B0", "B1", "B2"}
    assert json.loads(path.read_text()) == ["B0", "B1", "B2"]


# ------------------------------------------------------------------
# prepare_abo_assets.py: process-isolated packaging (OOM regression)
# ------------------------------------------------------------------

def test_package_all_runs_in_recycled_worker_processes(tmp_path, monkeypatch):
    """The per-object trimesh work runs in short-lived worker *processes*.

    It used to run on a thread pool, i.e. inside the packing process itself:
    each object's transient trimesh allocations ratcheted that process's RSS
    by a few MB (observed: >12 GB during the resume scan of the ~4,900
    already-prepared objects) until the kernel OOM-killed the run.  The fix
    isolates the work in worker processes and recycles the pool, so a process
    exit hands every retained byte back to the OS.
    """
    script = _load_prepare_script()
    ids = [f"B{i:09d}" for i in range(5)]
    for object_id in ids:
        _make_fake_object_dir(tmp_path, object_id)
    out_root = tmp_path / ABO_ASSETS_ROOT

    pools, worker_pids = [], set()
    real_pool = script.ProcessPoolExecutor

    class _RecordingPool(real_pool):  # noqa: N801
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

        def shutdown(self, *args, **kwargs):
            # Capture the worker pids before the pool clears them.
            worker_pids.update(p.pid for p in (self._processes or {}).values())
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(script, "ProcessPoolExecutor", _RecordingPool)
    # One task per epoch: every object must get a fresh pool (workers exit).
    monkeypatch.setattr(script, "_TASKS_PER_WORKER_EPOCH", 1)

    results, failures = script._package_all(
        out_root, {object_id: {} for object_id in ids}, ids,
        keep_glb=False, workers=1,
    )

    assert failures == {}
    assert list(results) == ids
    for graspable, extents, deviation in results.values():
        # The fake mesh is a unit cube: not graspable, no deviation on resume.
        assert graspable is False
        assert extents == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
        assert deviation == 0.0
    assert len(pools) == len(ids)
    # The work really ran in child processes, never in the parent.
    assert worker_pids and os.getpid() not in worker_pids


def test_package_all_reports_broken_objects_and_continues(tmp_path):
    """One unreadable object is a per-object failure, not the end of the run."""
    script = _load_prepare_script()
    ids = [f"B{i:09d}" for i in range(3)]
    for object_id in ids:
        _make_fake_object_dir(tmp_path, object_id)
    out_root = tmp_path / ABO_ASSETS_ROOT
    (out_root / ids[1] / "model.obj").unlink()  # resume path cannot read it

    results, failures = script._package_all(
        out_root, {object_id: {} for object_id in ids}, ids,
        keep_glb=False, workers=2,
    )

    assert set(results) == {ids[0], ids[2]}
    assert list(failures) == [ids[1]]
    assert failures[ids[1]]  # non-empty "ExceptionType: message" report


def test_package_all_survives_a_worker_that_dies(tmp_path, monkeypatch):
    """A worker killed mid-object (e.g. kernel OOM on a huge model) only fails
    the objects it swallowed; the run continues in a fresh pool.

    The worker is killed via a ``package_object`` patched in the test process,
    which is inherited by forked workers — hence the fork requirement.
    """
    if multiprocessing.get_start_method() != "fork":
        pytest.skip("needs the fork start method to inject a dying worker")
    script = _load_prepare_script()
    ids = [f"B{i:09d}" for i in range(4)]
    for object_id in ids:
        _make_fake_object_dir(tmp_path, object_id)
    out_root = tmp_path / ABO_ASSETS_ROOT
    real_package_object = script.package_object

    def _dying_package_object(object_id, *args, **kwargs):
        if object_id == ids[1]:
            os._exit(17)  # simulate a killed worker: no Python exception
        return real_package_object(object_id, *args, **kwargs)

    monkeypatch.setattr(script, "package_object", _dying_package_object)
    monkeypatch.setattr(script, "_TASKS_PER_WORKER_EPOCH", 1)  # one object per pool

    results, failures = script._package_all(
        out_root, {object_id: {} for object_id in ids}, ids,
        keep_glb=False, workers=1,
    )

    assert set(results) == {ids[0], ids[2], ids[3]}
    assert list(failures) == [ids[1]]
    assert "worker process died" in failures[ids[1]]


def test_select_object_ids():
    script = _load_prepare_script()
    index = {"B2": {}, "B1": {}, "B3": {}}
    assert script.select_object_ids(index) == ["B1", "B2", "B3"]
    assert script.select_object_ids(index, limit=2) == ["B1", "B2"]
    assert script.select_object_ids(index, objects="B3,B1") == ["B1", "B3"]
    with pytest.raises(ValueError, match="not both"):
        script.select_object_ids(index, objects="B1", limit=1)
    with pytest.raises(ValueError, match="Unknown ABO object id"):
        script.select_object_ids(index, objects="nope")
    with pytest.raises(ValueError, match="positive"):
        script.select_object_ids(index, limit=0)


# ------------------------------------------------------------------
# prepare_abo_assets.py: listings join
# ------------------------------------------------------------------

def test_find_listing_records_prefers_en_us(tmp_path, monkeypatch):
    script = _load_prepare_script()
    shard = _write_shard(
        tmp_path / "listings_0.json.gz",
        [
            {   # first occurrence, no English metadata
                "3dmodel_id": _GRASPABLE_ID,
                "item_name": [{"language_tag": "de_DE", "value": "deutsch"}],
            },
            _LISTINGS[_GRASPABLE_ID],  # same model id, en_US listing
            {"3dmodel_id": "B0000000XX", "item_name": []},
        ],
    )
    monkeypatch.setattr(
        script, "cached_resource", lambda cache_dir, key, refresh=False: shard,
    )
    records = script.find_listing_records(tmp_path, [_GRASPABLE_ID, _EN_GB_ID])
    assert script.localized_value(records[_GRASPABLE_ID]["item_name"]) == (
        _LISTINGS[_GRASPABLE_ID]["item_name"][1]["value"]
    )
    # No listing at all: the id is simply absent (the catalog falls back).
    assert _EN_GB_ID not in records


def test_find_listing_records_keeps_first_without_english(tmp_path, monkeypatch):
    script = _load_prepare_script()
    fallback = {
        "3dmodel_id": _GRASPABLE_ID,
        "item_name": [{"language_tag": "de_DE", "value": "deutsch"}],
    }
    shard = _write_shard(tmp_path / "listings_0.json.gz", [fallback])
    monkeypatch.setattr(
        script, "cached_resource", lambda cache_dir, key, refresh=False: shard,
    )
    records = script.find_listing_records(tmp_path, [_GRASPABLE_ID])
    # No English listing anywhere: the first occurrence is used as-is.
    assert records[_GRASPABLE_ID] == fallback
    assert script.localized_value(records[_GRASPABLE_ID]["item_name"]) == "deutsch"


def test_iter_listings_reads_gzip_jsonl(tmp_path):
    script = _load_prepare_script()
    shard = _write_shard(tmp_path / "shard.json.gz", [_LISTINGS[_EN_GB_ID]])
    assert list(script.iter_listings(shard)) == [_LISTINGS[_EN_GB_ID]]


# ------------------------------------------------------------------
# prepare_abo_assets.py: GLB → Z-up OBJ
# ------------------------------------------------------------------

def test_convert_glb_to_zup_obj_bakes_orientation(tmp_path):
    import trimesh

    script = _load_prepare_script()
    glb = _make_glb(tmp_path / "model.glb", extents=(0.2, 0.4, 0.1))
    obj_dir = tmp_path / "object"
    extents = script.convert_glb_to_zup_obj(glb.read_bytes(), obj_dir)

    # The GLB's up axis (Y, 0.4) becomes Z; the conversions' extents match the
    # source mesh, and the OBJ on disk is already Z-up (nothing to transform
    # at load time — otherwise the trimesh bounding box would disagree).
    assert extents == pytest.approx([0.2, 0.1, 0.4], abs=1e-6)
    loaded = trimesh.load(obj_dir / "model.obj", force="mesh")
    assert loaded.extents == pytest.approx([0.2, 0.1, 0.4], abs=1e-4)

    # Texture: the GLB material's name (quotes included) is gone, the OBJ
    # references the stable texture.png and the image is bound on reload.
    assert sorted(p.name for p in obj_dir.iterdir()) == [
        "material.mtl", "model.obj", "texture.png",
    ]
    material = (obj_dir / "material.mtl").read_text()
    assert "map_Kd texture.png" in material
    assert '"ABC".png' not in material  # the material name itself is kept
    assert loaded.visual.material.image is not None
    obj_text = (obj_dir / "model.obj").read_text()
    assert '"ABC".png' not in obj_text
    assert "mtllib material.mtl" in obj_text


def test_convert_glb_without_texture_writes_placeholder(tmp_path):
    from PIL import Image

    script = _load_prepare_script()
    glb = _make_glb(tmp_path / "model.glb", textured=False)
    obj_dir = tmp_path / "object"
    script.convert_glb_to_zup_obj(glb.read_bytes(), obj_dir)

    texture = obj_dir / "texture.png"
    assert texture.is_file()
    assert Image.open(texture).size == (4, 4)
    # A material-less GLB exports no material at all: the packager writes one
    # so the per-object layout stays uniform and self-consistent.
    material = (obj_dir / "material.mtl").read_text()
    assert material.endswith("map_Kd texture.png\n")
    obj = (obj_dir / "model.obj").read_text()
    assert "usemtl material_0" in obj and "mtllib material.mtl" in obj


def test_rewrite_texture_references_renames_and_rewrites(tmp_path):
    script = _load_prepare_script()
    obj_dir = tmp_path
    (obj_dir / '"ABC".png').write_bytes(b"first")
    (obj_dir / "second.png").write_bytes(b"second")
    (obj_dir / "model.obj").write_text(
        'mtllib material.mtl\nusemtl "ABC"\n'
    )
    (obj_dir / "material.mtl").write_text(
        'newmtl "ABC"\nmap_Kd "ABC".png\nnewmtl second\nmap_Kd second.png\n'
    )
    script.rewrite_texture_references(obj_dir)

    assert (obj_dir / "texture.png").read_bytes() == b"first"
    assert (obj_dir / "texture_1.png").read_bytes() == b"second"
    material = (obj_dir / "material.mtl").read_text()
    assert material == (
        'newmtl "ABC"\nmap_Kd texture.png\nnewmtl second\nmap_Kd texture_1.png\n'
    )
    # The material name (which references no file) is left untouched.
    assert (obj_dir / "model.obj").read_text() == 'mtllib material.mtl\nusemtl "ABC"\n'


# ------------------------------------------------------------------
# End-to-end Genesis scene load (CPU, headless)
# ------------------------------------------------------------------

@pytest.fixture
def genesis_cpu():
    """Genesis must be initialised before morphs/entities can be built."""
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu)
    return gs


def test_load_into_scene_places_z_up_entity(
    genesis_cpu, assets_dir, tmp_path,
):
    """Packed GLB → Z-up OBJ → Genesis AABB stays Z-up.

    The OBJ on disk is already Z-up (the packager bakes R_x(+90°)) and
    ``gs.morphs.Mesh`` must load it as such — ``file_meshes_are_zup``
    defaults to True, i.e. no implicit transform for OBJ.  A Genesis-side
    regression that applied the Y-up ``(X, Y, Z) → (X, -Z, Y)`` conversion
    would flip the 0.4 m tall axis to Y and move the z-min.
    """
    from unienv_genesis import GenesisWorld

    script = _load_prepare_script()
    # The source GLB is Y-up, tall along Y (0.4); after the packager's baked
    # rotation the object must be tall along Z (0.4).
    glb = _make_glb(tmp_path / "model.glb", extents=(0.2, 0.4, 0.1))
    obj_dir = assets_dir / ABO_ASSETS_ROOT / _GRASPABLE_ID
    script.convert_glb_to_zup_obj(glb.read_bytes(), obj_dir)

    obj = ABOObject(_GRASPABLE_ID)
    local = obj.get_local_bounding_box()  # trimesh-side prediction
    assert (local.high - local.low).tolist() == pytest.approx(
        [0.2, 0.1, 0.4], abs=1e-4,
    )

    world = GenesisWorld()
    try:
        # Stand the object on the ground: the entity origin goes one local
        # z-min above z=0, so the placed z-min must land back on z≈0.
        pos = (0.0, 0.0, -float(local.low[2]))
        entity = obj.load_into_scene(world, pos=pos)
        world.scene.build(n_envs=0)

        aabb = entity.get_AABB()  # Genesis ≥1.0 name (get_aabb is removed)
        low, high = aabb[0].cpu().tolist(), aabb[1].cpu().tolist()
        extents = [hi - lo for lo, hi in zip(low, high)]

        # Z-up: the largest extent (0.4) is along Z, never Y, and the placed
        # AABB matches the trimesh bbox translated by *pos*.
        assert extents == pytest.approx([0.2, 0.1, 0.4], abs=2e-3)
        assert extents[2] == max(extents)
        assert low == pytest.approx([-0.1, -0.05, 0.0], abs=2e-3)
        assert high == pytest.approx([0.1, 0.05, 0.4], abs=2e-3)
        # Standing on the ground plane: z-min at the placement height ≈ 0.
        assert low[2] == pytest.approx(0.0, abs=2e-3)
    finally:
        world.scene.destroy()
