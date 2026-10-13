"""Tests for ReplicaCAD furniture integration (no asset downloads).

The catalog ships with the assets as ``objects/replicacad/catalog.json``; tests
install the curated source (the ``CATALOG`` constant of
``scripts/prepare_replicacad_assets.py``) into a temporary assets dir, so
catalog content checks run against the real packaging data.

Covers:
  - Catalog integrity (annotations broad → detailed, tags incl.
    furniture/articulated, unit ``standing_quaternion``)
  - list_replicacad_objects tag filtering
  - make_replicacad_pool selection / sampling / validation
  - ReplicaCADFurniture construction and asset path resolution
  - Lazy asset completeness check (_ensure_assets) incl. partial-download repair
  - Offline FK union bbox sanity (urdf_link_meshes_at_config)
  - scripts/prepare_replicacad_assets.py: mesh-path rewriting, upstream
    comma-separated vector normalization, catalog cross-check, item repackage

A GPU smoke test (skipped unless CUDA is available) loads a tiny synthetic
URDF through ``load_into_scene`` and cross-checks the offline union bbox
against ``entity.get_vAABB()`` at the identity pose — the same gate the
prepare pipeline runs per item.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.articulated_object import (
    ArticulationError,
    urdf_link_meshes_at_config,
)
from unienv_genesis_collection.objects.hub_objects.articulated.replicacad_furniture import (
    REPLICACAD_ASSETS_ROOT,
    ReplicaCADFurniture,
    get_replicacad_catalog,
    list_replicacad_objects,
    make_replicacad_pool,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_replicacad_assets", _SCRIPTS_DIR / "prepare_replicacad_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_prepare_script()
_ITEM_IDS = list(_SCRIPT.ARTICULATED_ITEMS)


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------

# Welded base (visual mesh) + a door link on a revolute joint — the shape of
# the upstream ReplicaCAD URDFs, with directory-local filename references.
_URDF_TEMPLATE = """\
<robot name="{oid}">
    <link name="root">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry><mesh filename="body.glb" scale="1 1 1" /></geometry>
        </visual>
        <collision>
            <geometry><box size="0.6 0.6 1.8" /></geometry>
        </collision>
    </link>
    <joint name="door_hinge" type="revolute">
        <origin xyz="0.3 0 0" rpy="0 0 0" />
        <parent link="root" />
        <child link="door" />
        <axis xyz="0 0 1" />
        <limit effort="44.4" lower="0.0" upper="2.3" velocity="40" />
    </joint>
    <link name="door">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry><mesh filename="door.glb" scale="1 1 1" /></geometry>
        </visual>
        <collision>
            <geometry><box size="0.05 0.6 1.8" /></geometry>
        </collision>
    </link>
</robot>
"""

_ITEM_FILES = ("body.glb", "door.glb")
_BODY_EXTENTS = (0.6, 0.6, 1.8)
_DOOR_EXTENTS = (0.05, 0.6, 1.8)


def _write_box_glb(path: Path, extents: tuple[float, float, float]) -> None:
    """Write a real centered box mesh (GLB) to *path*."""
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=extents).export(path)


def _make_fake_item_dir(assets_dir: Path, object_id: str) -> Path:
    """Create a complete fake ReplicaCAD item directory under *assets_dir*."""
    import trimesh

    obj_dir = assets_dir / REPLICACAD_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    (obj_dir / f"{object_id}.urdf").write_text(_URDF_TEMPLATE.format(oid=object_id))
    _write_box_glb(obj_dir / "body.glb", _BODY_EXTENTS)
    _write_box_glb(obj_dir / "door.glb", _DOOR_EXTENTS)
    return obj_dir


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir holding the real curated catalog.json."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    catalog_file = tmp_path / REPLICACAD_ASSETS_ROOT / "catalog.json"
    catalog_file.parent.mkdir(parents=True)
    catalog_file.write_text(json.dumps(_SCRIPT.CATALOG))
    return tmp_path


@pytest.fixture
def fake_fridge(assets_dir):
    """A complete fake 'fridge' item directory."""
    return _make_fake_item_dir(assets_dir, "fridge")


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
    catalog = get_replicacad_catalog()
    assert len(catalog) == len(_ITEM_IDS)
    assert set(catalog) == set(_ITEM_IDS)
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert " " not in object_id
        # Multiple annotations, broad -> detailed, separate from tags.
        assert len(info.language_annotations) >= 2
        assert len(info.language_annotations[0]) <= len(info.language_annotations[-1])
        assert info.tags
        assert {"furniture", "articulated"} <= info.tags
        for tag in info.tags:
            assert tag == tag.lower() and " " not in tag
        # A unit (w, x, y, z) quaternion when present.
        if info.standing_quaternion is not None:
            quat = torch.tensor(info.standing_quaternion, dtype=torch.float64)
            assert torch.isclose(quat.norm(), torch.tensor(1.0, dtype=torch.float64))


def test_catalog_spot_checks(assets_dir):
    catalog = get_replicacad_catalog()
    fridge = catalog["fridge"]
    assert {"furniture", "kitchen", "appliance"} <= fridge.tags
    # Kitchen-tagged items are exactly the kitchen units/appliances.
    assert set(list_replicacad_objects(tags="kitchen")) == {
        "fridge", "kitchen_counter", "kitchenCupboard_01",
    }
    # Every item carries the furniture + articulated tags.
    assert len(list_replicacad_objects(tags={"furniture", "articulated"})) == len(catalog)


def test_catalog_is_cached(assets_dir, no_download):
    assert get_replicacad_catalog() is get_replicacad_catalog()


# ------------------------------------------------------------------
# list_replicacad_objects
# ------------------------------------------------------------------

def test_list_all_objects_sorted(assets_dir):
    ids = list_replicacad_objects()
    assert ids == sorted(ids)
    assert set(ids) == set(_ITEM_IDS)


def test_list_filter_by_string_tag(assets_dir):
    doors = list_replicacad_objects(tags="door")
    assert {"door1", "door2", "door3", "doubledoor"} <= set(doors)
    assert "fridge" not in doors


def test_list_filter_requires_all_tags(assets_dir):
    kitchen_appliances = list_replicacad_objects(tags={"kitchen", "appliance"})
    assert "fridge" in kitchen_appliances
    assert "kitchen_counter" not in kitchen_appliances  # kitchen but not an appliance
    assert set(kitchen_appliances) <= set(list_replicacad_objects(tags="kitchen"))


def test_list_unknown_tag_empty(assets_dir):
    assert list_replicacad_objects(tags="no_such_tag") == []


# ------------------------------------------------------------------
# make_replicacad_pool
# ------------------------------------------------------------------

def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    pool = make_replicacad_pool(["fridge", "door1", "cabinet"])
    assert [o.object_id for o in pool] == ["fridge", "door1", "cabinet"]


def test_pool_ids_and_tags_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError):
        make_replicacad_pool(["fridge"], tags="furniture")


def test_pool_sampling_reproducible(assets_dir, no_download):
    pool_a = make_replicacad_pool(tags="articulated", n=5,
                                  rng=torch.Generator().manual_seed(42))
    pool_b = make_replicacad_pool(tags="articulated", n=5,
                                  rng=torch.Generator().manual_seed(42))
    ids_a = [o.object_id for o in pool_a]
    assert ids_a == [o.object_id for o in pool_b]
    assert len(set(ids_a)) == 5  # distinct (sampling without replacement)


def test_pool_n_too_large(assets_dir):
    with pytest.raises(ValueError):
        make_replicacad_pool(["fridge", "door1"], n=3)


def test_pool_kwargs_forwarded(assets_dir, no_download):
    pool = make_replicacad_pool(["fridge"], scale=0.5, fixed=True)
    # The hub articulated base keeps scalar scales as-is and forwards them
    # (unlike the YCB mesh class, which normalizes to a per-axis tuple).
    assert pool[0].scale == 0.5
    assert pool[0]._morph_scale == 0.5
    assert pool[0].fixed is True


# ------------------------------------------------------------------
# ReplicaCADFurniture construction
# ------------------------------------------------------------------

def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="not_an_item"):
        ReplicaCADFurniture("not_an_item")


def test_construction_downloads_no_item_assets(assets_dir, no_download):
    item = ReplicaCADFurniture("fridge")
    assert item.object_id == "fridge"
    assert item.name == "fridge"
    assert item.language_annotations == tuple(
        _SCRIPT.CATALOG["fridge"]["language_annotations"]
    )
    assert item.fixed is True  # furniture default
    assert item.reference_qpos == {}
    assert item.standing_quaternion == tuple(
        _SCRIPT.CATALOG["fridge"]["standing_quaternion"]
    )
    assert not no_download  # only the (already local) catalog was touched


def test_paths_under_assets_dir(assets_dir):
    item = ReplicaCADFurniture("fridge")
    expected_dir = assets_dir / REPLICACAD_ASSETS_ROOT / "fridge"
    assert item.asset_dir == expected_dir
    assert item.description_file == expected_dir / "fridge.urdf"


def test_overrides(assets_dir):
    item = ReplicaCADFurniture(
        "fridge", name="my_fridge", language_annotations=["the fridge"],
        standing_quaternion=(0.0, 0.0, 0.0, 1.0),
    )
    assert item.name == "my_fridge"
    assert item.language_annotations == ("the fridge",)
    assert item.standing_quaternion == (0.0, 0.0, 0.0, 1.0)


# ------------------------------------------------------------------
# Asset completeness (_ensure_assets)
# ------------------------------------------------------------------

def test_ensure_assets_complete_no_download(fake_fridge, no_download):
    item = ReplicaCADFurniture("fridge")
    item._ensure_assets()  # complete dir -> no Hub access
    assert item._assets_ready


def test_ensure_assets_partial_dir_triggers_redownload(fake_fridge, monkeypatch):
    # Simulate an interrupted download: a referenced mesh missing, dir exists.
    (fake_fridge / "door.glb").unlink()
    calls = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        _write_box_glb(fake_fridge / "door.glb", _DOOR_EXTENTS)

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    item = ReplicaCADFurniture("fridge")
    item._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == f"{REPLICACAD_ASSETS_ROOT}/fridge/**"
    assert item._assets_ready


def test_ensure_assets_missing_referenced_mesh_fails(fake_fridge, monkeypatch):
    # A referenced mesh is missing and the (no-op) re-download does not
    # produce it -> hard failure, not a silently broken asset.
    (fake_fridge / "door.glb").unlink()
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    item = ReplicaCADFurniture("fridge")
    with pytest.raises(FileNotFoundError, match="door.glb"):
        item._ensure_assets()
    assert not item._assets_ready


def test_required_files_includes_all_references(fake_fridge):
    item = ReplicaCADFurniture("fridge")
    required = item._required_files()
    assert required[0] == item.description_file
    assert set(required) == {
        item.description_file,
        item.asset_dir / "body.glb",
        item.asset_dir / "door.glb",
    }
    assert len(required) == len(set(required))  # de-duplicated


# ------------------------------------------------------------------
# Offline FK union bbox
# ------------------------------------------------------------------

def test_link_meshes_and_union_bbox_offline_fk(fake_fridge):
    """get_local_bounding_box is the union of the per-link offline FK meshes."""
    item = ReplicaCADFurniture("fridge")
    meshes = item._link_meshes_at_reference()
    # Only links with visuals appear; both template links have one.
    assert set(meshes) == {"root", "door"}
    # The descriptor wires the shared FK helper (same links, same frames).
    helper_meshes = urdf_link_meshes_at_config(str(item.description_file))
    assert set(helper_meshes) == set(meshes)

    # Per-link boxes in the base (root-link) frame: the root visual is a
    # centered box, the door visual is offset by the joint origin (0.3, 0, 0).
    root_mesh = meshes["root"]
    assert torch.allclose(
        torch.tensor(root_mesh.bounds, dtype=torch.float32),
        torch.tensor([[-e / 2 for e in _BODY_EXTENTS], [e / 2 for e in _BODY_EXTENTS]],
                     dtype=torch.float32),
    )
    door_mesh = meshes["door"]
    assert torch.allclose(
        torch.tensor(door_mesh.bounds, dtype=torch.float32),
        torch.tensor([[0.3 - _DOOR_EXTENTS[0] / 2, -0.3, -0.9],
                      [0.3 + _DOOR_EXTENTS[0] / 2, 0.3, 0.9]], dtype=torch.float32),
    )

    # Union bbox = min/max over the per-link bounds, in the base-link frame.
    union = item.get_local_bounding_box()
    assert torch.allclose(
        union.low,
        torch.tensor([-0.3, -0.3, -0.9], dtype=torch.float32),
    )
    assert torch.allclose(
        union.high,
        torch.tensor([0.3 + _DOOR_EXTENTS[0] / 2, 0.3, 0.9], dtype=torch.float32),
    )


def test_reference_qpos_flows_into_offline_fk(fake_fridge):
    """A non-zero reference configuration rotates the FK'd link meshes."""
    closed = ReplicaCADFurniture("fridge").get_local_bounding_box()
    opened = ReplicaCADFurniture(
        "fridge", reference_qpos={"door_hinge": math.pi / 2},
    ).get_local_bounding_box()
    # The door (thin in x, wide in y) swings 90° about +z: the union grows
    # along +x and shrinks on the closed pose's +x edge (0.325 -> 0.6 vs 0.3).
    assert float(opened.high[0]) == pytest.approx(0.6, abs=1e-5)
    assert float(opened.low[0]) == pytest.approx(-0.3, abs=1e-5)
    assert float(closed.high[0]) == pytest.approx(0.3 + _DOOR_EXTENTS[0] / 2, abs=1e-5)
    assert not torch.allclose(closed.high, opened.high)


def test_unknown_reference_qpos_joint_raises(fake_fridge):
    # The offline FK validates reference_qpos lazily, on the first geometry
    # query (construction stays offline-safe and does not parse the URDF).
    item = ReplicaCADFurniture("fridge", reference_qpos={"no_such_joint": 0.1})
    with pytest.raises(ArticulationError, match="no_such_joint"):
        item.get_local_bounding_box()


# ------------------------------------------------------------------
# prepare_replicacad_assets.py
# ------------------------------------------------------------------

def test_rewrite_mesh_paths_strips_dir_prefixes():
    # Upstream refs are directory-local already; the rewrite is defensive and
    # strips urdf/<dir>/ prefixes, mirroring the YCB pipeline's rewrite.
    text = '<mesh filename="cabinet.glb" scale="1 1 1" />'
    assert _SCRIPT.rewrite_mesh_paths(text, "cabinet") == text
    prefixed = '<mesh filename="cabinet/cabinet.glb" /><mesh filename="urdf/cabinet/body.glb" />'
    out = _SCRIPT.rewrite_mesh_paths(prefixed, "cabinet")
    assert 'filename="cabinet.glb"' in out and 'filename="body.glb"' in out
    assert "urdf/" not in out


def test_normalize_vector_attribs_comma_separated_vectors():
    # The upstream URDFs contain comma-separated vectors that
    # genesis.ext.urdfpy cannot parse (np.fromstring(value, sep=" ")).
    text = '<mesh filename="cabinet.glb" scale="1, 1, 1"/>'
    assert _SCRIPT.normalize_vector_attribs(text) == (
        '<mesh filename="cabinet.glb" scale="1 1 1"/>'
    )
    mixed = '<box size="0.617 0.145, 0.894" />'
    assert _SCRIPT.normalize_vector_attribs(mixed) == '<box size="0.617 0.145 0.894" />'
    # Non-numeric values (filenames) are untouched.
    keep = '<mesh filename="a,b.glb" /><link name="x">'
    assert _SCRIPT.normalize_vector_attribs(keep) == keep


def test_referenced_mesh_files_deduplicated(fake_fridge):
    urdf = fake_fridge / "fridge.urdf"
    assert _SCRIPT.referenced_mesh_files(urdf) == ["body.glb", "door.glb"]


def test_write_catalog_covers_prepared_items(tmp_path):
    objects_root = tmp_path / "objects" / "replicacad"
    objects_root.mkdir(parents=True)
    _SCRIPT._write_catalog(objects_root, ["fridge", "cabinet"])
    written = json.loads((objects_root / "catalog.json").read_text())
    assert set(written) == {"fridge", "cabinet"}
    # JSON round-trip: tuples come back as lists.
    assert written["fridge"] == json.loads(json.dumps(_SCRIPT.CATALOG["fridge"]))


def test_write_catalog_missing_entry_fails(tmp_path):
    objects_root = tmp_path / "objects" / "replicacad"
    objects_root.mkdir(parents=True)
    with pytest.raises(KeyError, match="no_such_item"):
        _SCRIPT._write_catalog(objects_root, ["no_such_item"])


def test_write_catalog_merges_with_existing(tmp_path):
    """Incremental runs keep the entries of items outside the current run."""
    objects_root = tmp_path / "objects" / "replicacad"
    objects_root.mkdir(parents=True)
    _SCRIPT._write_catalog(objects_root, ["fridge"])
    _SCRIPT._write_catalog(objects_root, ["cabinet"])
    written = json.loads((objects_root / "catalog.json").read_text())
    assert set(written) == {"fridge", "cabinet"}
    assert written["fridge"] == json.loads(json.dumps(_SCRIPT.CATALOG["fridge"]))
    assert written["cabinet"] == json.loads(json.dumps(_SCRIPT.CATALOG["cabinet"]))


def test_repackage_item(tmp_path):
    """A synthetic upstream tree is repackaged into a self-contained item dir."""
    # Minimal upstream layout: urdf/<dir>/<urdf> + referenced meshes, with the
    # comma-separated scale quirk the normalizer must fix ("fridge" is a real
    # entry of ARTICULATED_ITEMS, mapping to urdf/fridge/fridge.urdf).
    urdf_text = (
        '<robot name="fridge">\n'
        '  <link name="root"><visual><geometry>'
        '<mesh filename="body.glb" scale="1, 1, 1"/></geometry></visual></link>\n'
        "</robot>\n"
    )
    source = tmp_path / "src"
    (source / "urdf" / "fridge").mkdir(parents=True)
    (source / "urdf" / "fridge" / "fridge.urdf").write_text(urdf_text)
    (source / "urdf" / "fridge" / "body.glb").write_bytes(b"glTF placeholder bytes")

    objects_root = tmp_path / "out" / "objects" / "replicacad"
    _SCRIPT.repackage_item(source, objects_root, "fridge")

    obj_dir = objects_root / "fridge"
    rewritten = (obj_dir / "fridge.urdf").read_text()
    assert 'scale="1 1 1"' in rewritten  # normalized
    assert "body.glb" in rewritten  # directory-local reference kept
    assert (obj_dir / "body.glb").read_bytes() == b"glTF placeholder bytes"


# ------------------------------------------------------------------
# GPU smoke test (skipped offline)
# ------------------------------------------------------------------

def _is_cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_into_scene_smoke_gpu(assets_dir, no_download):
    """Real GPU smoke: load the synthetic URDF via load_into_scene.

    Mirrors the prepare pipeline's sanity gate: the built entity keeps its
    named links (``merge_fixed_links=False``), has a non-empty joint set, and
    its ``get_vAABB`` matches the offline FK union bbox at the identity pose.
    """
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)

    _make_fake_item_dir(assets_dir, "fridge")

    item = ReplicaCADFurniture("fridge")
    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(type("WorldStub", (), {"scene": scene})())
        scene.build(n_envs=0)

        moving = [j for j in entity.joints if int(j.type) != gs.JOINT_TYPE.FIXED]
        assert [j.name for j in moving] == ["door_hinge"]
        # merge_fixed_links=False keeps the named links addressable.
        assert {link.name for link in entity.links} == {"root", "door"}

        # The packaging-time cross-check: offline FK union bbox vs runtime AABB.
        vaabb = entity.get_vAABB().cpu()
        local = item.get_local_bounding_box()
        assert torch.allclose(vaabb[0], local.low.to(vaabb.device), atol=2e-3)
        assert torch.allclose(vaabb[1], local.high.to(vaabb.device), atol=2e-3)
    finally:
        scene.destroy()
        # gs.init installs the backend device as torch's *ambient* default
        # (``torch.set_default_device``); leaving it on CUDA would silently
        # switch every later test's host-side fake tensors (torch.zeros(…)
        # with no explicit device) to CUDA — see the device-isolation note in
        # tests/test_room_builder.py.  Restore the host default (the offline
        # tests model a CPU world).
        torch.set_default_device("cpu")
