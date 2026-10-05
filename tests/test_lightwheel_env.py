"""Tests for the Lightwheel **static environment** family (no downloads).

Covers the new ``objects/lightwheel_env`` family added for the bundle's
``Locomotion/`` subset:

  - ``LightwheelEnv`` construction (paths, catalog annotations, scalar scale
    enforcement, description-file override) and asset completeness;
  - the pxr ``BBoxCache`` offline bbox (``trimesh`` cannot read ``.usd``);
  - ``load_into_scene`` morph construction through a duck-typed scene stub;
  - the packager's Locomotion path: the measured ``LOCOMOTION_ROOTS`` map,
    the per-category routing table, the sanitised ``2-StoryStaircase`` id,
    the closure-copy repackaging (junk excluded by construction, referenced
    payload kept), the Apartment UE-material-record warn-and-record policy,
    the static catalog entries and the CC BY-NC license/readme for the new
    tree.

The synthetic stages are authored with ``pxr`` (ASCII ``.usda``) in
``tmp_path``; ``--subset`` end-to-end packaging runs with the gate disabled.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.hub_objects.lightwheel_env import (
    LIGHTWHEEL_ENV_ASSETS_ROOT,
    LIGHTWHEEL_ENV_CATALOG_PATH,
    LightwheelEnv,
    get_lightwheel_env_catalog,
    list_lightwheel_env_objects,
    make_lightwheel_env_pool,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_lightwheel_assets", _SCRIPTS_DIR / "prepare_lightwheel_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_prepare_script()


# ------------------------------------------------------------------
# Synthetic assets
# ------------------------------------------------------------------

def _write_env_stage(path: Path, *, size=(8.0, 6.0, 0.2), terrain=True):
    """A jointless terrain-shaped stage: one collided mesh, no joints."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = UsdGeom.Xform.Define(stage, "/root")
    mesh = UsdGeom.Mesh.Define(stage, "/root/terrain")
    ex, ey, ez = size
    pts = [
        Gf.Vec3f(px * ex / 2, py * ey / 2, pz * ez / 2 - (ez / 2 if terrain else 0.0))
        for px in (-1.0, 1.0)
        for py in (-1.0, 1.0)
        for pz in (-1.0, 1.0)
    ]
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 1, 5, 7, 3, 0, 2, 6, 4]
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return root


def _make_env_object_dir(assets_dir: Path, object_id: str, *, size=(8.0, 6.0, 0.2)):
    obj_dir = assets_dir / LIGHTWHEEL_ENV_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    _write_env_stage(obj_dir / f"{object_id}.usd", size=size)
    return obj_dir


def _catalog_entry(object_id: str, *, tags=("environment", "locomotion", "terrain", "static")):
    return {
        "language_annotations": ["environment", "terrain", f"the {object_id} terrain"],
        "tags": sorted(tags),
    }


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir with a generated ``objects/lightwheel_env`` catalog."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    catalog_file = tmp_path / LIGHTWHEEL_ENV_CATALOG_PATH
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    catalog_file.write_text(json.dumps({
        "lightwheel_env_Grass": _catalog_entry("grass"),
        "lightwheel_env_SnowGround": _catalog_entry("snow"),
        "lightwheel_env_Apartment": _catalog_entry(
            "apartment", tags=("environment", "locomotion", "room", "interior", "static"),
        ),
    }))
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()
    return tmp_path


@pytest.fixture
def fake_grass(assets_dir):
    return _make_env_object_dir(assets_dir, "lightwheel_env_Grass")


@pytest.fixture
def no_download(monkeypatch):
    calls = []

    def _fail(**kwargs):
        calls.append(kwargs)
        raise AssertionError(f"Unexpected download attempt: {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail)
    return calls


# ------------------------------------------------------------------
# Catalog + pool
# ------------------------------------------------------------------

def test_catalog_integrity(assets_dir):
    catalog = get_lightwheel_env_catalog()
    assert set(catalog) == {
        "lightwheel_env_Grass", "lightwheel_env_SnowGround", "lightwheel_env_Apartment",
    }
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert info.tags  # environments are always tagged


def test_list_filter_by_tags_and_contains(assets_dir):
    assert list_lightwheel_env_objects() == sorted([
        "lightwheel_env_Grass", "lightwheel_env_SnowGround", "lightwheel_env_Apartment",
    ])
    assert list_lightwheel_env_objects(tags="terrain") == [
        "lightwheel_env_Grass", "lightwheel_env_SnowGround",
    ]
    assert list_lightwheel_env_objects(tags=["terrain", "static"]) == [
        "lightwheel_env_Grass", "lightwheel_env_SnowGround",
    ]
    assert list_lightwheel_env_objects(contains="apart") == ["lightwheel_env_Apartment"]
    assert list_lightwheel_env_objects(tags="terrain", contains="snow") == [
        "lightwheel_env_SnowGround",
    ]
    assert list_lightwheel_env_objects(tags="no_such_tag") == []


def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="Unknown Lightwheel environment id"):
        LightwheelEnv("lightwheel_env_Nope")


def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    pool = make_lightwheel_env_pool(["lightwheel_env_SnowGround", "lightwheel_env_Grass"])
    assert [item.object_id for item in pool] == [
        "lightwheel_env_SnowGround", "lightwheel_env_Grass",
    ]
    assert no_download == []


def test_pool_ids_and_tags_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError, match="not both"):
        make_lightwheel_env_pool(["lightwheel_env_Grass"], tags="terrain")


def test_pool_sampling_reproducible(assets_dir, no_download):
    first = make_lightwheel_env_pool(tags="terrain", n=1, rng=torch.Generator().manual_seed(0))
    second = make_lightwheel_env_pool(tags="terrain", n=1, rng=torch.Generator().manual_seed(0))
    assert [i.object_id for i in first] == [i.object_id for i in second]


def test_pool_n_too_large(assets_dir):
    with pytest.raises(ValueError, match="only 2 match"):
        make_lightwheel_env_pool(tags="terrain", n=5)


# ------------------------------------------------------------------
# Descriptor
# ------------------------------------------------------------------

def test_construction_paths_and_catalog_annotations(assets_dir, no_download):
    item = LightwheelEnv("lightwheel_env_Grass")
    assert item.asset_dir == asset_path.ASSETS_DIR / LIGHTWHEEL_ENV_ASSETS_ROOT / "lightwheel_env_Grass"
    assert item.description_file.name == "lightwheel_env_Grass.usd"
    assert item.fixed is True                      # environments are immovable
    assert item.language_annotations == get_lightwheel_env_catalog()["lightwheel_env_Grass"].language_annotations
    assert no_download == []                       # construction is offline-safe


def test_scalar_scale_only(assets_dir):
    with pytest.raises(ValueError, match="scalar"):
        LightwheelEnv("lightwheel_env_Grass", scale=(1.0, 2.0, 3.0))


def test_description_file_name_override(assets_dir):
    item = LightwheelEnv(
        "lightwheel_env_Grass", description_file_name="lightwheel_env_Grass.usda",
    )
    assert item.description_file.name == "lightwheel_env_Grass.usda"


def test_required_files_include_the_closure(assets_dir, fake_grass):
    item = LightwheelEnv("lightwheel_env_Grass")
    required = item._required_files()
    assert required[0] == fake_grass / "lightwheel_env_Grass.usd"
    assert all(isinstance(p, Path) for p in required)
    assert item._missing_files() == []


def test_ensure_assets_missing_stage_raises(assets_dir, monkeypatch):
    """A missing root layer triggers a download; a still-missing one raises."""
    item = LightwheelEnv("lightwheel_env_Grass")
    monkeypatch.setattr(asset_path, "download_assets", lambda **kwargs: None)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        item._ensure_assets()


def test_local_bounding_box_from_pxr(assets_dir, fake_grass, no_download):
    """The offline bbox is point-accurate over the collider subtree (metres, scaled)."""
    item = LightwheelEnv("lightwheel_env_Grass")
    box = item.get_local_bounding_box()
    low = box.low.detach().cpu().numpy()
    high = box.high.detach().cpu().numpy()
    assert np.allclose(high - low, [8.0, 6.0, 0.2], atol=1e-6)
    assert low[2] == pytest.approx(-0.2, abs=1e-6)  # terrain z-min ≤ 0
    # Scaled variant: the morph scale multiplies the stage bounds.
    scaled = LightwheelEnv("lightwheel_env_Grass", scale=2.0).get_local_bounding_box()
    assert np.allclose(
        scaled.high.detach().cpu().numpy() - scaled.low.detach().cpu().numpy(),
        [16.0, 12.0, 0.4], atol=1e-6,
    )
    assert no_download == []


def test_load_trimesh_unsupported(assets_dir, fake_grass):
    """trimesh cannot read USD — the failure is explicit, not a stray crash."""
    item = LightwheelEnv("lightwheel_env_Grass")
    with pytest.raises(NotImplementedError, match="trimesh cannot read USD"):
        item._load_trimesh()


@pytest.fixture
def genesis_cpu():
    """Genesis must be initialised before ``gs.morphs.*`` can be built."""
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu)
    return gs


class _SceneStub:
    def __init__(self):
        self.stage_morphs = []
        self.materials = []

    def add_stage(self, morph, material=None):
        self.stage_morphs.append(morph)
        self.materials.append(material)
        return ["entity"]


class _WorldStub:
    def __init__(self, scene):
        self.scene = scene


def test_load_into_scene_builds_usd_morph(assets_dir, fake_grass, no_download, genesis_cpu):
    """``load_into_scene`` uses ``gs.morphs.USD`` + ``scene.add_stage``."""
    gs = genesis_cpu
    scene = _SceneStub()
    item = LightwheelEnv("lightwheel_env_Grass", scale=1.5)
    entities = item.load_into_scene(_WorldStub(scene), pos=(1.0, 2.0, 0.0))
    assert entities == ["entity"]
    morph = scene.stage_morphs[0]
    assert isinstance(morph, gs.morphs.USD)
    assert morph.file == str(fake_grass / "lightwheel_env_Grass.usd")
    assert morph.fixed is True
    assert tuple(morph.pos) == (1.0, 2.0, 0.0)
    assert morph.scale == 1.5
    # Room-scale SDF cells (the int32 SDF solver layout overflow).
    material = scene.materials[0]
    assert isinstance(material, gs.materials.Rigid)
    assert material.sdf_cell_size == 0.15 and material.sdf_max_res == 96
    assert no_download == []


def test_load_into_scene_ignores_name_override(assets_dir, fake_grass, no_download, genesis_cpu):
    scene = _SceneStub()
    item = LightwheelEnv("lightwheel_env_Grass")
    item.load_into_scene(_WorldStub(scene), name_override="other")
    assert len(scene.stage_morphs) == 1  # warned + ignored, never forwarded


# ------------------------------------------------------------------
# Packager: Locomotion routing + closure-copy repackaging
# ------------------------------------------------------------------

def test_locomotion_root_map_is_the_measured_one():
    """§3a: the root-layer table is explicit and evidence-backed."""
    assert _SCRIPT.LOCOMOTION_ROOTS == {
        "2-StoryStaircase": "2-StoryStaircase.usd",
        "Apartment": "scene_04.usd",
        "Grass": "Grass.usd",
        "GravelGround": "GravelGround.usd",
        "KitchenRoom": "KitchenRoom_RSS.usd",
        "MudGround": "MudGround.usd",
        "SLATEGround": "SLATEGround.usd",
        "SnowGround": "SnowGround.usd",
    }
    # KitchenRoom.usd is unusable; Ground_stone.usd is a payload, not a root.
    assert "KitchenRoom.usd" not in _SCRIPT.LOCOMOTION_ROOTS.values()
    assert "Ground_stone.usd" not in _SCRIPT.LOCOMOTION_ROOTS.values()


def test_locomotion_routing_and_ids():
    """§4: KitchenRoom → articulated tree; the other seven → lightwheel_env."""
    assert _SCRIPT.locomotion_object_id("KitchenRoom") == "lightwheel_KitchenRoom_RSS"
    assert _SCRIPT.locomotion_family("KitchenRoom") == "articulated"
    assert _SCRIPT.locomotion_object_id("2-StoryStaircase") == "lightwheel_env_2_story_staircase"
    assert _SCRIPT.locomotion_object_id("GravelGround") == "lightwheel_env_GravelGround"
    assert _SCRIPT.locomotion_object_id("SLATEGround") == "lightwheel_env_SLATEGround"
    for category in _SCRIPT.LOCOMOTION_ROOTS:
        if category == "KitchenRoom":
            continue
        assert _SCRIPT.locomotion_family(category) == "static"
        assert _SCRIPT.locomotion_object_id(category).startswith("lightwheel_env_")
    out = Path("/tmp/out")
    assert _SCRIPT.locomotion_out_root("KitchenRoom", out) == out / "objects" / "lightwheel"
    assert _SCRIPT.locomotion_out_root("Grass", out) == out / "objects" / "lightwheel_env"


def test_gate_family_dispatch():
    assert _SCRIPT.gate_family_of("lightwheel_env_Grass") == "static"
    assert _SCRIPT.gate_family_of("lightwheel_Microwave043") == "articulated"
    assert _SCRIPT.gate_family_of("lightwheel_KitchenRoom_RSS") == "articulated"


def test_sanitise_category_token():
    assert _SCRIPT._sanitise_category_token("2-StoryStaircase") == "2_story_staircase"
    assert _SCRIPT._sanitise_category_token("GravelGround") == "GravelGround"
    assert _SCRIPT._sanitise_category_token("SLATEGround") == "SLATEGround"


def _make_locomotion_category(source: Path, category: str, root_layer: str) -> Path:
    """A synthetic Locomotion category: root + payload + junk dirs."""
    asset_dir = source / "Locomotion" / category
    (asset_dir / "草坪草地3D模型").mkdir(parents=True)
    _write_env_stage(asset_dir / root_layer, size=(8.0, 6.0, 0.2))
    # A referenced payload sublayer (kept) and unreferenced junk (dropped).
    payload = asset_dir / "草坪草地3D模型" / "detail.usda"
    payload.write_text("#usda 1.0\n")
    (asset_dir / "E").mkdir()
    (asset_dir / "E" / "junk.bin").write_bytes(b"junk")
    (asset_dir / ".thumbs").mkdir()
    (asset_dir / ".thumbs" / "thumb.png").write_bytes(b"png")
    (asset_dir / ".DS_Store").write_bytes(b"ds")
    (asset_dir / ".collect.mapping.json").write_text("{}")
    (asset_dir / "Table049").mkdir()
    (asset_dir / "Table049" / "Table049.usd").write_text("#usda 1.0\n")
    # Reference the payload from the root layer (the closure must keep it).
    from pxr import Usd

    stage = Usd.Stage.Open(str(asset_dir / root_layer))
    stage.GetRootLayer().subLayerPaths.append("./草坪草地3D模型/detail.usda")
    stage.GetRootLayer().Save()
    return asset_dir


def test_list_locomotion_assets_uses_the_root_map(tmp_path):
    source = tmp_path / "bundle"
    _make_locomotion_category(source, "Grass", "Grass.usd")
    _make_locomotion_category(source, "KitchenRoom", "KitchenRoom_RSS.usd")
    # A category absent from the map (a future upstream export) is skipped.
    _make_locomotion_category(source, "MarsGround", "MarsGround.usd")

    pairs = _SCRIPT.list_locomotion_assets(source)
    assert {oid: root for oid, _dir, root in pairs} == {
        "lightwheel_env_Grass": "Grass.usd",
        "lightwheel_KitchenRoom_RSS": "KitchenRoom_RSS.usd",
    }


def test_list_locomotion_requires_locomotion_dir(tmp_path):
    with pytest.raises(SystemExit, match="Locomotion"):
        _SCRIPT.list_locomotion_assets(tmp_path)


def test_repackage_scene_closure_copy_excludes_junk(tmp_path):
    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "Grass", "Grass.usd")
    out_root = tmp_path / "out" / "objects" / "lightwheel_env"

    packaged, n_converted, unresolvable = _SCRIPT.repackage_scene(
        asset_dir, "lightwheel_env_Grass", "Grass.usd", out_root,
    )

    names = sorted(str(p.relative_to(packaged)) for p in packaged.rglob("*") if p.is_file())
    assert names == [
        "lightwheel_env_Grass.usd",
        "草坪草地3D模型/detail.usda",
    ]
    assert n_converted == 0
    assert unresolvable == []
    # The referenced payload dir is kept; every unreferenced junk dir is gone.
    assert not (packaged / "E").exists()
    assert not (packaged / ".thumbs").exists()
    assert not (packaged / "Table049").exists()
    assert not (packaged / ".DS_Store").exists()
    assert not (packaged / ".collect.mapping.json").exists()


def test_repackage_scene_missing_usd_layer_is_fatal(tmp_path):
    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "Grass", "Grass.usd")
    (asset_dir / "草坪草地3D模型" / "detail.usda").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        _SCRIPT.repackage_scene(
            asset_dir, "lightwheel_env_Grass", "Grass.usd",
            tmp_path / "out" / "objects" / "lightwheel_env",
        )


def test_repackage_scene_warn_and_records_ue_material_records(tmp_path):
    """Apartment's ``Foo.Foo`` material records: warn-and-record, not fatal."""
    from pxr import Usd

    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "Apartment", "scene_04.usd")
    stage = Usd.Stage.Open(str(asset_dir / "scene_04.usd"))
    # A reference to a UE-export material record that exists nowhere.
    prim = stage.DefinePrim("/root/leaf", "Xform")
    prim.GetReferences().AddReference(
        "Game/AI_vol3_04/Materials/MI_leaf_32_1.MI_leaf_32_1"
    )
    stage.GetRootLayer().Save()

    packaged, _n, unresolvable = _SCRIPT.repackage_scene(
        asset_dir, "lightwheel_env_Apartment", "scene_04.usd",
        tmp_path / "out" / "objects" / "lightwheel_env",
    )
    assert unresolvable == ["Game/AI_vol3_04/Materials/MI_leaf_32_1.MI_leaf_32_1"]
    assert (packaged / "lightwheel_env_Apartment.usd").is_file()


def test_repackage_scene_missing_texture_is_fatal(tmp_path):
    """A missing *texture* stays fatal (only non-USD/non-image refs warn)."""
    from pxr import Sdf, Usd, UsdShade

    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "Grass", "Grass.usd")
    stage = Usd.Stage.Open(str(asset_dir / "Grass.usd"))
    material = UsdShade.Material.Define(stage, "/root/mat")
    shader = UsdShade.Shader.Define(stage, "/root/mat/tex")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("texture/missing.png")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "rgb")
    stage.GetRootLayer().Save()

    with pytest.raises(FileNotFoundError, match="missing"):
        _SCRIPT.repackage_scene(
            asset_dir, "lightwheel_env_Grass", "Grass.usd",
            tmp_path / "out" / "objects" / "lightwheel_env",
        )


def test_required_files_recurse_into_payload_textures(tmp_path):
    """``_usd_required_files`` is a fixpoint over *payload* layers too.

    A nested payload layer's own texture references (Lightwheel's
    ``tudi2/tudi2.usd`` → ``tudi2/textures/*.jpg``) must surface — a
    one-level walk of the root layer misses them and the packaged tree
    ships with dangling texture references.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "MudGround", "MudGround.usd")
    # Payload the root's terrain prim onto a nested layer that references
    # its own texture (the upstream layout).
    inner_dir = asset_dir / "tudi2"
    inner_dir.mkdir()
    inner = Usd.Stage.CreateNew(str(inner_dir / "tudi2.usd"))
    mesh = UsdGeom.Mesh.Define(inner, "/World/m")
    mesh.CreatePointsAttr([
        Gf.Vec3f(-1, 0, -1), Gf.Vec3f(1, 0, -1), Gf.Vec3f(1, 0, 1), Gf.Vec3f(-1, 0, 1),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    material = UsdShade.Material.Define(inner, "/World/mat")
    shader = UsdShade.Shader.Define(inner, "/World/mat/tex")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./textures/tex_u1_v1_diffuse.jpg")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "rgb")
    inner.GetRootLayer().Save()
    texture = inner_dir / "textures" / "tex_u1_v1_diffuse.jpg"
    texture.parent.mkdir()
    texture.write_bytes(b"jpg")

    # The root payloads the nested layer (the upstream scene layout).
    root_stage = Usd.Stage.Open(str(asset_dir / "MudGround.usd"))
    xform = UsdGeom.Xform.Define(root_stage, "/World/tudi2")
    xform.GetPrim().GetPayloads().AddPayload("./tudi2/tudi2.usd")
    root_stage.GetRootLayer().Save()

    from unienv_genesis_collection.objects.hub_objects.articulated.hub_articulated_object import (
        _usd_required_files,
    )

    required = _usd_required_files(asset_dir / "MudGround.usd", asset_dir)
    rels = {str(p.relative_to(asset_dir)) for p in required if p.is_file()}
    assert "tudi2/tudi2.usd" in rels                      # the payload layer
    assert "tudi2/textures/tex_u1_v1_diffuse.jpg" in rels  # its texture
    assert "草坪草地3D模型/detail.usda" in rels            # the root's sublayer


def test_required_files_skips_unsatisfiable_records(tmp_path):
    """``required_only=True`` drops the never-satisfiable UE material records.

    The descriptor's ``_required_files`` must not demand them (they exist
    nowhere in the bundle): a permanently-missing entry turns every
    ``_ensure_assets()`` into a Hub download and the offline completeness
    check into a permanent failure.  The default (packager) view keeps them
    so ``repackage_scene`` can warn-and-record them.
    """
    from pxr import Usd

    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "Apartment", "scene_04.usd")
    stage = Usd.Stage.Open(str(asset_dir / "scene_04.usd"))
    prim = stage.DefinePrim("/root/leaf", "Xform")
    prim.GetReferences().AddReference(
        "Game/AI_vol3_04/Materials/MI_leaf_32_1.MI_leaf_32_1"
    )
    stage.GetRootLayer().Save()

    from unienv_genesis_collection.objects.hub_objects.articulated.hub_articulated_object import (
        _usd_required_files,
    )

    full = _usd_required_files(asset_dir / "scene_04.usd", asset_dir)
    required = _usd_required_files(asset_dir / "scene_04.usd", asset_dir, required_only=True)
    full_names = {p.name for p in full}
    required_names = {p.name for p in required}
    assert "MI_leaf_32_1.MI_leaf_32_1" in full_names
    assert "MI_leaf_32_1.MI_leaf_32_1" not in required_names
    # Everything else (the root layer, the sublayer payload) stays required.
    assert {p.name for p in required} >= {"scene_04.usd", "detail.usda"}


def test_descriptor_required_files_agree_with_the_packed_tree(tmp_path, monkeypatch):
    """Apartment's packed directory satisfies the descriptor offline.

    Regression: the descriptor demanded the 20 warn-and-recorded UE material
    records, so every load attempted a Hub download — tripping the sanity
    gate's download stub ("the repackaged object directory is incomplete").
    """
    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "Apartment", "scene_04.usd")
    from pxr import Usd

    stage = Usd.Stage.Open(str(asset_dir / "scene_04.usd"))
    prim = stage.DefinePrim("/root/leaf", "Xform")
    prim.GetReferences().AddReference(
        "Game/AI_vol3_04/Materials/MI_leaf_32_1.MI_leaf_32_1"
    )
    stage.GetRootLayer().Save()

    packaged, _n, unresolvable = _SCRIPT.repackage_scene(
        asset_dir, "lightwheel_env_Apartment", "scene_04.usd",
        tmp_path / "out" / "objects" / "lightwheel_env",
    )
    assert unresolvable == ["Game/AI_vol3_04/Materials/MI_leaf_32_1.MI_leaf_32_1"]

    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path / "out")
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()
    catalog_file = tmp_path / "out" / LIGHTWHEEL_ENV_CATALOG_PATH
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    catalog_file.write_text(json.dumps({
        "lightwheel_env_Apartment": {
            "language_annotations": ["environment", "room"],
            "tags": ["environment", "room", "static"],
        },
    }))

    def _fail_download(**kwargs):
        raise AssertionError(f"Unexpected download attempt: {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail_download)

    item = LightwheelEnv("lightwheel_env_Apartment")
    item._ensure_assets()  # must not attempt a download
    assert item._missing_files() == []


def test_local_bounding_box_excludes_uninstantiated_geometry(tmp_path, assets_dir):
    """The offline bbox mirrors ``add_stage``'s geometry selection.

    The two traps that dropped Grass (0.14 m in Z) and the 2-storey
    staircase (20 m in X/Y):

    * visual-only meshes with no collider ancestor — ``add_stage`` never
      instantiates them, the offline bbox must not count them;
    * an authored ``extent`` hint coarser than the mesh — the bound cache
      transforms that box rigidly (a 45° rotation of an 8 m plate's inflated
      extent cube inflates X/Y to 11.3 m from the true 5.7 m), the actual
      ``points`` do not.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    object_id = "lightwheel_env_Grass"
    obj_dir = assets_dir / LIGHTWHEEL_ENV_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(obj_dir / f"{object_id}.usd"))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    # The collider: an 8 × 0.1 plate rotated 45°, carrying an *inflated*
    # authored extent (a ±4 m cube — coarser than the true point span).
    root = UsdGeom.Xform.Define(stage, "/root")
    xform = UsdGeom.Xform.Define(stage, "/root/cell")
    UsdPhysics.CollisionAPI.Apply(xform.GetPrim())
    UsdGeom.Xform(xform.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 45.0))
    plate = UsdGeom.Mesh.Define(stage, "/root/cell/plate")
    plate.CreatePointsAttr([
        Gf.Vec3f(-4.0, -0.05, 0.0), Gf.Vec3f(4.0, -0.05, 0.0),
        Gf.Vec3f(4.0, 0.05, 0.0), Gf.Vec3f(-4.0, 0.05, 0.0),
    ])
    plate.CreateFaceVertexCountsAttr([4])
    plate.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plate.CreateExtentAttr([Gf.Vec3f(-4.0, -4.0, -4.0), Gf.Vec3f(4.0, 4.0, 4.0)])

    # A visual-only sibling far outside the collider (no collider ancestor).
    far = UsdGeom.Mesh.Define(stage, "/distant_backdrop")
    far.CreatePointsAttr([
        Gf.Vec3f(-100, -100, 0), Gf.Vec3f(100, -100, 0),
        Gf.Vec3f(100, 100, 0), Gf.Vec3f(-100, 100, 0),
    ])
    far.CreateFaceVertexCountsAttr([4])
    far.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    stage.GetRootLayer().Save()

    item = LightwheelEnv(object_id)
    box = item.get_local_bounding_box()
    low = box.low.detach().cpu().numpy()
    high = box.high.detach().cpu().numpy()
    # The point-accurate AABB of the rotated 8 × 0.1 plate: 2·(4·cos45°+0.05·sin45°)
    # — a bound-cache box over the inflated extent cube would span 11.31 m in X/Y.
    expected_xy = 2.0 * (4.0 * np.cos(np.pi / 4) + 0.05 * np.sin(np.pi / 4))
    assert np.allclose(high - low, [expected_xy, expected_xy, 0.0], atol=2e-2)
    # The 200 m visual-only backdrop never enters the bound.
    assert np.abs(low).max() < 10.0 and np.abs(high).max() < 10.0


def test_repackage_scene_repairs_prefix_stripped_texture(tmp_path):
    """KitchenRoom's upstream path bug: the texture ships prefix-stripped.

    The crate references ``./textures/3d66Model-19913701-files-024.JPG`` but
    the bundle ships ``textures/19913701-files-024.JPG`` — the repair copies
    it under the referenced name instead of dropping the whole room.
    """
    from pxr import Sdf, Usd, UsdShade

    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "KitchenRoom", "KitchenRoom_RSS.usd")
    stage = Usd.Stage.Open(str(asset_dir / "KitchenRoom_RSS.usd"))
    material = UsdShade.Material.Define(stage, "/root/mat")
    shader = UsdShade.Shader.Define(stage, "/root/mat/tex")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput(
        "file", Sdf.ValueTypeNames.Asset,
    ).Set("./textures/3d66Model-19913701-files-024.JPG")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "rgb")
    stage.GetRootLayer().Save()
    texture_dir = asset_dir / "textures"
    texture_dir.mkdir()
    (texture_dir / "19913701-files-024.JPG").write_bytes(b"jpg")

    packaged, _n, unresolvable = _SCRIPT.repackage_scene(
        asset_dir, "lightwheel_KitchenRoom_RSS", "KitchenRoom_RSS.usd",
        tmp_path / "out" / "objects" / "lightwheel",
    )
    # The referenced name exists in the packaged tree (repaired), and the
    # reference closure is complete — nothing warn-and-recorded.
    assert (packaged / "textures" / "3d66Model-19913701-files-024.JPG").is_file()
    assert unresolvable == []


def test_repackage_scene_transcodes_exr_textures(tmp_path):
    """SnowRoad's .EXR maps are re-encoded to sibling .png files.

    The loader reads every ``UsdUVTexture.file`` through PIL, which cannot
    identify OpenEXR buffers — without the transcode the packaged stage
    loads with every road material failing.
    """
    import numpy as np
    from pxr import Sdf, Usd, UsdShade

    source = tmp_path / "bundle"
    asset_dir = _make_locomotion_category(source, "SnowGround", "SnowGround.usd")
    stage = Usd.Stage.Open(str(asset_dir / "SnowGround.usd"))
    material = UsdShade.Material.Define(stage, "/root/mat")
    shader = UsdShade.Shader.Define(stage, "/root/mat/tex")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./snow/road_D.EXR")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "rgb")
    stage.GetRootLayer().Save()
    textures = asset_dir / "snow"
    textures.mkdir()
    import imageio.v2 as imageio

    imageio.imwrite(str(textures / "road_D.EXR"), np.full((4, 4, 3), 0.5, dtype=np.float32))

    packaged, _n, unresolvable = _SCRIPT.repackage_scene(
        asset_dir, "lightwheel_env_SnowGround", "SnowGround.usd",
        tmp_path / "out" / "objects" / "lightwheel_env",
    )
    assert (packaged / "snow" / "road_D.png").is_file()
    # The reference closure keeps the EXR (the payload/original layer holds its opinion).
    assert (packaged / "snow" / "road_D.EXR").is_file()
    assert unresolvable == []
    # The converter-authored texture input points at the PNG.
    from unienv_genesis_collection.utils.usd_conversions import _relocate_texture
    root_dir = packaged.resolve()
    target = _relocate_texture(
        root_dir, "./snow/road_D.EXR", root_dir,
    )
    assert target == "./snow/road_D.png"


def test_annotate_locomotion_object_static_and_articulated():
    static_entry = _SCRIPT.annotate_locomotion_object("GravelGround", "static")
    assert static_entry["language_annotations"] == [
        "environment", "terrain", "a gravel ground terrain for locomotion",
    ]
    assert set(static_entry["tags"]) >= {"environment", "locomotion", "static", "terrain", "gravel"}

    articulated = _SCRIPT.annotate_locomotion_object("KitchenRoom", "articulated", 35, 23)
    assert "static" not in articulated["tags"]
    assert {"environment", "locomotion", "articulated", "kitchen"} <= set(articulated["tags"])
    assert articulated["language_annotations"][-1].endswith(
        "with 35 hinged parts, 23 sliding parts"
    )

    apartment = _SCRIPT.annotate_locomotion_object("Apartment", "static")
    assert apartment["language_annotations"][-1] == "an apartment interior scene"
    assert "apartment" in apartment["tags"]


def test_locomotion_license_and_readme():
    """The new tree carries the same CC BY-NC 4.0 license + a provenance README."""
    assert "Attribution-NonCommercial 4.0 International" in _SCRIPT.LICENSE_TEXT
    readme = _SCRIPT.ENV_PROVENANCE_README
    assert "CC BY-NC 4.0" in readme
    assert "lightwheel_env_2_story_staircase" in readme
    assert "scene_04.usd" in readme
    assert "unresolvable_refs" in readme
    # The upstream license header stays at the top of the packager script.
    script_text = Path(_SCRIPT.__file__).read_text()
    assert "CC BY-NC 4.0" in script_text.split("\n")[1]


def test_prepare_locomotion_end_to_end_without_gate(tmp_path):
    """`prepare(..., subset="locomotion")` writes both trees + logs (no GPU)."""
    source = tmp_path / "bundle"
    _make_locomotion_category(source, "Grass", "Grass.usd")
    _make_locomotion_category(source, "Apartment", "scene_04.usd")
    _make_locomotion_category(source, "KitchenRoom", "KitchenRoom_RSS.usd")
    output_dir = tmp_path / "out"

    out_root, kept = _SCRIPT.prepare(
        source, output_dir, gate=False, subset="locomotion",
    )

    env_root = output_dir / "objects" / "lightwheel_env"
    assert (env_root / "lightwheel_env_Grass" / "lightwheel_env_Grass.usd").is_file()
    assert (env_root / "lightwheel_env_Apartment" / "lightwheel_env_Apartment.usd").is_file()
    # KitchenRoom is routed into the articulated tree.
    assert (output_dir / "objects" / "lightwheel"
            / "lightwheel_KitchenRoom_RSS" / "lightwheel_KitchenRoom_RSS.usd").is_file()
    assert set(kept) == {
        "lightwheel_env_Grass", "lightwheel_env_Apartment", "lightwheel_KitchenRoom_RSS",
    }

    catalog = json.loads((env_root / "catalog.json").read_text())
    assert set(catalog) == {"lightwheel_env_Grass", "lightwheel_env_Apartment"}
    assert "static" in catalog["lightwheel_env_Grass"]["tags"]
    assert catalog["lightwheel_env_Apartment"]["language_annotations"][-1] == (
        "an apartment interior scene"
    )
    articulated_catalog = json.loads(
        (output_dir / "objects" / "lightwheel" / "catalog.json").read_text()
    )
    assert set(articulated_catalog) == {"lightwheel_KitchenRoom_RSS"}
    assert "articulated" in articulated_catalog["lightwheel_KitchenRoom_RSS"]["tags"]
    assert "kitchenroom" in articulated_catalog["lightwheel_KitchenRoom_RSS"]["tags"]

    log = json.loads((env_root / "packaging_log.json").read_text())
    assert log["subset"] == "locomotion"
    assert log["family"] == "static"
    assert sorted(log["packaged"]) == ["lightwheel_env_Apartment", "lightwheel_env_Grass"]
    assert (env_root / "LICENSE").read_text() == _SCRIPT.LICENSE_TEXT
    assert (env_root / "README.md").read_text() == _SCRIPT.ENV_PROVENANCE_README


def test_prepare_subset_defaults_to_manipulation(tmp_path):
    """Backward compatibility: the default subset is the Manipulation tree."""
    source = tmp_path / "bundle"
    asset_dir = source / "Manipulation" / "Microwave043"
    (asset_dir / "texture").mkdir(parents=True)
    _write_env_stage(asset_dir / "Microwave043.usd", terrain=False)
    (asset_dir / "texture" / "T_Body001_BC001.png").write_bytes(b"png")

    out_root, kept = _SCRIPT.prepare(source, tmp_path / "out", gate=False)
    assert out_root == tmp_path / "out" / "objects" / "lightwheel"
    assert kept == ["lightwheel_Microwave043"]
    assert (out_root / "lightwheel_Microwave043" / "lightwheel_Microwave043.usd").is_file()
    assert not (tmp_path / "out" / "objects" / "lightwheel_env").exists()


def test_prepare_subset_validation(tmp_path):
    with pytest.raises(SystemExit, match="Unknown --subset"):
        _SCRIPT.prepare(tmp_path, tmp_path / "out", gate=False, subset="nope")


def test_cli_accepts_subset(tmp_path):
    """The CLI exposes --subset and validates the source layout per subset."""
    source = tmp_path / "bundle"
    _make_locomotion_category(source, "Grass", "Grass.usd")
    output_dir = tmp_path / "out"

    assert _SCRIPT.main([
        "--source", str(source),
        "--output-dir", str(output_dir),
        "--subset", "locomotion",
        "--skip-gate",
    ]) == 0
    assert (output_dir / "objects" / "lightwheel_env" / "lightwheel_env_Grass").is_dir()

    # Manipulation subset on a Locomotion-only tree fails loudly.
    with pytest.raises(SystemExit, match="Manipulation"):
        _SCRIPT.main([
            "--source", str(source),
            "--output-dir", str(tmp_path / "out2"),
            "--subset", "manipulation",
            "--skip-gate",
        ])


# ------------------------------------------------------------------
# Static gate (offline assertions only — the GPU load check needs CUDA)
# ------------------------------------------------------------------

def test_static_gate_rejects_jointed_stage(assets_dir, tmp_path):
    """A misfiled articulated asset is routed out by the no-joints assert."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    object_id = "lightwheel_env_Grass"
    obj_dir = _make_env_object_dir(assets_dir, object_id)
    stage = Usd.Stage.Open(str(obj_dir / f"{object_id}.usd"))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/root")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/terrain")])
    joint.CreateAxisAttr("Z")
    stage.GetRootLayer().Save()

    item = LightwheelEnv(object_id)
    with pytest.raises(AssertionError, match="static"):
        _SCRIPT.sanity_check_static_env(item)


def test_static_gate_rejects_collisionless_stage(assets_dir):
    """A terrain without collision is a packaging bug."""
    from pxr import Usd, UsdPhysics

    object_id = "lightwheel_env_Grass"
    obj_dir = _make_env_object_dir(assets_dir, object_id)
    stage = Usd.Stage.Open(str(obj_dir / f"{object_id}.usd"))
    stage.GetPrimAtPath("/root/terrain").RemoveAPI(UsdPhysics.CollisionAPI)
    stage.GetRootLayer().Save()

    item = LightwheelEnv(object_id)
    with pytest.raises(AssertionError, match="CollisionAPI"):
        _SCRIPT.sanity_check_static_env(item)


def test_static_gate_rejects_floating_terrain(assets_dir):
    """A ``terrain``-tagged asset must sit at or below z = 0."""
    object_id = "lightwheel_env_Grass"
    obj_dir = _make_env_object_dir(assets_dir, object_id, size=(8.0, 6.0, 0.2))
    _write_env_stage(obj_dir / f"{object_id}.usd", size=(8.0, 6.0, 0.2), terrain=False)
    # Lift the terrain above the ground plane.
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(obj_dir / f"{object_id}.usd"))
    xform = UsdGeom.Xform(stage.GetPrimAtPath("/root"))
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))
    stage.GetRootLayer().Save()

    item = LightwheelEnv(object_id)
    with pytest.raises(AssertionError, match="z-min"):
        _SCRIPT.sanity_check_static_env(item)
