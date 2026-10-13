"""Tests for Google Scanned Objects integration (no downloads, no scenes).

The catalog ships with the assets as ``objects/gso/catalog.json``; tests
generate it from the checked-in Fuel metadata cache
(``scripts/data/gso_fuel_metadata.json``) via the prepare script's own
builder, so catalog content checks run against the real data.

Covers:
  - Generated catalog integrity (annotations broad → detailed, separate tags)
  - list_gso_objects tag / substring filtering
  - make_gso_pool selection / sampling / validation
  - GSOObject construction, path resolution, fixed variant
  - Realistic mass resolution (curated table / density fallback / override)
  - Texture binding forwarded to the scene (explicit diffuse texture surface)
  - Lazy asset completeness check incl. partial-download repair
  - Freejoint injection and catalog helpers in scripts/prepare_gso_assets.py
"""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.hub_objects import gso_object as gso_module
from unienv_genesis_collection.objects.hub_objects.gso_object import (
    GSO_ASSETS_ROOT,
    GSO_MASSES,
    GSO_MASS_BOUNDS,
    GSO_MASS_DENSITY,
    GSOObject,
    get_gso_catalog,
    list_gso_objects,
    make_gso_pool,
)
from unienv_genesis_collection.objects.mesh_object import MeshObject

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_FUEL_METADATA = _SCRIPTS_DIR / "data" / "gso_fuel_metadata.json"

_SAMPLE_ID = "3D_Dollhouse_Lamp"

_MJCF_TEMPLATE = """\
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

_OBJECT_FILES = (
    "model.obj",
    "texture.png",
    "model_collision_0.obj",
    "model_collision_1.obj",
)

_OBJ_TEMPLATE = """\
v {x0} {y0} {z0}
v {x1} {y0} {z0}
v {x0} {y1} {z0}
v {x0} {y0} {z1}
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_gso_assets", _SCRIPTS_DIR / "prepare_gso_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_object_dir(assets_dir: Path, object_id: str) -> Path:
    """Create a complete fake GSO object directory under *assets_dir*."""
    script = _load_prepare_script()
    obj_dir = assets_dir / GSO_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.xml").write_text(_MJCF_TEMPLATE)
    (obj_dir / "model_freejoint.xml").write_text(
        script.inject_freejoint(_MJCF_TEMPLATE)
    )
    for name in _OBJECT_FILES:
        (obj_dir / name).write_text("placeholder")
    _write_png(obj_dir / "texture.png")
    return obj_dir


def _write_png(path: Path, size=(2, 2)) -> None:
    """Write a real (tiny) PNG — ``gs.textures.ImageTexture`` decodes eagerly."""
    from PIL import Image

    Image.new("RGB", size, (255, 0, 0)).save(path)


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir with catalog.json built from the real Fuel metadata."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    script = _load_prepare_script()
    metadata = json.loads(_FUEL_METADATA.read_text())
    out_root = tmp_path / GSO_ASSETS_ROOT
    out_root.mkdir(parents=True)
    script.write_catalog(out_root, list(metadata), metadata)
    return tmp_path


@pytest.fixture
def fake_object(assets_dir):
    """A complete fake sample object directory."""
    return _make_fake_object_dir(assets_dir, _SAMPLE_ID)


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
    catalog = get_gso_catalog()
    assert len(catalog) == 1030
    assert _SAMPLE_ID in catalog
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert object_id and "/" not in object_id and " " not in object_id
        # Annotations broad -> detailed, separate from tags; same
        # conventions as YCB (lowercase, no bracketed qualifiers).
        assert info.language_annotations and all(info.language_annotations)
        for annotation in info.language_annotations:
            assert annotation == annotation.lower()
            assert "(" not in annotation and "[" not in annotation
        for tag in info.tags:
            assert tag == tag.lower() and " " not in tag
        # Objects with a category tag carry a broad term + the detailed one
        # (box_gripper_graspable is not a category and adds no annotation).
        if info.tags - {"box_gripper_graspable"}:
            assert len(info.language_annotations) >= 2

def test_catalog_spot_checks(assets_dir):
    lamp = get_gso_catalog()[_SAMPLE_ID]
    assert lamp.language_annotations == ("toy", "3d dollhouse lamp")
    assert lamp.tags == frozenset({"toys", "box_gripper_graspable"})
    # Categories follow the GSO paper's taxonomy; shoes dominate.
    assert len(list_gso_objects(tags="shoe")) > 200
    assert get_gso_catalog()["adistar_boost_m"].language_annotations[0] == "shoe"
    # Graspability is computed from the mesh; ~half of GSO objects qualify.
    n_graspable = len(list_gso_objects(tags="box_gripper_graspable"))
    assert 400 < n_graspable < 700


# ------------------------------------------------------------------
# list_gso_objects
# ------------------------------------------------------------------

def test_list_all_objects_sorted(assets_dir):
    ids = list_gso_objects()
    assert len(ids) == 1030
    assert ids == sorted(ids)

def test_list_filter_by_tag(assets_dir):
    toys = list_gso_objects(tags="toys")
    assert _SAMPLE_ID in toys
    catalog = get_gso_catalog()
    assert all("toys" in catalog[oid].tags for oid in toys)

def test_list_contains_filter_case_insensitive(assets_dir):
    lamps = list_gso_objects(contains="lamp")
    assert _SAMPLE_ID in lamps
    assert all("lamp" in oid.lower() for oid in lamps)

def test_list_tags_and_contains_combine(assets_dir):
    toy_lamps = list_gso_objects(tags="toys", contains="lamp")
    assert _SAMPLE_ID in toy_lamps
    assert set(toy_lamps) <= set(list_gso_objects(tags="toys"))
    assert set(toy_lamps) <= set(list_gso_objects(contains="lamp"))

def test_list_no_match(assets_dir):
    assert list_gso_objects(contains="zzz_no_such_object") == []
    assert list_gso_objects(tags="zzz_no_such_tag") == []


# ------------------------------------------------------------------
# make_gso_pool
# ------------------------------------------------------------------

def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    ids = ["Lalaloopsy_Peanut_Big_Top", _SAMPLE_ID, "adistar_boost_m"]
    ids = [i for i in ids if i in get_gso_catalog()][:2] + [_SAMPLE_ID]
    pool = make_gso_pool(ids)
    assert [o.object_id for o in pool] == ids

def test_pool_ids_and_filters_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError):
        make_gso_pool([_SAMPLE_ID], contains="lamp")
    with pytest.raises(ValueError):
        make_gso_pool([_SAMPLE_ID], tags="toys")

def test_pool_tags_filter(assets_dir, no_download):
    pool = make_gso_pool(tags="action_figures", n=3,
                         rng=torch.Generator().manual_seed(1))
    assert all("action_figures" in o.info.tags for o in pool)

def test_pool_sampling_reproducible(assets_dir, no_download):
    pool_a = make_gso_pool(n=6, rng=torch.Generator().manual_seed(7))
    pool_b = make_gso_pool(n=6, rng=torch.Generator().manual_seed(7))
    ids_a = [o.object_id for o in pool_a]
    assert ids_a == [o.object_id for o in pool_b]
    assert len(set(ids_a)) == 6

def test_pool_n_too_large(assets_dir):
    with pytest.raises(ValueError):
        make_gso_pool([_SAMPLE_ID], n=2)


# ------------------------------------------------------------------
# GSOObject construction
# ------------------------------------------------------------------

def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="not_an_object"):
        GSOObject("not_an_object")

def test_construction_downloads_no_object_assets(assets_dir, no_download):
    obj = GSOObject(_SAMPLE_ID)
    assert obj.object_id == _SAMPLE_ID
    assert obj.name == _SAMPLE_ID
    assert obj.language_annotations == ("toy", "3d dollhouse lamp")
    assert obj.info.tags == frozenset({"toys", "box_gripper_graspable"})
    assert not no_download  # only the (already local) catalog was touched

def test_paths_and_freejoint_variant(assets_dir):
    expected_dir = assets_dir / GSO_ASSETS_ROOT / _SAMPLE_ID
    movable = GSOObject(_SAMPLE_ID)
    assert movable.asset_dir == expected_dir
    assert movable.mjcf_file == expected_dir / "model_freejoint.xml"
    assert Path(movable.mesh_file) == expected_dir / "model.obj"
    welded = GSOObject(_SAMPLE_ID, fixed=True)
    assert welded.mjcf_file == expected_dir / "model.xml"

def test_anisotropic_scale_rejected(assets_dir):
    with pytest.raises(ValueError, match="uniform"):
        GSOObject(_SAMPLE_ID, scale=(1.0, 2.0, 1.0))

def test_overrides(assets_dir):
    obj = GSOObject(_SAMPLE_ID, name="lamp", language_annotations=["tiny lamp"])
    assert obj.name == "lamp"
    assert obj.language_annotations == ("tiny lamp",)


def test_local_bounding_box_uses_collision_mesh_bounds(fake_object, no_download):
    (fake_object / "model.obj").write_text(
        _OBJ_TEMPLATE.format(x0=0, y0=0, z0=0, x1=1, y1=1, z1=1)
    )
    (fake_object / "model_collision_0.obj").write_text(
        _OBJ_TEMPLATE.format(x0=-2, y0=-1, z0=-3, x1=-1, y1=0, z1=-2)
    )
    (fake_object / "model_collision_1.obj").write_text(
        _OBJ_TEMPLATE.format(x0=3, y0=4, z0=5, x1=6, y1=7, z1=8)
    )

    obj = GSOObject(_SAMPLE_ID)
    bbox = obj.get_local_bounding_box()
    assert torch.allclose(
        bbox.low,
        torch.tensor([-2.0, -1.0, -3.0], dtype=bbox.low.dtype),
    )
    assert torch.allclose(
        bbox.high,
        torch.tensor([6.0, 7.0, 8.0], dtype=bbox.high.dtype),
    )


def test_local_bounding_box_ignores_unreferenced_collision_mesh(fake_object, no_download):
    (fake_object / "model_collision_0.obj").write_text(
        _OBJ_TEMPLATE.format(x0=0, y0=0, z0=0, x1=1, y1=1, z1=1)
    )
    (fake_object / "model_collision_1.obj").write_text(
        _OBJ_TEMPLATE.format(x0=2, y0=2, z0=2, x1=3, y1=3, z1=3)
    )
    (fake_object / "model_collision_999.obj").write_text(
        _OBJ_TEMPLATE.format(x0=100, y0=100, z0=100, x1=101, y1=101, z1=101)
    )

    obj = GSOObject(_SAMPLE_ID)
    bbox = obj.get_local_bounding_box()
    assert torch.allclose(
        bbox.low,
        torch.tensor([0.0, 0.0, 0.0], dtype=bbox.low.dtype),
    )
    assert torch.allclose(
        bbox.high,
        torch.tensor([3.0, 3.0, 3.0], dtype=bbox.high.dtype),
    )


# ------------------------------------------------------------------
# Asset completeness (_ensure_assets)
# ------------------------------------------------------------------

def test_ensure_assets_complete_no_download(fake_object, no_download):
    obj = GSOObject(_SAMPLE_ID)
    obj._ensure_assets()
    assert obj._assets_ready

def test_ensure_assets_partial_dir_triggers_redownload(fake_object, monkeypatch):
    (fake_object / "texture.png").unlink()
    calls = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        (fake_object / "texture.png").write_text("placeholder")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    obj = GSOObject(_SAMPLE_ID)
    obj._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == (
        f"{GSO_ASSETS_ROOT}/{_SAMPLE_ID}/**"
    )
    assert obj._assets_ready

def test_ensure_assets_mjcf_meshes_checked(fake_object, monkeypatch):
    (fake_object / "model_collision_1.obj").unlink()
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    obj = GSOObject(_SAMPLE_ID)
    with pytest.raises(FileNotFoundError, match="model_collision_1.obj"):
        obj._ensure_assets()
    assert not obj._assets_ready


# ------------------------------------------------------------------
# prepare_gso_assets.py freejoint injection
# ------------------------------------------------------------------

def test_inject_freejoint():
    script = _load_prepare_script()
    out = script.inject_freejoint(_MJCF_TEMPLATE)
    root = ET.fromstring(out)
    body = root.find("./worldbody/body")
    assert body[0].tag == "freejoint"
    # Everything else is preserved.
    assert len(body.findall("geom")) == 3
    assert len(root.findall("./asset/mesh")) == 3

def test_inject_freejoint_rejects_jointed_body():
    script = _load_prepare_script()
    jointed = _MJCF_TEMPLATE.replace(
        '<body name="model">', '<body name="model"><freejoint/>',
    )
    with pytest.raises(ValueError, match="already has a joint"):
        script.inject_freejoint(jointed)


# ------------------------------------------------------------------
# prepare_gso_assets.py catalog helpers
# ------------------------------------------------------------------

def test_annotation_from_description():
    script = _load_prepare_script()
    fn = script.annotation_from_description
    # Bracketed qualifiers dropped, lowercased.
    assert fn("x", "Pokémon Conquest [Nintendo DS Game]") == "pokémon conquest"
    assert fn("x", "Lamp (small)") == "lamp"
    # Unbalanced brackets (closing half on a dropped line) truncate.
    assert fn("x", "Season Two (Includes Digital\nCopy)") == "season two"
    # Only the first line is used.
    assert fn("x", "Weisshai Great White Shark\nTest") == (
        "weisshai great white shark"
    )
    # Empty description falls back to the id.
    assert fn("Mens_Bahama_in_White", "") == "mens bahama in white"

def test_tag_from_category():
    script = _load_prepare_script()
    assert script.tag_from_category("Shoe") == "shoe"
    assert script.tag_from_category("Bottles and Cans and Cups") == (
        "bottles_and_cans_and_cups"
    )

def test_build_catalog_broad_to_detailed():
    script = _load_prepare_script()
    metadata = {
        "Schleich_Lion_Action_Figure": {
            "description": "Schleich Lion Action Figure",
            "categories": ["Action Figures"],
        },
        "Mystery_Item": {"description": "", "categories": []},
    }
    # graspable_ids={} isolates the category-tag/annotation logic.
    catalog = script.build_catalog(list(metadata), metadata, graspable_ids=set())
    lion = catalog["Schleich_Lion_Action_Figure"]
    assert lion["language_annotations"] == [
        "toy", "action figure", "schleich lion action figure",
    ]
    assert lion["tags"] == ["action_figures"]
    # No metadata categories -> id-derived annotation only, no tags.
    mystery = catalog["Mystery_Item"]
    assert mystery["language_annotations"] == ["mystery item"]
    assert mystery["tags"] == []

def test_build_catalog_injects_graspable_tag():
    script = _load_prepare_script()
    metadata = {
        "Toy_A": {"description": "Toy A", "categories": ["Toys"]},
        "Toy_B": {"description": "Toy B", "categories": ["Toys"]},
    }
    catalog = script.build_catalog(list(metadata), metadata, graspable_ids={"Toy_A"})
    assert "box_gripper_graspable" in catalog["Toy_A"]["tags"]
    assert "box_gripper_graspable" not in catalog["Toy_B"]["tags"]
    # Graspability is a tag only, never a language annotation.
    assert "box_gripper_graspable" not in catalog["Toy_A"]["language_annotations"]


# ------------------------------------------------------------------
# Realistic masses
# ------------------------------------------------------------------

def test_mass_argument_overrides(assets_dir, no_download):
    obj = GSOObject(_SAMPLE_ID, mass=0.42)
    assert obj.mass == 0.42
    # No asset access needed: the override wins outright.
    assert obj.resolve_mass() == 0.42
    assert not no_download


def test_mass_argument_must_be_positive(assets_dir):
    with pytest.raises(ValueError, match="positive"):
        GSOObject(_SAMPLE_ID, mass=0.0)
    with pytest.raises(ValueError, match="positive"):
        GSOObject(_SAMPLE_ID, mass=-1.0)


def test_mass_defaults_to_none_on_mesh_object(assets_dir):
    assert MeshObject(mesh_file="whatever.obj").mass is None


def test_curated_masses_are_known_objects_with_plausible_weights(assets_dir):
    catalog = get_gso_catalog()
    assert GSO_MASSES, "the curated mass table must not be empty"
    for object_id, mass in GSO_MASSES.items():
        assert object_id in catalog, object_id
        assert 0.0 < mass <= GSO_MASS_BOUNDS[1], (object_id, mass)


def test_curated_mass_is_used_without_asset_access(assets_dir, no_download):
    """A curated entry resolves offline — no hull volume needed."""
    assert _SAMPLE_ID in GSO_MASSES
    obj = GSOObject(_SAMPLE_ID)
    assert obj.mass is None
    assert obj.resolve_mass() == GSO_MASSES[_SAMPLE_ID]
    assert not no_download


def test_density_fallback_scales_with_hull_volume(assets_dir, no_download, monkeypatch):
    """Uncurated objects fall back to hull volume x density, clamped."""
    monkeypatch.setattr(gso_module, "GSO_MASSES", {})
    obj = GSOObject(_SAMPLE_ID)

    def _volume(value):
        return lambda: SimpleNamespace(volume=value)

    monkeypatch.setattr(obj, "_load_collision_trimesh", _volume(0.002))
    assert obj.resolve_mass() == pytest.approx(0.002 * GSO_MASS_DENSITY)

    low, high = GSO_MASS_BOUNDS
    monkeypatch.setattr(obj, "_load_collision_trimesh", _volume(1e-9))
    assert obj.resolve_mass() == pytest.approx(low)
    monkeypatch.setattr(obj, "_load_collision_trimesh", _volume(10.0))
    assert obj.resolve_mass() == pytest.approx(high)

    # The fallback density is a plausible household-object density, far below
    # the ~1000-2000 kg/m^3 the geometry-derived mass implies.
    assert 50.0 <= GSO_MASS_DENSITY <= 400.0


def test_apply_mass_sets_the_entity_mass(assets_dir, no_download):
    class _Entity:
        def __init__(self):
            self.masses = []

        def set_mass(self, mass):
            self.masses.append(mass)

    entity = _Entity()
    GSOObject(_SAMPLE_ID, mass=0.25).apply_mass(entity)
    GSOObject(_SAMPLE_ID).apply_mass(entity)  # curated table
    assert entity.masses == [0.25, GSO_MASSES[_SAMPLE_ID]]


def test_apply_mass_skips_fixed_objects(assets_dir, no_download):
    class _Entity:
        def set_mass(self, mass):  # pragma: no cover - must not be called
            raise AssertionError("set_mass must not be called for fixed objects")

    GSOObject(_SAMPLE_ID, fixed=True, mass=0.25).apply_mass(_Entity())


def test_apply_mass_is_a_noop_without_a_mass(assets_dir):
    class _Entity:
        def set_mass(self, mass):  # pragma: no cover - must not be called
            raise AssertionError("set_mass must not be called without a mass")

    MeshObject(mesh_file="whatever.obj").apply_mass(_Entity())


def test_cluttered_table_manager_applies_descriptor_masses():
    """The manager finalises every loaded entity's mass after the build."""
    from unienv_genesis_collection.scenes.cluttered_table_manager import (
        ClutteredTableManager,
    )

    applied = []
    descriptors = [
        SimpleNamespace(apply_mass=lambda entity: applied.append((0, entity))),
        SimpleNamespace(apply_mass=lambda entity: applied.append((1, entity))),
    ]
    manager = object.__new__(ClutteredTableManager)
    manager._loaded_objects = descriptors
    manager._loaded_entities = ["entity_0", "entity_1"]

    manager._apply_object_masses()
    assert applied == [(0, "entity_0"), (1, "entity_1")]


# ------------------------------------------------------------------
# Texture binding
# ------------------------------------------------------------------

class _FakeScene:
    def __init__(self):
        self.added = []

    def add_entity(self, morph, surface=None, name=None):
        self.added.append(SimpleNamespace(morph=morph, surface=surface, name=name))
        return self.added[-1]


def test_load_into_scene_binds_the_mjcf_texture_explicitly(fake_object, no_download):
    """The morph is added with an explicit diffuse-texture surface.

    A renderer that re-exports the scene (Nyx) otherwise serialises every
    MJCF mesh's texture to the same generic material file, so a scene's GSO
    objects would all share one texture; the explicit ``ImageTexture`` is
    exported per instance instead.
    """
    import genesis as gs

    captured = []

    class _MJCFSpy:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.kwargs = kwargs

    scene = _FakeScene()
    world = SimpleNamespace(scene=scene)
    obj = GSOObject(_SAMPLE_ID)

    with mock.patch.object(gs, "morphs", SimpleNamespace(MJCF=_MJCFSpy)):
        entity = obj.load_into_scene(world, pos=(0.1, 0.2, 0.3))

    assert len(captured) == 1, captured
    assert captured[0]["file"] == str(obj.mjcf_file)
    assert entity.name == _SAMPLE_ID

    surface = entity.surface
    assert isinstance(surface, gs.surfaces.Plastic), surface
    assert isinstance(surface.diffuse_texture, gs.textures.ImageTexture), surface
    assert Path(surface.diffuse_texture.image_path) == Path(obj.visual_texture_path)
    assert Path(obj.visual_texture_path) == obj.asset_dir / "texture.png"
