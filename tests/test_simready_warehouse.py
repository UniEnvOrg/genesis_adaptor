"""Tests for the SimReady Warehouse rigid-prop integration (no downloads).

Mirrors ``tests/test_artvip_furniture.py`` / ``tests/test_molmospaces_furniture.py``
with **synthetic USD stages** in ``tmp_path`` — plus the family's two
packaging converters (the ``RigidBodyAPI`` authoring and the
``metersPerUnit`` normalisation):

  - RigidBodyAPI authoring: applied on the top-level payload prim only (not
    the geometry/material prims), idempotent, and what makes the stage
    collapse into one rigid component.
  - Unit normalisation: mixed ``metersPerUnit`` (0.01 / 1.0) folds into the
    root prim's linear map — world size unchanged, one unit convention.
  - MDL → UsdPreviewSurface reuse (shared converter, covered in
    ``test_lightwheel_furniture.py``; here only the packaging wiring).
  - Catalog integrity + tag/contains filtering (CSV-derived annotations).
  - SimReadyWarehouseProp construction (``prop_``/``hand_`` ids, scalar
    scale, reference_qpos refusal) and asset completeness (partial-download
    repair on the ``materials/`` sidecar layout).
  - Offline FK (box proxy degenerates to the whole-prop box), clean names,
    empty joint set.
  - scripts/prepare_simready_warehouse_assets.py: CSV selection
    (classification classes, HandManipulation file stems), physics-overlay
    packaging (``.thumbs``/``.mdl`` dropping + rename), catalog generation,
    license text.
  - RoomBuilder._resolve_descriptor ``source="simready_warehouse"`` branch.

A GPU smoke test (skipped unless CUDA) loads the synthetic stage through
``load_into_scene`` (single entity, offline bbox vs ``get_vAABB``) — the
packaging gate's core assertion; **no moving joints are required** (rigid
props).
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.articulated_object import (
    ArticulationError,
    usd_link_meshes_at_config,
)
from unienv_genesis_collection.objects.hub_objects.articulated.simready_warehouse import (
    SIMREADY_WAREHOUSE_ASSETS_ROOT,
    SimReadyWarehouseProp,
    get_simready_warehouse_catalog,
    list_simready_warehouse_objects,
    make_simready_warehouse_pool,
)
from unienv_genesis_collection.scenes.room_builder import (
    FurniturePlacement,
    RoomBuilder,
    RoomLayoutSpec,
)
from unienv_genesis_collection.utils.usd_conversions import (
    convert_mdl_to_preview_surface,
    count_mdl_materials,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_simready_warehouse_assets",
        _SCRIPTS_DIR / "prepare_simready_warehouse_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_prepare_script()


def _usd_runtime_is_safe() -> bool:
    return "gs_nyx" not in __import__("sys").modules


@pytest.fixture(autouse=True)
def _skip_when_usd_runtime_poisoned():
    if not _usd_runtime_is_safe():
        pytest.skip("gs_nyx DSO present — the usd-core Tf runtime is poisoned")


# ------------------------------------------------------------------
# Synthetic USD stage authoring (pxr)
# ------------------------------------------------------------------


def _write_box_mesh(stage, path: str, extents):
    """A centered-box Mesh prim with a PhysicsCollisionAPI."""
    from pxr import Gf, UsdGeom, UsdPhysics

    ex, ey, ez = extents
    pts = [
        Gf.Vec3f(px * ex / 2, py * ey / 2, pz * ez / 2)
        for px in (-1.0, 1.0)
        for py in (-1.0, 1.0)
        for pz in (-1.0, 1.0)
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 1, 5, 7, 3, 0, 2, 6, 4]
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh.GetPrim()


def _write_warehouse_stage(path: Path, *, meters_per_unit: float = 1.0,
                           mdl: bool = False) -> Path:
    """A Warehouse-shaped *visual* stage: /RootNode + meshes + materials.

    With ``mdl=True`` the material is an OmniPBR MDL graph (as upstream);
    the stage is written at *meters_per_unit* (the dataset mixes 0.01 and
    1.0).  This is the VISUAL layer — the packaged stage is the
    ``_physics`` overlay over it (the colliders), which the prep script
    synthesises; a plain ``<id>_physics.usd`` overlay is authored too.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", float(meters_per_unit))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("defaultPrim", "RootNode")

    root = UsdGeom.Xform.Define(stage, "/RootNode")
    UsdGeom.XformCommonAPI(root.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
    UsdGeom.Scope.Define(stage, "/RootNode/Geometry")
    UsdGeom.Scope.Define(stage, "/RootNode/Materials")

    if mdl:
        material = UsdShade.Material.Define(stage, "/RootNode/Materials/Cardboard_C1")
        shader = UsdShade.Shader.Define(stage, "/RootNode/Materials/Cardboard_C1/Shader")
        shader.CreateImplementationSourceAttr("sourceAsset")
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("./materials/Cardboard_C1.mdl"))
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
        ).Set("OmniPBR")
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./materials/textures/T_Cardbox_Albedo.png")
        )
        shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./materials/textures/T_Cardbox_ORM.png")
        )
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./materials/textures/T_Cardbox_Normal.png")
        )
    else:
        material = UsdShade.Material.Define(stage, "/RootNode/Materials/Cardboard_C1")
        surface = UsdShade.Shader.Define(
            stage, "/RootNode/Materials/Cardboard_C1/UsdPreviewSurface"
        )
        surface.CreateIdAttr("UsdPreviewSurface")
        texture = UsdShade.Shader.Define(
            stage, "/RootNode/Materials/Cardboard_C1/Tex_diffuse"
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./materials/textures/T_Cardbox_Albedo.png")
        )
        reader = UsdShade.Shader.Define(
            stage, "/RootNode/Materials/Cardboard_C1/Tex_diffuse/stReader"
        )
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    # Box extents in *stage units* (upstream authors centimetres): the
    # meters-per-unit factor carries the geometry to its metre size, so the
    # packaged metersPerUnit→1.0 normalisation preserves the world size.
    unit = float(meters_per_unit)
    extents_stage_units = (0.3 / unit, 0.2 / unit, 0.4 / unit)
    box = _write_box_mesh(stage, "/RootNode/Geometry/Cardbox_C1", extents_stage_units)
    UsdShade.MaterialBindingAPI.Apply(box)
    UsdShade.MaterialBindingAPI(box).Bind(material)

    stage.GetRootLayer().Save()

    # The `_physics` overlay variant: sublayers the visual and applies the
    # convex-hull collision over the same prim path (upstream convention —
    # the upstream overlay files are `<Asset>_physics.usd`).
    physics = path.parent / f"{path.stem}_physics.usd"
    physics.write_text(
        "#usda 1.0\n"
        "(\n"
        f"    metersPerUnit = {float(meters_per_unit)}\n"
        "    upAxis = \"Z\"\n"
        f"    subLayers = [@./{path.name}@]\n"
        ")\n\n"
        "over \"RootNode\"\n"
        "{\n"
        "    over \"Geometry\"\n"
        "    {\n"
        "        over \"Cardbox_C1\" (\n"
        "            prepend apiSchemas = [\"PhysicsCollisionAPI\", "
        "\"PhysicsMeshCollisionAPI\"]\n"
        "        )\n"
        "        {\n"
        "            uniform token physics:approximation = \"convexHull\"\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    return physics


def _make_fake_prop_dir(assets_dir: Path, object_id: str, *, meters_per_unit: float = 0.01,
                        mdl: bool = False) -> Path:
    """A complete fake SimReady Warehouse prop directory (physics overlay).

    The packaged-stage edits (MDL → UsdPreviewSurface, ``RigidBodyAPI`` on
    the root, ``metersPerUnit`` → 1.0) run through the prep script's own
    converters — the fixture materialises the *packaged* shape the
    descriptor loads.
    """
    obj_dir = assets_dir / SIMREADY_WAREHOUSE_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    visual = obj_dir / "Cardbox_C1.usda"
    physics = _write_warehouse_stage(
        visual, meters_per_unit=meters_per_unit, mdl=mdl,
    )
    packaged_stage = obj_dir / f"{object_id}.usd"
    shutil.copy2(physics, packaged_stage)
    physics.unlink()

    if mdl:
        convert_mdl_to_preview_surface(str(visual))
        convert_mdl_to_preview_surface(str(packaged_stage))
    from pxr import Usd

    stage = Usd.Stage.Open(str(packaged_stage))
    _SCRIPT._apply_root_rigid_body(stage)
    _SCRIPT._normalise_units(stage)
    stage.GetRootLayer().Save()

    # The material texture sidecar (a real 2×2 PNG — the Genesis material
    # parser opens the referenced files).
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (160, 110, 70)).save(buffer, format="PNG")
    for name in ("T_Cardbox_Albedo.png", "T_Cardbox_ORM.png", "T_Cardbox_Normal.png"):
        textures = obj_dir / "materials" / "textures"
        textures.mkdir(parents=True, exist_ok=True)
        (textures / name).write_bytes(buffer.getvalue())
    return obj_dir


def _catalog_entry(stem: str, *, label: str = "cardboard box", mass: str = "1.0",
                   classification: str = "Prop general"):
    """A CSV-derived catalog entry (via the packaging annotation path)."""
    record = dict(classification=classification, label=label, q_code="Q1395006",
                  mass=mass, asset_name=stem)
    entry = _SCRIPT.annotate_object(record)[1]
    return entry


def _make_catalog(assets_dir: Path, entries: dict) -> None:
    catalog_file = assets_dir / SIMREADY_WAREHOUSE_ASSETS_ROOT / "catalog.json"
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    catalog_file.write_text(json.dumps(entries))


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir with a generated (template-path) catalog."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    _make_catalog(assets_dir=tmp_path, entries={
        "prop_cardbox_c1": _catalog_entry("Cardbox_C1"),
        "prop_cardbox_a1": _catalog_entry(
            "Cardbox_A1", label="cardboard box", classification="Prop general",
        ),
        "prop_trolley_wheel_base": _catalog_entry(
            "Trolley_wheel_base", label="cart", classification="Prop general",
        ),
        "hand_sm_book_open_a01_simready_01": _catalog_entry(
            "sm_book_open_a01_simready_01", label="book",
            classification="Prop general hand manipulation",
        ),
    })
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()
    return tmp_path


@pytest.fixture
def fake_cardbox(assets_dir):
    """A complete fake 'prop_cardbox_c1' prop dir (physics overlay .usda)."""
    return _make_fake_prop_dir(assets_dir, "prop_cardbox_c1", meters_per_unit=0.01)


@pytest.fixture
def no_download(monkeypatch):
    calls = []

    def _fail(**kwargs):
        calls.append(kwargs)
        raise AssertionError(f"Unexpected download attempt: {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail)
    return calls


# ------------------------------------------------------------------
# The RigidBodyAPI + unit converters
# ------------------------------------------------------------------

def test_rigid_body_authoring(tmp_path):
    from pxr import Usd, UsdPhysics

    physics = _write_warehouse_stage(
        tmp_path / "Cardbox_C1.usda", meters_per_unit=1.0, mdl=False,
    )
    stage = Usd.Stage.Open(str(physics))
    count = _SCRIPT._apply_root_rigid_body(stage)
    assert count == 1  # exactly the /RootNode payload prim
    assert stage.GetPrimAtPath("/RootNode").HasAPI(UsdPhysics.RigidBodyAPI)
    assert not stage.GetPrimAtPath("/RootNode/Geometry/Cardbox_C1").HasAPI(
        UsdPhysics.RigidBodyAPI
    )
    # Idempotent.
    assert _SCRIPT._apply_root_rigid_body(stage) == 0


def test_unit_normalisation_preserves_world_size(tmp_path):
    """A 0.01 stage keeps its world extents when normalised to metersPerUnit 1."""
    from pxr import Usd, UsdGeom

    physics = _write_warehouse_stage(
        tmp_path / "Cardbox_C1.usda", meters_per_unit=0.01, mdl=False,
    )
    stage = Usd.Stage.Open(str(physics))
    before = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False
    ).ComputeWorldBound(stage.GetPrimAtPath("/RootNode"))
    aligned_before = before.ComputeAlignedRange()

    old = _SCRIPT._normalise_units(stage)
    assert old == pytest.approx(0.01, abs=1e-6)
    after = UsdGeom.GetStageMetersPerUnit(stage)
    assert after == pytest.approx(1.0, abs=1e-9)
    bounds_after = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False
    ).ComputeWorldBound(stage.GetPrimAtPath("/RootNode"))
    aligned_after = bounds_after.ComputeAlignedRange()

    # Stage-unit extents shrink by the factor; the *world* size (metres)
    # is unchanged: extent × metersPerUnit before and after.
    size_before = aligned_before.GetSize()[0] * old
    size_after = aligned_after.GetSize()[0] * after
    assert size_after == pytest.approx(size_before, abs=1e-6)


# ------------------------------------------------------------------
# Catalog
# ------------------------------------------------------------------

def test_catalog_integrity(assets_dir):
    catalog = get_simready_warehouse_catalog()
    assert len(catalog) == 4
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert object_id.startswith(("prop_", "hand_"))
        assert len(info.language_annotations) >= 2
        assert info.tags
        assert {"rigid"} <= info.tags
        for tag in info.tags:
            assert tag == tag.lower() and " " not in tag


def test_catalog_is_cached(assets_dir, no_download):
    assert get_simready_warehouse_catalog() is get_simready_warehouse_catalog()


def test_list_all_objects_sorted(assets_dir):
    ids = list_simready_warehouse_objects()
    assert ids == sorted(ids)
    assert ids == [
        "hand_sm_book_open_a01_simready_01", "prop_cardbox_a1",
        "prop_cardbox_c1", "prop_trolley_wheel_base",
    ]


def test_list_filter_by_tags_and_contains(assets_dir):
    hands = list_simready_warehouse_objects(tags="hand")
    assert hands == ["hand_sm_book_open_a01_simready_01"]
    assert list_simready_warehouse_objects(contains="CARDBOX") == [
        "prop_cardbox_a1", "prop_cardbox_c1",
    ]
    assert list_simready_warehouse_objects(tags="no_such_tag") == []


def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="not_an_object"):
        SimReadyWarehouseProp("not_an_object")


# ------------------------------------------------------------------
# make_simready_warehouse_pool
# ------------------------------------------------------------------

def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    pool = make_simready_warehouse_pool(["prop_cardbox_c1", "prop_trolley_wheel_base"])
    assert [o.object_id for o in pool] == ["prop_cardbox_c1", "prop_trolley_wheel_base"]


def test_pool_ids_and_tags_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError, match="not both"):
        make_simready_warehouse_pool(["prop_cardbox_c1"], tags="prop")


def test_pool_sampling_reproducible(assets_dir, no_download):
    pool_a = make_simready_warehouse_pool(
        tags="rigid", n=3, rng=torch.Generator().manual_seed(42)
    )
    pool_b = make_simready_warehouse_pool(
        tags="rigid", n=3, rng=torch.Generator().manual_seed(42)
    )
    assert [o.object_id for o in pool_a] == [o.object_id for o in pool_b]
    assert len({o.object_id for o in pool_a}) == 3


def test_pool_kwargs_forwarded(assets_dir, no_download):
    pool = make_simready_warehouse_pool(
        ["prop_cardbox_c1"], scale=0.5, fixed=False,
    )
    assert pool[0].scale == 0.5
    assert pool[0].fixed is False


# ------------------------------------------------------------------
# Construction + completeness
# ------------------------------------------------------------------

def test_construction_downloads_no_object_assets(assets_dir, no_download):
    item = SimReadyWarehouseProp("prop_cardbox_c1")
    assert item.object_id == "prop_cardbox_c1"
    assert item.description_file.name == "prop_cardbox_c1.usd"
    assert item.fixed is True
    assert item.reference_qpos == {}
    assert not no_download


def test_reference_qpos_refused(assets_dir):
    with pytest.raises(ValueError, match="rigid prop"):
        SimReadyWarehouseProp("prop_cardbox_c1", reference_qpos={"x": 0.0})


def test_scalar_scale_only(assets_dir):
    with pytest.raises(ValueError, match="uniform"):
        SimReadyWarehouseProp("prop_cardbox_c1", scale=(1.0, 2.0, 3.0))


def test_required_files_materials_sidecar(fake_cardbox):
    item = SimReadyWarehouseProp("prop_cardbox_c1")
    required = item._required_files()
    assert required[0] == item.description_file
    assert item.asset_dir / "materials" / "textures" / "T_Cardbox_Albedo.png" in required
    assert len(required) == len(set(required))


def test_ensure_assets_partial_dir_triggers_redownload(assets_dir, monkeypatch):
    _make_fake_prop_dir(assets_dir, "prop_cardbox_c1")
    texture = (assets_dir / SIMREADY_WAREHOUSE_ASSETS_ROOT / "prop_cardbox_c1"
               / "materials" / "textures" / "T_Cardbox_Albedo.png")
    texture.unlink()
    calls: list = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        texture.write_bytes(b"png")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    item = SimReadyWarehouseProp("prop_cardbox_c1")
    item._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == (
        f"{SIMREADY_WAREHOUSE_ASSETS_ROOT}/prop_cardbox_c1/**"
    )
    assert item._assets_ready


# ------------------------------------------------------------------
# Offline FK (rigid degenerate) + clean names + empty joints
# ------------------------------------------------------------------

def test_fk_returns_the_whole_prop_box(fake_cardbox):
    """The jointless root FK degenerates to one box for the single body."""
    item = SimReadyWarehouseProp("prop_cardbox_c1")
    meshes = item._link_meshes_at_reference()
    assert len(meshes) == 1
    bounds = next(iter(meshes.values())).bounds
    # The 0.3×0.2×0.4 box centred at the stage origin (in metres after the
    # packaged normalisation).
    assert bounds[0][0] == pytest.approx(-0.15, abs=1e-5)
    assert bounds[1][2] == pytest.approx(0.2, abs=1e-5)
    union = item.get_local_bounding_box()
    assert torch.allclose(
        _host(union.low), _host(torch.tensor([-0.15, -0.1, -0.2], dtype=torch.float32))
    )
    assert torch.allclose(
        _host(union.high), _host(torch.tensor([0.15, 0.1, 0.2], dtype=torch.float32))
    )


def _host(tensor):
    return tensor.detach().to("cpu") if tensor.is_cuda else tensor


def test_jointless_contract(fake_cardbox):
    """No joints; a single link; link-local geometry works for part grasps."""
    item = SimReadyWarehouseProp("prop_cardbox_c1")
    assert item.list_joint_names() == []
    assert len(item.list_link_names()) == 1
    link_name = item.list_link_names()[0]
    local = item.get_link_local_bounding_box(link_name)
    assert local is not None
    # Joint queries raise the standard resolution error (nothing to drive).
    from pxr import UsdPhysics  # noqa: F401 — the error message lists clean names

    with pytest.raises(ArticulationError):
        item._to_runtime_name("no_such_joint", "joint")


def test_fk_unknown_joint_raises(fake_cardbox):
    with pytest.raises(ArticulationError) as exc_info:
        usd_link_meshes_at_config(
            str(fake_cardbox / "prop_cardbox_c1.usd"), {"no_such_joint": 0.1}
        )
    assert "Available joints" in str(exc_info.value)


# ------------------------------------------------------------------
# RoomBuilder descriptor resolution
# ------------------------------------------------------------------

def test_room_builder_resolves_simready_warehouse_source(assets_dir, fake_cardbox,
                                                         no_download):
    world = type("WorldStub", (), {"batch_size": None})()
    room = RoomBuilder(
        world,
        name="room",
        layout=RoomLayoutSpec(room_size=(4.0, 4.0, 2.7), furniture=()),
    )
    placement = FurniturePlacement(
        object_id="prop_cardbox_c1",
        source="simready_warehouse",
        pos=(0.0, 0.0, 0.0),
        yaw=0.0,
    )
    descriptor = room._resolve_descriptor(placement)
    assert isinstance(descriptor, SimReadyWarehouseProp)
    assert descriptor.object_id == "prop_cardbox_c1"

    with pytest.raises(ValueError, match="simready_warehouse"):
        room._resolve_descriptor(FurniturePlacement(
            object_id="x", source="unsuppo", pos=(0, 0, 0), yaw=0.0,
        ))


# ------------------------------------------------------------------
# prepare_simready_warehouse_assets.py
# ------------------------------------------------------------------

def _make_csv(tmp_path: Path, rows) -> Path:
    csv_file = tmp_path / "physical_ai_simready_warehouse_01.csv"
    header = ["asset_preview", "asset_name", "relative_path", "classification",
              "label", "q_code", "mass", "thumbnail_path"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(
            str(row.get(column, "")) for column in header
        ))
    csv_file.write_text("\n".join(lines) + "\n")
    return csv_file


def _make_source_props(source: Path, dirs):
    """Upstream prop dirs under Props/<group>/<Dir>/ (visual + physics)."""
    for group, stem in dirs:
        asset_dir = source / "Props" / group / stem
        asset_dir.mkdir(parents=True)
        _write_warehouse_stage(
            asset_dir / f"{stem}.usd", meters_per_unit=0.01, mdl=True,
        )
        (asset_dir / ".thumbs").mkdir(exist_ok=True)
        (asset_dir / ".thumbs" / "thumb.png").write_bytes(b"png")
        mdl_dir = asset_dir / "materials"
        mdl_dir.mkdir(parents=True, exist_ok=True)
        (mdl_dir / "Cardboard_C1.mdl").write_bytes(b"mdl")
        textures = asset_dir / "materials" / "textures"
        textures.mkdir(parents=True, exist_ok=True)
        for name in ("T_Cardbox_Albedo.png", "T_Cardbox_ORM.png", "T_Cardbox_Normal.png"):
            (textures / name).write_bytes(b"png")
    return source


def test_csv_selection_groups_by_directory(tmp_path):
    csv_file = _make_csv(tmp_path, [
        dict(asset_name="Cardbox_C1_physics",
             relative_path="Props/general/Cardbox_C1/Cardbox_C1_physics.usd",
             classification="Prop general", label="Cardboard Box", mass="1.0"),
        dict(asset_name="sm_book_open_a01_simready_01",
             relative_path="Props/general/HandManipulation/book_open_a/"
                           "sm_book_open_a01_simready_01.usd",
             classification="Prop general hand manipulation", label="book"),
        dict(asset_name="Pallets_A1_physics",
             relative_path="Props/assembly/Pallets_A1/Pallets_A1_physics.usd",
             classification="Assembly", label="pallet"),
        dict(asset_name="physical_ai_simready_warehouse_01",
             relative_path="physical_ai_simready_warehouse_01.usd",
             classification="Scenario", label="warehouse"),
    ])
    rows = _SCRIPT.read_csv_catalog(csv_file)
    assert len(rows) == 3  # the scene row is not a per-prop row

    props = dict(_SCRIPT.select_csv_objects(rows, ["prop", "handmanip"]))
    assert set(props) == {"prop_cardbox_c1", "hand_sm_book_open_a01_simready_01"}
    assemblies = dict(_SCRIPT.select_csv_objects(rows, ["assembly"]))
    assert set(assemblies) == {"prop_pallets_a1"}
    everything = dict(_SCRIPT.select_csv_objects(rows, _SCRIPT.ALL_KNOWN_CATEGORIES))
    assert set(everything) == set(props) | set(assemblies)

    # The shared selection path (the no-`--source` download planning and the
    # packaging both call it) agrees with the raw class selection.
    assert dict(_SCRIPT.select_records(rows, _SCRIPT.ALL_KNOWN_CATEGORIES)) == everything
    assert dict(_SCRIPT.select_records(rows, ["assembly"], objects=["pallet"])) == assemblies
    with pytest.raises(SystemExit, match="--objects matched none"):
        _SCRIPT.select_records(rows, ["assembly"], objects=["no_such_prop"])


def test_hub_fetch_plan_groups_rows_by_prop_directory(tmp_path):
    """The no-`--source` download plan is per prop subtree, not the dataset."""
    rows = _SCRIPT.read_csv_catalog(_make_csv(tmp_path, [
        dict(asset_name="Cardbox_C1_physics",
             relative_path="Props/general/Cardbox_C1/Cardbox_C1_physics.usd",
             classification="Prop general", label="Cardboard Box"),
        dict(asset_name="Cardbox_C1",
             relative_path="Props/general/Cardbox_C1/Cardbox_C1.usd",
             classification="Prop general", label="Cardboard Box"),
        dict(asset_name="Pallets_A1_physics",
             relative_path="Props/assembly/Pallets_A1/Pallets_A1_physics.usd",
             classification="Assembly", label="pallet"),
        dict(asset_name="physical_ai_simready_warehouse_01",
             relative_path="physical_ai_simready_warehouse_01.usd",
             classification="Scenario", label="warehouse"),
    ]))
    plan = _SCRIPT._hub_fetch_plan([row["relative_path"] for row in rows])
    assert plan == {
        "Props/general/Cardbox_C1": ("Props/general/Cardbox_C1/**",),
        "Props/assembly/Pallets_A1": ("Props/assembly/Pallets_A1/**",),
    }


def test_fetch_dataset_downloads_the_assembly_payload_closure(tmp_path, monkeypatch):
    """No `--source`: an assembly's payload props are fetched, not left dangling.

    ``fetch_dataset`` never touches the network here — a fake
    ``snapshot_download`` materialises exactly the subtree each plan names
    (the Hub snapshot semantics), so the closure round is observable.
    """
    output_dir = tmp_path / "hub"
    calls = []

    def _fake_snapshot_download(repo_id, *, repo_type, allow_patterns, local_dir):
        calls.append(sorted(allow_patterns))
        for pattern in allow_patterns:
            subtree = pattern.removesuffix("/**")
            target = Path(local_dir) / subtree
            target.mkdir(parents=True, exist_ok=True)
            if subtree == "Props/assembly/Pallets_A1":
                (target / "Pallets_A1_physics.usd").write_text(
                    "#usda 1.0\n"
                    "(\n"
                    '    defaultPrim = "World"\n'
                    ")\n\n"
                    'def Xform "World"\n'
                    "{\n"
                    '    def Xform "Pallet_C1_01"\n'
                    "    (\n"
                    "        prepend payload = "
                    "@../../general/Pallet_C1/Pallet_C1_physics.usd@\n"
                    "    )\n"
                    "    {\n"
                    "    }\n"
                    "}\n"
                )
            else:
                (target / f"{Path(subtree).name}_physics.usd").write_text("#usda 1.0\n")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)
    source = _SCRIPT.fetch_dataset(
        output_dir,
        relative_paths=["Props/assembly/Pallets_A1/Pallets_A1_physics.usd"],
    )
    assert source == output_dir
    # Two rounds: the assembly subtree (+ the shared physics-materials
    # subtree added up front), then the payload prop it references.
    assert calls == [
        ["Props/assembly/Pallets_A1/**", "Props/materials/physics/**"],
        ["Props/general/Pallet_C1/**"],
    ]
    assert (output_dir / "Props/general/Pallet_C1/Pallet_C1_physics.usd").is_file()


def test_repackage_authors_rigid_body_and_normalises_units(tmp_path):
    source = tmp_path / "snapshot"
    _make_source_props(source, [("general", "Cardbox_C1")])
    csv_file = _make_csv(tmp_path, [
        dict(asset_name="Cardbox_C1_physics",
             relative_path="Props/general/Cardbox_C1/Cardbox_C1_physics.usd",
             classification="Prop general", label="Cardboard Box", mass="1.0"),
    ])

    out_root = tmp_path / "out" / "objects" / "simready_warehouse"
    _SCRIPT.repackage_object(
        source, ["Props/general/Cardbox_C1/Cardbox_C1_physics.usd"],
        "prop_cardbox_c1", out_root,
    )

    packaged = out_root / "prop_cardbox_c1"
    names = sorted(str(p.relative_to(packaged)) for p in packaged.rglob("*") if p.is_file())
    # `.thumbs` previews and the upstream *.mdl recipes are dropped; the
    # physics overlay is renamed to <object_id>.usd and sublayers the visual.
    assert names == [
        "Cardbox_C1.usd",
        "materials/textures/T_Cardbox_Albedo.png",
        "materials/textures/T_Cardbox_Normal.png",
        "materials/textures/T_Cardbox_ORM.png",
        "prop_cardbox_c1.usd",
    ]
    assert not list(packaged.glob("materials/*.mdl"))
    assert not list(packaged.glob(".thumbs/*"))

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(packaged / "prop_cardbox_c1.usd"))
    assert stage.GetPrimAtPath("/RootNode").HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0, abs=1e-6)
    assert count_mdl_materials(str(packaged / "prop_cardbox_c1.usd")) == 0
    # The packaged visual sidecar was converted too.
    assert count_mdl_materials(str(packaged / "Cardbox_C1.usd")) == 0


def test_annotate_object_from_csv_record(tmp_path):
    record = dict(classification="Prop general hand manipulation", label="book",
                  q_code="Q571", mass="0.4", asset_name="sm_book_open_a01_simready_01")
    category, entry = _SCRIPT.annotate_object(record)
    assert category == "hand"
    assert entry["language_annotations"][0] == "warehouse hand"
    assert "a book" in entry["language_annotations"]
    assert {"hand", "rigid", "warehouse", "book"}.issubset(set(entry["tags"]))


def test_write_catalog_merges_with_existing(tmp_path):
    out_root = tmp_path / "objects" / "simready_warehouse"
    out_root.mkdir(parents=True)
    entry_a = _SCRIPT.annotate_object(
        dict(classification="Prop general", label="Cardboard Box", mass="1.0"),
    )[1]
    _SCRIPT.write_catalog(out_root, [dict(object_id="prop_cardbox_c1",
                                          catalog_entry=entry_a)])
    entry_b = _SCRIPT.annotate_object(
        dict(classification="Prop general", label="Cart", mass="12.5"),
    )[1]
    _SCRIPT.write_catalog(out_root, [dict(object_id="prop_trolley",
                                          catalog_entry=entry_b)])
    written = json.loads((out_root / "catalog.json").read_text())
    assert set(written) == {"prop_cardbox_c1", "prop_trolley"}


def test_unknown_categories_cli_refusal(tmp_path):
    with pytest.raises(SystemExit, match="Unknown SimReady Warehouse"):
        _SCRIPT.main([
            "--output-dir", str(tmp_path / "out"),
            "--categories", "scenario",
        ])


def test_license_and_readme_ship_attribution():
    assert "CC-BY-4.0" in _SCRIPT.LICENSE_TEXT
    assert "commits **no LICENSE file**" in _SCRIPT.LICENSE_TEXT
    assert "RigidBodyAPI" in _SCRIPT.PROVENANCE_README
    assert "no Rigid Bodies" in _SCRIPT.PROVENANCE_README


def test_repackage_strips_unresolvable_omniverse_physics_reference(tmp_path):
    """A hand prop whose simready stage references a physics layer by an
    ``omniverse://`` URL packages without dangling references.

    The NVIDIA warehouse authors the physics material (friction/restitution)
    as a prim carrying an internal reference to an absolute Nucleus URL that
    can never resolve off Omniverse; the pipeline re-authors its own rigid
    body, so the prim is stripped and the verifier sees a complete tree.
    """
    from pxr import Usd

    source = tmp_path / "snapshot"
    asset_dir = source / "Props" / "general" / "HandManipulation" / "book_open_a"
    asset_dir.mkdir(parents=True)
    asset_dir.joinpath("sm_book_open_a01_simready_01.usd").write_text(
        "#usda 1.0\n"
        "(\n"
        "    metersPerUnit = 0.01\n"
        '    defaultPrim = "RootNode"\n'
        ")\n\n"
        'def Xform "RootNode"\n'
        "{\n"
        '    def "Materials"\n'
        "    {\n"
        '        def "physics_cardboard" (\n'
        "            prepend references = @omniverse://simready.ov.nvidia.com/"
        "NVIDIA/Assets/simready_content/materials/physics/physics_cardboard.usda@\n"
        "        )\n"
        "        {\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    out_root = tmp_path / "out" / "objects" / "simready_warehouse"
    _SCRIPT.repackage_object(
        source,
        ["Props/general/HandManipulation/book_open_a/sm_book_open_a01_simready_01.usd"],
        "hand_sm_book_open_a01_simready_01", out_root,
    )
    packaged = out_root / "hand_sm_book_open_a01_simready_01"
    stage_file = packaged / "hand_sm_book_open_a01_simready_01.usd"
    assert stage_file.is_file()
    stage = Usd.Stage.Open(str(stage_file))
    # The physics_material prim (and its omniverse:// reference) is gone.
    assert not stage.GetPrimAtPath("/RootNode/Materials/physics_cardboard")
    assert stage.GetPrimAtPath("/RootNode").IsValid()
    # No dangling references remain in the packaged tree.
    from unienv_genesis_collection.objects.hub_objects.articulated.hub_articulated_object import (
        _usd_required_files,
    )
    assert not [p for p in _usd_required_files(stage_file, packaged) if not p.is_file()]


# ------------------------------------------------------------------
# GPU smoke test (skipped offline)
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_into_scene_smoke_gpu(assets_dir, fake_cardbox, no_download):
    """Real GPU smoke: a rigid prop loads as one entity (no joints)."""
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)

    item = SimReadyWarehouseProp("prop_cardbox_c1")
    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(type("WorldStub", (), {"scene": scene})())
        scene.build(n_envs=0)
        assert entity is not None

        # Rigid: no moving joints, one body, vgeoms with the bound texture.
        assert not [j for j in entity.joints if int(j.n_dofs) > 0]
        vaabb = entity.get_vAABB()
        local = item.get_local_bounding_box()
        assert torch.allclose(vaabb[0], local.low.to(vaabb.device), atol=2e-3)
        assert torch.allclose(vaabb[1], local.high.to(vaabb.device), atol=2e-3)

        # fixed=False loads the same prop as a dynamic rigid body.
        dynamic_scene = gs.Scene(show_viewer=False,
                                 renderer=gs.options.renderers.Rasterizer())
        try:
            dynamic = SimReadyWarehouseProp("prop_cardbox_c1", fixed=False)
            dynamic_entity = dynamic.load_into_scene(
                type("WorldStub", (), {"scene": dynamic_scene})()
            )
            dynamic_scene.build(n_envs=0)
            assert dynamic_entity is not None
            assert len(dynamic_entity.links) == 1
        finally:
            dynamic_scene.destroy()
    finally:
        scene.destroy()
        torch.set_default_device("cpu")


def test_repackage_assembly_copies_payload_closure(tmp_path):
    """Assemblies keep the upstream subtree layout + their payload props.

    The packaged root is a thin sublayer of the preserved assembly stage;
    the payload'd prop directories ride along so the ``../../general/…``
    arcs resolve inside the packaged tree.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "snapshot"
    _make_source_props(source, [("general", "Cardbox_C1")])
    # An assembly whose payloads point at the (present) prop — the upstream
    # convention: two payload arcs under a World root.
    assembly_dir = source / "Props" / "assembly" / "RackLarge_A1"
    assembly_dir.mkdir(parents=True)
    assembly_dir.joinpath("RackLarge_A1_physics.usd").write_text(
        "#usda 1.0\n"
        "(\n"
        "    metersPerUnit = 0.01\n"
        '    defaultPrim = "World"\n'
        '    upAxis = "Z"\n'
        ")\n\n"
        'def Xform "World"\n'
        "{\n"
        '    def Xform "Cardbox_A"\n'
        "    (\n"
        "        prepend payload = @../../general/Cardbox_C1/Cardbox_C1_physics.usd@\n"
        "    )\n"
        "    {\n"
        "    }\n"
        "}\n"
    )

    out_root = tmp_path / "out" / "objects" / "simready_warehouse"
    _SCRIPT.repackage_object(
        source, ["Props/assembly/RackLarge_A1/RackLarge_A1_physics.usd"],
        "prop_racklarge_a1", out_root,
    )
    packaged = out_root / "prop_racklarge_a1"
    names = sorted(str(p.relative_to(packaged)) for p in packaged.rglob("*") if p.is_file())
    assert names == [
        "Props/assembly/RackLarge_A1/RackLarge_A1_physics.usd",
        "Props/general/Cardbox_C1/Cardbox_C1.usd",
        "Props/general/Cardbox_C1/Cardbox_C1_physics.usd",
        "Props/general/Cardbox_C1/materials/textures/T_Cardbox_Albedo.png",
        "Props/general/Cardbox_C1/materials/textures/T_Cardbox_Normal.png",
        "Props/general/Cardbox_C1/materials/textures/T_Cardbox_ORM.png",
        "prop_racklarge_a1.usd",
    ]
    # The same pruning as the non-assembly path: the copied closure carries
    # no `.thumbs` previews and no upstream `*.mdl` recipes.
    assert not list(packaged.rglob(".thumbs"))
    assert not list(packaged.rglob("*.mdl"))
    # The composed stage resolves through the payload closure and collapses
    # to one rigid body with metersPerUnit = 1.
    stage = Usd.Stage.Open(str(packaged / "prop_racklarge_a1.usd"))
    assert stage.GetDefaultPrim().IsValid()
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0, abs=1e-6)
    assert count_mdl_materials(str(packaged / "prop_racklarge_a1.usd")) == 0


# ------------------------------------------------------------------
# Recoverable-drop fixes: UDIM textures + degenerate 3-point meshes
# ------------------------------------------------------------------

from unienv_genesis_collection.utils.usd_conversions import _relocate_texture  # noqa: E402

_converters = importlib.import_module("unienv_genesis_collection.utils.usd_conversions")


def _first_udim_stage_file(path: Path, *, meters_per_unit: float = 1.0) -> Path:
    """A Warehouse-shaped MDL stage whose textures are ``<UDIM>`` templates."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", float(meters_per_unit))
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("defaultPrim", "RootNode")

    UsdGeom.Xform.Define(stage, "/RootNode")
    UsdGeom.Scope.Define(stage, "/RootNode/Geometry")
    UsdGeom.Scope.Define(stage, "/RootNode/Materials")
    material = UsdShade.Material.Define(stage, "/RootNode/Materials/Rack")
    shader = UsdShade.Shader.Define(stage, "/RootNode/Materials/Rack/Shader")
    shader.CreateImplementationSourceAttr("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./materials/Rack.mdl"))
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
    ).Set("OmniPBR")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./materials/textures/T_Rack_A1_Albedo.<UDIM>.png")
    )
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./materials/textures/T_Rack_A1_ORM.<UDIM>.png")
    )
    _write_box_mesh(stage, "/RootNode/Geometry/Rack_C1", (3.0, 4.0, 0.3))
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/RootNode/Geometry/Rack_C1"))
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/RootNode/Geometry/Rack_C1")).Bind(material)
    stage.GetRootLayer().Save()

    physics = path.parent / f"{path.stem}_physics.usd"
    physics.write_text(
        "#usda 1.0\n"
        "(\n"
        f"    metersPerUnit = {float(meters_per_unit)}\n"
        "    upAxis = \"Z\"\n"
        f"    subLayers = [@./{path.name}@]\n"
        ")\n"
    )
    return physics


def test_relocate_rewrites_udim_template_to_first_tile(tmp_path):
    """``….<UDIM>.png`` inputs point at the concrete first existing tile."""
    root_dir = tmp_path / "prop_rack_a1"
    textures = root_dir / "materials" / "textures"
    textures.mkdir(parents=True)
    for tile in (1001, 1002, 1003):
        (textures / f"T_Rack_A1_Albedo.{tile}.png").write_bytes(b"png")
        (textures / f"T_Rack_A1_ORM.{tile}.png").write_bytes(b"png")

    relocated = _relocate_texture(
        root_dir, "./materials/textures/T_Rack_A1_Albedo.<UDIM>.png", root_dir,
    )
    assert relocated == "./materials/textures/T_Rack_A1_Albedo.1001.png"

    # A tile-set starting past 1001 still resolves (numerically first tile).
    for name in textures.iterdir():
        name.unlink()
    (textures / "T_Rack_A1_ORM.1005.png").write_bytes(b"png")
    assert _relocate_texture(
        root_dir, "./materials/textures/T_Rack_A1_ORM.<UDIM>.png", root_dir,
    ) == "./materials/textures/T_Rack_A1_ORM.1005.png"

    # No concrete tile: the template passes through (the required-file check
    # flags it) instead of rewriting to a file that does not exist.
    assert _relocate_texture(
        root_dir, "./materials/textures/T_Rack_A1_Albedo.<UDIM>.png", root_dir,
    ) == "./materials/textures/T_Rack_A1_Albedo.<UDIM>.png"


def test_repackage_rewrites_udim_textures(tmp_path):
    """The packaged stage's texture inputs name concrete tiles (load-correct).

    usd-core does not expand ``<UDIM>`` — Genesis's ``UsdUVTexture`` reader
    raises on the template, so the inputs must be re-authored to a real tile.
    Documented caveat: the props' UVs may span several tiles, so the
    single-tile pick is load-correct but texture-lossy.
    """
    from pxr import Sdf, Usd, UsdShade

    source = tmp_path / "snapshot"
    asset_dir = source / "Props" / "general" / "RackLarge_A1"
    asset_dir.mkdir(parents=True)
    _first_udim_stage_file(asset_dir / "RackLarge_A1.usd", meters_per_unit=0.01)
    textures = asset_dir / "materials" / "textures"
    textures.mkdir(parents=True)
    for tile in (1001, 1002):
        for kind in ("Albedo", "ORM"):
            (textures / f"T_Rack_A1_{kind}.{tile}.png").write_bytes(b"png")

    out_root = tmp_path / "out" / "objects" / "simready_warehouse"
    _SCRIPT.repackage_object(
        source,
        ["Props/general/RackLarge_A1/RackLarge_A1_physics.usd"],
        "prop_racklarge_a1", out_root,
    )

    packaged_stage = out_root / "prop_racklarge_a1" / "prop_racklarge_a1.usd"
    stage = Usd.Stage.Open(str(packaged_stage))
    files = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        file_input = UsdShade.Shader(prim).GetInput("file")
        if file_input is not None and file_input.Get() is not None:
            files.append(file_input.Get().path)
    assert files, "the converted material lost its texture inputs"
    assert files == ["./materials/textures/T_Rack_A1_Albedo.1001.png",
                     "./materials/textures/T_Rack_A1_ORM.1001.png",
                     "./materials/textures/T_Rack_A1_ORM.1001.png"]
    assert not any("<UDIM>" in path for path in files)


def test_hide_degenerate_mesh_prims_hides_sub_4_point_meshes(tmp_path):
    """Sub-4-point meshes are made invisible; real meshes stay visible."""
    from pxr import Gf, Usd, UsdGeom

    stage_file = tmp_path / "prop_handtruck_a01.usd"
    stage = Usd.Stage.CreateNew(str(stage_file))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.Xform.Define(stage, "/RootNode")
    UsdGeom.Scope.Define(stage, "/RootNode/Geometry")
    # The 3-point sticker-plane mesh the dataset ships (qhull needs 4).
    degenerate = UsdGeom.Mesh.Define(stage, "/RootNode/Geometry/Plastic_Rough_Black_A")
    degenerate.CreatePointsAttr([
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0),
    ])
    degenerate.CreateFaceVertexCountsAttr([3])
    degenerate.CreateFaceVertexIndicesAttr([0, 1, 2])
    _write_box_mesh(stage, "/RootNode/Geometry/Frame", (0.3, 0.2, 0.4))
    stage.GetRootLayer().Save()

    assert _converters.hide_degenerate_mesh_prims(str(stage_file)) == 1

    reopened = Usd.Stage.Open(str(stage_file))
    hidden = UsdGeom.Imageable(reopened.GetPrimAtPath("/RootNode/Geometry/Plastic_Rough_Black_A"))
    assert hidden.ComputeVisibility() == "invisible"
    frame = UsdGeom.Imageable(reopened.GetPrimAtPath("/RootNode/Geometry/Frame"))
    assert frame.ComputeVisibility() != "invisible"
    # Idempotent: nothing left to hide.
    assert _converters.hide_degenerate_mesh_prims(str(stage_file)) == 0


def test_repackage_hides_degenerate_mesh_prims(tmp_path):
    """The non-assembly packaging path hides the degenerate mesh prims."""
    from pxr import Usd, UsdGeom

    source = tmp_path / "snapshot"
    asset_dir = source / "Props" / "general" / "HandTruck_A01"
    asset_dir.mkdir(parents=True)
    visual = asset_dir / "HandTruck_A01.usd"
    stage = Usd.Stage.CreateNew(str(visual))
    stage.SetMetadata("metersPerUnit", 0.01)
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("defaultPrim", "RootNode")
    UsdGeom.Xform.Define(stage, "/RootNode")
    UsdGeom.Scope.Define(stage, "/RootNode/Geometry")
    degenerate = UsdGeom.Mesh.Define(stage, "/RootNode/Geometry/Plastic_Rough_Black_A")
    from pxr import Gf

    degenerate.CreatePointsAttr([
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0),
    ])
    degenerate.CreateFaceVertexCountsAttr([3])
    degenerate.CreateFaceVertexIndicesAttr([0, 1, 2])
    _write_box_mesh(stage, "/RootNode/Geometry/Frame", (30.0, 20.0, 40.0))
    stage.GetRootLayer().Save()

    asset_dir.joinpath("HandTruck_A01_physics.usd").write_text(
        "#usda 1.0\n"
        "(\n"
        "    metersPerUnit = 0.01\n"
        "    upAxis = \"Z\"\n"
        "    subLayers = [@./HandTruck_A01.usd@]\n"
        ")\n"
    )

    out_root = tmp_path / "out" / "objects" / "simready_warehouse"
    _SCRIPT.repackage_object(
        source, ["Props/general/HandTruck_A01/HandTruck_A01_physics.usd"],
        "prop_sm_handtruck_a01", out_root,
    )

    packaged_stage = out_root / "prop_sm_handtruck_a01" / "prop_sm_handtruck_a01.usd"
    packaged = Usd.Stage.Open(str(packaged_stage))
    hidden = UsdGeom.Imageable(
        packaged.GetPrimAtPath("/RootNode/Geometry/Plastic_Rough_Black_A"),
    )
    assert hidden.ComputeVisibility() == "invisible"
    frame = UsdGeom.Imageable(packaged.GetPrimAtPath("/RootNode/Geometry/Frame"))
    assert frame.ComputeVisibility() != "invisible"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_degenerate_stage_smoke_gpu(tmp_path, monkeypatch):
    """GPU regression: a stage with a 3-point mesh loads (was QhullError).

    The loader's name-pattern fallback makes a non-``Visual``/``Collision``
    prim name BOTH visual and collision, so the 3-point mesh reaches qhull
    (``QhullError: QH6214``) unless it is hidden first — the packaging fix.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    obj_dir = tmp_path / "objects" / "simready_warehouse" / "prop_sm_handtruck_a01"
    obj_dir.mkdir(parents=True)
    visual = obj_dir / "HandTruck_A01.usd"
    stage = Usd.Stage.CreateNew(str(visual))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("defaultPrim", "RootNode")
    UsdGeom.Xform.Define(stage, "/RootNode")
    UsdGeom.Scope.Define(stage, "/RootNode/Geometry")
    degenerate = UsdGeom.Mesh.Define(stage, "/RootNode/Geometry/Plastic_Rough_Black_A")
    degenerate.CreatePointsAttr([
        Gf.Vec3f(-0.0005, 0, 0), Gf.Vec3f(0.0005, 0, 0), Gf.Vec3f(0, 0.0005, 0),
    ])
    degenerate.CreateFaceVertexCountsAttr([3])
    degenerate.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(degenerate.GetPrim())
    _write_box_mesh(stage, "/RootNode/Geometry/Frame", (0.3, 0.2, 0.4))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/RootNode/Geometry/Frame"))
    stage.GetRootLayer().Save()

    packaged_stage = obj_dir / "prop_sm_handtruck_a01.usd"
    packaged_stage.write_text(
        "#usda 1.0\n"
        "(\n"
        "    metersPerUnit = 1\n"
        "    upAxis = \"Z\"\n"
        "    subLayers = [@./HandTruck_A01.usd@]\n"
        ")\n"
    )
    assert _converters.hide_degenerate_mesh_prims(str(packaged_stage)) == 1
    object_stage = Usd.Stage.Open(str(packaged_stage))
    _SCRIPT._apply_root_rigid_body(object_stage)
    _SCRIPT._normalise_units(object_stage)
    object_stage.GetRootLayer().Save()

    _make_catalog(tmp_path, {
        "prop_sm_handtruck_a01": _catalog_entry(
            "HandTruck_A01", label="convertible steel hand truck",
        ),
    })
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()

    import genesis as gs
    from genesis.constants import backend as gs_backend

    if getattr(gs, "_initialized", False):
        gs.destroy()
    gs.init(backend=gs_backend.cuda)
    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        item = SimReadyWarehouseProp("prop_sm_handtruck_a01")
        entity = item.load_into_scene(type("WorldStub", (), {"scene": scene})())
        scene.build(n_envs=0)
        assert entity is not None
        assert len(entity.links) == 1
        vaabb = entity.get_vAABB()
        local = item.get_local_bounding_box()
        assert torch.allclose(vaabb[0], local.low.to(vaabb.device), atol=2e-3)
        assert torch.allclose(vaabb[1], local.high.to(vaabb.device), atol=2e-3)
    finally:
        scene.destroy()
        torch.set_default_device("cpu")
