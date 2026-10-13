"""Tests for the Lightwheel USD furniture integration (no asset downloads).

Mirrors ``tests/test_artvip_furniture.py`` / ``tests/test_molmospaces_furniture.py``
with **synthetic USD stages** in ``tmp_path`` — plus the family's critical
converter tests (the OmniPBR MDL → UsdPreviewSurface rewrite on a
synthetic MDL material):

  - MDL conversion: diffuse/ORM (roughness g, metallic b)/normal maps
    re-wired into a ``UsdPreviewSurface`` with ``outputs:surface`` bound,
    the MDL shader prim removed (together with the ``outputs:mdl:*``
    connections pointing at it), no-op on already-preview materials,
    count_mdl_materials before/after.
  - No joint-body swap needed (Lightwheel authors body0 = parent): the
    synthetic stage is Genesis-ordered and loads as one entity offline.
  - Catalog integrity + tag/contains filtering (entries generated through
    the prepare script's ``annotate_object`` template path).
  - LightwheelFurniture construction (``lightwheel_``-prefixed ids, scalar
    scale) and asset completeness (partial-download repair).
  - Offline FK + clean names (shared ArtVIP machinery).
  - scripts/prepare_lightwheel_assets.py: asset listing, stage rename +
    texture sidecar copy, conversion invocation, catalog generation,
    license text.
  - RoomBuilder._resolve_descriptor ``source="lightwheel"`` branch.

A GPU smoke test (skipped unless CUDA) loads the synthetic stage through
``load_into_scene`` — the packaging gate's core assertion.
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
    usd_clean_names,
    usd_link_meshes_at_config,
)
from unienv_genesis_collection.objects.hub_objects.articulated.lightwheel_furniture import (
    LIGHTWHEEL_ASSETS_ROOT,
    LightwheelFurniture,
    get_lightwheel_catalog,
    list_lightwheel_objects,
    make_lightwheel_pool,
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
        "prepare_lightwheel_assets", _SCRIPTS_DIR / "prepare_lightwheel_assets.py",
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


def _write_box_mesh(stage, path: str, extents, translation=None):
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
    if translation is not None:
        UsdGeom.Xform(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(*translation))
    return mesh.GetPrim()


def _write_body(stage, path: str, *, body_extents, body_translation=None, rigid=True,
                material_path=None):
    """A RigidBodyAPI Xform link holding one box mesh at its own origin."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    prim = UsdGeom.Xform.Define(stage, path)
    if body_translation is not None:
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*body_translation))
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
    mesh = _write_box_mesh(stage, f"{path}/box", body_extents)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    if material_path is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh)
        UsdShade.MaterialBindingAPI(mesh).Bind(
            UsdShade.Material(stage.GetPrimAtPath(str(material_path)))
        )
    return prim.GetPrim()


def _write_lightwheel_stage(path: Path, *, mdl: bool = False):
    """A Lightwheel-shaped stage: welded base + revolute door + texture.

    Lightwheel authors ``body0`` = parent already (no swap needed).  With
    ``mdl=True`` the material is an OmniPBR MDL graph (``info:mdl:sourceAsset
    = @OmniPBR.mdl@``, ``inputs:diffuse_texture``/``inputs:ORM_texture``/
    ``inputs:normalmap_texture`` bound to a relative ``texture/`` file) —
    the shape the packaging conversion must rewrite.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    root_scope = UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(root_scope.GetPrim())

    if mdl:
        material = UsdShade.Material.Define(stage, "/root/Looks/material_Body001_material")
        shader = UsdShade.Shader.Define(stage, "/root/Looks/material_Body001_material/Shader")
        shader.CreateImplementationSourceAttr("sourceAsset")
        shader_prim = shader.GetPrim()
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_prim = shader.GetPrim()
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
        ).Set("OmniPBR")
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./texture/T_Body001_BC001.png")
        )
        shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./texture/T_Body001_ORM001.png")
        )
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./texture/T_Body001_N001.png")
        )
    else:
        # The post-conversion shape: UsdPreviewSurface + UsdUVTexture whose
        # file input is the relative ./texture/*.png asset (the required-file
        # closure must name it even when it is missing).
        material = UsdShade.Material.Define(stage, "/root/Looks/material_Body001_material")
        surface = UsdShade.Shader.Define(
            stage, "/root/Looks/material_Body001_material/UsdPreviewSurface"
        )
        surface.CreateIdAttr("UsdPreviewSurface")
        texture = UsdShade.Shader.Define(
            stage, "/root/Looks/material_Body001_material/Tex_diffuse"
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./texture/T_Body001_BC001.png")
        )
        reader = UsdShade.Shader.Define(
            stage, "/root/Looks/material_Body001_material/Tex_diffuse/stReader"
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

    base_material_path = material.GetPath()
    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 0.4),
                material_path=base_material_path)
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 0.4),
                body_translation=(0.2, 0.0, 0.0), material_path=base_material_path)
    door_joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/microjoint")
    door_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    door_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
    door_joint.CreateLocalPos0Attr(Gf.Vec3f(0.2, 0.0, 0.0))
    door_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    door_joint.CreateAxisAttr("X")
    door_joint.CreateLowerLimitAttr(-90.0)
    door_joint.CreateUpperLimitAttr(0.0)

    stage.GetRootLayer().Save()
    return stage


def _make_fake_object_dir(assets_dir: Path, object_id: str, *, mdl: bool = False) -> Path:
    """A complete fake Lightwheel object directory (synthetic stage).

    The texture sidecars are **valid** (2×2 PNGs): the Genesis material
    parser opens the referenced files, so placeholders would raise
    ``UnidentifiedImageError`` in the GPU smoke.
    """
    obj_dir = assets_dir / LIGHTWHEEL_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    _write_lightwheel_stage(obj_dir / f"{object_id}.usd", mdl=mdl)
    textures = obj_dir / "texture"
    textures.mkdir(parents=True, exist_ok=True)
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (200, 100, 50)).save(buffer, format="PNG")
    for name in ("T_Body001_BC001.png", "T_Body001_ORM001.png", "T_Body001_N001.png"):
        (textures / name).write_bytes(buffer.getvalue())
    return obj_dir


def _catalog_entry(asset_name: Path, *, n_revolute: int = 1, n_prismatic: int = 0):
    """A catalog entry generated through the packaging templates."""
    entry = _SCRIPT.annotate_object(asset_name.name, n_revolute, n_prismatic)[1]
    entry["tags"] = sorted(set(entry["tags"]) | {"articulated"})
    return entry


def _make_catalog(assets_dir: Path, entries: dict) -> None:
    catalog_file = assets_dir / LIGHTWHEEL_ASSETS_ROOT / "catalog.json"
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    catalog_file.write_text(json.dumps(entries))


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir with a generated (template-path) catalog."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    upstream = tmp_path / "upstream" / "Microwave043"
    upstream.mkdir(parents=True)
    _make_catalog(assets_dir=tmp_path, entries={
        "lightwheel_Microwave043": _catalog_entry(upstream),
        "lightwheel_Stovetop002": _catalog_entry(
            tmp_path / "upstream" / "Stovetop002", n_revolute=4, n_prismatic=0,
        ),
        "lightwheel_Toaster003": _catalog_entry(
            tmp_path / "upstream" / "Toaster003", n_revolute=1, n_prismatic=2,
        ),
    })
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()
    return tmp_path


@pytest.fixture
def fake_microwave(assets_dir):
    """A complete fake 'lightwheel_Microwave043' object dir (synthetic .usd)."""
    return _make_fake_object_dir(assets_dir, "lightwheel_Microwave043")


@pytest.fixture
def no_download(monkeypatch):
    calls = []

    def _fail(**kwargs):
        calls.append(kwargs)
        raise AssertionError(f"Unexpected download attempt: {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail)
    return calls


# ------------------------------------------------------------------
# The MDL → UsdPreviewSurface converter
# ------------------------------------------------------------------

def _first_surface_shader(stage, material_prim_path: str):
    from pxr import UsdShade

    material = UsdShade.Material(stage.GetPrimAtPath(material_prim_path))
    node, _output_name, _context = material.GetOutput("surface").GetConnectedSource()
    return UsdShade.Shader(node.GetPrim())


def test_convert_maps_diffuse_orm_and_normal(tmp_path):
    """diffuse_texture→diffuseColor, ORM g/b→roughness/metallic, normal→normal."""
    from pxr import Sdf, Usd, UsdShade

    path = tmp_path / "omnipbr.usda"
    stage = Usd.Stage.CreateNew(str(path))
    material = UsdShade.Material.Define(stage, "/Looks/material_Mat")
    shader = UsdShade.Shader.Define(stage, "/Looks/material_Mat/Shader")
    shader.CreateImplementationSourceAttr("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
    ).Set("OmniPBR")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./texture/T_Microwave043_BC001.png"))
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./texture/T_Microwave043_ORM001.png"))
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./texture/T_Microwave043_N001.png"))
    stage.GetRootLayer().Save()

    assert count_mdl_materials(path) == 1
    converted = convert_mdl_to_preview_surface(path)
    assert converted == 1
    assert count_mdl_materials(path) == 0  # the MDL shader prim is gone

    reopened = Usd.Stage.Open(str(path))
    surface = _first_surface_shader(reopened, "/Looks/material_Mat")
    assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
    inputs = {shader_input.GetBaseName(): shader_input for shader_input in surface.GetInputs()}
    assert inputs_connected(inputs, "diffuseColor")
    assert inputs_connected(inputs, "roughness")
    assert inputs_connected(inputs, "metallic")
    assert inputs_connected(inputs, "normal")
    # The MDL shader prim itself was removed (no dangling @…mdl@ reference).
    mdl_prim = reopened.GetPrimAtPath("/Looks/material_Mat/Shader")
    assert not mdl_prim.IsValid()


def inputs_connected(inputs: dict, name: str) -> bool:
    return name in inputs and inputs[name].HasConnectedSource()


def test_convert_binds_surface_output_and_clears_mdl_outputs(tmp_path):
    """``outputs:surface`` is bound and the dangling ``outputs:mdl:*`` are gone."""
    from pxr import Gf, Sdf, Usd, UsdShade

    path = tmp_path / "stage.usda"
    _write_lightwheel_stage(path, mdl=True)
    assert count_mdl_materials(path) == 1
    assert convert_mdl_to_preview_surface(path) == 1

    reopened = Usd.Stage.Open(str(path))
    material = UsdShade.Material(reopened.GetPrimAtPath(
        "/root/Looks/material_Body001_material"))
    outputs = {output.GetName() for output in reopened.GetPrimAtPath(
        "/root/Looks/material_Body001_material").GetAttributes()
        if output.GetName().startswith("outputs:mdl")}
    assert outputs == set()  # no dangling mdl outputs
    surface = _first_surface_shader(reopened, "/root/Looks/material_Body001_material")
    assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
    # The converted graph's texture slots all carry image connections and the
    # shared 'st' primvar reader (the Genesis-native shape).
    texture_slots = {
        shader_input.GetBaseName(): shader_input for shader_input in surface.GetInputs()
    }
    assert inputs_connected(texture_slots, "diffuseColor")
    assert inputs_connected(texture_slots, "roughness")
    assert inputs_connected(texture_slots, "metallic")
    assert inputs_connected(texture_slots, "normal")


def material_dict_has_textures(material_dict: dict) -> bool:
    return (
        material_dict.get("color_texture") is not None
        and material_dict.get("roughness_texture") is not None
        and material_dict.get("metallic_texture") is not None
    )


def test_convert_skips_non_mdl_materials(tmp_path, fake_microwave):
    """Materials already carrying a preview surface are left untouched."""
    from pxr import Usd

    path = fake_microwave / "lightwheel_Microwave043.usd"
    assert count_mdl_materials(path) == 0
    before_stage = Usd.Stage.Open(str(path))  # keep the stage alive below
    before_id = _first_surface_shader(
        before_stage, "/root/Looks/material_Body001_material",
    ).GetIdAttr().Get()
    assert convert_mdl_to_preview_surface(path) == 0
    # The stage still opens and the preview surface shader is unchanged
    # (the export round trip rewrites binary metadata, not the graph).
    reopened = Usd.Stage.Open(str(path))
    surface_after = _first_surface_shader(reopened,
                                          "/root/Looks/material_Body001_material")
    assert surface_after.GetIdAttr().Get() == before_id == "UsdPreviewSurface"
    assert count_mdl_materials(path) == 0


def test_mdl_constant_fallbacks(tmp_path):
    """diffuse_color_constant / opacity_constant feed the PreviewSurface factors."""
    from pxr import Gf, Sdf, Usd, UsdShade

    path = tmp_path / "constants.usda"
    stage = Usd.Stage.CreateNew(str(path))
    material = UsdShade.Material.Define(stage, "/Looks/material_Mat")
    shader = UsdShade.Shader.Define(stage, "/Looks/material_Mat/Shader")
    shader.CreateImplementationSourceAttr("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.4, 0.6))
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(0.4)
    stage.GetRootLayer().Save()

    assert convert_mdl_to_preview_surface(path) == 1

    reopened = Usd.Stage.Open(str(path))
    surface = _first_surface_shader(reopened, "/Looks/material_Mat")
    diffuse = surface.GetInput("diffuseColor")
    assert diffuse.Get() == Gf.Vec3f(0.2, 0.4, 0.6)
    assert not diffuse.HasConnectedSource()  # a constant, not a texture
    opacity = surface.GetInput("opacity")
    assert opacity is not None
    assert opacity.Get() == pytest.approx(0.4, abs=1e-6)


# ------------------------------------------------------------------
# Catalog
# ------------------------------------------------------------------

def test_catalog_integrity(assets_dir):
    catalog = get_lightwheel_catalog()
    assert len(catalog) == 3
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert object_id.startswith("lightwheel_")  # the attribution format
        assert len(info.language_annotations) >= 2
        assert {"articulated"} <= info.tags
        assert info.standing_quaternion is None


def test_catalog_is_cached(assets_dir, no_download):
    assert get_lightwheel_catalog() is get_lightwheel_catalog()


def test_list_all_objects_sorted(assets_dir):
    ids = list_lightwheel_objects()
    assert ids == sorted(ids)
    assert ids == [
        "lightwheel_Microwave043", "lightwheel_Stovetop002", "lightwheel_Toaster003",
    ]


def test_list_filter_by_tags_and_contains(assets_dir):
    kitchen = list_lightwheel_objects(tags="kitchen")
    assert "lightwheel_Microwave043" in kitchen
    assert list_lightwheel_objects(contains="MICROWAVE") == ["lightwheel_Microwave043"]
    assert list_lightwheel_objects(tags="no_such_tag") == []


def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="not_an_object"):
        LightwheelFurniture("not_an_object")


# ------------------------------------------------------------------
# make_lightwheel_pool
# ------------------------------------------------------------------

def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    pool = make_lightwheel_pool(["lightwheel_Microwave043", "lightwheel_Toaster003"])
    assert [o.object_id for o in pool] == [
        "lightwheel_Microwave043", "lightwheel_Toaster003",
    ]


def test_pool_ids_and_tags_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError, match="not both"):
        make_lightwheel_pool(["lightwheel_Microwave043"], tags="appliance")


def test_pool_sampling_reproducible(assets_dir, no_download):
    pool_a = make_lightwheel_pool(
        tags="articulated", n=2, rng=torch.Generator().manual_seed(42)
    )
    pool_b = make_lightwheel_pool(
        tags="articulated", n=2, rng=torch.Generator().manual_seed(42)
    )
    assert [o.object_id for o in pool_a] == [o.object_id for o in pool_b]
    assert len({o.object_id for o in pool_a}) == 2


def test_pool_kwargs_forwarded(assets_dir, no_download):
    pool = make_lightwheel_pool(["lightwheel_Microwave043"], scale=0.5, fixed=False)
    assert pool[0].scale == 0.5
    assert pool[0].fixed is False


# ------------------------------------------------------------------
# Construction + completeness
# ------------------------------------------------------------------

def test_construction_downloads_no_object_assets(assets_dir, no_download):
    item = LightwheelFurniture("lightwheel_Microwave043")
    assert item.object_id == "lightwheel_Microwave043"
    assert item.description_file.name == "lightwheel_Microwave043.usd"
    assert item.fixed is True
    assert item.reference_qpos == {}
    assert not no_download


def test_scalar_scale_only(assets_dir):
    with pytest.raises(ValueError, match="uniform"):
        LightwheelFurniture("lightwheel_Microwave043", scale=(1.0, 2.0, 3.0))


def test_required_files_usda_branch(fake_microwave):
    item = LightwheelFurniture("lightwheel_Microwave043")
    required = item._required_files()
    assert required[0] == item.description_file
    # The texture the converted stage references through ./texture/.
    assert item.asset_dir / "texture" / "T_Body001_BC001.png" in required
    assert len(required) == len(set(required))


def test_ensure_assets_partial_dir_triggers_redownload(assets_dir, monkeypatch):
    _make_fake_object_dir(assets_dir, "lightwheel_Microwave043")
    texture = (assets_dir / LIGHTWHEEL_ASSETS_ROOT / "lightwheel_Microwave043"
               / "texture" / "T_Body001_BC001.png")
    texture.unlink()
    calls: list = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        texture.write_bytes(b"png")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    item = LightwheelFurniture("lightwheel_Microwave043")
    item._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == (
        f"{LIGHTWHEEL_ASSETS_ROOT}/lightwheel_Microwave043/**"
    )
    assert item._assets_ready


# ------------------------------------------------------------------
# Offline FK + clean names (shared ArtVIP machinery)
# ------------------------------------------------------------------

def test_fk_closed_pose_and_displacement(fake_microwave):
    item = LightwheelFurniture("lightwheel_Microwave043")
    meshes = item._link_meshes_at_reference()
    assert set(meshes) == {"base", "door"}
    # The door is a 0.05-thick box at x ∈ [0.175, 0.225] in the closed pose.
    door_bounds = meshes["door"].bounds
    assert door_bounds[0][0] == pytest.approx(0.175, abs=1e-5)
    assert door_bounds[1][0] == pytest.approx(0.225, abs=1e-5)

    opened = LightwheelFurniture(
        "lightwheel_Microwave043", reference_qpos={"microjoint": math.pi / 2.0},
    )
    opened_bounds = opened._link_meshes_at_reference()["door"].bounds
    # The 0.05×0.6×0.4 door rotates 90° about the X hinge at (0.2, 0, 0):
    # the y (0.6) and z (0.4) extents swap.
    assert opened_bounds[0][1] == pytest.approx(-0.2, abs=1e-5)
    assert opened_bounds[1][1] == pytest.approx(0.2, abs=1e-5)
    assert opened_bounds[0][2] == pytest.approx(-0.3, abs=1e-5)
    assert opened_bounds[1][2] == pytest.approx(0.3, abs=1e-5)


def test_fk_unknown_joint_raises_listing_clean_names(fake_microwave):
    with pytest.raises(ArticulationError) as exc_info:
        usd_link_meshes_at_config(
            str(fake_microwave / "lightwheel_Microwave043.usd"),
            {"no_such_joint": 0.1},
        )
    assert "microjoint" in str(exc_info.value)


def test_clean_name_translation(assets_dir, fake_microwave):
    item = LightwheelFurniture("lightwheel_Microwave043")
    assert item._to_runtime_name("microjoint", "joint") == "/root/door/microjoint"
    assert item.clean_name_of("/root/door/microjoint") == "microjoint"
    with pytest.raises(ArticulationError, match="no_such_joint"):
        item._to_runtime_name("no_such_joint", "joint")
    assert set(item.list_joint_names()) == {"microjoint"}
    assert set(item.list_link_names()) == {"base", "door"}


# ------------------------------------------------------------------
# RoomBuilder descriptor resolution
# ------------------------------------------------------------------

def test_room_builder_resolves_lightwheel_source(assets_dir, fake_microwave, no_download):
    world = type("WorldStub", (), {"batch_size": None})()
    room = RoomBuilder(
        world,
        name="room",
        layout=RoomLayoutSpec(room_size=(4.0, 4.0, 2.7), furniture=()),
    )
    placement = FurniturePlacement(
        object_id="lightwheel_Microwave043",
        source="lightwheel",
        pos=(0.0, 0.0, 0.0),
        yaw=0.0,
    )
    descriptor = room._resolve_descriptor(placement)
    assert isinstance(descriptor, LightwheelFurniture)
    assert descriptor.object_id == "lightwheel_Microwave043"

    with pytest.raises(ValueError, match="lightwheel"):
        room._resolve_descriptor(FurniturePlacement(
            object_id="x", source="unsuppo", pos=(0, 0, 0), yaw=0.0,
        ))


# ------------------------------------------------------------------
# prepare_lightwheel_assets.py
# ------------------------------------------------------------------

def _make_bundle_asset(source: Path, asset_name: str, *, mdl: bool = False) -> Path:
    """A bundle-shaped Manipulation/<Asset>/<Asset>.usd + texture/ dir."""
    asset_dir = source / "Manipulation" / asset_name
    asset_dir.mkdir(parents=True)
    _write_lightwheel_stage(asset_dir / f"{asset_name}.usd", mdl=mdl)
    textures = asset_dir / "texture"
    textures.mkdir(parents=True, exist_ok=True)
    for name in ("T_Body001_BC001.png", "T_Body001_ORM001.png", "T_Body001_N001.png"):
        (textures / name).write_bytes(b"png")
    return asset_dir


def test_repackage_renames_and_converts(tmp_path):
    source = tmp_path / "bundle"
    _make_bundle_asset(source, "Microwave043", mdl=True)

    out_root = tmp_path / "out" / "objects" / "lightwheel"
    packaged, n_converted = _SCRIPT.repackage_object(
        source / "Manipulation" / "Microwave043", "lightwheel_Microwave043", out_root,
    )

    names = sorted(str(p.relative_to(packaged)) for p in packaged.rglob("*") if p.is_file())
    assert names == [
        "lightwheel_Microwave043.usd",
        "texture/T_Body001_BC001.png",
        "texture/T_Body001_N001.png",
        "texture/T_Body001_ORM001.png",
    ]
    assert n_converted == 1
    assert count_mdl_materials(packaged / "lightwheel_Microwave043.usd") == 0


def test_list_bundle_assets(tmp_path):
    source = tmp_path / "bundle"
    _make_bundle_asset(source, "Microwave043")
    _make_bundle_asset(source, "Stovetop002")
    pairs = dict(_SCRIPT.list_lightwheel_assets(source))
    assert list(pairs) == ["lightwheel_Microwave043", "lightwheel_Stovetop002"]


def test_list_bundle_requires_manipulation_dir(tmp_path):
    with pytest.raises(SystemExit, match="Manipulation"):
        _SCRIPT.list_lightwheel_assets(tmp_path)


def test_catalog_written_from_gate_records(tmp_path):
    out_root = tmp_path / "objects" / "lightwheel"
    out_root.mkdir(parents=True)
    entry = _SCRIPT.annotate_object("Microwave043", 2, 0)[1]
    entry["tags"] = sorted(set(entry["tags"]) | {"articulated"})
    _SCRIPT.write_catalog(out_root, [
        dict(object_id="lightwheel_Microwave043", catalog_entry=entry),
    ])

    written = json.loads((out_root / "catalog.json").read_text())
    stored = written["lightwheel_Microwave043"]
    assert stored["language_annotations"][0] == "appliance"
    assert stored["language_annotations"][-1].startswith("the microwave asset")
    assert {"appliance", "articulated", "kitchen", "microwave"}.issubset(set(stored["tags"]))


def test_license_is_cc_by_nc_and_readme_flags_it():
    assert "Attribution-NonCommercial 4.0 International" in _SCRIPT.LICENSE_TEXT
    assert "lightwheel_{asset_name}" in _SCRIPT.LICENSE_TEXT
    assert "NON-COMMERCIAL USE ONLY" in _SCRIPT.LICENSE_TEXT
    assert "CC BY-NC 4.0" in _SCRIPT.PROVENANCE_README
    assert "convert_mdl_to_preview_surface" in _SCRIPT.PROVENANCE_README


def test_attribution_ids_carry_the_lightwheel_prefix():
    pairs = dict(_SCRIPT.list_lightwheel_assets(
        Path("/tmp/cookie_agent_quantumcookie/lightwheel/lw_open/Lightwheel_OpenSource"),
    )) if Path(
        "/tmp/cookie_agent_quantumcookie/lightwheel/lw_open/Lightwheel_OpenSource/Manipulation",
    ).is_dir() else {}
    for object_id in pairs:
        assert object_id.startswith("lightwheel_")


# ------------------------------------------------------------------
# GPU smoke test (skipped offline)
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_into_scene_smoke_gpu(assets_dir, fake_microwave, no_download):
    """Real GPU smoke: load the synthetic stage via load_into_scene."""
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)

    item = LightwheelFurniture("lightwheel_Microwave043")
    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(type("WorldStub", (), {"scene": scene})())
        scene.build(n_envs=0)
        assert entity is not None

        moving = [j for j in entity.joints if int(j.n_dofs) > 0]
        assert [j.name for j in moving] == ["/root/door/microjoint"]

        vaabb = entity.get_vAABB()
        local = item.get_local_bounding_box()
        assert torch.allclose(vaabb[0], local.low.to(vaabb.device), atol=2e-3)
        assert torch.allclose(vaabb[1], local.high.to(vaabb.device), atol=2e-3)

        item.set_open_fraction(entity, "microjoint", torch.tensor(0.5))
        fraction = item.get_open_fraction(entity, "microjoint")
        assert float(fraction.squeeze().cpu()) == pytest.approx(0.5, abs=1e-2)
    finally:
        scene.destroy()
        torch.set_default_device("cpu")


# ------------------------------------------------------------------
# Joint body retargeting (CoffeeMachine103 / RangeHood010 / Stove070 shapes)
# ------------------------------------------------------------------

def _write_retarget_stage(path: Path) -> Path:
    """A Lightwheel-shaped stage with the three retargeting defects.

    * ``/root/door/door_joint``: a revolute whose ``body1`` targets the bare
      *Visuals mesh* prim ``/root/door/Visuals/door_mesh`` (no RigidBodyAPI)
      — the mixed-entity shape (CoffeeMachine103/RangeHood010).
    * ``/root/door/Visuals/window``: a ``PhysicsFixedJoint`` whose ``body1``
      targets ``/root/door_window`` — a path that does not exist at all
      (ToasterOven056's stale weld).
    * ``/root/part``: a non-root rigid body welded by a fixed joint with an
      **empty** ``body0`` (a weld-to-world) — Stove070's split-off Body002.
    * ``/root/base/FixedJoint``: the *legitimate* root world weld (must stay
      untouched).
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    UsdGeom.Scope.Define(stage, "/root")
    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 0.4))
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 0.4),
                body_translation=(0.2, 0.0, 0.0))
    _write_body(stage, "/root/part", body_extents=(0.1, 0.1, 0.1),
                body_translation=(0.0, 0.0, 0.5))

    # Bare mesh prims (not rigid bodies).
    _write_box_mesh(stage, "/root/door/Visuals/door_mesh", (0.04, 0.5, 0.3))

    door_joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/door_joint")
    door_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    # body1 targets the bare Visuals mesh prim (not a RigidBodyAPI prim).
    door_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door/Visuals/door_mesh")])
    door_joint.CreateAxisAttr("X")

    stale_joint = UsdPhysics.FixedJoint.Define(stage, "/root/door/Visuals/window")
    stale_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    stale_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door_window")])

    part_joint = UsdPhysics.FixedJoint.Define(stage, "/root/part/part_joint")
    # empty body0 = weld-to-world on a non-root body (Stove070's Body002)
    part_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/part")])

    root_joint = UsdPhysics.FixedJoint.Define(stage, "/root/base/FixedJoint")
    # empty body0 on the ROOT body — the legitimate world weld.
    root_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/base")])

    stage.GetRootLayer().Save()
    return stage


def test_retarget_joint_bodies_repairs_all_three_defects(tmp_path):
    stage_file = tmp_path / "lightwheel_Mixer001.usd"
    _write_retarget_stage(stage_file)

    retargeted, deactivated = _SCRIPT.retarget_joint_bodies(stage_file, asset_name="base")

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(stage_file))
    rigid = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    door_joint = UsdPhysics.Joint(stage.GetPrimAtPath("/root/door/door_joint"))
    assert [str(t) for t in door_joint.GetBody1Rel().GetTargets()] == ["/root/door"]
    # The mesh prim itself stays in the stage (visual geometry is untouched).
    assert stage.GetPrimAtPath("/root/door/Visuals/door_mesh").IsValid()
    # The stale fixed joint (nonexistent body1 path) is deactivated.
    stale = stage.GetPrimAtPath("/root/door/Visuals/window")
    assert stale.IsValid() and not stale.IsActive()
    assert deactivated == 1
    # Stove070 shape: the non-root body's world weld becomes a root weld.
    part_joint = UsdPhysics.Joint(stage.GetPrimAtPath("/root/part/part_joint"))
    assert [str(t) for t in part_joint.GetBody0Rel().GetTargets()] == ["/root/base"]
    # The legitimate root world weld is untouched.
    root_joint = UsdPhysics.Joint(stage.GetPrimAtPath("/root/base/FixedJoint"))
    assert list(root_joint.GetBody0Rel().GetTargets()) == []
    assert retargeted == 2
    # Every joint target now resolves onto a rigid body (or is the empty
    # root world weld).
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()):
            targets = list(rel.GetTargets())
            for target in targets:
                assert str(target) in rigid


def test_retarget_joint_bodies_is_idempotent(tmp_path):
    stage_file = tmp_path / "lightwheel_Mixer001.usd"
    _write_retarget_stage(stage_file)
    _SCRIPT.retarget_joint_bodies(stage_file, asset_name="base")
    assert _SCRIPT.retarget_joint_bodies(stage_file, asset_name="base") == (0, 0)


def test_repackage_applies_the_joint_body_retargeting(tmp_path):
    """repackage_object repairs the mesh-targeted joints of a bundle asset."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    asset_dir = tmp_path / "bundle" / "Manipulation" / "Mixer001"
    asset_dir.mkdir(parents=True)
    _write_retarget_stage(asset_dir / "Mixer001.usd")
    textures = asset_dir / "texture"
    textures.mkdir()
    for name in ("T_Body001_BC001.png", "T_Body001_ORM001.png", "T_Body001_N001.png"):
        (textures / name).write_bytes(b"png")

    out_root = tmp_path / "out" / "objects" / "lightwheel"
    packaged, _ = _SCRIPT.repackage_object(asset_dir, "lightwheel_Mixer001", out_root)

    stage = Usd.Stage.Open(str(packaged / "lightwheel_Mixer001.usd"))
    door_joint = UsdPhysics.Joint(stage.GetPrimAtPath("/root/door/door_joint"))
    assert [str(t) for t in door_joint.GetBody1Rel().GetTargets()] == ["/root/door"]
    assert not stage.GetPrimAtPath("/root/door/Visuals/window").IsActive()


def test_relocate_rewrites_windows_drive_letter_texture(tmp_path):
    """``E:/360MoveData/…`` textures rewrite onto the packaged texture dir.

    On POSIX ``PurePosixPath("E:/…").is_absolute()`` is False, so the drive
    path must be detected by pattern and relocated by basename (the
    ElectricKettle007/Sink004 exporters ship the basename in ``texture/``).
    """
    from unienv_genesis_collection.utils.usd_conversions import _relocate_texture

    root_dir = tmp_path / "lightwheel_Kettle007"
    (root_dir / "texture").mkdir(parents=True)
    (root_dir / "texture" / "T_Kettle_BC001.png").write_bytes(b"png")
    (root_dir / "texture" / "T_Kettle_ORM001.png").write_bytes(b"png")

    relocated = _relocate_texture(
        root_dir,
        "E:/360MoveData/Users/PC/Desktop/005/5.29/Kettle007/Kettle007/texture/T_Kettle_BC001.png",
        root_dir,
    )
    assert relocated == "./texture/T_Kettle_BC001.png"

    backslash_form = _relocate_texture(
        root_dir,
        "E:\\360MoveData\\Kettle007\\texture\\T_Kettle_ORM001.png",
        root_dir,
    )
    assert backslash_form == "./texture/T_Kettle_ORM001.png"

    # A drive path with no packaged copy is returned verbatim (the
    # required-file check flags it instead of crashing the rewrite).
    unmatched = _relocate_texture(root_dir, "E:/somewhere/texture/T_Missing_N001.png", root_dir)
    assert unmatched == "E:/somewhere/texture/T_Missing_N001.png"


def test_convert_drops_dangling_normal_map_input(tmp_path):
    """Stove070's ``…_N001`` inputs point at a texture that never shipped.

    The converter drops the normal slot (and its reference) instead of
    authoring a missing file that would fail the required-file check and
    drop the object; the diffuse/ORM slots stay authoritative.
    """
    from pxr import Usd, UsdShade

    obj_dir = tmp_path / "lightwheel_Stove070"
    obj_dir.mkdir()
    _write_lightwheel_stage(obj_dir / "lightwheel_Stove070.usd", mdl=True)
    textures = obj_dir / "texture"
    textures.mkdir()
    for name in ("T_Body001_BC001.png", "T_Body001_ORM001.png"):
        (textures / name).write_bytes(b"png")  # deliberately no T_Body001_N001.png

    converted = convert_mdl_to_preview_surface(
        obj_dir / "lightwheel_Stove070.usd", drop_missing_normal_maps=True,
    )
    assert converted == 1

    stage = Usd.Stage.Open(str(obj_dir / "lightwheel_Stove070.usd"))
    material = stage.GetPrimAtPath("/root/Looks/material_Body001_material")
    labels = sorted(
        prim.GetName().removeprefix("Tex_")
        for prim in material.GetChildren() if prim.GetName().startswith("Tex_")
    )
    assert labels == ["diffuse", "orm_b", "orm_g"]
    assert not any("N001" in str(a.Get()) for a in material.GetPrim().GetAttributes())


def test_convert_keeps_an_existing_normal_map(tmp_path):
    """The dangling-normal drop must not touch resolvable normal maps."""
    from pxr import Usd

    obj_dir = tmp_path / "lightwheel_Kettle001"
    obj_dir.mkdir()
    _write_lightwheel_stage(obj_dir / "lightwheel_Kettle001.usd", mdl=True)
    textures = obj_dir / "texture"
    textures.mkdir()
    for name in ("T_Body001_BC001.png", "T_Body001_ORM001.png", "T_Body001_N001.png"):
        (textures / name).write_bytes(b"png")

    convert_mdl_to_preview_surface(
        obj_dir / "lightwheel_Kettle001.usd", drop_missing_normal_maps=True,
    )
    stage = Usd.Stage.Open(str(obj_dir / "lightwheel_Kettle001.usd"))
    labels = [
        prim.GetName()
        for prim in stage.GetPrimAtPath("/root/Looks/material_Body001_material").GetChildren()
    ]
    assert "Tex_normal" in labels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sanity_gate_handles_multi_dof_and_unbounded_joints_gpu(assets_dir, no_download):
    """GPU regression: the halfway-opened FK cross-check skips SPHERICAL
    (3-DOF generic ``UsdPhysics.Joint``) joints and clamps the midpoint of
    unbounded (±inf limit) DOFs to the closed pose — the Sink0xx crash and
    Stovetop015's ``math domain error`` shapes, on one synthetic stage.
    """
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)

    obj_dir = assets_dir / LIGHTWHEEL_ASSETS_ROOT / "lightwheel_SinkStove001"
    obj_dir.mkdir(parents=True)
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(obj_dir / "lightwheel_SinkStove001.usd"))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root_scope = UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(root_scope.GetPrim())
    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 0.4))
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 0.4),
                body_translation=(0.2, 0.0, 0.0))
    _write_body(stage, "/root/handle", body_extents=(0.1, 0.1, 0.1),
                body_translation=(0.0, 0.0, 0.5))
    _write_body(stage, "/root/knob", body_extents=(0.1, 0.1, 0.1),
                body_translation=(0.0, 0.0, -0.5))

    bounded = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/microjoint")
    bounded.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    bounded.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
    bounded.CreateAxisAttr("X")
    bounded.CreateLowerLimitAttr(-90.0)
    bounded.CreateUpperLimitAttr(0.0)

    # A generic UsdPhysics.Joint → Genesis SPHERICAL, 3 DOFs (the Lightwheel
    # Sink0xx shape: rotational Drive/Limit configs, no translational ones).
    generic = UsdPhysics.Joint.Define(stage, "/root/handle/handle_joint")
    generic.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    generic.CreateBody1Rel().SetTargets([Sdf.Path("/root/handle")])
    for dof_name in ("rotX", "rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(generic.GetPrim(), dof_name)
        limit.CreateLowAttr(-1.0)
        limit.CreateHighAttr(1.0)

    unbounded = UsdPhysics.RevoluteJoint.Define(stage, "/root/knob/knob_joint")
    unbounded.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    unbounded.CreateBody1Rel().SetTargets([Sdf.Path("/root/knob")])
    unbounded.CreateAxisAttr("X")
    unbounded.CreateLowerLimitAttr(0.0)
    unbounded.CreateUpperLimitAttr(float("inf"))
    stage.GetRootLayer().Save()

    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()
    entry = _catalog_entry(obj_dir / "lightwheel_SinkStove001")
    import json as _json

    catalog_file = assets_dir / LIGHTWHEEL_ASSETS_ROOT / "catalog.json"
    entries = _json.loads(catalog_file.read_text())
    entries["lightwheel_SinkStove001"] = entry
    catalog_file.write_text(_json.dumps(entries))
    catalog_module._CACHE.clear()

    item = LightwheelFurniture("lightwheel_SinkStove001")
    record = _SCRIPT.sanity_check_object(item, render_dir=None)

    reference = record["bbox"]["opened"]["reference_qpos"]
    # The 3-DOF spherical joint has no opened reference (the offline FK
    # walker models revolute/prismatic only); the unbounded DOF is clamped
    # to the closed pose — and every reference value stays finite.
    assert sorted(reference) == ["knob_joint", "microjoint"]
    assert float(reference["knob_joint"]) == 0.0
    assert all(math.isfinite(value) for value in reference.values())
    assert record["bbox"]["opened"]["max_delta"] <= 0.03
    scene_records = record["joints"]
    assert {j["type"]: j["n_dofs"] for j in scene_records} == {
        "REVOLUTE": 1, "SPHERICAL": 3,
    }
    torch.set_default_device("cpu")


def test_repackage_survives_a_missing_normal_texture(tmp_path):
    """Stove070 shape through the real repackage path: N001 absent upstream.

    The asset used to die at the required-file check (a missing *image*
    reference is fatal); the repackage's `drop_missing_normal_maps` opt-in
    drops the 11 dangling normal inputs instead.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    asset_dir = tmp_path / "bundle" / "Manipulation" / "Stove070"
    asset_dir.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(asset_dir / "Stove070.usd"))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Scope.Define(stage, "/root").GetPrim())
    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 0.4))
    material = UsdShade.Material.Define(stage, "/root/Looks/material_Body001_material")
    shader = UsdShade.Shader.Define(stage, "/root/Looks/material_Body001_material/Shader")
    shader.CreateImplementationSourceAttr("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token,
    ).Set("OmniPBR")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./texture/T_Stove070_BC001.png"))
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./texture/T_Stove070_ORM001.png"))
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./texture/T_Stove070_N001.png"))
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/root/base/box"))
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/root/base/box")).Bind(material)
    stage.GetRootLayer().Save()
    textures = asset_dir / "texture"
    textures.mkdir()
    (textures / "T_Stove070_BC001.png").write_bytes(b"png")
    (textures / "T_Stove070_ORM001.png").write_bytes(b"png")

    packaged, n_converted = _SCRIPT.repackage_object(asset_dir, "lightwheel_Stove070", out_root=tmp_path / "out" / "objects" / "lightwheel")

    from pxr import Usd

    reopened = Usd.Stage.Open(str(packaged / "lightwheel_Stove070.usd"))
    material_children = [
        prim.GetName()
        for prim in reopened.GetPrimAtPath("/root/Looks/material_Body001_material").GetChildren()
    ]
    assert "Tex_normal" not in material_children
    assert "Tex_diffuse" in material_children
    assert not list(packaged.rglob("*N001*"))
