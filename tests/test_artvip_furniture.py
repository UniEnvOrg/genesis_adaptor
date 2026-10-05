"""Tests for the ArtVIP USD furniture integration (no asset downloads).

Mirrors ``tests/test_replicacad_furniture.py`` with **synthetic USD stages**
authored with ``pxr`` (ASCII ``.usda`` — the shipped assets keep ``.usd``;
``_required_files``/the loader accept all USD suffixes) in ``tmp_path``:

  - Catalog integrity + tag/contains filtering (entries generated through the
    prepare script's ``annotate_object`` template path, like the packaging).
  - list_artvip_objects / make_artvip_pool selection, sampling, validation.
  - ArtVIPFurniture construction (paths, catalog annotations, scalar scale
    enforcement) and asset completeness (_ensure_assets, partial-download
    repair, ``_required_files`` .usd/.usda branch).
  - Offline FK (usd_link_meshes_at_config): closed pose == authored extents,
    reference_qpos displacement with clean joint names, link_local mode,
    unknown-joint errors.
  - Clean-name sanitisation: leaf names, duplicate-leaf disambiguation,
    ``"/"``-prefixed runtime pass-through, unknown-clean-name errors.
  - scripts/prepare_artvip_assets.py: ``.mdl``/thumbs dropping, ``model_…``
    rename, layer-closure verification failure on a truncated tree,
    catalog generation from the category templates.
  - RoomBuilder._resolve_descriptor ``source="artvip"`` branch.

A GPU smoke test (skipped unless CUDA) loads the synthetic stage through
``load_into_scene`` (single entity, DOF, offline bbox vs ``get_vAABB``) —
the packaging gate's core assertion.
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
    usd_clean_names,
    usd_link_meshes_at_config,
)
from unienv_genesis_collection.objects.hub_objects.articulated.artvip_furniture import (
    ARTVIP_ASSETS_ROOT,
    ArtVIPFurniture,
    get_artvip_catalog,
    list_artvip_objects,
    make_artvip_pool,
)
from unienv_genesis_collection.scenes.room_builder import (
    FurniturePlacement,
    RoomBuilder,
    RoomLayoutSpec,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_artvip_assets", _SCRIPTS_DIR / "prepare_artvip_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_prepare_script()

import numpy as np  # noqa: E402  (RC-3/RC-4 fixup tests)


# Cross-suite isolation: the Nyx renderer tests (a neighbouring module)
# load ``gs_nyx``, which bundles its own monolithic OpenUSD runtime that
# corrupts the usd-core Tf notice registry these tests rely on (see
# camera_node.py's DSO note).  When a gs_nyx DSO is already loaded, the USD
# geometry tests skip rather than crash the whole run — run the Nyx set and
# the USD set in separate pytest invocations.
def _usd_runtime_is_safe() -> bool:
    return "gs_nyx" not in __import__("sys").modules


@pytest.fixture(autouse=True)
def _skip_when_usd_runtime_poisoned():
    if not _usd_runtime_is_safe():
        pytest.skip("gs_nyx DSO present — the usd-core Tf runtime is poisoned")

# ------------------------------------------------------------------
# Synthetic USD stage authoring (pxr)
# ------------------------------------------------------------------


def _write_box_mesh(stage, path: str, extents, translation=None, rotation_z=None):
    """A centered-box Mesh prim with a PhysicsCollisionAPI (visual geometry:
    Gen's default pattern = collision-and-visual for unmatched prim names)."""
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
    if (translation, rotation_z) != (None, None):
        xform = UsdGeom.Xform(mesh.GetPrim())
        if rotation_z is not None:
            xform.AddRotateZOp().Set(float(rotation_z))
        if translation is not None:
            xform.AddTranslateOp().Set(Gf.Vec3f(*translation))
    return mesh.GetPrim()


def _write_body(stage, path: str, *, body_extents, body_translation=None, rigid=True):
    """A RigidBodyAPI Xform link holding one box mesh at its own origin."""
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = UsdGeom.Xform.Define(stage, path)
    if body_translation is not None:
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*body_translation))
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
    _write_box_mesh(stage, f"{path}/box", body_extents)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    return prim.GetPrim()


def _write_artvip_stage(path: Path):
    """A minimal ArtVIP-shaped stage: base + hinged door + sliding drawer.

    Mirrors the URDF test's drawer-door template: the door link's *mesh* is
    offset/rotated inside the link frame, so FK displacement is observable;
    the joint anchors sit at the door/front face like the fridge asset.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    root_scope = UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(root_scope.GetPrim())

    # Root body, anchored at the stage origin (furniture convention).
    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 1.8))
    # Door link: box translated inside the link frame by (0.025, 0, 0)... it is
    # authored at the LINK ORIGIN with a 0.05-thick box; the hinge anchor sits
    # at its left edge.
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 1.8),
                body_translation=(0.2, 0.0, 0.0))
    door_joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/revolute_1")
    door_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    door_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
    door_joint.CreateLocalPos0Attr(Gf.Vec3f(0.2, 0.0, 0.0))
    door_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    door_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    door_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    door_joint.CreateAxisAttr("Z")
    door_joint.CreateLowerLimitAttr(0.0)
    door_joint.CreateUpperLimitAttr(120.0)
    # Drawer link: sliding body at the stage origin (authored closed pose),
    # translation along tuvo the stage's front axis.
    _write_body(stage, "/root/drawer", body_extents=(0.5, 0.4, 0.2),
                body_translation=(0.0, 0.0, 1.2))
    drawer_joint = UsdPhysics.PrismaticJoint.Define(stage, "/root/drawer/drawer_slide")
    drawer_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    drawer_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/drawer")])
    drawer_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 1.2))
    drawer_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    drawer_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    drawer_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    drawer_joint.CreateAxisAttr("Y")
    drawer_joint.CreateLowerLimitAttr(-0.4)
    drawer_joint.CreateUpperLimitAttr(0.0)

    # A sublayer (the ArtVIP assets compose resource/** payload layers) whose
    # material carries an *asset-valued* texture reference — the closure of
    # _required_files must name the texture even when it is missing.
    sublayer = path.parent / "resource" / "model" / "payload.usda"
    sublayer.parent.mkdir(parents=True, exist_ok=True)
    sublayer.write_text(
        "#usda 1.0\n"
        "(\n    defaultPrim = \"payload\"\n)\n\n"
        "def Material \"material\"\n"
        "{\n"
        "    def Shader \"tex\"\n"
        "    {\n"
        "        token info:id = \"UsdPreviewSurface\"\n"
        "        custom asset inputs:file = @../img/texture.png@\n"
        "    }\n"
        "}\n"
    )
    stage.GetRootLayer().subLayerPaths.append("./resource/model/payload.usda")

    stage.GetRootLayer().Save()
    return stage


def _write_colliding_stage(path: Path):
    """A stage whose *leaf* prim names collide: two doors each carrying a
    joint named ``hinge`` (the D2 disambiguation fixture)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    root_scope = UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(root_scope.GetPrim())

    _write_body(stage, "/root/body", body_extents=(0.6, 0.6, 0.6))
    for index in (1, 2):
        _write_body(stage, f"/root/E_door{index}", body_extents=(0.05, 0.6, 0.6),
                    body_translation=(0.3 + 0.1 * index, 0.0, 0.0))
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/root/E_door{index}/hinge")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/body")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/root/E_door{index}")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.3 + 0.1 * index, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(90.0)
    stage.GetRootLayer().Save()
    return stage


def _make_fake_object_dir(assets_dir: Path, object_id: str) -> Path:
    """A complete fake ArtVIP object directory (synthetic stage + resource)."""
    obj_dir = assets_dir / ARTVIP_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    _write_artvip_stage(obj_dir / f"{object_id}.usda")
    # A resource/ texture sidecar (the required-file glob's second arm).
    img_dir = obj_dir / "resource" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "texture.png").write_bytes(b"png placeholder")
    return obj_dir


def _make_fake_colliding_object_dir(assets_dir: Path, object_id: str) -> Path:
    """A complete fake object directory from the colliding-leaf stage."""
    obj_dir = assets_dir / ARTVIP_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    _write_colliding_stage(obj_dir / f"{object_id}.usda")
    (obj_dir / "resource" / "img").mkdir(parents=True, exist_ok=True)
    (obj_dir / "resource" / "img" / "texture.png").write_bytes(b"png placeholder")
    return obj_dir


def _catalog_entry(upstream: Path, *, n_revolute: int = 1, n_prismatic: int = 1):
    """A catalog entry generated through the packaging templates."""
    entry = _SCRIPT.annotate_object(upstream, n_revolute, n_prismatic)[2]
    entry["tags"] = sorted(set(entry["tags"]) | {"articulated"})
    return entry


def _make_catalog(assets_dir: Path, entries: dict) -> None:
    (assets_dir / ARTVIP_ASSETS_ROOT / "catalog.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (assets_dir / ARTVIP_ASSETS_ROOT / "catalog.json").write_text(
        json.dumps(entries)
    )


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir with a generated (template-path) catalog."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    upstream = tmp_path / "upstream" / "Articulated_objects" / "major_appliances" / "refrigerator" / "fridge"
    upstream.mkdir(parents=True)
    _make_catalog(assets_dir=tmp_path, entries={
        "refrigerator_fridge": _catalog_entry(upstream),
        "refrigerator_freezer": _catalog_entry(
            tmp_path / "upstream" / "Articulated_objects" / "major_appliances"
            / "freezer" / "freezer_1",
            n_revolute=0, n_prismatic=2,
        ),
        "cabinet_kitchen_unit": _catalog_entry(
            tmp_path / "upstream" / "Articulated_objects" / "small_furniture"
            / "cabinet" / "cabinet_1",
        ),
        "washing_machine_unit": _catalog_entry(
            tmp_path / "upstream" / "Articulated_objects" / "major_appliances"
            / "washing_machine" / "washing_machine_2",
        ),
    })
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()
    return tmp_path


@pytest.fixture
def fake_fridge(assets_dir):
    """A complete fake 'refrigerator_fridge' object directory (synthetic .usda)."""
    return _make_fake_object_dir(assets_dir, "refrigerator_fridge")


@pytest.fixture
def no_download(monkeypatch):
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
    catalog = get_artvip_catalog()
    assert len(catalog) == 4
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert " " not in object_id
        # Multiple annotations, broad → detailed, separate from tags.
        assert len(info.language_annotations) >= 2
        assert len(info.language_annotations[0]) <= len(info.language_annotations[-1])
        assert info.tags
        # Every generated entry carries the packaging's baseline tags: kind +
        # subcategory from the templates (+ "articulated" from the gate pass).
        assert {"articulated"} <= info.tags
        assert info.language_annotations[0] in ("appliance", "furniture")
        for tag in info.tags:
            assert tag == tag.lower() and " " not in tag
        # No standing_quaternion authored: ArtVIP stages are Z-up already.
        assert info.standing_quaternion is None


def test_catalog_detailed_annotation_carries_joint_counts(assets_dir):
    """The generated detailed line records the gate-observed DOF kinds."""
    catalog = get_artvip_catalog()
    fridge = catalog["refrigerator_fridge"]
    assert fridge.language_annotations[-1].startswith("a refrigerator with")
    assert "hinged door" in fridge.language_annotations[-1]
    freezer = catalog["refrigerator_freezer"]
    # The freezer (no hinges) carries its sliding-parts number instead.
    assert "sliding part" in freezer.language_annotations[-1]
    assert "hinged" not in freezer.language_annotations[-1]


def test_catalog_is_cached(assets_dir, no_download):
    assert get_artvip_catalog() is get_artvip_catalog()


def test_list_all_objects_sorted(assets_dir):
    ids = list_artvip_objects()
    assert ids == sorted(ids)
    assert ids == [
        "cabinet_kitchen_unit", "refrigerator_freezer", "refrigerator_fridge",
        "washing_machine_unit",
    ]


def test_list_filter_by_tags_and_contains(assets_dir):
    # Room hints flow into tags through the templates.
    kitchen = list_artvip_objects(tags="kitchen")
    assert "refrigerator_fridge" in kitchen
    assert "washing_machine_unit" not in kitchen  # laundry hint, not kitchen
    # Substring filter, combinable with tags.
    assert list_artvip_objects(contains="FRIDGE") == ["refrigerator_fridge"]
    assert "refrigerator_fridge" in list_artvip_objects(
        tags="kitchen", contains="refrigerator"
    )
    assert list_artvip_objects(tags="no_such_tag") == []


def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="not_an_object"):
        ArtVIPFurniture("not_an_object")


# ------------------------------------------------------------------
# make_artvip_pool
# ------------------------------------------------------------------

def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    pool = make_artvip_pool(["refrigerator_fridge", "cabinet_kitchen_unit"])
    assert [o.object_id for o in pool] == ["refrigerator_fridge", "cabinet_kitchen_unit"]


def test_pool_ids_and_tags_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError, match="not both"):
        make_artvip_pool(["refrigerator_fridge"], tags="furniture")


def test_pool_sampling_reproducible(assets_dir, no_download):
    pool_a = make_artvip_pool(
        tags="articulated", n=3, rng=torch.Generator().manual_seed(42)
    )
    pool_b = make_artvip_pool(
        tags="articulated", n=3, rng=torch.Generator().manual_seed(42)
    )
    ids_a = [o.object_id for o in pool_a]
    assert ids_a == [o.object_id for o in pool_b]
    assert len(set(ids_a)) == 3  # distinct (sampling without replacement)


def test_pool_n_too_large(assets_dir):
    with pytest.raises(ValueError, match="only 2"):
        make_artvip_pool(["refrigerator_fridge", "cabinet_kitchen_unit"], n=3)


def test_pool_kwargs_forwarded(assets_dir, no_download):
    pool = make_artvip_pool(["refrigerator_fridge"], scale=0.5, fixed=True)
    assert pool[0].scale == 0.5
    assert pool[0]._morph_scale == 0.5
    assert pool[0].fixed is True


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_construction_downloads_no_object_assets(assets_dir, no_download):
    item = ArtVIPFurniture("refrigerator_fridge")
    assert item.object_id == "refrigerator_fridge"
    assert item.name == "refrigerator_fridge"
    assert item.language_annotations == tuple(
        get_artvip_catalog()["refrigerator_fridge"].language_annotations
    )
    assert item.fixed is True
    assert item.reference_qpos == {}
    assert not no_download  # only the (already local) catalog was touched


def test_paths_under_assets_dir(assets_dir):
    item = ArtVIPFurniture("refrigerator_fridge")
    expected_dir = assets_dir / ARTVIP_ASSETS_ROOT / "refrigerator_fridge"
    assert item.asset_dir == expected_dir
    assert item.description_file == expected_dir / "refrigerator_fridge.usd"


def test_description_file_name_override(assets_dir):
    """Locally-authored assets/tests may rename the description file (e.g.
    the ``<id>.usda`` fixtures — the packaged assets ship ``<id>.usd``)."""
    _make_fake_object_dir(assets_dir, "refrigerator_fridge")
    item = ArtVIPFurniture("refrigerator_fridge", description_file_name="refrigerator_fridge.usda")
    assert item.description_file.name == "refrigerator_fridge.usda"
    item._ensure_assets()
    assert item._assets_ready


def test_overrides(assets_dir):
    item = ArtVIPFurniture(
        "refrigerator_fridge", name="my_fridge", language_annotations=["the fridge"],
        standing_quaternion=(0.0, 0.0, 0.0, 1.0),
    )
    assert item.name == "my_fridge"
    assert item.language_annotations == ("the fridge",)
    assert item.standing_quaternion == (0.0, 0.0, 0.0, 1.0)


def test_scalar_scale_only(assets_dir):
    with pytest.raises(ValueError, match="uniform"):
        ArtVIPFurniture("refrigerator_fridge", scale=(1.0, 2.0, 3.0))


# ------------------------------------------------------------------
# Asset completeness (_ensure_assets / _required_files)
# ------------------------------------------------------------------

def test_required_files_usda_branch(fake_fridge):
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda"
    )
    required = item._required_files()
    assert required[0] == item.description_file
    # The pxr layer stack (the sublayer the synthetic stage composes) plus the
    # resource texture glob.
    assert set(required) == {
        item.description_file,
        item.asset_dir / "resource" / "model" / "payload.usda",
        item.asset_dir / "resource" / "img" / "texture.png",
    }
    assert len(required) == len(set(required))  # de-duplicated


def test_ensure_assets_complete_no_download(fake_fridge, no_download):
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda"
    )
    item._ensure_assets()  # complete dir -> no Hub access
    assert item._assets_ready


def test_ensure_assets_partial_dir_triggers_redownload(assets_dir, monkeypatch):
    """Simulated-interrupted-download repair, mirroring the ReplicaCAD tests."""
    fake_fridge = _make_fake_object_dir(assets_dir, "refrigerator_fridge")
    # Break one required file but keep the description file (an interrupted
    # download's observable symptom).
    (fake_fridge / "resource" / "img" / "texture.png").unlink()
    calls: list = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        (fake_fridge / "resource" / "img" / "texture.png").write_bytes(b"png")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda"
    )
    item._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == f"{ARTVIP_ASSETS_ROOT}/refrigerator_fridge/**"
    assert item._assets_ready


def test_ensure_assets_missing_required_texture_fails(assets_dir, monkeypatch):
    _make_fake_object_dir(assets_dir, "refrigerator_fridge")
    (assets_dir / ARTVIP_ASSETS_ROOT / "refrigerator_fridge" / "resource"
     / "img" / "texture.png").unlink()
    monkeypatch.setattr(asset_path, "download_assets", lambda **kw: None)
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda"
    )
    with pytest.raises(FileNotFoundError, match="texture.png"):
        item._ensure_assets()
    assert not item._assets_ready


# ------------------------------------------------------------------
# Offline FK (usd_link_meshes_at_config)
# ------------------------------------------------------------------

# Authored geometry of the synthetic stage: base 0.6×0.6×1.8 centered at the
# origin; door link framed at (0.2, 0, 0) with a 0.05×0.6×1.8 box (cuboid
# 0.2±0.025 in x); drawer 0.5×0.4×0.2 at (0, 0, 1.2).
_BASE_BOUNDS = ([-0.3, -0.3, -0.9], [0.3, 0.3, 0.9])
_DOOR_CLOSED_BOUNDS = ([0.175, -0.3, -0.9], [0.225, 0.3, 0.9])
_DRAWER_CLOSED_BOUNDS = ([-0.25, -0.2, 1.1], [0.25, 0.2, 1.3])


# Device isolation: the GPU smoke test in *this* module (or a neighbouring
# CUDA-initialising test module) leaves the torch *ambient* default device
# installed by ``genesis.init`` (see the device-isolation note in
# tests/test_room_builder.py) — the FK tests' expected bounds are host-side
# constants pinned to the same device as the actual result tensors.
def _host(tensor):
    return tensor.detach().to("cpu") if tensor.is_cuda else tensor


def test_fk_closed_pose_matches_authored_extents(fake_fridge):
    """q = 0 == the authored closed pose: box bounds only re-expressed."""
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda"
    )
    meshes = item._link_meshes_at_reference()
    assert set(meshes) == {"base", "door", "drawer"}
    assert torch.allclose(
        _host(torch.tensor(meshes["base"].bounds, dtype=torch.float32)),
        _host(torch.tensor(_BASE_BOUNDS, dtype=torch.float32)),
        atol=1e-5,
    )
    assert torch.allclose(
        _host(torch.tensor(meshes["door"].bounds, dtype=torch.float32)),
        _host(torch.tensor(_DOOR_CLOSED_BOUNDS, dtype=torch.float32)),
        atol=1e-5,
    )
    assert torch.allclose(
        _host(torch.tensor(meshes["drawer"].bounds, dtype=torch.float32)),
        _host(torch.tensor(_DRAWER_CLOSED_BOUNDS, dtype=torch.float32)),
        atol=1e-5,
    )
    # The union bbox is the min/max over the per-link bounds.
    union = item.get_local_bounding_box()
    assert torch.allclose(
        _host(union.low), _host(torch.tensor([-0.3, -0.3, -0.9], dtype=torch.float32))
    )
    assert torch.allclose(
        _host(union.high), _host(torch.tensor([0.3, 0.3, 1.3], dtype=torch.float32))
    )


def test_fk_reference_qpos_moves_links_with_clean_names(fake_fridge):
    """A nonzero reference configuration displaces the links: the door swings
    90° about +z through the hinge anchor (0.2, 0, 0), the drawer slides.

    Exact 90° numbers: the door's 0.05×0.6 box rotates to a 0.6×0.05
    arrangement — world x through the anchor [-0.1, 0.5], world y ±0.025.
    """
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda",
        reference_qpos={"revolute_1": math.pi / 2.0, "drawer_slide": 0.1},
    )
    meshes = item._link_meshes_at_reference()
    door = meshes["door"]
    assert torch.allclose(
        _host(torch.tensor(door.bounds, dtype=torch.float32)),
        _host(torch.tensor([[-0.1, -0.025, -0.9], [0.5, 0.025, 0.9]], dtype=torch.float32)),
        atol=1e-5,
    )
    # The drawer slides along y by +0.1 m.
    drawer = meshes["drawer"]
    assert torch.allclose(
        _host(torch.tensor(drawer.bounds, dtype=torch.float32)),
        _host(torch.tensor([[-0.25, -0.1, 1.1], [0.25, 0.3, 1.3]], dtype=torch.float32)),
        atol=1e-5,
    )
    # The base stays put.
    assert torch.allclose(
        _host(torch.tensor(meshes["base"].bounds, dtype=torch.float32)),
        _host(torch.tensor(_BASE_BOUNDS, dtype=torch.float32)),
        atol=1e-5,
    )


def test_fk_unknown_joint_raises_listing_clean_names(fake_fridge):
    stage_path = fake_fridge / "refrigerator_fridge.usda"
    with pytest.raises(ArticulationError) as exc_info:
        usd_link_meshes_at_config(str(stage_path), {"no_such_joint": 0.1})
    # The error lists the CLEAN (leaf) joint names.
    assert "revolute_1" in str(exc_info.value)
    assert "drawer_slide" in str(exc_info.value)


def test_fk_link_local_mode(fake_fridge):
    """link_local=True returns the links' own-frame boxes (no FK transform)."""
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda",
        reference_qpos={"revolute_1": math.pi / 2.0},
    )
    meshes = item._link_local_meshes_at_reference()
    assert set(meshes) == {"base", "door", "drawer"}
    # The door's own-frame box is configuration-independent: exactly its authored
    # local extents around the link origin.
    assert torch.allclose(
        _host(torch.tensor(meshes["door"].bounds, dtype=torch.float32)),
        _host(torch.tensor([[-0.025, -0.3, -0.9], [0.025, 0.3, 0.9]], dtype=torch.float32)),
        atol=1e-5,
    )
    fk_meshes = item._link_meshes_at_reference()
    assert not torch.allclose(  # the FK'd variant differs (the door is rotated)
        _host(torch.tensor(meshes["door"].bounds, dtype=torch.float32)),
        _host(torch.tensor(fk_meshes["door"].bounds, dtype=torch.float32)),
        atol=1e-4,
    )


def test_fk_skips_geometry_less_links(assets_dir):
    """Links without drawable bounds are omitted from the mapping."""
    from pxr import Usd, UsdGeom, UsdPhysics

    obj_id = "refrigerator_fridge"
    _make_fake_object_dir(assets_dir, obj_id)
    usd_path = assets_dir / ARTVIP_ASSETS_ROOT / obj_id / f"{obj_id}.usda"
    usd_stage = Usd.Stage.Open(str(usd_path))
    naked = UsdGeom.Xform.Define(usd_stage, "/root/naked_link")
    UsdPhysics.RigidBodyAPI.Apply(naked.GetPrim())
    usd_stage.GetRootLayer().Save()

    meshes = usd_link_meshes_at_config(str(usd_path))
    assert set(meshes) == {"base", "door", "drawer"}
    assert "naked_link" not in meshes


# ------------------------------------------------------------------
# Clean-name sanitisation
# ------------------------------------------------------------------

def test_usd_clean_names_disambiguates_duplicate_leaves(tmp_path):
    """Colliding leaf names get the parent-chain suffix (shortest-first)."""
    from pxr import Usd, UsdPhysics

    path = tmp_path / "colliding.usda"
    _write_colliding_stage(path)
    usd_stage = Usd.Stage.Open(str(path))
    joints = [
        prim for prim in usd_stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    ]
    links = [
        prim for prim in usd_stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    joint_clean = usd_clean_names(joints)
    link_clean = usd_clean_names(links)
    # Both joints' leaf names collide ("hinge"): disambiguated by the parent.
    assert joint_clean == ["E_door1/hinge", "E_door2/hinge"]
    # The non-colliding ("body") and unique-after-one-growth doors stay plain.
    assert link_clean == ["body", "E_door1", "E_door2"]


@pytest.fixture
def colliding_assets(assets_dir):
    """Assets dir with a catalog entry + object dir for the colliding stage."""
    _make_fake_colliding_object_dir(assets_dir, "test_cabinet")
    existing = json.loads(
        (assets_dir / ARTVIP_ASSETS_ROOT / "catalog.json").read_text()
    )
    existing["test_cabinet"] = {
        "language_annotations": ["furniture", "a cabinet"],
        "tags": ["furniture", "cabinet"],
    }
    _make_catalog(assets_dir=assets_dir, entries=existing)
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()
    return assets_dir


def test_descriptor_clean_name_translation(assets_dir, fake_fridge):
    """Task-facing clean names ↔ the runtime full-path names."""
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda"
    )
    # Clean → runtime: the USD joint prim path under the door.
    assert item._to_runtime_name("revolute_1", "joint") == "/root/door/revolute_1"
    # Already-runtime ("path-form") names pass through unchanged.
    assert (
        item._to_runtime_name("/root/door/revolute_1", "joint")
        == "/root/door/revolute_1"
    )
    # The reverse lookup: gate records address joints by *runtime path*.
    assert item.clean_name_of("/root/door/revolute_1") == "revolute_1"
    assert item.clean_name_of("/root/drawer/drawer_slide", "joint") == "drawer_slide"
    # Unknown clean names raise listing the *clean* names (not the runtime paths).
    with pytest.raises(ArticulationError, match="no_such_joint"):
        item._to_runtime_name("no_such_joint", "joint")
    assert set(item.list_joint_names()) == {"revolute_1", "drawer_slide"}
    assert set(item.list_link_names()) == {"base", "door", "drawer"}


def test_descriptor_disambiguated_names(colliding_assets):
    """Duplicate-leaf stage through the descriptor: suffix clean names."""
    item = ArtVIPFurniture("test_cabinet", description_file_name="test_cabinet.usda")
    assert set(item.list_joint_names()) == {"E_door1/hinge", "E_door2/hinge"}
    assert set(item.list_link_names()) == {"body", "E_door1", "E_door2"}
    # The resolve_on_entity-level translation maps the disambiguated name to
    # its own prim path.
    assert item._to_runtime_name("E_door2/hinge", "joint") == "/root/E_door2/hinge"


def test_descriptor_unknown_clean_name_error_lists_clean(assets_dir, fake_fridge):
    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda"
    )
    with pytest.raises(ArticulationError) as exc_info:
        item._to_runtime_name("no_such_joint", "joint")
    message = str(exc_info.value)
    assert "no_such_joint" in message
    assert "revolute_1" in message and "drawer_slide" in message
    assert "/root/door/revolute_1" not in message  # clean names, not runtime


# ------------------------------------------------------------------
# RoomBuilder descriptor resolution
# ------------------------------------------------------------------

def test_room_builder_resolves_artvip_source(assets_dir, fake_fridge, no_download):
    world = type("WorldStub", (), {"batch_size": None})()
    room = RoomBuilder(
        world,
        name="room",
        layout=RoomLayoutSpec(room_size=(4.0, 4.0, 2.7), furniture=()),
    )
    placement = FurniturePlacement(
        object_id="refrigerator_fridge",
        source="artvip",
        pos=(0.0, 0.0, 0.0),
        yaw=0.0,
    )
    descriptor = room._resolve_descriptor(placement)
    assert isinstance(descriptor, ArtVIPFurniture)
    assert descriptor.object_id == "refrigerator_fridge"

    with pytest.raises(ValueError, match="artvip"):
        room._resolve_descriptor(FurniturePlacement(
            object_id="x", source="unsuppo", pos=(0, 0, 0), yaw=0.0,
        ))


# ------------------------------------------------------------------
# prepare_artvip_assets.py
# ------------------------------------------------------------------

def _make_upstream_object_dir(source: Path, category: str, subcategory: str,
                              upstream_id: str) -> Path:
    """A minimal upstream object dir: model_<id>.usda + resource + sidecars."""
    obj_dir = source / "Articulated_objects" / category / subcategory / upstream_id
    (obj_dir / "resource" / "img").mkdir(parents=True)
    _write_artvip_stage(obj_dir / f"model_{upstream_id}.usda")
    (obj_dir / "resource" / "img" / "texture.png").write_bytes(b"png")
    # upstream sidecars dropped at packaging
    (obj_dir / ".collect.mapping.json").write_text("{}")
    (obj_dir / ".thumbs").mkdir()
    (obj_dir / ".thumbs" / "thumb.png").write_bytes(b"png")
    (obj_dir / "Isaac Sim 4.5").mkdir()
    (obj_dir / "Isaac Sim 4.5" / "glass.mdl").write_bytes(b"mdl")
    return obj_dir


def test_repackage_drops_sidecars_and_renames(tmp_path):
    source = tmp_path / "snap"
    upstream = _make_upstream_object_dir(source, "major_appliances", "refrigerator", "fridge_4")

    out_root = tmp_path / "out" / "objects" / "artvip"
    _SCRIPT.repackage_object(upstream, "refrigerator_fridge_4", out_root)

    packaged = out_root / "refrigerator_fridge_4"
    names = sorted(str(p.relative_to(packaged)) for p in packaged.rglob("*") if p.is_file())
    # A2: the root layer is renamed to ``<object_id>.usd`` UNCONDITIONALLY —
    # an upstream ``.usda``/``.usdc`` root ships as ``.usd`` (pxr sniffs the
    # layer format by content), because the gate/descriptor hard-code the
    # ``<object_id>.usd`` name.
    assert names == [
        "refrigerator_fridge_4.usd",
        "resource/img/texture.png",
        "resource/model/payload.usda",
    ]
    # no upstream sidecars, no pytest noise:
    assert not list(packaged.glob("*.mdl"))
    assert not list(packaged.glob(".collect*"))
    assert not list(packaged.glob(".thumbs/*"))
    # (module docstring tests cover the sublayer closure separately)


def test_repackage_layer_closure_failure_dropped(tmp_path):
    """A truncated upstream tree fails loudly in repackage (caller drops)."""
    source = tmp_path / "snap"
    upstream = _make_upstream_object_dir(source, "major_appliances", "refrigerator", "fridge_x")
    # The synthetic stage composes a resource/model payload sublayer —
    # simulate a *partial upstream download* by cutting it off:
    (upstream / "resource" / "model" / "payload.usda").unlink()
    with pytest.raises(FileNotFoundError, match="references files missing"):
        _SCRIPT.repackage_object(
            upstream, "refrigerator_fridge_x", tmp_path / "out" / "objects" / "artvip"
        )


# ------------------------------------------------------------------
# A5 / A6 / A7 — packaged-stage fixups
# ------------------------------------------------------------------

def _write_loft_bed_stage(path: Path):
    """A stage with an articulated door + an *unjointed* pillow rigid body.

    Mirrors the ArtVIP loft beds (``loft_bed_with_desk_bed_and_table00{2,3,6,9}``):
    Genesis raises "Mixed entity detected ... has N joints but also has M rigid
    bodies not referenced by joints" for this shape.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    _write_body(stage, "/root/base", body_extents=(1.0, 0.6, 0.4))
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 0.4),
                body_translation=(0.5, 0.0, 0.0))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.5, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(90.0)
    # The pillow: a rigid body no joint references (the mixed-entity trigger).
    _write_body(stage, "/root/pillow", body_extents=(0.4, 0.3, 0.1),
                body_translation=(0.0, 0.0, 0.5))
    stage.GetRootLayer().Save()
    return stage


def _write_degenerate_collider_stage(path: Path):
    """A stage whose base carries a 3-point "collider" (no 3D hull).

    ``trimesh.convex.convex_hull`` aborts on it with
    ``QH6214 ... not enough points(3)`` — the subwaydoor gate crash.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    _write_body(stage, "/root/base", body_extents=(1.0, 0.6, 0.4))
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 0.4),
                body_translation=(0.5, 0.0, 0.0))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.5, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(90.0)

    # A degenerate collider inside the door link: three collinear points.
    mesh = UsdGeom.Mesh.Define(stage, "/root/door/degenerate")
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.1), Gf.Vec3f(0.0, 0.0, 0.2)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return stage


def test_repackage_deactivates_unjointed_rigid_bodies(tmp_path):
    """A5: unjointed RigidBodyAPI prims are deactivated (mixed-entity fix)."""
    from pxr import Usd, UsdPhysics

    source = tmp_path / "snap"
    upstream = source / "Articulated_objects" / "large_furniture" / "loft_bed" / "bed002"
    (upstream / "resource" / "img").mkdir(parents=True)
    _write_loft_bed_stage(upstream / "model_bed002.usda")
    (upstream / "resource" / "img" / "texture.png").write_bytes(b"png")

    out_root = tmp_path / "out" / "objects" / "artvip"
    packaged = _SCRIPT.repackage_object(upstream, "bed002", out_root)

    stage = Usd.Stage.Open(str(packaged / "bed002.usd"))
    bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert bodies == ["/root/base", "/root/door"]  # the pillow is gone
    # …and the prim is still *defined* upstream (deactivated, not deleted, so
    # the referenced link layer's spec is neutralised too).
    assert not stage.GetPrimAtPath("/root/pillow").IsActive()
    # The joint is untouched: the articulation still has its moving DOF.
    assert [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)] == [
        "/root/door/hinge"
    ]


def test_repackage_prunes_degenerate_colliders(tmp_path):
    """A6: collision meshes without four non-coplanar points lose CollisionAPI."""
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "snap"
    upstream = source / "Articulated_objects" / "industrial_machinery" / "subway" / "door"
    (upstream / "resource" / "img").mkdir(parents=True)
    _write_degenerate_collider_stage(upstream / "model_door.usda")
    (upstream / "resource" / "img" / "texture.png").write_bytes(b"png")

    out_root = tmp_path / "out" / "objects" / "artvip"
    packaged = _SCRIPT.repackage_object(upstream, "door", out_root)

    stage = Usd.Stage.Open(str(packaged / "door.usd"))
    degenerate = stage.GetPrimAtPath("/root/door/degenerate")
    assert not degenerate.HasAPI(UsdPhysics.CollisionAPI)  # collider pruned
    assert degenerate.IsA(UsdGeom.Mesh)                    # geometry kept
    # The healthy colliders survive.
    remaining = [
        str(p.GetPath())
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert "/root/base/box" in remaining and "/root/door/box" in remaining


def test_is_degenerate_collider_rules():
    """A6 helper: <4 points or a coplanar set ⇒ degenerate; tetra ⇒ fine."""
    import numpy as np

    assert _SCRIPT._is_degenerate_collider(np.zeros((3, 3)))
    assert _SCRIPT._is_degenerate_collider(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    )
    assert _SCRIPT._is_degenerate_collider(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]] * 4)
    )
    assert not _SCRIPT._is_degenerate_collider(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert _SCRIPT._is_degenerate_collider(np.full((4, 3), np.nan))


def test_repackage_oversized_asset_dropped(tmp_path):
    """A7: a tens-of-metres stage is dropped before the expensive load."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source = tmp_path / "snap"
    upstream = source / "Articulated_objects" / "industrial_machinery" / "subway" / "car"
    (upstream / "resource" / "img").mkdir(parents=True)
    stage = _write_degenerate_collider_stage(upstream / "model_car.usda")
    # Stretch the base body to 60 m (the subwaycar001 shape).
    _write_box_mesh(stage, "/root/base/box", (60.0, 1.0, 1.0))
    stage.GetRootLayer().Save()
    (upstream / "resource" / "img" / "texture.png").write_bytes(b"png")

    out_root = tmp_path / "out" / "objects" / "artvip"
    with pytest.raises(_SCRIPT.OversizedAssetError, match="exceeds the 20.0 m"):
        _SCRIPT.repackage_object(upstream, "car", out_root)
    # The guard sits above the two shipped ~13.8 m ArtVIP assets.
    assert _SCRIPT.MAX_STAGE_DIAGONAL_M > 13.8
    assert _SCRIPT._stage_diagonal(upstream / "model_car.usda") > 20.0


def test_stage_diagonal_of_small_stage(tmp_path):
    """A7 helper: the extent probe reports the real stage size."""
    upstream = tmp_path / "obj"
    (upstream / "resource" / "img").mkdir(parents=True)
    stage_file = upstream / "model_small.usda"
    _write_loft_bed_stage(stage_file)
    diagonal = _SCRIPT._stage_diagonal(stage_file)
    assert 1.0 < diagonal < 3.0  # a ~1.1 x 0.6 x 0.4 m bench


def test_list_flattens_ids_and_skips_missing_categories(tmp_path):
    source = tmp_path / "snap"
    _make_upstream_object_dir(source, "major_appliances", "refrigerator", "fridge_4")
    _make_upstream_object_dir(source, "major_appliances", "refrigerator", "refrigerator002")
    # An id that does NOT start with its subcategory gets the subcat prefix;
    # one that does (cabinet/cabinet_12) collapses to the upstream spelling.
    _make_upstream_object_dir(source, "small_furniture", "cabinet", "cabinet_12")
    _make_upstream_object_dir(source, "small_furniture", "door", "door005")
    pairs = dict(_SCRIPT.list_artvip_objects(source, ["major_appliances", "small_furniture", "lab_items"]))
    assert pairs == {
        "refrigerator_fridge_4": source / "Articulated_objects/major_appliances/refrigerator/fridge_4",
        "refrigerator002": source / "Articulated_objects/major_appliances/refrigerator/refrigerator002",
        "cabinet_12": source / "Articulated_objects/small_furniture/cabinet/cabinet_12",
        "door005": source / "Articulated_objects/small_furniture/door/door005",
    }
    # missing categories are documented, not fatal
    assert "lab_items" not in pairs


def test_repackage_duplicate_id_fails(tmp_path):
    """Two upstream objects flattening to the same id abort the listing."""
    source = tmp_path / "snap"
    _make_upstream_object_dir(source, "major_appliances", "refrigerator", "fridge")
    # Same subcategory name under another parent: the flattened id collides.
    _make_upstream_object_dir(source, "small_furniture", "refrigerator", "fridge")
    with pytest.raises(SystemExit, match="Duplicate packaged object id"):
        _SCRIPT.list_artvip_objects(source, ["major_appliances", "small_furniture"])


# ------------------------------------------------------------------
# A1 — 2-level categories + sidecar dirs (Ikea_furniture, Medical_equipment)
# ------------------------------------------------------------------

def _make_two_level_object_dir(source: Path, category: str, product: str) -> Path:
    """The 2-level upstream layout: ``<category>/<product>/model_*.usd``.

    The product dir *is* the object dir; ``resource/``, ``origin/``,
    ``NVIDIA/``, ``img/`` and ``v_*`` sit next to it as sidecars (the
    pre-fix code recursed into them and packaged ``resource/`` as the
    object, so the real asset was never packaged).
    """
    obj_dir = source / "Articulated_objects" / category / product
    (obj_dir / "resource" / "v_69323").mkdir(parents=True)
    _write_artvip_stage(obj_dir / f"model_{product}.usd")
    (obj_dir / "resource" / "v_69323" / "link_0.usd").write_text("#usda 1.0\n")
    (obj_dir / "resource" / "img").mkdir()
    (obj_dir / "resource" / "img" / "texture.png").write_bytes(b"png")
    (obj_dir / "origin").mkdir()
    (obj_dir / "origin" / "orig.fbx").write_bytes(b"fbx")
    (obj_dir / "NVIDIA").mkdir()
    (obj_dir / "NVIDIA" / "preview.png").write_bytes(b"png")
    (obj_dir / "img").mkdir()
    (obj_dir / "img" / "thumb.png").write_bytes(b"png")
    (obj_dir / "v_1").mkdir()
    (obj_dir / "v_1" / "element_path.json").write_text("{}")
    (obj_dir / "v_1" / "model_other.usd").write_text("#usda 1.0\n")
    return obj_dir


def test_list_artvip_objects_two_level_categories(tmp_path):
    """A1: 2-level categories list the product dir, never its sidecars."""
    source = tmp_path / "snap"
    product = "EKET_Cabinet_with_2_drawers_white_70x35x35cm"
    obj_dir = _make_two_level_object_dir(source, "Ikea_furniture", product)
    pairs = dict(_SCRIPT.list_artvip_objects(source, ["Ikea_furniture"]))
    # 2-level categories have no subcategory level, so the product dir name
    # is the packaged id (spaces → underscores, case preserved).
    assert pairs == {product: obj_dir}
    # no ``..._resource``/``..._origin``/``..._NVIDIA`` pseudo-objects
    assert not [oid for oid in pairs if oid.endswith(("_resource", "_origin", "_img"))]
    # the v_* dir's stray model_other.usd is not picked up either
    assert not [oid for oid in pairs if "v_1" in oid]

    out_root = tmp_path / "out" / "objects" / "artvip"
    packaged = _SCRIPT.repackage_object(obj_dir, product, out_root)
    assert (packaged / f"{product}.usd").is_file()
    assert (packaged / "resource" / "img" / "texture.png").is_file()
    # ``origin/``, ``NVIDIA/``, ``img/`` and ``v_*`` never enter the pack.
    assert not (packaged / "origin").exists()
    assert not (packaged / "NVIDIA").exists()
    assert not (packaged / "img").exists()
    assert not list(packaged.glob("v_*"))


def test_find_model_usd_requires_model_stem(tmp_path):
    """A1: ``material.usd`` (or any non-``model_*`` layer) is never the root."""
    obj_dir = tmp_path / "obj"
    obj_dir.mkdir()
    (obj_dir / "material.usd").write_text("#usda 1.0\n")
    assert _SCRIPT._find_model_usd(obj_dir) is None
    (obj_dir / "model_thing.usda").write_text("#usda 1.0\n")
    assert _SCRIPT._find_model_usd(obj_dir) == obj_dir / "model_thing.usda"
    # two ``model_*`` layers are ambiguous → dropped
    (obj_dir / "model_other.usd").write_text("#usda 1.0\n")
    assert _SCRIPT._find_model_usd(obj_dir) is None


def test_all_categories_packagable():
    """No category ban: Ikea_furniture / Medical_equipment are packagable.

    The license caveat (dataset card claims Apache-2.0 but no LICENSE file is
    committed) is flagged in the module docstring + provenance README, and the
    upstream license header is kept at the top of the packager script — per
    the collection maintainer's decision, all categories are now packaged.
    """
    assert set(_SCRIPT.EXCLUDED_CATEGORIES) == set()
    assert "Ikea_furniture" in _SCRIPT.ALL_KNOWN_CATEGORIES
    assert "Medical_equipment" in _SCRIPT.ALL_KNOWN_CATEGORIES
    # The upstream license header is kept at the top of the adaptor script.
    script_text = Path(_SCRIPT.__file__).read_text()
    assert "Upstream asset license" in script_text.split("\n")[1]


def test_catalog_written_from_gate_records(tmp_path):
    """write_catalog: gate-merged entries, template tags, joint-set signal."""
    out_root = tmp_path / "objects" / "artvip"
    out_root.mkdir(parents=True)
    upstream = tmp_path / "upstream" / "Articulated_objects" / "small_furniture" / "cabinet" / "cab"
    upstream.mkdir(parents=True)
    entry = _SCRIPT.annotate_object(upstream, 2, 1)[2]
    entry["tags"] = sorted(set(entry["tags"]) | {"articulated"})
    _SCRIPT.write_catalog(out_root, [dict(object_id="cabinet_cab", catalog_entry=entry)])

    written = json.loads((out_root / "catalog.json").read_text())
    assert set(written) == {"cabinet_cab"}
    stored = written["cabinet_cab"]
    assert stored["language_annotations"][0] == "furniture"
    assert stored["language_annotations"][1] == "a cabinet"
    assert stored["language_annotations"][-1].startswith("a cabinet with")
    assert {"furniture", "articulated", "living_room"}.issubset(set(stored["tags"]))
    assert "cabinet" in stored["tags"]


def test_write_catalog_merges_with_existing(tmp_path):
    out_root = tmp_path / "objects" / "artvip"
    out_root.mkdir(parents=True)
    upstream = tmp_path / "upstream" / "Articulated_objects" / "major_appliances" / "refrigerator" / "fridge"
    upstream.mkdir(parents=True)
    entry_a = _SCRIPT.annotate_object(upstream, 2, 0)[2]
    entry_a["tags"] = sorted(set(entry_a["tags"]) | {"articulated"})
    _SCRIPT.write_catalog(out_root, [dict(object_id="refrigerator_fridge", catalog_entry=entry_a)])
    # Second incremental run writes another object — the earlier entry survives.
    entry_b = _SCRIPT.annotate_object(upstream.parent / "free", 1, 1)[2]
    entry_b["tags"] = sorted(set(entry_b["tags"]) | {"articulated"})
    _SCRIPT.write_catalog(out_root, [dict(object_id="refrigerator_free", catalog_entry=entry_b)])
    written = json.loads((out_root / "catalog.json").read_text())
    assert set(written) == {"refrigerator_fridge", "refrigerator_free"}


def test_write_catalog_template_detail_no_joint_gate(tmp_path):
    """Pre-gate (zero-count) annotations keep the template detail line."""
    out_root = tmp_path / "objects" / "artvip"
    out_root.mkdir(parents=True)
    upstream = tmp_path / "upstream" / "Articulated_objects" / "major_appliances" / "refrigerator" / "fridge"
    upstream.mkdir(parents=True)
    entry = _SCRIPT.annotate_object(upstream, 0, 0)[2]
    _SCRIPT.write_catalog(out_root, [dict(object_id="refrigerator_fridge", catalog_entry=entry)])
    written = json.loads((out_root / "catalog.json").read_text())
    assert written["refrigerator_fridge"]["language_annotations"][-1] == (
        "a plain refrigerator variant"
    )


# ------------------------------------------------------------------
# GPU smoke test (skipped offline)
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_into_scene_smoke_gpu(assets_dir, fake_fridge, no_download):
    """Real GPU smoke: load the synthetic stage via load_into_scene.

    Mirrors the prepare pipeline's gate: the stage splits into a single
    entity, the revolute joint carries DOFs, the clean joint name resolves
    against the entity, and the offline FK union bbox matches
    ``entity.get_vAABB()`` at the identity morph pose (USD stages are not
    re-centred onto the root body — the packaging gate's frame contract).
    """
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)

    item = ArtVIPFurniture(
        "refrigerator_fridge", description_file_name="refrigerator_fridge.usda"
    )
    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(type("WorldStub", (), {"scene": scene})())
        scene.build(n_envs=0)
        assert entity is not None

        moving = [j for j in entity.joints if int(j.n_dofs) > 0]
        assert [j.name for j in moving] == ["/root/door/revolute_1", "/root/drawer/drawer_slide"]
        # Runtime names are the USD paths; the clean names resolve to them.
        assert item._to_runtime_name("revolute_1", "joint") == "/root/door/revolute_1"
        assert torch.as_tensor(
            [list(d) for d in [item.resolve_joint_dofs(entity, "revolute_1")]]
        ).flatten().shape == (1,)

        # The packaging-time cross-check: offline FK union bbox vs runtime AABB
        # (identity morph pose ⇒ the stage's world frame == the entity frame).
        vaabb = entity.get_vAABB()
        local = item.get_local_bounding_box()
        assert torch.allclose(
vaabb[0], local.low.to(vaabb.device), atol=2e-3)
        assert torch.allclose(
vaabb[1], local.high.to(vaabb.device), atol=2e-3)

        # Open-fraction helpers on the *clean* name.
        item.set_open_fraction(entity, "revolute_1", torch.tensor(0.5))
        fraction = item.get_open_fraction(entity, "revolute_1")
        assert float(fraction.squeeze().cpu()) == pytest.approx(0.5, abs=1e-2)
    finally:
        scene.destroy()
        # Restore the host ambient device (see the replicacad GPU smoke's
        # teardown note — tests/test_room_builder.py's device-isolation note):
        torch.set_default_device("cpu")


# ------------------------------------------------------------------
# Halfway-opened FK cross-check: multi-DOF skip + non-finite clamp
# ------------------------------------------------------------------

def test_opened_reference_joints_skip_multi_dof():
    """Only single-DOF (revolute/prismatic) joints take part in the opened loop.

    A generic ``UsdPhysics.Joint`` maps onto a 3-DOF SPHERICAL/FREE — its
    halfway midpoint is a 3-vector and ``float(...item())`` raises.
    """
    moving = [
        dict(name="a", type="REVOLUTE", n_dofs=1),
        dict(name="b", type="SPHERICAL", n_dofs=3),
        dict(name="c", type="PRISMATIC", n_dofs=1),
        dict(name="d", type="FREE", n_dofs=6),
        dict(name="e", type="FIXED", n_dofs=0),
    ]
    assert [record["name"] for record in _SCRIPT._opened_reference_joints(moving)] == ["a", "c"]


def test_finite_midpoint_clamps_non_finite_limits():
    """An unbounded DOF's midpoint is inf/NaN → clamped to the closed pose.

    ``0.5 * (0, inf)`` is inf (which feeds ``math.sin(inf)`` into the FK) and
    ``0.5 * (-inf, inf)`` is NaN (which would make the bbox comparison
    silently pass) — both must come out as the finite closed pose (0.0).
    """
    zero = _SCRIPT._finite_midpoint(torch.tensor([0.0]), torch.tensor([float("inf")]))
    assert torch.isfinite(zero).all() and float(zero.squeeze()) == 0.0

    nan = _SCRIPT._finite_midpoint(torch.tensor([float("-inf")]), torch.tensor([float("inf")]))
    assert torch.isfinite(nan).all() and float(nan.squeeze()) == 0.0

    midpoint = _SCRIPT._finite_midpoint(torch.tensor([-90.0]), torch.tensor([0.0]))
    assert torch.isfinite(midpoint).all() and float(midpoint.squeeze()) == pytest.approx(-45.0)


def test_opened_gate_helpers_agree_across_the_packagers():
    """The three packager copies of the gate helpers behave identically."""
    import importlib.util as _ilu

    expected_filter = ["revolute", "prismatic"]
    moving = [
        dict(name="revolute", type="REVOLUTE", n_dofs=1),
        dict(name="spherical", type="SPHERICAL", n_dofs=3),
        dict(name="prismatic", type="PRISMATIC", n_dofs=1),
    ]
    for name in ("prepare_artvip_assets", "prepare_lightwheel_assets",
                 "prepare_molmospaces_assets"):
        module = _SCRIPT if name == "prepare_artvip_assets" else (
            _ilu.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
            and _ilu.module_from_spec(_ilu.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py"))
        )
        if module is not _SCRIPT:
            module.__spec__.loader.exec_module(module)
        assert [r["name"] for r in module._opened_reference_joints(moving)] == expected_filter
        clamped = module._finite_midpoint(torch.tensor([0.0]), torch.tensor([float("inf")]))
        assert float(clamped.squeeze()) == 0.0


# ------------------------------------------------------------------
# RC-2 (Fix A): Genesis-parity visibility filter
# ------------------------------------------------------------------

def _write_visibility_stage(path: Path):
    """base + hinged door; the base carries an extra INVISIBLE box that sticks
    out 0.4 m to +x (the RC-2 shape: an invisible authored site/receptacle
    prim must not widen the offline box), and a link whose ONLY drawable is
    invisible (which Genesis builds with no visual at all)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/root"))

    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 0.6))
    # A visible box (within the base's own extents) + an invisible box that
    # sticks 0.4 m out to +x.  Same pattern as Lightwheel's invisible
    # `Sites/*` gprims (purpose default, visibility invisible).
    _write_box_mesh(stage, "/root/base/visible_box", (0.2, 0.2, 0.2))
    ghost = _write_box_mesh(stage, "/root/base/ghost_box", (0.9, 0.2, 0.2),
                            translation=(0.65, 0.0, 0.0))
    UsdGeom.Imageable(ghost).MakeInvisible()
    assert str(UsdGeom.Imageable(ghost).ComputeVisibility()) == "invisible"

    # An all-invisible link: Genesis gives it no visual → no offline mesh.
    _write_body(stage, "/root/sites", body_extents=(0.3, 0.3, 0.3),
                body_translation=(0.0, 1.0, 0.0))
    UsdGeom.Imageable(stage.GetPrimAtPath("/root/sites/box")).MakeInvisible()

    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 0.6),
                body_translation=(0.3, 0.0, 0.0))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.3, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(90.0)
    stage.GetRootLayer().Save()
    return stage


def test_fk_excludes_invisible_drawables(tmp_path):
    """RC-2: an invisible drawable must not widen the offline box (Genesis's
    is_visible gate applies to BOTH the visual and the collision set)."""
    from pxr import Usd

    stage_path = tmp_path / "visible.usda"
    _write_visibility_stage(stage_path)

    meshes = usd_link_meshes_at_config(str(stage_path))
    # The ghost box (0.65±0.45 in x) is invisible: the base box stays 0.6 wide.
    base = meshes["base"]
    assert base.bounds[0][0] == pytest.approx(-0.3, abs=1e-6)
    assert base.bounds[1][0] == pytest.approx(0.3, abs=1e-6)


def test_fk_omits_all_invisible_links(tmp_path):
    """RC-2: a link whose drawables are ALL invisible has no Genesis geometry
    (Genesis drops every invisible gprim from the visual set) — omitted."""
    stage_path = tmp_path / "visible.usda"
    _write_visibility_stage(stage_path)

    meshes = usd_link_meshes_at_config(str(stage_path))
    assert "sites" not in meshes  # its only drawable is invisible
    assert set(meshes) == {"base", "door"}


# ------------------------------------------------------------------
# RC-1 (Fix B): rigid-frame FK mirroring Genesis
# ------------------------------------------------------------------

def _write_scaled_group_stage(path: Path, ancestor_scale=(0.01, 0.01, 0.01)):
    """base + hinged door UNDER a scale-carrying intermediate scope.

    The RC-1 ×1000 / ×0.03937 shape: an ancestor's authored scale must ride
    in the geometry (like Genesis's vgeoms) but never leak into the link
    frames the FK composes.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/root"))

    group = UsdGeom.Xform.Define(stage, "/root/group")
    UsdGeom.XformCommonAPI(group.GetPrim()).SetScale(Gf.Vec3f(*ancestor_scale))
    _write_body(stage, "/root/group/base", body_extents=(0.6, 0.6, 1.8))
    _write_body(stage, "/root/group/door", body_extents=(0.05, 0.6, 1.8),
                body_translation=(0.2, 0.0, 0.0))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/group/door/hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/group/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/group/door")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.2, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(120.0)
    stage.GetRootLayer().Save()
    return stage


def test_fk_scaled_ancestor_does_not_inherit_scale(tmp_path):
    """RC-1 #1: the door's FK frame must not inherit the ancestor's scale.

    Old walker: frames[door] composed INSIDE the ×s ancestor frame, so the
    opened door's box was displaced/scaled by s (SUNDVIK's ×1000 → the
    149.8 m union; table_1/drawers_chest ×0.03937).  The rigid frames make
    the door swing about its own anchor, scale folded into the geometry.
    """
    from pxr import Gf

    stage_path = tmp_path / "scaled.usda"
    _write_scaled_group_stage(stage_path, ancestor_scale=(0.01, 0.01, 0.01))

    scale = 0.01
    closed = usd_link_meshes_at_config(str(stage_path))
    # Closed pose: the authored (scaled) world geometry — extents 0.01× the
    # authored box, the door at x = 0.2·0.01.
    door = closed["door"]
    assert door.bounds[0][0] == pytest.approx(0.2 * scale - 0.05 * scale / 2.0, abs=1e-6)
    assert door.bounds[1][0] == pytest.approx(0.2 * scale + 0.05 * scale / 2.0, abs=1e-6)
    assert door.bounds[1][2] == pytest.approx(1.8 * scale / 2.0, abs=1e-6)

    opened = usd_link_meshes_at_config(str(stage_path), {"hinge": math.pi / 2.0})
    opened_door = opened["door"]
    # The door box (0.05×0.6) rotates to 0.6×0.05 — in the SCALED geometry
    # (extents ×0.01), about the anchor at the door's own origin (the joint
    # frame's anchor coincides with the link origin here).
    assert opened_door.bounds[0][0] == pytest.approx(0.002 - 0.6 * scale / 2.0, abs=1e-6)
    assert opened_door.bounds[1][0] == pytest.approx(0.002 + 0.6 * scale / 2.0, abs=1e-6)
    assert opened_door.bounds[1][1] == pytest.approx(0.05 * scale / 2.0, abs=1e-6)


def test_fk_non_uniform_ancestor_scale(tmp_path):
    """RC-1 #1b: a NON-uniform ancestor scale must not shear/leak into the
    door's FK frame either."""
    stage_path = tmp_path / "scaled_nu.usda"
    _write_scaled_group_stage(stage_path, ancestor_scale=(2.0, 0.5, 1.0))

    meshes = usd_link_meshes_at_config(str(stage_path))
    base = meshes["base"]
    # The base box (0.6³ at the group origin) is scaled by (2, 0.5, 1).
    assert base.bounds[0][0] == pytest.approx(-0.6, abs=1e-6)
    assert base.bounds[1][0] == pytest.approx(0.6, abs=1e-6)
    assert base.bounds[0][1] == pytest.approx(-0.15, abs=1e-6)
    assert base.bounds[1][1] == pytest.approx(0.15, abs=1e-6)
    door = meshes["door"]
    # The door (0.05-thick box at x = 0.2·2 = 0.4) — the scale rides in the
    # points, and the door's rigid frame sits at the scaled translation.
    assert door.bounds[0][0] == pytest.approx(0.4 - 0.05, abs=1e-6)
    assert door.bounds[1][0] == pytest.approx(0.4 + 0.05, abs=1e-6)

    opened = usd_link_meshes_at_config(str(stage_path), {"hinge": math.pi / 2.0})
    opened_door = opened["door"]
    # The door's local box (the closed-pose world box folded by its rigid
    # frame: extents 0.01-free… i.e. 0.05·2 × 0.6·0.5 × 1.8 around the origin)
    # swings 90° about z through the door's own origin → the x/y extents swap.
    assert opened_door.bounds[0][0] == pytest.approx(0.4 - 0.15, abs=1e-6)
    assert opened_door.bounds[1][0] == pytest.approx(0.4 + 0.15, abs=1e-6)
    assert opened_door.bounds[0][1] == pytest.approx(-0.05, abs=1e-6)
    assert opened_door.bounds[1][1] == pytest.approx(0.05, abs=1e-6)


def _write_rotated_rest_stage(path: Path):
    """A door link AUTHORED rotated 180° away from its joint frame.

    The RC-1 #2 shape (door001): the joint frames are identity, the child
    link sits at world (1, 0, 0) rotated 180° about z, its mesh offset
    (0.1, 0, 0) inside the link frame — the old walker substituted
    R(localRot0) (identity) for the authored link rotation, so the offset
    mesh FK'd to (1.1, 0, 0) instead of (0.9, 0, 0)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/root"))

    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 0.6))
    leaf = UsdGeom.Xform.Define(stage, "/root/leaf")
    UsdPhysics.RigidBodyAPI.Apply(leaf.GetPrim())
    xform = UsdGeom.XformCommonAPI(leaf.GetPrim())
    xform.SetTranslate(Gf.Vec3d(1.0, 0.0, 0.0))
    xform.SetRotate(Gf.Vec3f(0.0, 0.0, 180.0))
    # The mesh offset inside the link frame: the 180° authored rotation puts
    # its world centre at (0.9, 0, 0).
    _write_box_mesh(stage, "/root/leaf/panel", (0.05, 0.1, 0.2),
                    translation=(0.1, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/root/leaf/panel"))

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/leaf/hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/leaf")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(1.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(90.0)
    stage.GetRootLayer().Save()
    return stage


def test_fk_keeps_authored_rest_rotation(tmp_path):
    """RC-1 #2: the child's q = 0 pose is its AUTHORED rigid relative pose.

    A link authored 180° away from its joint frame keeps that rotation (the
    old walker substituted R(localRot0) and lost it)."""
    stage_path = tmp_path / "rotated.usda"
    _write_rotated_rest_stage(stage_path)

    closed = usd_link_meshes_at_config(str(stage_path))
    panel = closed["leaf"]
    # World centre (0.9, 0, 0) — the authored 180° rotation applied to the
    # (0.1, 0, 0) offset mesh.
    assert panel.bounds[0][0] == pytest.approx(0.875, abs=1e-6)
    assert panel.bounds[1][0] == pytest.approx(0.925, abs=1e-6)

    opened = usd_link_meshes_at_config(str(stage_path), {"hinge": math.pi / 2.0})
    opened_panel = opened["leaf"]
    # The DOF swings the leaf about the anchor at its own origin (the joint
    # frame's anchor coincides with the link origin): +90° about the leaf's
    # z from the 180° rest → the offset point (0.1, 0, 0) heads to the leaf's
    # −y at (1, −0.1, 0), extents swap to 0.1×0.05.
    assert opened_panel.bounds[0][0] == pytest.approx(0.95, abs=1e-5)
    assert opened_panel.bounds[1][0] == pytest.approx(1.05, abs=1e-5)
    assert opened_panel.bounds[0][1] == pytest.approx(-0.125, abs=1e-5)
    assert opened_panel.bounds[1][1] == pytest.approx(-0.075, abs=1e-5)


def test_fk_link_local_is_the_rigid_frame(tmp_path):
    """link_local=True returns the link's RIGID own-frame box: the scale
    rides in the geometry (Genesis's vgeom split), the frame maps back to
    the world through the runtime link pose only."""
    stage_path = tmp_path / "scaled.usda"
    _write_scaled_group_stage(stage_path, ancestor_scale=(0.01, 0.01, 0.01))

    local = usd_link_meshes_at_config(str(stage_path), {"hinge": math.pi / 2.0}, link_local=True)
    door = local["door"]
    # The door's rigid frame: at its own origin (the scaled translation
    # folded out), the box in the frame carries the extracted scale —
    # extents 0.01×(0.05, 0.6, 1.8) in the AUTHORED arrangement
    # (configuration-independent: the FK transform is skipped).
    assert door.bounds[0][0] == pytest.approx(-0.05 * 0.01 / 2.0, abs=1e-6)
    assert door.bounds[1][0] == pytest.approx(0.05 * 0.01 / 2.0, abs=1e-6)
    assert door.bounds[1][1] == pytest.approx(0.6 * 0.01 / 2.0, abs=1e-6)
    assert door.bounds[1][2] == pytest.approx(1.8 * 0.01 / 2.0, abs=1e-6)


def test_usd_extract_scale_polar_split():
    """The rigid/scale split matches Genesis's extract_scale: a rotated,
    uniformly scaled frame splits into (rotation, uniform scale); a pure
    translation is rigid with unit scale."""
    from unienv_genesis_collection.objects.articulated_object import (
        _usd_extract_scale,
    )

    angle = 0.7
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform = np.eye(4)
    transform[:3, :3] = 1000.0 * rotation
    transform[:3, 3] = (1.0, 2.0, 3.0)
    rigid, scale = _usd_extract_scale(transform)
    assert np.abs(rigid[:3, :3] - rotation).max() < 1e-9
    assert np.allclose(scale, np.diag([1000.0, 1000.0, 1000.0]), atol=1e-6)
    assert np.allclose(rigid[:3, 3], (1.0, 2.0, 3.0))

    # metersPerUnit is a positive uniform factor (Genesis multiplies the whole
    # linear block by it): it can only land in the scale, never in the rigid
    # part (the unit-handling note's claim).
    scaled_transform = np.eye(4)
    scaled_transform[:3, :3] = 0.01 * transform[:3, :3]
    rigid_m, scale_m = _usd_extract_scale(scaled_transform)
    assert np.abs(rigid_m[:3, :3] - rotation).max() < 1e-9
    assert np.allclose(scale_m, np.diag([10.0, 10.0, 10.0]), atol=1e-6)


# ------------------------------------------------------------------
# RC-3 / RC-4 — packaged-stage fixups
# ------------------------------------------------------------------

def _write_colliders_stage(path: Path):
    """Three colliders: an inverted closed shell (the RC-3 crash), a benign
    negative-volume NON-watertight mesh, and a healthy positive-volume one."""
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/root"))
    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 0.6))
    # A hinged child body, so the A5 fixup (which deactivates rigid bodies no
    # joint references) has a joint referencing the base and leaves the stage —
    # and its colliders — alone.
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 0.6),
                body_translation=(0.5, 0.0, 0.0))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.5, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(90.0)

    import trimesh

    def add_mesh(name, vertices, faces):
        mesh = UsdGeom.Mesh.Define(stage, f"/root/base/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        return mesh.GetPrim()

    cube = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    healthy = trimesh.Trimesh(cube.vertices, cube.faces, process=False)
    assert healthy.is_watertight and healthy.volume > 0.0
    add_mesh("healthy", healthy.vertices, healthy.faces)

    # The genuinely-inverted shell: the same cube with reversed winding —
    # welded-watertight, consistent winding, NEGATIVE volume, zero probe hits
    # (the exact microwave_2 failure).
    inverted = trimesh.Trimesh(cube.vertices, cube.faces[:, ::-1], process=False)
    assert inverted.is_watertight and inverted.volume < 0.0
    add_mesh("inverted", inverted.vertices, inverted.faces)

    # Benign: a LOCALLY flipped shell (part of the faces reversed — the
    # refrigerator_fridge shape).  Its manual signed volume is negative, but
    # the winding is inconsistent (and the probe's rays still hit), so the
    # runtime import never crashes on it and the fixup must not flip it.
    flipped_faces = np.asarray(cube.faces).copy()
    flipped_faces[:8] = flipped_faces[:8, ::-1]
    benign = trimesh.Trimesh(cube.vertices, flipped_faces, process=False)
    assert not benign.is_winding_consistent
    tri = cube.vertices[flipped_faces]
    manual_volume = float(
        (np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0).sum()
    )
    assert manual_volume < 0.0
    add_mesh("benign", benign.vertices, benign.faces)
    stage.GetRootLayer().Save()
    return stage


def test_fixup_flips_only_genuinely_inverted_colliders(tmp_path):
    """RC-3: the zero-hit-probe predicate flips the inverted closed shell and
    leaves the benign negative-volume (open) and healthy colliders alone."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage_path = tmp_path / "colliders.usda"
    _write_colliders_stage(stage_path)
    before_stage = Usd.Stage.Open(str(stage_path))  # bound: prims are weak proxies
    before = {
        name: list(
            UsdGeom.Mesh(before_stage.GetPrimAtPath(f"/root/base/{name}"))
            .GetFaceVertexIndicesAttr().Get()
        )
        for name in ("healthy", "inverted", "benign")
    }

    tally = _SCRIPT._fixup_packaged_stage(stage_path, "colliders")
    assert tally["inverted_colliders"] == ["/root/base/inverted"]
    assert tally["unjointed_bodies"] == []  # nothing else touched

    stage = Usd.Stage.Open(str(stage_path))
    flipped = np.asarray(
        UsdGeom.Mesh(stage.GetPrimAtPath("/root/base/inverted"))
        .GetFaceVertexIndicesAttr().Get(), dtype=np.int64
    ).reshape(-1, 3)
    # Each face's winding is reversed (the triangle's own vertex order).
    assert flipped.tolist() == np.asarray(before["inverted"]).reshape(-1, 3)[:, ::-1].tolist()
    # The healthy and benign colliders keep their winding.
    assert list(UsdGeom.Mesh(stage.GetPrimAtPath("/root/base/healthy"))
                .GetFaceVertexIndicesAttr().Get()) == before["healthy"]
    assert list(UsdGeom.Mesh(stage.GetPrimAtPath("/root/base/benign"))
                .GetFaceVertexIndicesAttr().Get()) == before["benign"]

    # Idempotent: a second run is a no-op (the flipped shell probes fine now).
    second = _SCRIPT._fixup_packaged_stage(stage_path, "colliders")
    assert second["inverted_colliders"] == []


def _write_world_welded_stage(path: Path):
    """The RC-4 shape: a main root body, a hinged knob — and a DOOR jointed
    to the WORLD (empty body0, localPos0 = the door's world position)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/root"))

    _write_body(stage, "/root/body", body_extents=(0.6, 0.6, 0.6))
    # The orphan door: body at world (0.5, 0, 0), hinge anchored at its own
    # origin, WELDED TO THE WORLD (empty body0).
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 0.6),
                body_translation=(0.5, 0.0, 0.0))
    orphan = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/orphan_hinge")
    orphan.CreateBody0Rel().SetTargets([])  # world
    orphan.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
    orphan.CreateLocalPos0Attr(Gf.Vec3f(0.5, 0.0, 0.0))  # world frame
    orphan.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    orphan.CreateAxisAttr("Z")
    orphan.CreateLowerLimitAttr(0.0)
    orphan.CreateUpperLimitAttr(90.0)
    # A knob referencing the MAIN root as body0 (the root disambiguator).
    _write_body(stage, "/root/knob", body_extents=(0.1, 0.1, 0.1),
                body_translation=(0.0, 0.4, 0.0))
    knob = UsdPhysics.RevoluteJoint.Define(stage, "/root/knob/knob_hinge")
    knob.CreateBody0Rel().SetTargets([Sdf.Path("/root/body")])
    knob.CreateBody1Rel().SetTargets([Sdf.Path("/root/knob")])
    knob.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.4, 0.0))
    knob.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    knob.CreateAxisAttr("Z")
    knob.CreateLowerLimitAttr(0.0)
    knob.CreateUpperLimitAttr(90.0)
    stage.GetRootLayer().Save()
    return stage


def test_fixup_reroots_world_welded_orphan_joints(tmp_path):
    """RC-4: a body0-less joint on a NON-root body is re-rooted onto the main
    articulation root, keeping the joint frame's authored world placement."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from unienv_genesis_collection.objects.articulated_object import _usd_joint_frame

    stage_path = tmp_path / "welded.usda"
    _write_world_welded_stage(stage_path)
    before_stage = Usd.Stage.Open(str(stage_path))
    orphan = UsdPhysics.Joint(before_stage.GetPrimAtPath("/root/door/orphan_hinge"))
    pos0_before = list(orphan.GetLocalPos0Attr().Get())

    tally = _SCRIPT._fixup_packaged_stage(stage_path, "welded")
    assert tally["rerooted_joints"] == ["/root/door/orphan_hinge"]

    stage = Usd.Stage.Open(str(stage_path))
    joint = UsdPhysics.Joint(stage.GetPrimAtPath("/root/door/orphan_hinge"))
    # body0 now targets the main root; body1 untouched.
    assert [str(t) for t in joint.GetBody0Rel().GetTargets()] == ["/root/body"]
    assert [str(t) for t in joint.GetBody1Rel().GetTargets()] == ["/root/door"]
    # localPos0 re-expressed: world (0.5, 0, 0) in the root's frame — the root
    # sits at the origin unrotated, so the frame's WORLD placement is exact.
    pos0 = np.asarray(joint.GetLocalPos0Attr().Get(), dtype=float)
    assert pos0 == pytest.approx(pos0_before, abs=1e-6)
    frame_world = (
        UsdGeom.XformCache(Usd.TimeCode.Default())
        .GetLocalToWorldTransform(stage.GetPrimAtPath("/root/body"))
    )
    joint_frame_world = _usd_joint_frame(pos0, joint.GetLocalRot0Attr().Get())
    placed = np.array([
        [frame_world[i][j] for j in range(4)] for i in range(4)
    ]).T @ joint_frame_world
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    door_matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/root/door"))
    door_world = np.array([
        [door_matrix[i][j] for j in range(4)] for i in range(4)
    ]).T
    assert placed[:3, 3] == pytest.approx(door_world[:3, 3], abs=1e-6)

    # Idempotent: a second run is a no-op.
    second = _SCRIPT._fixup_packaged_stage(stage_path, "welded")
    assert second["rerooted_joints"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fixup_world_welded_stage_loads_as_single_entity_gpu(assets_dir, tmp_path, monkeypatch):
    """GPU smoke for RC-4: the re-rooted stage builds ONE entity (the
    pre-fix stage raised ``ArticulationError: split into 2 entities``)."""
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)

    # Package the synthetic stage into the assets dir (the repackage applies
    # the fixups through repackage_object → _fixup_packaged_stage).
    source = tmp_path / "snap"
    upstream = source / "Articulated_objects" / "small_furniture" / "door" / "door_x"
    (upstream / "resource" / "img").mkdir(parents=True)
    _write_world_welded_stage(upstream / "model_door_x.usda")
    (upstream / "resource" / "img" / "texture.png").write_bytes(b"png")
    out_root = assets_dir / ARTVIP_ASSETS_ROOT
    packaged = _SCRIPT.repackage_object(upstream, "welded_door", out_root)
    assert (packaged / "welded_door.usd").is_file()

    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog = json.loads((out_root / "catalog.json").read_text())
    catalog["welded_door"] = {"language_annotations": ["a door"], "tags": ["articulated"]}
    (out_root / "catalog.json").write_text(json.dumps(catalog))
    catalog_module._CACHE.clear()

    item = ArtVIPFurniture("welded_door")
    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(type("WS", (), {"scene": scene})(), pos=(0.0, 0.0, 0.0))
        scene.build(n_envs=0)
        # ONE entity: the world-welded door did not split off.
        assert len(entity.links) == 3
        moving = [j for j in entity.joints if int(j.n_dofs) > 0]
        assert len(moving) == 2
        # The offline walker now parents the door onto the root as well.
        meshes = usd_link_meshes_at_config(str(item.description_file))
        assert set(meshes) == {"body", "door", "knob"}
    finally:
        scene.destroy()
        catalog_module._CACHE.clear()
        torch.set_default_device("cpu")


# ------------------------------------------------------------------
# Worker hardening — an unknown id is a per-object drop, not a batch kill
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_sanity_gate_records_unknown_id_as_drop(fake_fridge, no_download):
    """One unknown id must not kill the batch: it is recorded as that
    object's drop (the KeyError used to escape ``run_sanity_gate`` and kill
    the worker, taking every other object's record with it)."""
    # The gate (and the descriptor) hard-code the packaged ``<id>.usd`` root;
    # the fixture authors ``<id>.usda`` — ship the .usd root too.
    shutil.copy2(
        fake_fridge / "refrigerator_fridge.usda",
        fake_fridge / "refrigerator_fridge.usd",
    )
    # The gate drives descriptors from *output_dir* — the packaged tree root
    # (the parent of ``objects/artvip``, i.e. the fixture's assets dir).
    kept, passed, dropped = _SCRIPT.run_sanity_gate(
        fake_fridge.parent.parent.parent,
        ["definitely_not_a_packaged_id", "refrigerator_fridge"],
        render_dir=None,
    )
    assert kept == ["refrigerator_fridge"]
    assert [record["object_id"] for record in dropped] == [
        "definitely_not_a_packaged_id"
    ]
    assert "KeyError" in dropped[0]["error"]
    assert passed and passed[0]["object_id"] == "refrigerator_fridge"
