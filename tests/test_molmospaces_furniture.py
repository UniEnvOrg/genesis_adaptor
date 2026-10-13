"""Tests for the MolmoSpaces USD furniture integration (no asset downloads).

Mirrors ``tests/test_artvip_furniture.py`` with **synthetic USD stages**
authored with ``pxr`` in ``tmp_path`` — plus the family's critical
converter tests (the joint-body0/body1 swap on synthetic joints):

  - Body-swap correctness: relationship targets swapped AND the per-side
    ``localPos0/1``/``localRot0/1`` frames re-attributed (the values must
    travel with their body), fixed joints included, no-op on non-joints,
    idempotence (swapping twice restores the original).
  - Catalog integrity + tag/contains filtering (entries generated through
    the prepare script's ``annotate_object`` template path).
  - list_molmospaces_objects / make_molmospaces_pool selection, sampling.
  - MolmoSpacesFurniture construction (paths, scalar scale enforcement) and
    asset completeness (_ensure_assets, partial-download repair).
  - Offline FK + clean-name sanitisation (shared ArtVIP machinery).
  - scripts/prepare_molmospaces_assets.py: variant pick (``_mesh`` first),
    swap-in-repackage, stage rename, layer-closure failure, family
    templates.
  - RoomBuilder._resolve_descriptor ``source="molmospaces"`` branch.

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
from unienv_genesis_collection.objects.hub_objects.articulated.molmospaces_furniture import (
    MOLMOSPACES_ASSETS_ROOT,
    MolmoSpacesFurniture,
    get_molmospaces_catalog,
    list_molmospaces_objects,
    make_molmospaces_pool,
)
from unienv_genesis_collection.scenes.room_builder import (
    FurniturePlacement,
    RoomBuilder,
    RoomLayoutSpec,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_molmospaces_assets", _SCRIPTS_DIR / "prepare_molmospaces_assets.py",
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


def _write_box_mesh(stage, path: str, extents, translation=None, rotation_z=None):
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
    if (translation, rotation_z) != (None, None):
        xform = UsdGeom.Xform(mesh.GetPrim())
        if rotation_z is not None:
            xform.AddRotateZOp().Set(float(rotation_z))
        if translation is not None:
            xform.AddTranslateOp().Set(Gf.Vec3f(*translation))
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


def _write_molmo_stage(path: Path, *, inverted: bool = False):
    """A MolmoSpaces-shaped stage: base + hinged door + sliding drawer + welded shelf.

    With ``inverted=True`` the joints are authored in the *upstream*
    MolmoSpaces convention (body0 = child, body1 = parent) — the state the
    packaging swap must fix.  With ``inverted=False`` the stage is already
    Genesis-ordered (the post-swap shape the descriptor loads).

    The **shelf** is a fourth body welded to the door by a
    ``UsdPhysics.FixedJoint`` (upstream exports weld fixed decoration this
    way): it has no DOF of its own, but the FK walker must still place it
    through the joint-frames equation so it rides the door and lands in the
    union bbox.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    root_scope = UsdGeom.Scope.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(root_scope.GetPrim())

    # A UsdPreviewSurface material referencing ./Textures/Albedo.png — the
    # MolmoSpaces convention (100 % UsdPreviewSurface, relative textures);
    # the required-file closure must name the texture even when it is absent.
    material = UsdShade.Material.Define(stage, "/root/Materials/material_Metal_Mat")
    texture = UsdShade.Shader.Define(stage, "/root/Materials/material_Metal_Mat/tex")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./Textures/Albedo.png")
    )
    material.CreateSurfaceOutput().ConnectToSource(texture.ConnectableAPI(), "rgb")

    _write_body(stage, "/root/base", body_extents=(0.6, 0.6, 1.8),
                material_path=material.GetPath())
    _write_body(stage, "/root/door", body_extents=(0.05, 0.6, 1.8),
                body_translation=(0.2, 0.0, 0.0))
    door_joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/door_1")
    # Hinge anchor at the door's left edge: (0.2, 0, 0) in the *base* frame,
    # (0, 0, 0) in the door's own frame.  Inverted stages author
    # body0 = door (child) with its frame in localPos0 — the state the
    # packaging swap must fix; Genesis-ordered stages author
    # body0 = base (parent).
    if inverted:
        door_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/door")])
        door_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/base")])
        door_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        door_joint.CreateLocalPos1Attr(Gf.Vec3f(0.2, 0.0, 0.0))
    else:
        door_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
        door_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
        door_joint.CreateLocalPos0Attr(Gf.Vec3f(0.2, 0.0, 0.0))
        door_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    door_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    door_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    door_joint.CreateAxisAttr("Z")
    door_joint.CreateLowerLimitAttr(0.0)
    door_joint.CreateUpperLimitAttr(120.0)

    _write_body(stage, "/root/drawer", body_extents=(0.5, 0.4, 0.2),
                body_translation=(0.0, 0.0, 1.2))
    drawer_joint = UsdPhysics.PrismaticJoint.Define(stage, "/root/drawer/drawer_1")
    if inverted:
        drawer_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/drawer")])
        drawer_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/base")])
        drawer_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 1.2))
        drawer_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    else:
        drawer_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/base")])
        drawer_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/drawer")])
        drawer_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 1.2))
        drawer_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    drawer_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    drawer_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    drawer_joint.CreateAxisAttr("Y")
    drawer_joint.CreateLowerLimitAttr(-0.4)
    drawer_joint.CreateUpperLimitAttr(0.0)

    # A third body welded to the door by a UsdPhysics.FixedJoint: no DOF of
    # its own, but the FK walker's fixed-weld branch must place it through
    # the joint-frames equation (frame composition, no displacement) so it
    # follows the door and its box lands in the FK union bbox.  The joint
    # frames reproduce the shelf's authored pose at q = 0, like the two
    # moving joints.
    _write_body(stage, "/root/shelf", body_extents=(0.4, 0.1, 0.5),
                body_translation=(0.4, 0.1, 0.7))
    shelf_joint = UsdPhysics.FixedJoint.Define(stage, "/root/shelf/shelf_weld")
    if inverted:
        shelf_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/shelf")])
        shelf_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/door")])
        shelf_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        shelf_joint.CreateLocalPos1Attr(Gf.Vec3f(0.2, 0.1, 0.7))
    else:
        shelf_joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/door")])
        shelf_joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/shelf")])
        shelf_joint.CreateLocalPos0Attr(Gf.Vec3f(0.2, 0.1, 0.7))
        shelf_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    shelf_joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    shelf_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    stage.GetRootLayer().Save()
    return stage


def _make_fake_object_dir(assets_dir: Path, object_id: str, *, inverted: bool = False) -> Path:
    """A complete fake MolmoSpaces object directory (synthetic stage)."""
    obj_dir = assets_dir / MOLMOSPACES_ASSETS_ROOT / object_id
    obj_dir.mkdir(parents=True)
    _write_molmo_stage(obj_dir / f"{object_id}.usda", inverted=inverted)
    # The texture the stage references through ./Textures/Albedo.png.
    textures = obj_dir / "Textures"
    textures.mkdir(parents=True, exist_ok=True)
    (textures / "Albedo.png").write_bytes(b"png placeholder")
    return obj_dir


def _catalog_entry(upstream: Path, *, n_revolute: int = 1, n_prismatic: int = 1):
    """A catalog entry generated through the packaging templates."""
    entry = _SCRIPT.annotate_object(upstream.name, n_revolute, n_prismatic)[1]
    entry["tags"] = sorted(set(entry["tags"]) | {"articulated"})
    return entry


def _make_catalog(assets_dir: Path, entries: dict) -> None:
    catalog_file = assets_dir / MOLMOSPACES_ASSETS_ROOT / "catalog.json"
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    catalog_file.write_text(json.dumps(entries))


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Temp assets dir with a generated (template-path) catalog."""
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    upstream = tmp_path / "upstream" / "Fridge_26_mesh"
    upstream.mkdir(parents=True)
    _make_catalog(assets_dir=tmp_path, entries={
        "fridge_26": _catalog_entry(upstream),
        "microwave_12": _catalog_entry(
            tmp_path / "upstream" / "Microwave_12_mesh", n_revolute=1, n_prismatic=0,
        ),
        "ikeacabinet_1": _catalog_entry(
            tmp_path / "upstream" / "IKEACabinet_1_mesh", n_revolute=2, n_prismatic=1,
        ),
        "laptop_6": _catalog_entry(
            tmp_path / "upstream" / "Laptop_6_mesh", n_revolute=1, n_prismatic=0,
        ),
    })
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    catalog_module._CACHE.clear()
    return tmp_path


@pytest.fixture
def fake_fridge(assets_dir):
    """A complete fake 'fridge_26' object directory (synthetic .usda)."""
    return _make_fake_object_dir(assets_dir, "fridge_26")


@pytest.fixture
def no_download(monkeypatch):
    calls = []

    def _fail(**kwargs):
        calls.append(kwargs)
        raise AssertionError(f"Unexpected download attempt: {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail)
    return calls


# ------------------------------------------------------------------
# The joint-body swap converter
# ------------------------------------------------------------------

def test_swap_flips_targets_and_local_frames(tmp_path):
    """body0 ⇄ body1 targets AND localPos0/1 + localRot0/1 values swap."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from unienv_genesis_collection.utils.usd_conversions import (
        swap_usd_joint_bodies,
    )

    # A synthetic MolmoSpaces-style stage: body0 = child, body1 = parent,
    # the child's hinge offset authored in localPos0 (the child-side frame).
    stage = Usd.Stage.CreateNew(str(tmp_path / "molmo.usda"))
    stage.SetMetadata("metersPerUnit", 1.0)
    root = UsdGeom.Xform.Define(stage, "/root")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    _write_body(stage, "/root/base", body_extents=(0.4, 0.4, 0.4))
    _write_body(stage, "/root/door", body_extents=(0.05, 0.4, 0.4))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/root/door/joint_1")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/root/door")])   # child (upstream)
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/root/base")])   # parent
    joint.CreateLocalPos0Attr(Gf.Vec3f(1.0, 2.0, 3.0))            # child-side frame
    joint.CreateLocalPos1Attr(Gf.Vec3f(4.0, 5.0, 6.0))            # parent-side frame
    joint.CreateLocalRot0Attr(Gf.Quatf(0.9238795, Gf.Vec3f(0.0, 0.0, 0.38268343)))
    joint.CreateLocalRot1Attr(Gf.Quatf(0.7071068, Gf.Vec3f(0.7071068, 0.0, 0.0)))
    joint.CreateAxisAttr("Z")
    stage.GetRootLayer().Save()

    assert swap_usd_joint_bodies(tmp_path / "molmo.usda") == 1

    reopened = Usd.Stage.Open(str(tmp_path / "molmo.usda"))
    joint = UsdPhysics.RevoluteJoint(reopened.GetPrimAtPath("/root/door/joint_1"))
    assert [str(t) for t in joint.GetBody0Rel().GetTargets()] == ["/root/base"]
    assert [str(t) for t in joint.GetBody1Rel().GetTargets()] == ["/root/door"]
    # The frames travelled with their bodies: localPos0 now carries the
    # *parent-side* offset (the old localPos1 value).
    assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(4.0, 5.0, 6.0)
    assert joint.GetLocalPos1Attr().Get() == Gf.Vec3f(1.0, 2.0, 3.0)
    assert joint.GetLocalRot0Attr().Get() == Gf.Quatf(0.7071068, Gf.Vec3f(0.7071068, 0.0, 0.0))
    assert joint.GetLocalRot1Attr().Get() == Gf.Quatf(0.9238795, Gf.Vec3f(0.0, 0.0, 0.38268343))
    # The DOF axis/limits are untouched (joint-frame symmetric).
    assert joint.GetAxisAttr().Get() == "Z"


def test_swap_is_a_no_op_on_non_joints_and_defaults(tmp_path):
    """Meshes are untouched; joints without body relationships pass through."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from unienv_genesis_collection.utils.usd_conversions import (
        swap_usd_joint_bodies,
    )

    stage = Usd.Stage.CreateNew(str(tmp_path / "stage.usda"))
    stage.SetMetadata("metersPerUnit", 1.0)
    _write_body(stage, "/root/base", body_extents=(0.4, 0.4, 0.4))
    # A joint with no body relationships authored at all.
    orphan = UsdPhysics.RevoluteJoint.Define(stage, "/root/orphan_joint")
    orphan.CreateAxisAttr("X")
    stage.GetRootLayer().Save()

    assert swap_usd_joint_bodies(tmp_path / "stage.usda") == 1  # the orphan only

    reopened = Usd.Stage.Open(str(tmp_path / "stage.usda"))
    orphan = reopened.GetPrimAtPath("/root/orphan_joint")
    assert not orphan.GetAttribute("physics:body0").HasValue()
    assert orphan.GetAttribute("physics:axis").Get() == "X"
    # The mesh survives byte-identical (its points were never touched).
    mesh = reopened.GetPrimAtPath("/root/base/box")
    points = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
    assert len(points) == 8


def test_swap_is_idempotent_in_pairs(tmp_path):
    """Swapping twice restores the original (MolmoSpaces round trip)."""
    from pxr import Usd, UsdPhysics
    from unienv_genesis_collection.utils.usd_conversions import (
        swap_usd_joint_bodies,
    )

    path = tmp_path / "molmo.usda"
    _write_molmo_stage(path, inverted=True)

    first = swap_usd_joint_bodies(path)
    stage = Usd.Stage.Open(str(path))
    joint = [
        UsdPhysics.Joint(prim) for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    ][0]
    after_one = ([str(t) for t in joint.GetBody0Rel().GetTargets()],
                 joint.GetLocalPos0Attr().Get())
    stage = None

    second = swap_usd_joint_bodies(path)
    stage = Usd.Stage.Open(str(path))
    joint = [
        UsdPhysics.Joint(prim) for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    ][0]
    after_two = ([str(t) for t in joint.GetBody0Rel().GetTargets()],
                 joint.GetLocalPos0Attr().Get())

    assert first == second == 3  # revolute + prismatic + the fixed weld
    assert after_one != after_two  # the door's hinge offset moved sides


def test_post_swap_stage_matches_the_verified_convention(fake_fridge):
    """The synthetic post-swap stage loads like the probe's swapped samples.

    MolmoSpaces samples load 18/18 single-entity after the swap with the
    *base* link being the body that is never a joint child — mirrored here
    through the offline FK walker.
    """
    item = MolmoSpacesFurniture("fridge_26")
    meshes = item._link_meshes_at_reference()
    assert set(meshes) == {"base", "door", "drawer", "shelf"}


# ------------------------------------------------------------------
# Catalog
# ------------------------------------------------------------------

def test_catalog_integrity(assets_dir):
    catalog = get_molmospaces_catalog()
    assert len(catalog) == 4
    for object_id, info in catalog.items():
        assert info.object_id == object_id
        assert " " not in object_id
        assert len(info.language_annotations) >= 2
        assert info.tags
        assert {"articulated"} <= info.tags
        assert info.language_annotations[0] in ("appliance", "furniture",
                                                "household item", "object")
        for tag in info.tags:
            assert tag == tag.lower() and " " not in tag
        assert info.standing_quaternion is None  # MolmoSpaces stages are Z-up


def test_catalog_detailed_annotation_carries_joint_counts(assets_dir):
    catalog = get_molmospaces_catalog()
    assert "hinged door" in catalog["fridge_26"].language_annotations[-1]
    cabinet = catalog["ikeacabinet_1"]
    assert cabinet.language_annotations[-1].startswith("a cabinet with")


def test_catalog_is_cached(assets_dir, no_download):
    assert get_molmospaces_catalog() is get_molmospaces_catalog()


def test_list_all_objects_sorted(assets_dir):
    ids = list_molmospaces_objects()
    assert ids == sorted(ids)
    assert ids == ["fridge_26", "ikeacabinet_1", "laptop_6", "microwave_12"]


def test_list_filter_by_tags_and_contains(assets_dir):
    # Room hints flow into tags through the family templates.
    kitchen = list_molmospaces_objects(tags="kitchen")
    assert "fridge_26" in kitchen
    assert "laptop_6" not in kitchen  # office hint, not kitchen
    assert list_molmospaces_objects(contains="FRIDGE") == ["fridge_26"]
    assert "fridge_26" in list_molmospaces_objects(tags="kitchen", contains="fridge")
    assert list_molmospaces_objects(tags="no_such_tag") == []


def test_unknown_id_raises(assets_dir):
    with pytest.raises(KeyError, match="not_an_object"):
        MolmoSpacesFurniture("not_an_object")


# ------------------------------------------------------------------
# make_molmospaces_pool
# ------------------------------------------------------------------

def test_pool_explicit_ids_order_preserved(assets_dir, no_download):
    pool = make_molmospaces_pool(["fridge_26", "laptop_6"])
    assert [o.object_id for o in pool] == ["fridge_26", "laptop_6"]


def test_pool_ids_and_tags_mutually_exclusive(assets_dir):
    with pytest.raises(ValueError, match="not both"):
        make_molmospaces_pool(["fridge_26"], tags="furniture")


def test_pool_sampling_reproducible(assets_dir, no_download):
    pool_a = make_molmospaces_pool(
        tags="articulated", n=3, rng=torch.Generator().manual_seed(42)
    )
    pool_b = make_molmospaces_pool(
        tags="articulated", n=3, rng=torch.Generator().manual_seed(42)
    )
    ids_a = [o.object_id for o in pool_a]
    assert ids_a == [o.object_id for o in pool_b]
    assert len(set(ids_a)) == 3  # distinct (sampling without replacement)


def test_pool_n_too_large(assets_dir):
    with pytest.raises(ValueError, match="only 2"):
        make_molmospaces_pool(["fridge_26", "laptop_6"], n=3)


def test_pool_kwargs_forwarded(assets_dir, no_download):
    pool = make_molmospaces_pool(["fridge_26"], scale=0.5, fixed=True)
    assert pool[0].scale == 0.5
    assert pool[0]._morph_scale == 0.5
    assert pool[0].fixed is True


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_construction_downloads_no_object_assets(assets_dir, no_download):
    item = MolmoSpacesFurniture("fridge_26")
    assert item.object_id == "fridge_26"
    assert item.name == "fridge_26"
    assert item.language_annotations == tuple(
        get_molmospaces_catalog()["fridge_26"].language_annotations
    )
    assert item.fixed is True
    assert item.reference_qpos == {}
    assert not no_download


def test_paths_under_assets_dir(assets_dir):
    item = MolmoSpacesFurniture("fridge_26")
    expected_dir = assets_dir / MOLMOSPACES_ASSETS_ROOT / "fridge_26"
    assert item.asset_dir == expected_dir
    assert item.description_file == expected_dir / "fridge_26.usda"


def test_overrides(assets_dir):
    item = MolmoSpacesFurniture(
        "fridge_26", name="my_fridge", language_annotations=["the fridge"],
        standing_quaternion=(0.0, 0.0, 0.0, 1.0),
    )
    assert item.name == "my_fridge"
    assert item.language_annotations == ("the fridge",)
    assert item.standing_quaternion == (0.0, 0.0, 0.0, 1.0)


def test_scalar_scale_only(assets_dir):
    with pytest.raises(ValueError, match="uniform"):
        MolmoSpacesFurniture("fridge_26", scale=(1.0, 2.0, 3.0))


# ------------------------------------------------------------------
# Asset completeness
# ------------------------------------------------------------------

def test_required_files_usda_branch(fake_fridge):
    item = MolmoSpacesFurniture("fridge_26")
    required = item._required_files()
    assert required[0] == item.description_file
    # The texture the synthetic stage references through ./Textures/.
    assert set(required) == {
        item.description_file,
        item.asset_dir / "Textures" / "Albedo.png",
    }


def test_ensure_assets_complete_no_download(fake_fridge, no_download):
    item = MolmoSpacesFurniture("fridge_26")
    item._ensure_assets()
    assert item._assets_ready


def test_ensure_assets_partial_dir_triggers_redownload(assets_dir, monkeypatch):
    _make_fake_object_dir(assets_dir, "fridge_26")
    (assets_dir / MOLMOSPACES_ASSETS_ROOT / "fridge_26" / "Textures"
     / "Albedo.png").unlink()
    calls: list = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        (assets_dir / MOLMOSPACES_ASSETS_ROOT / "fridge_26" / "Textures"
         / "Albedo.png").write_bytes(b"png")

    monkeypatch.setattr(asset_path, "download_assets", _fake_download)
    item = MolmoSpacesFurniture("fridge_26")
    item._ensure_assets()
    assert calls and calls[0]["allow_patterns"] == f"{MOLMOSPACES_ASSETS_ROOT}/fridge_26/**"
    assert item._assets_ready


# ------------------------------------------------------------------
# Offline FK + clean names (shared ArtVIP machinery)
# ------------------------------------------------------------------

_BASE_BOUNDS = ([-0.3, -0.3, -0.9], [0.3, 0.3, 0.9])
_DOOR_CLOSED_BOUNDS = ([0.175, -0.3, -0.9], [0.225, 0.3, 0.9])
_DRAWER_CLOSED_BOUNDS = ([-0.25, -0.2, 1.1], [0.25, 0.2, 1.3])
# The FixedJoint-welded shelf: authored pose (q = 0) and 90°-open door.
_SHELF_CLOSED_BOUNDS = ([0.2, 0.05, 0.45], [0.6, 0.15, 0.95])
_SHELF_OPEN_BOUNDS = ([0.05, 0.0, 0.45], [0.15, 0.4, 0.95])


def _host(tensor):
    return tensor.detach().to("cpu") if tensor.is_cuda else tensor


def test_fk_closed_pose_matches_authored_extents(fake_fridge):
    item = MolmoSpacesFurniture("fridge_26")
    meshes = item._link_meshes_at_reference()
    assert set(meshes) == {"base", "door", "drawer", "shelf"}
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


def test_fk_reference_qpos_moves_links_with_clean_names(fake_fridge):
    item = MolmoSpacesFurniture(
        "fridge_26", reference_qpos={"door_1": math.pi / 2.0, "drawer_1": 0.1},
    )
    meshes = item._link_meshes_at_reference()
    door = meshes["door"]
    assert torch.allclose(
        _host(torch.tensor(door.bounds, dtype=torch.float32)),
        _host(torch.tensor([[-0.1, -0.025, -0.9], [0.5, 0.025, 0.9]], dtype=torch.float32)),
        atol=1e-5,
    )
    drawer = meshes["drawer"]
    assert torch.allclose(
        _host(torch.tensor(drawer.bounds, dtype=torch.float32)),
        _host(torch.tensor([[-0.25, -0.1, 1.1], [0.25, 0.3, 1.3]], dtype=torch.float32)),
        atol=1e-5,
    )


def test_fk_fixed_weld_body_is_placed_and_in_the_union_bbox(fake_fridge):
    """The ``UsdPhysics.FixedJoint``-welded body joins the FK (no DOF of its own).

    The fixed-weld branch composes the joint frames with a zero displacement,
    so the welded shelf reproduces its authored pose at ``q = 0`` (the FK
    rest frame), rides the door through a 90° opening, and its box is part of
    the FK union bbox the descriptor exposes as ``get_local_bounding_box``.
    """
    item = MolmoSpacesFurniture("fridge_26")
    meshes = item._link_meshes_at_reference()
    assert "shelf" in meshes  # a link, not a moving joint
    assert torch.allclose(
        _host(torch.tensor(meshes["shelf"].bounds, dtype=torch.float32)),
        _host(torch.tensor(_SHELF_CLOSED_BOUNDS, dtype=torch.float32)),
        atol=1e-5,
    )
    # The FK union bbox includes the welded body: the shelf's +x edge is the
    # furthest point of the whole union (base/door/drawer stop short of it).
    local = item.get_local_bounding_box()
    low = _host(local.low).float()
    high = _host(local.high).float()
    shelf_low, shelf_high = (
        torch.tensor(bound, dtype=torch.float32) for bound in _SHELF_CLOSED_BOUNDS
    )
    assert bool(torch.all(low <= shelf_low + 1e-5))
    assert bool(torch.all(high >= shelf_high - 1e-5))
    assert float(high[0]) == pytest.approx(_SHELF_CLOSED_BOUNDS[1][0], abs=1e-5)

    # Opened 90°: the welded body follows the door (the joint-frame
    # composition of the fixed-weld branch, not its authored pose).
    opened = usd_link_meshes_at_config(
        str(item.description_file), {"door_1": math.pi / 2.0},
    )
    assert torch.allclose(
        _host(torch.tensor(opened["shelf"].bounds, dtype=torch.float32)),
        _host(torch.tensor(_SHELF_OPEN_BOUNDS, dtype=torch.float32)),
        atol=1e-5,
    )


def test_fk_unknown_joint_raises_listing_clean_names(fake_fridge):
    with pytest.raises(ArticulationError) as exc_info:
        usd_link_meshes_at_config(
            str(fake_fridge / "fridge_26.usda"), {"no_such_joint": 0.1}
        )
    assert "door_1" in str(exc_info.value)
    assert "drawer_1" in str(exc_info.value)


def test_clean_name_translation(assets_dir, fake_fridge):
    item = MolmoSpacesFurniture("fridge_26")
    assert item._to_runtime_name("door_1", "joint") == "/root/door/door_1"
    assert item._to_runtime_name("/root/door/door_1", "joint") == "/root/door/door_1"
    assert item.clean_name_of("/root/door/door_1") == "door_1"
    with pytest.raises(ArticulationError, match="no_such_joint"):
        item._to_runtime_name("no_such_joint", "joint")
    assert set(item.list_joint_names()) == {"door_1", "drawer_1"}
    assert set(item.list_link_names()) == {"base", "door", "drawer", "shelf"}


# ------------------------------------------------------------------
# RoomBuilder descriptor resolution
# ------------------------------------------------------------------

def test_room_builder_resolves_molmospaces_source(assets_dir, fake_fridge, no_download):
    world = type("WorldStub", (), {"batch_size": None})()
    room = RoomBuilder(
        world,
        name="room",
        layout=RoomLayoutSpec(room_size=(4.0, 4.0, 2.7), furniture=()),
    )
    placement = FurniturePlacement(
        object_id="fridge_26",
        source="molmospaces",
        pos=(0.0, 0.0, 0.0),
        yaw=0.0,
    )
    descriptor = room._resolve_descriptor(placement)
    assert isinstance(descriptor, MolmoSpacesFurniture)
    assert descriptor.object_id == "fridge_26"

    with pytest.raises(ValueError, match="molmospaces"):
        room._resolve_descriptor(FurniturePlacement(
            object_id="x", source="unsuppo", pos=(0, 0, 0), yaw=0.0,
        ))


# ------------------------------------------------------------------
# prepare_molmospaces_assets.py
# ------------------------------------------------------------------

def _make_upstream_variant_dir(source: Path, name: str) -> Path:
    """One upstream variant dir: <Name>/<Name>.usda + Textures/.

    The upstream MolmoSpaces stage is authored *inverted* (body0 = child) —
    the swap in repackage turns it Genesis-ordered.
    """
    variant_dir = source / name
    variant_dir.mkdir(parents=True)
    _write_molmo_stage(variant_dir / f"{name}.usda", inverted=True)
    textures = variant_dir / "Textures"
    textures.mkdir(parents=True, exist_ok=True)
    (textures / "Albedo.png").write_bytes(b"png placeholder")
    return variant_dir


def test_repackage_picks_mesh_variant_and_swaps(tmp_path):
    source = tmp_path / "snapshot"
    _make_upstream_variant_dir(source, "Fridge_26")
    _make_upstream_variant_dir(source, "Fridge_26_prim")
    mesh_dir = _make_upstream_variant_dir(source, "Fridge_26_mesh")

    out_root = tmp_path / "out" / "objects" / "molmospaces"
    packaged, n_swapped = _SCRIPT.repackage_object(mesh_dir, "fridge_26", out_root)

    names = sorted(str(p.relative_to(packaged)) for p in packaged.rglob("*") if p.is_file())
    assert names == ["Textures/Albedo.png", "fridge_26.usda"]
    # The swap ran on the packaged stage: body0 = the base (parent) now.
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(packaged / "fridge_26.usda"))
    joint = [
        UsdPhysics.Joint(prim) for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    ][0]
    assert [str(t) for t in joint.GetBody0Rel().GetTargets()] == ["/root/base"]
    assert n_swapped >= 1


def test_list_flattens_ids_and_skips_static(tmp_path):
    source = tmp_path / "snapshot"
    _make_upstream_variant_dir(source, "Fridge_26")
    _make_upstream_variant_dir(source, "Fridge_26_mesh")
    _make_upstream_variant_dir(source, "Fridge_26_prim")
    _make_upstream_variant_dir(source, "Microwave_12")
    _make_upstream_variant_dir(source, "IKEACabinet_1_mesh")
    articulated = {"Fridge_26": True, "Microwave_12": False, "IKEACabinet_1": True}
    pairs = dict(_SCRIPT.list_molmo_objects(source, _SCRIPT.ALL_KNOWN_CATEGORIES, articulated))
    # Only the _mesh variant is claimed per object; the flagged-static
    # microwave is skipped.
    assert set(pairs) == {"fridge_26", "ikeacabinet_1"}
    assert pairs["fridge_26"].name.endswith("_mesh")


def test_list_falls_back_when_no_mesh_variant(tmp_path):
    source = tmp_path / "snapshot"
    base_dir = _make_upstream_variant_dir(source, "Toaster_1")
    pairs = dict(_SCRIPT.list_molmo_objects(source, _SCRIPT.ALL_KNOWN_CATEGORIES))
    assert pairs == {"toaster_1": base_dir}


def test_catalog_written_from_gate_records(tmp_path):
    out_root = tmp_path / "objects" / "molmospaces"
    out_root.mkdir(parents=True)
    entry = _SCRIPT.annotate_object("IKEACabinet_1_mesh", 2, 1)[1]
    entry["tags"] = sorted(set(entry["tags"]) | {"articulated"})
    _SCRIPT.write_catalog(out_root, [dict(object_id="ikeacabinet_1", catalog_entry=entry)])

    written = json.loads((out_root / "catalog.json").read_text())
    stored = written["ikeacabinet_1"]
    assert stored["language_annotations"][0] == "furniture"
    assert stored["language_annotations"][-1].startswith("a cabinet with")
    assert {"furniture", "articulated", "kitchen", "cabinet"}.issubset(set(stored["tags"]))


def test_write_catalog_merges_with_existing(tmp_path):
    out_root = tmp_path / "objects" / "molmospaces"
    out_root.mkdir(parents=True)
    entry_a = _SCRIPT.annotate_object("Fridge_26_mesh", 1, 0)[1]
    entry_a["tags"] = sorted(set(entry_a["tags"]) | {"articulated"})
    _SCRIPT.write_catalog(out_root, [dict(object_id="fridge_26", catalog_entry=entry_a)])
    entry_b = _SCRIPT.annotate_object("Microwave_12_mesh", 1, 0)[1]
    entry_b["tags"] = sorted(set(entry_b["tags"]) | {"articulated"})
    _SCRIPT.write_catalog(out_root, [dict(object_id="microwave_12", catalog_entry=entry_b)])
    written = json.loads((out_root / "catalog.json").read_text())
    assert set(written) == {"fridge_26", "microwave_12"}


def test_default_categories_exclude_unknown_families():
    for category in _SCRIPT.DEFAULT_CATEGORIES:
        assert category in _SCRIPT.ALL_KNOWN_CATEGORIES
    # The Objaverse subset (ODC-BY with per-object NC entries) is never in
    # the pack: the fetcher only touches thor_ archives.
    assert not any("objaverse" in c for c in _SCRIPT.ALL_KNOWN_CATEGORIES)


def test_license_and_readme_are_shipped(tmp_path):
    assert "CC BY 4.0" in _SCRIPT.LICENSE_TEXT
    assert "ODC-BY" in _SCRIPT.LICENSE_TEXT  # the exclusion is documented
    assert "joint-body0/body1 swap" in _SCRIPT.PROVENANCE_README


# ------------------------------------------------------------------
# GPU smoke test (skipped offline)
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_into_scene_smoke_gpu(assets_dir, fake_fridge, no_download):
    """Real GPU smoke: load the synthetic (already Genesis-ordered) stage."""
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)

    item = MolmoSpacesFurniture("fridge_26")
    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(type("WorldStub", (), {"scene": scene})())
        scene.build(n_envs=0)
        assert entity is not None

        moving = [j for j in entity.joints if int(j.n_dofs) > 0]
        assert [j.name for j in moving] == ["/root/door/door_1", "/root/drawer/drawer_1"]

        vaabb = entity.get_vAABB()
        local = item.get_local_bounding_box()
        assert torch.allclose(vaabb[0], local.low.to(vaabb.device), atol=2e-3)
        assert torch.allclose(vaabb[1], local.high.to(vaabb.device), atol=2e-3)

        item.set_open_fraction(entity, "door_1", torch.tensor(0.5))
        fraction = item.get_open_fraction(entity, "door_1")
        assert float(fraction.squeeze().cpu()) == pytest.approx(0.5, abs=1e-2)
    finally:
        scene.destroy()
        torch.set_default_device("cpu")


def test_list_skips_marker_only_group_archive_dirs(tmp_path):
    """Group archives extract to a marker-only dir → no phantom object id.

    ``thor_Cabinet.tar.zst``'s members are the *sibling* dirs
    (``Cabinet_Corner_01/``, …); ``extract_root/Cabinet`` holds nothing but
    the ``.extracted`` marker.  Before the fix the scan turned those dirs
    into phantom objects with no ``*.usda`` (repackage raised
    ``FileNotFoundError``); now they never appear, while the real sibling
    objects (and their variant preference) are untouched.
    """
    source = tmp_path / "extract"
    for name in ("Cabinet", "DecorativeBox", "Side_Table_201"):
        marker_dir = source / name
        marker_dir.mkdir(parents=True)
        (marker_dir / ".extracted").write_text("ok\n")
    _make_upstream_variant_dir(source, "Cabinet_Corner_01")
    _make_upstream_variant_dir(source, "Cabinet_Corner_01_mesh")
    _make_upstream_variant_dir(source, "Cabinet_Corner_01_prim")
    _make_upstream_variant_dir(source, "DecorativeBox_A_01_mesh")

    pairs = dict(_SCRIPT.list_molmo_objects(source, _SCRIPT.ALL_KNOWN_CATEGORIES))

    assert set(pairs) == {"cabinet_corner_01", "decorativebox_a_01"}
    # The variant preference is unaffected by the marker-dir skip.
    assert pairs["cabinet_corner_01"].name.endswith("_mesh")
    assert pairs["decorativebox_a_01"].name.endswith("_mesh")


def test_list_skips_empty_extraction_dir(tmp_path):
    """A dir with neither a stage nor a marker is skipped the same way."""
    source = tmp_path / "extract"
    (source / "Half_Extracted").mkdir(parents=True)
    _make_upstream_variant_dir(source, "Fridge_26_mesh")

    pairs = dict(_SCRIPT.list_molmo_objects(source, _SCRIPT.ALL_KNOWN_CATEGORIES))
    assert set(pairs) == {"fridge_26"}


def test_opened_gate_helpers_skip_multi_dof_and_clamp_non_finite():
    """The MolmoSpaces packager's opened-FK helpers match the family contract."""
    moving = [
        dict(name="door", type="REVOLUTE", n_dofs=1),
        dict(name="lid", type="SPHERICAL", n_dofs=3),
    ]
    assert [record["name"] for record in _SCRIPT._opened_reference_joints(moving)] == ["door"]
    clamped = _SCRIPT._finite_midpoint(torch.tensor([0.0]), torch.tensor([float("inf")]))
    assert torch.isfinite(clamped).all() and float(clamped.squeeze()) == 0.0
    midpoint = _SCRIPT._finite_midpoint(torch.tensor([-0.1]), torch.tensor([0.1]))
    assert float(midpoint.squeeze()) == pytest.approx(0.0)
