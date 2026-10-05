"""Unit tests for the shared ``scripts/_packaging`` library.

These cover the reconciled single-source-of-truth pieces that the packager
test suites only exercise through the (CUDA-gated) packager paths:

* ``_packaging.catalog.write_catalog`` — the merge/drop/overrides/summary
  parameter variants (the four former per-family implementations).
* ``_packaging.usd_verify`` — the asset-completeness check and lightwheel's
  missing-texture repair hook.
* ``_packaging.sanity.sanity_check_articulated`` — the gate semantics
  (duplicate-leaf contract, the moving-joint assert + the molmo
  joint-body-swap hint, the closed-bbox cross-check tolerances, the opened
  record shape) driven against a stub Genesis, so they run without CUDA.
* ``_packaging.gate_common`` — the shared helper surface.
* ``_packaging._gate_batch`` — the in-process fallback of the batch driver.

The heavy `import genesis as gs`/pxr pieces inside the gate bodies are
stubbed via ``sys.modules``; the pxr leaf-name walk runs against real
(usd-core) synthetic stages authored in ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_packaging():
    """Load the shared ``scripts/_packaging`` package by path.

    The directory name (``_packaging``) is not importable unless scripts/ is
    on sys.path — the tests load it under the package's own name exactly as
    the packager shims do.
    """
    name = "_packaging"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, _SCRIPTS_DIR / "_packaging" / "__init__.py",
            submodule_search_locations=[str(_SCRIPTS_DIR / "_packaging")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_PACKAGING = _load_packaging()
_catalog = _PACKAGING.catalog
_usd_verify = _PACKAGING.usd_verify
_sanity = _PACKAGING.sanity
_gate_common = _PACKAGING.gate_common
_gate_batch = _PACKAGING._gate_batch


# ---------------------------------------------------------------------------
# write_catalog — the reconciled parameter variants
# ---------------------------------------------------------------------------


def _entry(annotations=("appliance", "a fridge"), tags=("fridge", "kitchen")):
    return {
        "language_annotations": list(annotations),
        "tags": sorted(set(tags)),
    }


def test_write_catalog_fresh_write_and_return(tmp_path, capsys):
    records = [dict(object_id="fridge_26", catalog_entry=_entry())]
    assert _catalog.write_catalog(tmp_path, records) == 1
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert catalog == {"fridge_26": _entry()}
    # The default summary counts the entries tagged "articulated".
    assert "(0 articulated)." in capsys.readouterr().out


def test_write_catalog_merges_and_rerewrites(tmp_path):
    _catalog.write_catalog(
        tmp_path, [dict(object_id="fridge_26", catalog_entry=_entry())]
    )
    # Incremental run: an untouched object's entry survives, the re-run one
    # is rewritten.
    _catalog.write_catalog(
        tmp_path,
        [dict(object_id="microwave_12", catalog_entry=_entry(("appliance", "a microwave"),
                                                          tags=("microwave", "kitchen")))],
    )
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert set(catalog) == {"fridge_26", "microwave_12"}


def test_write_catalog_drop_removes_published_entry(tmp_path):
    _catalog.write_catalog(
        tmp_path, [dict(object_id="fridge_26", catalog_entry=_entry())]
    )
    # A gate re-run that drops the object removes its (now-stale) entry.
    n = _catalog.write_catalog(
        tmp_path, [dict(object_id="microwave_12", catalog_entry=_entry())],
        drop=["fridge_26"],
    )
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert set(catalog) == {"microwave_12"}
    assert n == 1


def test_write_catalog_overrides_add_tags_and_annotations(tmp_path):
    _catalog.write_catalog(
        tmp_path, [dict(object_id="fridge_26", catalog_entry=_entry())]
    )
    overrides = {
        "fridge_26": {"add_tags": ["white"], "language_annotations": ["a white fridge"]},
        "empty": None,  # empty overrides are skipped
    }
    _catalog.write_catalog(
        tmp_path, [dict(object_id="microwave_12", catalog_entry=_entry())],
        overrides=overrides,
    )
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert "empty" not in catalog
    entry = catalog["fridge_26"]
    assert entry["language_annotations"] == ["a white fridge"]
    assert "white" in entry["tags"] and "fridge" in entry["tags"]


def test_write_catalog_summary_labels(tmp_path, capsys):
    records = [
        dict(object_id="a", catalog_entry=_entry(tags=("articulated", "x"))),
        dict(object_id="b", catalog_entry=_entry(tags=("kitchen",))),
    ]
    _catalog.write_catalog(tmp_path, records)  # default: the articulated count
    assert "(1 articulated)." in capsys.readouterr().out
    _catalog.write_catalog(tmp_path, records, merge=False, summary_label="rigid props")
    assert "(rigid props)." in capsys.readouterr().out
    _catalog.write_catalog(tmp_path, records, merge=False, summary_label=None)
    assert "entries." in capsys.readouterr().out


def test_write_catalog_rejects_incomplete_entry(tmp_path):
    # A merged (incremental) entry missing annotations is refused — every
    # published entry must carry both.
    (tmp_path / "catalog.json").write_text(json.dumps(
        {"stale": {"tags": ["x"]}}))
    with pytest.raises(KeyError, match="missing annotations/tags"):
        _catalog.write_catalog(tmp_path, [])


# ---------------------------------------------------------------------------
# usd_verify
# ---------------------------------------------------------------------------


def _write_stage(path: Path, extra_reference: str | None = None) -> Path:
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateNew(str(path))
    if extra_reference:
        prim = stage.DefinePrim("/root", "Xform")
        prim.GetReferences().AddReference(Sdf.Reference(extra_reference))
    stage.Save()
    return path


def test_is_fatal_missing_ref():
    assert _usd_verify.is_fatal_missing_ref(Path("a.usd"))
    assert _usd_verify.is_fatal_missing_ref(Path("a.JPG"))
    assert not _usd_verify.is_fatal_missing_ref(Path("Foo.Foo"))


def test_verify_object_assets_missing_layer_is_fatal(tmp_path):
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateNew(str(tmp_path / "obj.usd"))
    stage.DefinePrim("/root", "Xform").GetReferences().AddReference(
        Sdf.Reference("./missing_layer.usd")
    )
    stage.Save()
    with pytest.raises(FileNotFoundError, match="missing_layer.usd"):
        _usd_verify.verify_object_assets(tmp_path / "obj.usd", tmp_path)


def test_verify_object_assets_tolerates_unresolvable_refs(tmp_path):
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateNew(str(tmp_path / "obj.usd"))
    stage.DefinePrim("/root", "Xform").GetReferences().AddReference(
        Sdf.Reference("Game/Level/MI_leaf.MI_leaf")
    )
    stage.Save()
    tolerated = _usd_verify.verify_object_assets(
        tmp_path / "obj.usd", tmp_path, allow_unresolvable_refs=True
    )
    assert [p.name for p in tolerated] == ["MI_leaf.MI_leaf"]
    # Without the flag the same missing ref is fatal.
    with pytest.raises(FileNotFoundError):
        _usd_verify.verify_object_assets(tmp_path / "obj.usd", tmp_path)


def test_recoverable_texture_prefix_match(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    shipped = textures / "19913701-files-024.JPG"
    shipped.write_bytes(b"fake")
    missing = textures / "3d66Model-19913701-files-024.JPG"
    assert _usd_verify.recoverable_texture(missing, tmp_path) == shipped
    # Two candidates (ambiguous) or no candidate → not recoverable.
    (textures / "024.JPG").write_bytes(b"fake")  # a second suffix match → ambiguous
    assert _usd_verify.recoverable_texture(missing, tmp_path) is None
    assert _usd_verify.recoverable_texture(Path("missing_layer.usd"), tmp_path) is None


def test_verify_object_assets_recovers_through_the_hook(tmp_path):
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateNew(str(tmp_path / "obj.usd"))
    stage.DefinePrim("/root", "Xform").GetReferences().AddReference(
        Sdf.Reference("./textures/T_Stove_BC001.png")
    )
    stage.Save()
    textures = tmp_path / "textures"
    textures.mkdir()
    (textures / "Stove_BC001.png").write_bytes(b"png")
    # The hook performs the repair copy; verify then passes clean.
    tolerated = _usd_verify.verify_object_assets(
        tmp_path / "obj.usd", tmp_path,
        recover_texture=_usd_verify.recoverable_texture,
    )
    assert tolerated == []
    assert (textures / "T_Stove_BC001.png").is_file()


# ---------------------------------------------------------------------------
# sanity_check_articulated — gate semantics against a stub Genesis
# ---------------------------------------------------------------------------


class _FakeGenesis(types.ModuleType):
    class JOINT_TYPE:
        REVOLUTE = "REVOLUTE"
        PRISMATIC = "PRISMATIC"
        FIXED = "FIXED"
        FREE = "FREE"
        SPHERICAL = "SPHERICAL"

    class options:
        class renderers:
            Rasterizer = staticmethod(lambda: "rasterizer")

    _initialized = False
    cuda = "cuda"
    cpu = "cpu"

    def __init__(self):
        super().__init__("genesis")

    def init(self, backend=None, **kwargs):
        self._initialized = True
        return None

    def destroy(self):
        self._initialized = False

    def Scene(self, **kwargs):
        return _FakeScene()


class _FakeScene:
    """Duck-typed ``gs.Scene``: build/destroy only (no simulation)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cameras = []

    def add_entity(self, morph):
        return SimpleNamespace(morph=morph)

    def add_camera(self, **kwargs):
        camera = SimpleNamespace(**kwargs)
        self.cameras.append(camera)
        return camera

    def build(self, n_envs=0):
        pass

    def destroy(self):
        pass


class _FakeJoint:
    def __init__(self, name, joint_type, n_dofs):
        self.name = name
        self.type = joint_type
        self.n_dofs = n_dofs


class _FakeLink:
    def __init__(self, name):
        self.name = name


class _FakeEntity:
    def __init__(self, joints, links, aabb):
        self.joints = joints
        self.links = links
        self._aabb = aabb

    def get_vAABB(self):
        return self._aabb

    def get_dofs_limit(self, dofs_idx_local):
        return torch.tensor([-2.0]), torch.tensor([2.0])


class _FakeItem:
    """Duck-typed descriptor for the offline parts of the gate."""

    name = "unit_object"

    def __init__(self, description_file, offline_aabb, runtime_aabb, joints, links,
                 opened_aabb=None):
        self.description_file = Path(description_file)
        self._offline = offline_aabb
        self._runtime = runtime_aabb
        self._joints = joints
        self._links = links
        self._opened = opened_aabb if opened_aabb is not None else runtime_aabb

    def get_local_bounding_box(self):
        return SimpleNamespace(low=torch.tensor(self._offline[0]),
                               high=torch.tensor(self._offline[1]))

    def load_into_scene(self, world, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                        **kwargs):
        # The first load is the closed scene, the second the opened one.
        return self._entity_stack.pop(0)

    def clean_name_of(self, runtime_name, kind="joint"):
        return runtime_name.rsplit("/", 1)[-1]

    def _to_runtime_name(self, name, kind):
        return f"/runtime/{name}"

    def resolve_joint_dofs(self, entity, clean):
        return [0]

    def set_dofs_position(self, entity, midpoint, clean):
        pass

    def list_link_names(self):
        return [link.name if hasattr(link, "name") else link for link in self._links]


def _install_fake_genesis(monkeypatch):
    fake = _FakeGenesis()
    monkeypatch.setitem(sys.modules, "genesis", fake)
    return fake


@pytest.fixture()
def fake_gate_world(monkeypatch):
    """Stub genesis + the offline FK walker for the shared gate."""
    _install_fake_genesis(monkeypatch)

    from unienv_genesis_collection.objects import articulated_object

    meshes = {}
    monkeypatch.setattr(
        articulated_object, "usd_link_meshes_at_config",
        lambda usd_file, reference_qpos=None: meshes,
    )
    return meshes


def _make_item(tmp_path, monkeypatch, *, joints, offline, runtime, opened=None,
               prim_paths=("/root/body", "/root/door")):
    from pxr import Usd, UsdPhysics

    stage_file = tmp_path / "unit_object.usda"
    stage = Usd.Stage.CreateNew(str(stage_file))
    UsdGeom_Xform(stage)
    for path in prim_paths:
        prim = stage.DefinePrim(path, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim)
    stage.Save()

    leaf_links = [_FakeLink(path.rsplit("/", 1)[-1]) for path in prim_paths]
    entity = _FakeEntity(
        joints=[_FakeJoint(f"/root/{n}", "REVOLUTE", 1) for n in joints],
        links=leaf_links,
        aabb=(torch.tensor(runtime[0]), torch.tensor(runtime[1])),
    )
    opened_entity = _FakeEntity(
        joints=[_FakeJoint(f"/root/{n}", "REVOLUTE", 1) for n in joints],
        links=leaf_links,
        aabb=(torch.tensor((opened or runtime)[0]), torch.tensor((opened or runtime)[1])),
    )
    item = _FakeItem(stage_file, offline, runtime, joints, leaf_links)
    item._entity_stack = [entity, opened_entity]
    return item, entity, opened_entity


def UsdGeom_Xform(stage):
    from pxr import UsdGeom

    return UsdGeom.Xform.Define(stage, "/root")


def test_sanity_check_articulated_duplicate_leaf_names(tmp_path, monkeypatch, fake_gate_world):
    item, _entity, _opened = _make_item(
        tmp_path, monkeypatch,
        joints=["door"], offline=((-1, -1, -1), (1, 1, 1)), runtime=((-1, -1, -1), (1, 1, 1)),
        prim_paths=("/root/body", "/root/extra/body"),
    )
    with pytest.raises(AssertionError, match="duplicate link/joint leaf names upstream"):
        _sanity.sanity_check_articulated(item, render_dir=None)


def test_sanity_check_articulated_requires_moving_joints(tmp_path, monkeypatch, fake_gate_world):
    item, _entity, _opened = _make_item(
        tmp_path, monkeypatch, joints=[], offline=((-1, -1, -1), (1, 1, 1)),
        runtime=((-1, -1, -1), (1, 1, 1)),
    )
    # The default gate carries the molmo joint-body-swap hint.
    with pytest.raises(AssertionError, match="the joint-body swap may be missing"):
        _sanity.sanity_check_articulated(item, render_dir=None)
    # ...and the knob turns it off (the simready-style path).
    record = _sanity.sanity_check_articulated(
        item, require_moving_joints=False, render_dir=None
    )
    assert record["n_joints"] == 0
    assert record["bbox"].get("opened") is None


def test_sanity_check_articulated_bbox_cross_check(tmp_path, monkeypatch, fake_gate_world):
    # Offline bbox 0.1 m off on every axis: over the 0.03 m absolute tolerance.
    item, _entity, _opened = _make_item(
        tmp_path, monkeypatch, joints=["door"],
        offline=((-1, -1, -1), (1, 1, 1)), runtime=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)),
    )
    with pytest.raises(AssertionError, match="offline-FK bbox deviates"):
        _sanity.sanity_check_articulated(item, render_dir=None)


def test_sanity_check_articulated_passes_with_opened_record(tmp_path, monkeypatch, fake_gate_world):
    item, _entity, _opened = _make_item(
        tmp_path, monkeypatch, joints=["door"],
        offline=((-1, -1, -1), (1, 1, 1)), runtime=((-1, -1, -1), (1, 1, 1)),
        opened=((-1, -1, -1), (1, 1, 1)),
    )
    fake_gate_world["/root/body"] = SimpleNamespace(
        bounds=(torch.tensor((-1, -1, -1)).numpy(), torch.tensor((1, 1, 1)).numpy())
    )
    record = _sanity.sanity_check_articulated(item, render_dir=None)
    assert record["status"] == "passed"
    assert record["n_joints"] == 1
    assert record["clean_joint_names"] == ["door"]
    assert record["clean_link_names"] == ["body", "door"]
    assert record["bbox"]["closed"]["max_delta"] == pytest.approx(0.0, abs=1e-6)
    assert record["bbox"]["opened"]["max_delta"] == pytest.approx(0.0, abs=1e-6)
    assert record["bbox"]["opened"]["reference_qpos"] == {"door": 0.0}
    assert record["links"] == ["body", "door"]
    assert record["render"] is None


def test_sanity_check_articulated_render_is_best_effort(tmp_path, monkeypatch, fake_gate_world, capsys):
    def _boom(descriptor, object_id, render_dir):
        raise RuntimeError("no CUDA")

    monkeypatch.setattr(_sanity, "_render_thumbnail", _boom)
    item, _entity, _opened = _make_item(
        tmp_path, monkeypatch, joints=["door"],
        offline=((-1, -1, -1), (1, 1, 1)), runtime=((-1, -1, -1), (1, 1, 1)),
    )
    fake_gate_world["/root/body"] = SimpleNamespace(
        bounds=(torch.tensor((-1, -1, -1)).numpy(), torch.tensor((1, 1, 1)).numpy())
    )
    record = _sanity.sanity_check_articulated(item, render_dir=tmp_path / "renders")
    assert record["render"] is None
    assert "render FAILED (object kept)" in capsys.readouterr().out


def test_run_sanity_gate_records_unknown_id_as_drop(tmp_path, monkeypatch, fake_gate_world, capsys):
    _install_fake_genesis(monkeypatch)

    def factory(object_id):
        if object_id == "unit_object":
            item, _entity, _opened = _make_item(
                tmp_path, monkeypatch, joints=["door"],
                offline=((-1, -1, -1), (1, 1, 1)), runtime=((-1, -1, -1), (1, 1, 1)),
            )
            fake_gate_world["/root/body"] = SimpleNamespace(
                bounds=(torch.tensor((-1, -1, -1)).numpy(), torch.tensor((1, 1, 1)).numpy())
            )
            return item
        raise KeyError(f"Unknown object id: {object_id!r}")

    from unienv_genesis_collection import asset_path
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    kept, passed, dropped = _sanity.run_sanity_gate(
        tmp_path, ["unknown_id", "unit_object"],
        descriptor_factory=factory, check=_sanity.sanity_check_articulated,
        render_dir=None,
    )
    assert kept == ["unit_object"]
    assert [record["object_id"] for record in dropped] == ["unknown_id"]
    assert "KeyError" in dropped[0]["error"]
    assert passed and passed[0]["object_id"] == "unit_object"
    # The asset_path patching is restored (single-process gate contract).
    assert asset_path.download_assets.__name__ != "_no_download"
    catalog_module._CACHE.clear()


def test_run_sanity_gate_pre_check_drop_appended_and_check_skipped(
    tmp_path, monkeypatch, fake_gate_world, capsys
):
    """``pre_check`` drop: the record is appended and the check never runs.

    The artvip oversized-asset guard (A7) drops an object *before* the
    (potentially huge) Genesis load — the drop goes through the hook's own
    record into ``dropped_records``, and ``check_fn`` must never be called.
    A second object whose pre-check returns ``None`` must still run the
    normal check, so the hook is a per-object guard, not a batch filter.
    """
    calls: list[str] = []

    def check_fn(item, *, bbox_tol_abs, bbox_tol_rel, render_dir):
        calls.append(item.name)
        return dict(status="passed", links=[], joints=[], n_joints=0,
                    clean_joint_names=[], clean_link_names=[],
                    bbox=dict(closed=dict(max_delta=0.0, tolerance_per_axis=[0.0])),
                    render=None)

    oversized, _e1, _o1 = _make_item(
        tmp_path, monkeypatch, joints=["door"],
        offline=((-1, -1, -1), (1, 1, 1)), runtime=((-1, -1, -1), (1, 1, 1)),
    )
    regular, _e2, _o2 = _make_item(
        tmp_path / "regular", monkeypatch, joints=["door"],
        offline=((-1, -1, -1), (1, 1, 1)), runtime=((-1, -1, -1), (1, 1, 1)),
    )
    # Per-instance names: the pre-check keys on them like the packager's
    # _gate_pre_check keys on the item's stage extent.
    oversized.name = "oversized_item"
    regular.name = "regular_item"

    def pre_check(object_id, item):
        # The oversized item carries the marker the packager's guard keys on.
        if item.name == "oversized_item":
            return dict(
                object_id=object_id, status="dropped",
                error=f"OversizedAsset: {object_id} exceeds the size limit",
            )
        assert calls == [], "pre_check(None) only after the check ran once"
        return None

    kept, passed, dropped = _sanity.run_sanity_gate(
        tmp_path, ["oversized", "regular"],
        descriptor_factory=lambda _id: oversized if _id == "oversized" else regular,
        check=check_fn, pre_check=pre_check, render_dir=None,
    )
    assert kept == ["regular"]
    assert calls == ["regular_item"], "the dropped item's check must never run"
    assert [record["object_id"] for record in dropped] == ["oversized"]
    assert dropped[0]["status"] == "dropped"
    assert dropped[0]["error"].startswith("OversizedAsset:")
    assert [record["object_id"] for record in passed] == ["regular"]
    # The hook's record (not a generic "Exception: ..." drop) is what got in.
    assert "Exception" not in dropped[0]["error"]


def test_run_sanity_gate_descriptor_factory_tuple_dispatches_check(
    tmp_path, monkeypatch, fake_gate_world, capsys
):
    """A ``(item, check_fn)`` factory pair runs that check, not the default.

    Lightwheel ships static props *and* articulated furniture in one tree:
    its ``descriptor_factory`` returns the per-item ``(item, check)`` pair
    (the static/articulated dispatch seam) and the shared loop must dispatch
    on it, exactly like lightwheel's former in-loop fork did — with the flat
    (default-*check*) shape still working in the same batch.
    """
    from pxr import Usd, UsdPhysics

    _install_fake_genesis(monkeypatch)
    called: list[str] = []

    def _make_fake_item(prim_paths, name):
        stage_file = tmp_path / f"{name}.usda"
        stage = Usd.Stage.CreateNew(str(stage_file))
        UsdGeom_Xform(stage)
        for path in prim_paths:
            prim = stage.DefinePrim(path, "Xform")
            UsdPhysics.RigidBodyAPI.Apply(prim)
        stage.Save()
        return _FakeItem(stage_file, ((-1, -1, -1), (1, 1, 1)),
                         ((-1, -1, -1), (1, 1, 1)),
                         [], [_FakeLink("body")], [])

    def default_check(item, *, bbox_tol_abs, bbox_tol_rel, render_dir):
        called.append(item.name)
        return dict(status="passed", links=[], joints=[], n_joints=0,
                    clean_joint_names=[], clean_link_names=[],
                    bbox=dict(closed=dict(max_delta=0.0, tolerance_per_axis=[0.0])),
                    render=None, via="default-check")

    def static_check(item, *, bbox_tol_abs, bbox_tol_rel, render_dir):
        called.append(item.name)
        return dict(status="passed", links=[], joints=[], n_moving_joints=0,
                    clean_joint_names=[], clean_link_names=[],
                    bbox=dict(closed=dict(max_delta=0.0, tolerance_per_axis=[0.0])),
                    render=None, via="per-item-check")

    rigid = _make_fake_item(("/root/body",), "rigid_prop")
    articulated = _make_fake_item(("/root/body",), "artic_item")
    rigid.name, articulated.name = "rigid_prop", "artic_item"

    def descriptor_factory(object_id):
        # The lightwheel factory shape: static props ship as a flat item
        # (the default gate applies), articulated ones as the ``(item,
        # check)`` pair the shared loop dispatches on.
        if object_id == "static_prop":
            return rigid
        return (articulated, static_check)

    kept, passed, dropped = _sanity.run_sanity_gate(
        tmp_path, ["static_prop", "artic_item"],
        descriptor_factory=descriptor_factory, check=default_check,
        render_dir=None,
    )
    assert kept == ["static_prop", "artic_item"]
    assert not dropped
    # The seam, asserted: the flat item falls through to the default check,
    # the 2-tuple dispatches to the factory's own per-item check — if the
    # loop ignored the tuple (or always used it) exactly one `via` flips.
    assert [record["via"] for record in passed] == ["default-check", "per-item-check"]
    assert called == ["rigid_prop", "artic_item"]


# ---------------------------------------------------------------------------
# gate_common helpers
# ---------------------------------------------------------------------------


def test_quat_to_matrix_identity_and_quarter_turn():
    identity = _gate_common.quat_to_matrix((1.0, 0.0, 0.0, 0.0))
    assert identity == pytest.approx(torch.eye(3).numpy(), abs=1e-12)
    quarter = _gate_common.quat_to_matrix(
        (torch.cos(torch.tensor(0.25 * torch.pi)).item(), 0.0, 0.0,
         torch.sin(torch.tensor(0.25 * torch.pi)).item())
    )
    assert quarter[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert quarter[1, 0] == pytest.approx(1.0, abs=1e-6)


def test_finite_midpoint_clamps_non_finite_limits():
    zero = _gate_common._finite_midpoint(torch.tensor([0.0]), torch.tensor([float("inf")]))
    assert float(zero.squeeze()) == 0.0
    midpoint = _gate_common._finite_midpoint(torch.tensor([-0.1]), torch.tensor([0.1]))
    assert float(midpoint.squeeze()) == pytest.approx(0.0)


def test_opened_reference_joints_skip_multi_dof():
    moving = [
        dict(name="revolute", type="REVOLUTE", n_dofs=1),
        dict(name="spherical", type="SPHERICAL", n_dofs=3),
        dict(name="prismatic", type="PRISMATIC", n_dofs=1),
    ]
    assert [record["name"] for record in _gate_common._opened_reference_joints(moving)] == [
        "revolute", "prismatic"
    ]


# ---------------------------------------------------------------------------
# _gate_batch — in-process fallback
# ---------------------------------------------------------------------------


def test_gate_batch_in_process_fallback(tmp_path, monkeypatch):
    calls = []

    def runner(output_dir, object_ids, *, render_dir, bbox_tol_abs, bbox_tol_rel):
        calls.append(list(object_ids))
        return list(object_ids), [], []

    monkeypatch.setenv("GATE_BATCH_SIZE", "0")
    kept, passed, dropped = _gate_batch.run_sanity_gate_batched(
        __file__, runner, tmp_path, ["a", "b", "c"],
        render_dir=None, bbox_tol_abs=0.03, bbox_tol_rel=0.12,
    )
    assert kept == ["a", "b", "c"] and calls == [["a", "b", "c"]]


def test_gate_batch_small_batches_stay_in_process(tmp_path, monkeypatch):
    calls = []

    def runner(output_dir, object_ids, *, render_dir, bbox_tol_abs, bbox_tol_rel):
        calls.append(list(object_ids))
        return list(object_ids), [], []

    kept, _passed, _dropped = _gate_batch.run_sanity_gate_batched(
        __file__, runner, tmp_path, ["a", "b"],
        render_dir=None, bbox_tol_abs=0.03, bbox_tol_rel=0.12,
    )
    assert kept == ["a", "b"]
    assert calls == [["a", "b"]]
