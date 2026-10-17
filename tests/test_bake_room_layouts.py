"""Tests for ``scripts/bake_room_layouts.py`` (S5: arrangement/layout baking).

Offline except the marked smoke tests: the ReplicaCAD conversion runs against a
**synthetic assets tree** under ``tmp_path`` (a synthetic stage GLB box, a
synthetic item URDF, an injected synthetic catalog) with
``asset_path.ASSETS_DIR`` repointed and ``download_assets`` stubbed; the *real*
assets tree is used only for the baked-file round-trip and the spot checks (no
Hub downloads — every referenced file ships in the collection).

Covers (plan §3.4 / §6 S5):

  - Stage → room frame mapping: the stage GLBs are glTF (Y-up), the room frame
    is Z-up with the floor at z=0 → ``p_room = (p_x, -p_z, p_y)``.
  - Yaw extraction: the scene rotation composes the item's upright (standing)
    orientation — the catalog's ``standing_quaternion``, which maps the item's
    authored up axis onto the stage's up axis (+y) — with a yaw about that up
    axis.  The bake removes the standing orientation
    (``R_standing_stage = Mᵀ · R_standing_room``) and reads the residual's yaw
    about the up axis; a tilted residual (roll/pitch beyond the tolerance)
    cannot be expressed by a yaw-only spec and is rejected.
  - COM → base-link origin: a ``translation_origin: "COM"`` translation is
    corrected by the URDF root-link inertial offset
    (``t_base = t_com − R_stage · com_local``); an inertial-less root link
    needs no correction (every shipped ReplicaCAD item).
  - Scale handling: non-unit ``uniform_scale`` instances are reported and
    skipped (``FurniturePlacement`` has no scale field and the room builder
    loads catalog descriptors at scale 1.0) — never silently dropped.
  - Unknown template ids / movable (``fixed_base: false``) instances skipped.
  - ``room_size``: computed from the stage shell GLB's bounding box — the
    largest floor rectangle centred on the shell origin inside the shell bbox
    (the stage GLB asset origins are not the centres of their floor
    rectangles).
  - Baked ids: ``RoomBuilder(arrangement="replicacad/<name>")`` resolves
    ``scenes/rooms/replicacad/arrangements/<name>.json`` and
    ``RoomBuilder(layout_yaml="robocasa_kitchen/<style>")`` resolves
    ``scenes/rooms/robocasa_kitchen/layouts/<style>.json`` (via
    ``baked_spec_relative_path``; the upstream inputs keep their
    ``.scene_instance.json`` names, the baked outputs drop it).
  - Every baked file (all 91 arrangements + both layouts) round-trips through
    ``load_baked_room_layout_spec``.
  - Baked determinism: re-baking reproduces byte-identical JSON.
  - RoboCasa layouts: the fixtures stand on the floor, backs flush with the
    counter run's room-side face, fronts facing the room interior, laid out
    centred and non-overlapping along the run, all in-bounds.
  - CPU smoke: ``RoomBuilder(arrangement="replicacad/apt_0")`` builds the shell
    + the furniture in a real minimal Genesis scene and the fridge stands
    upright (its world AABB is taller in z than in y, bottom at the baked
    float height).
  - GPU smoke (``skipif``): ``RoomBuilder(layout_yaml="robocasa_kitchen/open_kitchen")``
    builds the MJCF shell + the fixtures.

Device isolation: ``gs.init()`` installs the compute device as torch's
*ambient* default and it stays there for the rest of the process, so the
offline assertions read tensors through ``.detach().cpu()`` / Python floats
instead of assuming a host device.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.hub_objects.articulated.replicacad_furniture import (
    REPLICACAD_ASSETS_ROOT,
    get_replicacad_catalog,
)
from unienv_genesis_collection.scenes.room_builder import (
    baked_spec_relative_path,
    load_baked_room_layout_spec,
    load_room_layout_spec_json,
    placement_load_quat,
)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_REAL_ASSETS_DIR = (
    Path(__file__).parent.parent / "unienv_genesis_collection" / "assets"
)

# The baked-file round-trips and spot checks read the real scene assets (~13 GB,
# downloaded from the Hub on demand); skip them where those aren't present (e.g. CI).
requires_scene_assets = pytest.mark.skipif(
    not (_REAL_ASSETS_DIR / "scenes" / "rooms").is_dir(),
    reason="real scene assets (unienv_genesis_collection/assets/scenes/rooms) are not downloaded",
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "bake_room_layouts", _SCRIPTS_DIR / "bake_room_layouts.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the script's dataclasses look their module up in
    # sys.modules at class-creation time.
    sys.modules["bake_room_layouts"] = module
    spec.loader.exec_module(module)
    return module


BAKE = _load_script()

# The standing (upright) orientations of the real ReplicaCAD catalog, for the
# baked-file spot checks.
_STANDINGS = {
    object_id: info.standing_quaternion
    for object_id, info in get_replicacad_catalog().items()
}


@pytest.fixture
def no_download(monkeypatch):
    def _fail(**kwargs):
        raise AssertionError(f"Unexpected download attempt: {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail)


# ---------------------------------------------------------------------------
# Synthetic assets tree (offline)
# ---------------------------------------------------------------------------

SYNTH_STAGE_ID = "synthetic_stage"
SYNTH_ITEM = "synth_item"
# The synthetic item's upright orientation: +90° roll about x (the ReplicaCAD
# y-up authoring convention — the item's height runs along root +y).
SYNTH_STANDING = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
# The synthetic item's root-link inertial COM offset, in the root-link frame.
SYNTH_COM_LOCAL = (0.0, 0.1, 0.0)


class _CatalogInfo:
    """Minimal catalog entry for the synthetic offline conversion tests."""

    def __init__(self, standing_quaternion):
        self.standing_quaternion = standing_quaternion


def _synthetic_catalog():
    """The offline catalog: one synthetic item with a +90°-roll standing quat."""
    return {SYNTH_ITEM: _CatalogInfo(SYNTH_STANDING)}


def _write_synthetic_item(assets_dir: Path, with_root_inertial: bool) -> Path:
    """A synthetic ReplicaCAD item directory: just the (weighted) root URDF."""
    urdf_text = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<robot name="synth_item">\n'
        '  <link name="root">\n'
        + (
            "    <inertial>\n"
            f"      <origin xyz=\"{' '.join(f'{v:g}' for v in SYNTH_COM_LOCAL)}\" rpy=\"0 0 0\"/>\n"
            "      <mass value=\"10.0\"/>\n"
            '      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>\n'
            "    </inertial>\n"
            if with_root_inertial
            else ""
        )
        + "  </link>\n"
        '  <joint name="root_rotation" type="fixed">\n'
        '    <origin rpy="1.5707963267948966 0 0" xyz="0 0 0"/>\n'
        '    <parent link="root"/>\n'
        '    <child link="body"/>\n'
        "  </joint>\n"
        '  <link name="body">\n'
        "    <visual>\n"
        '      <origin rpy="0 0 0" xyz="0 0 0"/>\n'
        '      <geometry><mesh filename="body.glb" scale="1 1 1"/></geometry>\n'
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )
    item_dir = assets_dir / REPLICACAD_ASSETS_ROOT / SYNTH_ITEM
    item_dir.mkdir(parents=True, exist_ok=True)
    urdf = item_dir / f"{SYNTH_ITEM}.urdf"
    urdf.write_text(urdf_text, encoding="utf-8")
    return urdf


def _write_synthetic_stage_glb(assets_dir: Path, extents, centre) -> Path:
    """A synthetic stage shell GLB: one box mesh."""
    import trimesh

    stage_dir = assets_dir / "scenes" / "rooms" / "replicacad" / SYNTH_STAGE_ID
    stage_dir.mkdir(parents=True, exist_ok=True)
    glb = stage_dir / f"{SYNTH_STAGE_ID}.glb"
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(centre)
    mesh.export(glb)
    return glb


def _write_synthetic_assets(assets_dir: Path, *, with_root_inertial: bool) -> None:
    """A synthetic stage GLB box + item URDF (the catalog is injected)."""
    _write_synthetic_item(assets_dir, with_root_inertial=with_root_inertial)
    _write_synthetic_stage_glb(
        assets_dir, extents=(5.0, 3.0, 4.0), centre=(1.0, 0.0, 0.0)
    )


def _write_scene_instance(path: Path, instances) -> None:
    data = {
        "stage_instance": {"template_name": f"stages/{SYNTH_STAGE_ID}"},
        "object_instances": [],
        "articulated_object_instances": list(instances),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Independent frame-matrix helpers (the expectations, not the script's own)
# ---------------------------------------------------------------------------


def _stage_yaw_matrix(yaw: float) -> np.ndarray:
    """R_y(yaw): a yaw about the stage's up axis (+y)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _stage_standing_matrix(standing_room) -> np.ndarray:
    """The item's upright orientation, expressed in the stage (Y-up) frame.

    ``R_standing_room = M · R_standing_stage``
    → ``R_standing_stage = Mᵀ · R_standing_room``.
    """
    if standing_room is None:
        return np.eye(3)
    return BAKE._STAGE_TO_ROOM.T @ BAKE.quat_to_matrix(
        tuple(float(v) for v in standing_room)
    )


def _scene_rotation(yaw: float, standing_room) -> tuple:
    """Scene quaternion for ``R_stage = R_y(yaw) · R_standing_stage``."""
    return BAKE.matrix_to_quat(
        _stage_yaw_matrix(yaw) @ _stage_standing_matrix(standing_room)
    )


@pytest.fixture
def synth_assets(tmp_path, monkeypatch):
    """A synthetic ReplicaCAD assets tree, with ASSETS_DIR repointed at it."""
    _write_synthetic_item(tmp_path, with_root_inertial=True)
    _write_synthetic_stage_glb(
        tmp_path, extents=(5.0, 3.0, 4.0), centre=(1.0, 0.0, 0.0)
    )
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    monkeypatch.setattr(BAKE, "_load_replicacad_catalog", _synthetic_catalog)
    return tmp_path


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


def test_stage_to_room_pos_maps_yup_stage_to_zup_room():
    # Stage (x, y_up, z) → room (x, -z, y): the stage's up axis (+y) becomes the
    # room's +z and the stage's +z becomes the room's -y (a proper rotation).
    assert BAKE.stage_to_room_pos((1.0, 2.0, 3.0)) == (1.0, -3.0, 2.0)
    assert BAKE.stage_to_room_pos((-2.0, 0.5, 4.0)) == (-2.0, -4.0, 0.5)


def test_stage_room_frame_from_synthetic_glb(synth_assets, no_download):
    """room_size comes from the stage shell GLB bbox mapped into the room."""
    glb = (
        synth_assets
        / "scenes"
        / "rooms"
        / "replicacad"
        / SYNTH_STAGE_ID
        / f"{SYNTH_STAGE_ID}.glb"
    )
    width, depth, wall_height = BAKE._shell_room_frame(glb)
    # Stage bbox: x ∈ [-1.5, 3.5] → width = 2·min(1.5, 3.5) = 3.0;
    #             z ∈ [-2.0, 2.0] → depth = 2·min(2.0, 2.0) = 4.0;
    #             y ∈ [-1.5, 1.5] → wall height = max y = 1.5.
    assert width == pytest.approx(3.0)
    assert depth == pytest.approx(4.0)
    assert wall_height == pytest.approx(1.5)


def test_yaw_extraction_removes_standing_orientation(no_download):
    """A pure stage yaw survives removing the catalog's standing orientation."""
    standing = _STANDINGS["fridge"]  # a +90° roll about x (a y-up item)
    rotation = _scene_rotation(0.7, standing)
    yaw, tilt = BAKE.yaw_about_up_axis(tuple(float(v) for v in rotation), standing)
    assert yaw == pytest.approx(0.7, abs=1e-9)
    assert tilt < 1e-9


def test_yaw_extraction_rejects_tilted_rotation(no_download):
    """A tilted instance (roll/pitch beyond the yaw) is rejected."""
    standing = _STANDINGS["fridge"]
    # R_stage = R_x(30°) · R_y(0.7) · R_standing_stage — a 30° pitch on top.
    c, s = math.cos(0.5), math.sin(0.5)
    pitch = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    rotation = BAKE.matrix_to_quat(
        pitch @ (_stage_yaw_matrix(0.7) @ _stage_standing_matrix(standing))
    )
    _yaw, tilt = BAKE.yaw_about_up_axis(tuple(float(v) for v in rotation), standing)
    assert tilt > 1e-3


# ---------------------------------------------------------------------------
# Synthetic arrangement conversion
# ---------------------------------------------------------------------------


def test_bake_arrangement_pose_com_scale_and_skips(
    tmp_path, monkeypatch, no_download
):
    """End-to-end synthetic conversion: pose, COM→base, scale and skip reasons."""
    _write_synthetic_item(tmp_path, with_root_inertial=True)
    _write_synthetic_stage_glb(
        tmp_path, extents=(5.0, 3.0, 4.0), centre=(1.0, 0.0, 0.0)
    )
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    monkeypatch.setattr(BAKE, "_load_replicacad_catalog", _synthetic_catalog)

    # The synthetic item's root link carries an inertial COM offset, so the
    # COM→base correction is exercised: com_local = (0, 0.1, 0) in the root
    # frame; the scene places the COM at (3, 0.6, 4) with a pure yaw of 0.7 →
    # the base origin = com − R_stage · com_local.
    standing = SYNTH_STANDING
    yaw = 0.7
    com_local = np.array(SYNTH_COM_LOCAL, dtype=float)
    R_stage = BAKE.quat_to_matrix(_scene_rotation(yaw, standing))
    t_com = np.array([3.0, 0.6, 4.0])
    t_base = t_com - R_stage @ com_local
    scene_file = (
        tmp_path
        / "scenes"
        / "rooms"
        / "replicacad"
        / "arrangements"
        / "apt_x.scene_instance.json"
    )
    _write_scene_instance(
        scene_file,
        [
            {
                "template_name": SYNTH_ITEM,
                "translation_origin": "COM",
                "fixed_base": True,
                "uniform_scale": 1.0,
                "translation": [float(v) for v in t_com],
                "rotation": [float(v) for v in _scene_rotation(yaw, standing)],
            },
            # A non-unit scale instance must be skipped, never silently baked.
            {
                "template_name": SYNTH_ITEM,
                "translation_origin": "COM",
                "fixed_base": True,
                "uniform_scale": 0.4,
                "translation": [10.0, 0.0, 10.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            # A movable instance must be skipped too.
            {
                "template_name": SYNTH_ITEM,
                "translation_origin": "COM",
                "fixed_base": False,
                "translation": [10.0, 0.0, 10.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            # An unknown template id must be skipped.
            {
                "template_name": "no_such_item",
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
        ],
    )

    report = BAKE.bake_replicacad_arrangement(scene_file, tmp_path, tmp_path)

    assert len(report.spec.furniture) == 1
    placement = report.spec.furniture[0]
    assert placement.object_id == SYNTH_ITEM
    assert placement.source == "replicacad"
    assert placement.yaw == pytest.approx(yaw, abs=1e-9)
    # The room frame: (stage_x, -stage_z, stage_y) — the stage is Y-up.
    assert placement.pos[0] == pytest.approx(float(t_base[0]), abs=1e-9)
    assert placement.pos[1] == pytest.approx(-float(t_base[2]), abs=1e-9)
    assert placement.pos[2] == pytest.approx(float(t_base[1]), abs=1e-9)
    # The skipped instances are reported with their reason, never silently.
    skipped_ids = [template for template, _reason in report.skipped]
    assert skipped_ids.count(SYNTH_ITEM) == 2  # the non-unit scale + the movable
    assert any("uniform_scale" in reason for _template, reason in report.skipped)
    assert any("fixed_base" in reason for _template, reason in report.skipped)
    assert any(
        "not a ReplicaCAD catalog" in reason for _template, reason in report.skipped
    )
    # The room_size comes from the synthetic shell GLB bbox (the same
    # mapping as the frame test: the centred inner rect + the max y).
    assert report.spec.room_size == pytest.approx((3.0, 4.0, 1.5), rel=1e-6)


def test_bake_arrangement_inertialless_root_needs_no_correction(
    tmp_path, monkeypatch, no_download
):
    """A root link without <inertial>: the translation *is* the base origin."""
    _write_synthetic_item(tmp_path, with_root_inertial=False)
    _write_synthetic_stage_glb(
        tmp_path, extents=(4.0, 4.0, 3.0), centre=(0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    monkeypatch.setattr(BAKE, "_load_replicacad_catalog", _synthetic_catalog)

    translation = (2.0, 1.0, -3.0)
    scene_file = (
        tmp_path
        / "scenes"
        / "rooms"
        / "replicacad"
        / "arrangements"
        / "apt_y.scene_instance.json"
    )
    _write_scene_instance(
        scene_file,
        [
            {
                "template_name": SYNTH_ITEM,
                "translation_origin": "COM",
                "fixed_base": True,
                "translation": list(translation),
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
        ],
    )
    report = BAKE.bake_replicacad_arrangement(scene_file, tmp_path, tmp_path)
    placement = report.spec.furniture[0]
    # com_local = 0 → the base origin = the translation.
    assert placement.pos == pytest.approx(
        (translation[0], -translation[2], translation[1]), abs=1e-9
    )


def test_bake_arrangement_skips_tilted_instance(
    tmp_path, monkeypatch, no_download
):
    """A tilted instance is reported and skipped, never baked with a wrong yaw."""
    _write_synthetic_item(tmp_path, with_root_inertial=True)
    _write_synthetic_stage_glb(
        tmp_path, extents=(4.0, 4.0, 3.0), centre=(0.0, 0.0, 0.0)
    )
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    monkeypatch.setattr(BAKE, "_load_replicacad_catalog", _synthetic_catalog)

    # R_stage = R_x(30°) · R_y(0.7) · R_standing_stage — a 30° pitch on top.
    c, s = math.cos(0.5), math.sin(0.5)
    pitch = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    rotation = BAKE.matrix_to_quat(
        pitch @ (_stage_yaw_matrix(0.7) @ _stage_standing_matrix(SYNTH_STANDING))
    )
    scene_file = (
        tmp_path
        / "scenes"
        / "rooms"
        / "replicacad"
        / "arrangements"
        / "apt_z.scene_instance.json"
    )
    _write_scene_instance(
        scene_file,
        [
            {
                "template_name": SYNTH_ITEM,
                "translation_origin": "COM",
                "fixed_base": True,
                "translation": [0.0, 0.0, 0.0],
                "rotation": [float(v) for v in rotation],
            },
        ],
    )
    report = BAKE.bake_replicacad_arrangement(scene_file, tmp_path, tmp_path)
    assert report.spec.furniture == ()
    # Tilted rotations are warned (a yaw-only spec cannot carry roll/pitch).
    assert any("not upright" in message for message in report.warnings)


# ---------------------------------------------------------------------------
# The real baked files: id lookup + round-trip + spot checks
# ---------------------------------------------------------------------------


@requires_scene_assets
def test_baked_arrangement_id_lookup(no_download):
    """``arrangement="replicacad/<name>"`` resolves the baked file."""
    rel = baked_spec_relative_path("replicacad/apt_0", kind="arrangement")
    assert rel == ("scenes", "rooms", "replicacad", "arrangements", "apt_0.json")
    assert (_REAL_ASSETS_DIR.joinpath(*rel)).is_file()


@requires_scene_assets
def test_baked_layout_id_lookup(no_download):
    rel = baked_spec_relative_path("robocasa_kitchen/open_kitchen", kind="layout")
    assert rel == (
        "scenes", "rooms", "robocasa_kitchen", "layouts", "open_kitchen.json",
    )
    assert (_REAL_ASSETS_DIR.joinpath(*rel)).is_file()


@requires_scene_assets
def test_every_baked_arrangement_round_trips(no_download):
    import glob

    files = sorted(
        path
        for path in glob.glob(
            str(_REAL_ASSETS_DIR / "scenes/rooms/replicacad/arrangements/*.json")
        )
        if not path.endswith(".scene_instance.json")
    )
    assert len(files) == 91  # one baked spec per upstream arrangement
    for path in files:
        spec = load_room_layout_spec_json(path)
        name = Path(path).name[: -len(".json")]
        loaded = load_baked_room_layout_spec(f"replicacad/{name}", kind="arrangement")
        assert loaded.furniture == spec.furniture
        assert loaded.room_size == spec.room_size
        assert loaded.shell_id == spec.shell_id


@requires_scene_assets
def test_every_baked_layout_round_trips(no_download):
    for style in ("open_kitchen", "galley"):
        spec = load_baked_room_layout_spec(f"robocasa_kitchen/{style}", kind="layout")
        assert spec.furniture, style
        assert spec.shell_id == f"robocasa_kitchen/{style}"


@requires_scene_assets
def test_baked_apt_0_spot_checks(no_download):
    """apt_0's baked poses follow from the upstream arrangement's data model."""
    spec = load_baked_room_layout_spec("replicacad/apt_0", kind="arrangement")
    assert spec.shell_id == "replicacad/frl_apartment_stage/frl_apartment_stage.glb"
    assert len(spec.furniture) == 3
    by_id = {placement.object_id: placement for placement in spec.furniture}
    assert sorted(by_id) == ["cabinet", "door2", "fridge"]

    upstream = json.loads(
        (
            _REAL_ASSETS_DIR
            / "scenes/rooms/replicacad/arrangements/apt_0.scene_instance.json"
        ).read_text()
    )
    instances = {
        instance["template_name"]: instance
        for instance in upstream["articulated_object_instances"]
    }

    # The fridge: an inertial-less root link → the translation *is* the base
    # origin; the identity scene rotation is the pure upright yaw 0.
    fridge = by_id["fridge"]
    translation = np.asarray(instances["fridge"]["translation"], dtype=float)
    assert fridge.pos == pytest.approx(
        (float(translation[0]), -float(translation[2]), float(translation[1])),
        abs=1e-9,
    )
    assert fridge.yaw == pytest.approx(0.0, abs=1e-9)

    # The cabinet: a pure stage yaw of ≈ 2.5 rad; the base origin sits 5 cm
    # above the stage floor (its bbox bottom then rests on the floor).
    cabinet = by_id["cabinet"]
    translation = np.asarray(instances["cabinet"]["translation"], dtype=float)
    assert cabinet.pos == pytest.approx(
        (float(translation[0]), -float(translation[2]), float(translation[1])),
        abs=1e-9,
    )
    assert cabinet.yaw == pytest.approx(2.5000009278662345, abs=1e-9)

    # The door: its base origin sits at the top of the door (2.4 m) with its
    # authored geometry running down to the floor.
    door = by_id["door2"]
    translation = np.asarray(instances["door2"]["translation"], dtype=float)
    assert door.pos == pytest.approx(
        (float(translation[0]), -float(translation[2]), float(translation[1])),
        abs=1e-9,
    )
    assert door.yaw == pytest.approx(0.0, abs=1e-9)


@requires_scene_assets
def test_baked_apt_0_room_size_from_shell_bbox(no_download):
    import trimesh

    spec = load_baked_room_layout_spec("replicacad/apt_0", kind="arrangement")
    width, depth, wall_height = spec.room_size
    mesh = trimesh.load(
        str(
            _REAL_ASSETS_DIR
            / "scenes/rooms/replicacad/frl_apartment_stage/frl_apartment_stage.glb"
        ),
        force="mesh",
    )
    lo, hi = mesh.bounds  # glTF: Y-up
    assert width == pytest.approx(2.0 * min(abs(lo[0]), abs(hi[0])), rel=1e-9)
    # Stage z → room -y (the mapping negates the z axis).
    assert depth == pytest.approx(2.0 * min(abs(lo[2]), abs(hi[2])), rel=1e-9)
    assert wall_height == pytest.approx(float(hi[1]), rel=1e-9)


@requires_scene_assets
def test_baked_robocasa_open_kitchen_spot_checks(no_download):
    spec = load_baked_room_layout_spec("robocasa_kitchen/open_kitchen", kind="layout")
    assert spec.room_size == pytest.approx((6.2, 5.2, 2.7))
    assert [placement.object_id for placement in spec.furniture] == [
        "cabinet_hinge",
        "drawer",
        "stove_simple_gas",
    ]
    assert all(placement.source == "robocasa" for placement in spec.furniture)
    # The fixtures stand on the floor (their bbox bottoms at z=0).
    geometry = BAKE.load_fixture_geometry(
        [placement.object_id for placement in spec.furniture], _REAL_ASSETS_DIR
    )
    for placement in spec.furniture:
        assert placement.pos[2] == pytest.approx(
            -geometry[placement.object_id]["z_lo"]
        )


@requires_scene_assets
def test_baked_robocasa_layouts_are_deterministic(tmp_path, no_download):
    """Re-baking the layouts reproduces byte-identical JSON."""
    for style in ("open_kitchen", "galley"):
        baked = (
            _REAL_ASSETS_DIR
            / "scenes/rooms/robocasa_kitchen/layouts"
            / f"{style}.json"
        ).read_bytes()
        fresh = tmp_path / "scenes/rooms/robocasa_kitchen/layouts" / f"{style}.json"
        BAKE.bake_robocasa_layout(style, _REAL_ASSETS_DIR, tmp_path)
        assert fresh.read_bytes() == baked


# ---------------------------------------------------------------------------
# Minimal real Genesis scenes (CPU + a CUDA-marked variant)
# ---------------------------------------------------------------------------


def _real_world():
    from unienv_genesis import GenesisWorld

    return GenesisWorld()


def _host(tensor):
    return torch.as_tensor(tensor).detach().cpu()


def test_room_builder_arrangement_smoke_cpu():
    """End-to-end smoke: RoomBuilder(arrangement="replicacad/apt_0") builds."""
    import genesis as gs
    from genesis.constants import backend as gs_backend
    from unienv_interface.world import WorldEnv

    from unienv_genesis_collection.scenes.room_builder import RoomBuilder

    if not gs._initialized:
        gs.init(backend=gs_backend.cpu)

    world = _real_world()
    room = RoomBuilder(world, name="room", arrangement="replicacad/apt_0")
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    spec = room.layout
    assert spec is not None and spec.shell_id.startswith("replicacad/")
    assert len(room.shell_entities) == 1  # the baked stage shell mesh
    assert sorted(room.furniture) == ["cabinet", "door2", "fridge"]
    assert {
        placement.source for placement in room.placements.values()
    } == {"replicacad"}

    # The fridge's world AABB stands upright: the ReplicaCAD URDFs author the
    # item lying down (height along root +y), so the room builder composes the
    # catalog's standing orientation on top of the placement's yaw.
    fridge = room.furniture["fridge"]
    entity = room.furniture_entities["fridge"]
    aabb = fridge.get_bounding_box(entity)
    low = _host(aabb.low).tolist()
    high = _host(aabb.high).tolist()
    extents = [high[i] - low[i] for i in range(3)]
    assert extents[2] > extents[1]  # upright: tall along +z, not along ±y
    assert extents[2] == pytest.approx(1.921, abs=2e-2)
    # The baked float: the fridge's base origin sits at z ≈ 0.976 (its centre)
    # and its bbox bottom rests at the authored ~2 cm float above the floor.
    assert _host(entity.get_pos()).tolist()[2] == pytest.approx(
        0.9755649566650391, abs=1e-4
    )
    assert 0.0 <= low[2] <= 0.06

    # The task-facing API works off the baked layout (world-device tensors).
    samples = room.sample_free_floor_positions(8, None)
    assert samples.shape == (8, 3)
    assert samples.device == world.device

    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_room_builder_layout_yaml_smoke_gpu():
    """GPU smoke: RoomBuilder(layout_yaml=...) builds the MJCF shell + fixtures."""
    import genesis as gs
    from genesis.constants import backend as gs_backend
    from unienv_interface.world import WorldEnv

    from unienv_genesis_collection.scenes.room_builder import RoomBuilder

    if not gs._initialized:
        gs.init(backend=gs_backend.cuda)

    world = _real_world()
    room = RoomBuilder(
        world, name="room", layout_yaml="robocasa_kitchen/open_kitchen"
    )
    env = WorldEnv(world, [room], render_mode=None)
    env.reset(seed=0, reload=True)

    assert room.layout.shell_id == "robocasa_kitchen/open_kitchen"
    assert room.room_size == pytest.approx((6.2, 5.2, 2.7))
    assert len(room.shell_entities) == 1  # the MJCF shell
    assert [descriptor.name for descriptor in room.furniture.values()] == [
        "cabinet_hinge",
        "drawer",
        "stove_simple_gas",
    ]
    # Every fixture stands on the floor (its world AABB bottom at z ≈ 0).
    for name, descriptor in room.furniture.items():
        bbox = descriptor.get_bounding_box(room.furniture_entities[name])
        low = _host(bbox.low).tolist()
        assert low[2] == pytest.approx(0.0, abs=2e-2)
    env.close()
