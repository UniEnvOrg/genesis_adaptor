#!/usr/bin/env python3
"""Bake ReplicaCAD arrangements / RoboCasa kitchen layouts to RoomLayoutSpec JSON.

Plan §3.4 / §6 S5.  Room frame: origin at the floor centre, ``z`` up.

--------------------------------------------------------------------------------
Frame convention (derived from the shipped data, verified against Genesis 1.1.2)
--------------------------------------------------------------------------------

:class:`~unienv_genesis_collection.scenes.room_builder.RoomLayoutSpec` uses a
**room frame: origin at the floor centre, ``z`` up**.  The two source families
need different conversions:

* **ReplicaCAD** (``scenes/rooms/replicacad/arrangements/<name>.scene_instance.json``):

  - The *stage frame* the scene-instance translations live in is the stage
    GLB's own frame.  The stage GLBs are glTF assets — **Y-up** (verified:
    ``frl_apartment_stage.glb`` spans 7.3 × 3.1 × 13.0 m, thin along ``+y``),
    so the stage's up axis is ``+y`` and its horizontal plane is X/Z.
  - ``translation`` / ``rotation`` are the pose of the articulated object's
    **root (base) link** in the stage frame.  ``translation_origin: "COM"``
    (habitat semantics) means the translation locates the root link's
    **centre of mass**, so the base-link origin is
    ``t_base = t_com − R_stage · com_local`` with ``com_local`` the root
    link's inertial offset in its own frame.  Every ReplicaCAD item shipped by
    this collection has an **inertial-less root link** (``com_local = 0``), so
    the translations *are* base-origin poses; the correction is implemented
    (and unit-tested on a synthetic weighted-root URDF) so a future item with
    a weighted root link converts faithfully.  Cross-checks against the data:
    the floor-standing ``chestOfDrawers_01`` carries ``translation[1] = 0.0``
    exactly (the stage floor is the stage's ``y = 0`` plane) and the upright
    fridge's base origin sits at ``y ≈ 0.98`` — half its 1.92 m height.
  - The scene ``rotation`` composes the item's **upright orientation** (the
    catalog's ``standing_quaternion``, which maps the item's authored up axis
    — ``+y`` for the fridge/cabinet/counter/doors, ``+z`` for the
    chest/cupboard — onto the stage's ``+y``) with a yaw **about the stage's up
    axis**.  The bake removes the standing orientation
    (``R_standing_stage = Mᵀ · R_standing_room``) and takes the residual's
    angle about the up axis as the baked ``yaw`` — asserting the residual is a
    pure up-axis rotation (tilt ≈ 0).  A tilted instance cannot be expressed
    by a yaw-only spec, so it is reported and *skipped*.
  - Stage → room mapping (a proper rotation, M = R_x(+90°)):
    ``pos_room = (t_stage_x, −t_stage_z, t_stage_y − floor_y)`` with
    ``floor_y = 0`` — the shell is loaded at the world origin
    (``RoomBuilder._build_shell`` passes ``pos=(0, 0, 0)``), so the room
    origin *is* the shell origin and the baked ``z`` is the base-link origin's
    height above the shell floor.  The baked ``yaw`` is the item's *upright*
    yaw; :meth:`RoomBuilder.reload` composes the descriptor's catalog
    ``standing_quaternion`` on top of it, which makes the furniture stand
    upright in the built room (verified: the fridge entity's AABB stands
    1.92 m tall with its bottom ~2 cm above the floor, matching the authored
    float).
  - **``uniform_scale``**: ``FurniturePlacement`` has no scale field and
    ``RoomBuilder._resolve_descriptor`` constructs catalog descriptors at
    their default scale, so a non-unit-scale instance cannot be baked
    faithfully.  Such instances are reported and *skipped*.  In this
    collection the scale is a per-item constant — ``kitchen_counter`` 0.98,
    ``kitchenCupboard_01`` 0.38, ``chestOfDrawers_01`` 0.4 — so every
    arrangement keeps its fridge/cabinet/door2 and drops its counter,
    cupboard and chest.

* **RoboCasa kitchen shells** (``scenes/rooms/robocasa_kitchen/<style>/model.xml``):
  static MJCF, already Z-up with the floor top at ``z = 0`` and centred on the
  origin, so ``room_size = (floor width, floor depth, wall top height)`` comes
  straight out of the XML.  The fixtures are laid out **deterministically in
  the counter run**: standing on the floor in front of the run, back face
  flush with the run's room-side face, fronts facing the room interior, spaced
  along the run centred and non-overlapping, using each fixture's own
  offline-FK footprint — the same bbox ``RoboCasaFixture`` exposes to the room
  builder.  Fixtures that do not fit the run's length are skipped in id order.

Baked ids (plan §3.4 / :func:`~...room_builder.baked_spec_relative_path`):

* ``RoomBuilder(arrangement="replicacad/<name>")`` →
  ``load_baked_room_layout_spec("replicacad/<name>", kind="arrangement")`` →
  ``scenes/rooms/replicacad/arrangements/<name>.json``.  The upstream inputs
  keep their ``.scene_instance.json`` names; the baked outputs drop it.
* ``RoomBuilder(layout_yaml="robocasa_kitchen/<style>")`` →
  ``load_baked_room_layout_spec(..., kind="layout")`` →
  ``scenes/rooms/robocasa_kitchen/layouts/<style>.json``.

Shell ids: ``_resolve_shell_file`` accepts a mesh/MJCF *file* id or a
*directory* id containing one recognised shell file (``shell.glb`` /
``model.xml`` / …).  The RoboCasa shells keep the directory form
(``robocasa_kitchen/<style>`` — ``model.xml``).  The ReplicaCAD stage
directories hold only ``<stage_id>.glb`` (no ``shell.glb`` candidate), so
their baked shell ids point at the GLB *file* itself
(``replicacad/<stage_id>/<stage_id>.glb``) — the loader's supported file form.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from unienv_genesis_collection.asset_path import ASSETS_DIR
from unienv_genesis_collection.objects.hub_objects.articulated.replicacad_furniture import (
    REPLICACAD_ASSETS_ROOT,
    get_replicacad_catalog,
)
from unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture import (
    ROBOCASA_ASSETS_ROOT,
)
from unienv_genesis_collection.scenes.room_builder import (
    FurniturePlacement,
    RoomLayoutSpec,
    room_layout_spec_to_json_dict,
)

__all__ = [
    "BAKED_ROTATION_TOL",
    "ROBOCASA_FIXTURE_PLAN",
    "bake_replicacad_arrangement",
    "bake_replicacad_arrangements",
    "bake_robocasa_layout",
    "bake_robocasa_layouts",
    "stage_to_room_pos",
    "urdf_root_com_offset",
    "main",
]

# Residual tilt tolerated for "furniture stands upright": the removed-standing
# rotation's residual must be a pure rotation about the stage's up axis
# (max(|q_x|, |q_z|) ≈ 0).  ReplicaCAD authored quaternions are exact.
BAKED_ROTATION_TOL = 1e-3

# Scale values within this of 1.0 count as unit (the spec has no scale field).
_SCALE_TOL = 1e-4

# Deterministic gap between fixtures laid out along a counter run (metres).
_FIXTURE_GAP = 0.02

# Deterministic fixture plan per RoboCasa kitchen style: one list of fixture
# ids per counter run (the runs are ordered wall-adjacent-first, i.e. the
# north/back run first).  Ids must exist in objects/robocasa/catalog.json.
ROBOCASA_FIXTURE_PLAN: Dict[str, List[List[str]]] = {
    "open_kitchen": [
        ["cabinet_hinge", "drawer", "stove_simple_gas"],  # single back run
    ],
    "galley": [
        ["cabinet_hinge", "drawer", "stove_simple_gas"],  # north (back) run
        ["cabinet_single", "sink_white_sink", "drawer"],  # south run
    ],
}


# ---------------------------------------------------------------------------
# Quaternion helpers (offline-safe, no Genesis import)
# ---------------------------------------------------------------------------


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` → 3×3 rotation matrix (Hamilton)."""
    w, x, y, z = (float(v) for v in quat[:4])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError(f"Degenerate quaternion: {tuple(quat)!r}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """3×3 rotation matrix → unit quaternion ``(w, x, y, z)`` (Shepperd)."""
    R = np.asarray(matrix, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        qw, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError(f"Degenerate rotation matrix: {R.tolist()}")
    quat /= norm
    return tuple(float(v) for v in quat)


# The stage→room mapping: the stage GLBs are glTF (Y-up), the room frame is
# Z-up.  room = (stage_x, -stage_z, stage_y) is a proper rotation (M = R_x(+90°))
# that maps the stage's up axis (+y) onto the room's +z and keeps the stage's
# +x axis as the room's +x.
_STAGE_TO_ROOM = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Yaw about the stage's up axis (+y) and yaw about the room's +z carry the same
# angle: M · R_y(θ) · Mᵀ = R_z(θ), so a yaw extracted in the stage frame is the
# baked yaw in the room frame.
assert np.allclose(
    _STAGE_TO_ROOM @ np.array([[math.cos(0.7), 0, math.sin(0.7)], [0, 1, 0], [-math.sin(0.7), 0, math.cos(0.7)]]) @ _STAGE_TO_ROOM.T,
    np.array([[math.cos(0.7), -math.sin(0.7), 0], [math.sin(0.7), math.cos(0.7), 0], [0, 0, 1]]),
)

# Per-stage / per-item caches: the stage GLBs are multi-MB meshes and the
# arrangement conversion runs per arrangement, so the shell frames and the
# URDF root-link COM offsets are parsed once per process.
_SHELL_ROOM_FRAME_CACHE: Dict[Path, Tuple[float, float, float]] = {}
_URDF_ROOT_COM_CACHE: Dict[Path, Tuple[float, float, float]] = {}


def yaw_about_up_axis(
    rotation: Sequence[float],
    standing_quaternion: Optional[Sequence[float]] = None,
    *,
    tol: float = BAKED_ROTATION_TOL,
) -> Tuple[float, float]:
    """Decompose a scene-instance rotation into ``(yaw, tilt)``.

    The scene rotation composes the item's upright (standing) orientation with a
    yaw about the stage's up axis: ``R_stage = R_y(θ) · R_standing_stage``.  The
    bake removes the standing orientation (expressed in the stage frame via
    ``R_standing_stage = Mᵀ · R_standing_room``, since ``RoomBuilder`` composes
    the room-frame ``standing_quaternion`` on top of the baked yaw) and reads
    the residual's yaw about the up axis.

    Args:
        rotation: Scene-instance quaternion ``(w, x, y, z)`` (stage frame).
        standing_quaternion: The item's catalog ``standing_quaternion``
            ``(w, x, y, z)`` in the Z-up room frame, or ``None`` when the item's
            base-link frame is already upright.
        tol: Maximum accepted residual tilt (``max(|q_x|, |q_z|)``).

    Returns:
        ``(yaw, tilt)`` — the yaw in radians normalised to ``(-pi, pi]`` and the
        residual tilt.  A ``tilt > tol`` instance is tilted: it cannot be
        expressed as a yaw-only placement and must be skipped.

    Raises:
        ValueError: If *rotation* is not a unit quaternion.
    """
    R_stage = quat_to_matrix(tuple(float(v) for v in rotation[:4]))
    standing_room = (
        None if standing_quaternion is None else tuple(float(v) for v in standing_quaternion[:4])
    )
    # The item's upright orientation, expressed in the stage (Y-up) frame:
    # R_standing_room = M · R_standing_stage  →  R_standing_stage = Mᵀ · R_standing_room.
    R_standing_stage = _STAGE_TO_ROOM.T @ (np.eye(3) if standing_room is None else quat_to_matrix(standing_room))
    residual = R_stage @ R_standing_stage.T
    qw, qx, qy, qz = matrix_to_quat(residual)
    tilt = max(abs(qx), abs(qz))
    yaw = 2.0 * math.atan2(qy, qw)
    if yaw > math.pi:
        yaw -= 2.0 * math.pi
    elif yaw <= -math.pi:
        yaw += 2.0 * math.pi
    return yaw, tilt


def stage_to_room_pos(stage_pos: Sequence[float]) -> Tuple[float, float, float]:
    """Stage-frame position (Y-up) → room-frame position (Z-up, floor at z=0)."""
    x, y, z = (float(v) for v in stage_pos[:3])
    return (x, -z, y)


# ---------------------------------------------------------------------------
# ReplicaCAD arrangement baking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrangementReport:
    """What one arrangement bake produced (printed verbatim by the CLI)."""

    name: str
    spec: RoomLayoutSpec
    kept: Tuple[str, ...]
    skipped: Tuple[Tuple[str, str], ...]
    warnings: Tuple[str, ...]
    baked_path: Path


def urdf_root_com_offset(urdf_file: Path) -> Tuple[float, float, float]:
    """Root-link inertial COM offset in the root-link's own frame (metres).

    Every ReplicaCAD item in this collection has an inertial-less root link, so
    the offset is ``(0, 0, 0)`` and a ``translation_origin: "COM"`` translation
    *is* the base-link origin position.  A weighted root link is honoured:
    ``t_base = t_com − R_stage · com_local``.
    """
    cached = _URDF_ROOT_COM_CACHE.get(urdf_file)
    if cached is not None:
        return cached
    root = ET.parse(urdf_file).getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = [link for link in root.findall("link") if link.get("name") not in children]
    if len(roots) != 1:
        raise ValueError(
            f"{urdf_file}: expected exactly one root link, got "
            f"{[link.get('name') for link in roots]}."
        )
    inertial = roots[0].find("inertial")
    if inertial is None:
        return (0.0, 0.0, 0.0)
    origin = inertial.find("origin")
    if origin is None or origin.get("xyz") is None:
        return (0.0, 0.0, 0.0)
    values = [float(v) for v in str(origin.get("xyz")).replace(",", " ").split()]
    if len(values) != 3:
        raise ValueError(f"{urdf_file}: root inertial origin must be xyz, got {values!r}.")
    offset = tuple(values)
    _URDF_ROOT_COM_CACHE[urdf_file] = offset
    return offset


def stage_shell_glb(shells_dir: Path, stage_id: str) -> Path:
    """The stage shell GLB under ``scenes/rooms/replicacad/<stage_id>/``.

    ``RoomBuilder._resolve_shell_file`` needs either a recognised candidate
    name (``shell.glb`` / …) or a direct file id; the stage directories hold a
    single ``<stage_id>.glb``, which the baked ``shell_id`` points at directly.
    """
    stage_dir = shells_dir / stage_id
    glbs = sorted(stage_dir.glob("*.glb"))
    if len(glbs) != 1:
        raise FileNotFoundError(
            f"Stage shell directory {stage_dir} must hold exactly one .glb, "
            f"found {[p.name for p in glbs]}."
        )
    return glbs[0]


def _shell_room_frame(glb: Path) -> Tuple[float, float, float]:
    """Room-frame ``(width, depth, wall_height)`` of a stage shell GLB.

    The GLB is glTF (Y-up).  The shell is loaded at the world origin, so the
    room origin *is* the shell origin; ``room_size`` is the largest
    ``(width, depth)`` rectangle centred on the shell origin that still lies
    inside the shell's bounding box (so ``sample_free_floor_positions`` never
    samples outside the shell), and the wall height is the shell's
    floor-to-max height (the stage floor is the ``y = 0`` plane).
    """
    cached = _SHELL_ROOM_FRAME_CACHE.get(glb)
    if cached is not None:
        return cached
    import trimesh

    mesh = trimesh.load(str(glb), force="mesh")
    lo, hi = mesh.bounds  # glTF: Y-up
    width = 2.0 * min(abs(float(lo[0])), abs(float(hi[0])))
    # Stage z → room -y (the mapping negates the z axis).
    depth = 2.0 * min(abs(float(lo[2])), abs(float(hi[2])))
    wall_height = max(float(hi[1]), 1e-3)
    frame = (width, depth, wall_height)
    _SHELL_ROOM_FRAME_CACHE[glb] = frame
    return frame


def _load_replicacad_catalog() -> Dict[str, object]:
    """The ReplicaCAD furniture catalog (the collection's loader).

    A thin indirection so the offline tests can inject a synthetic catalog.
    """
    return get_replicacad_catalog()


def bake_replicacad_arrangement(
    scene_instance_file: Path,
    source_assets: Path,
    out_dir: Path,
    *,
    shells_dir: Optional[Path] = None,
    tilt_tol: float = BAKED_ROTATION_TOL,
) -> ArrangementReport:
    """Convert one ReplicaCAD ``<name>.scene_instance.json`` to a baked spec.

    The ``articulated_object_instances`` are the furniture (the rigid
    ``object_instances`` are clutter and skipped by design).  Each instance
    contributes its base-link origin pose in the room frame:

    * non-unit ``uniform_scale`` → skipped (``FurniturePlacement`` has no scale
      field; ``RoomBuilder`` loads catalog descriptors at scale 1.0);
    * ``fixed_base: false`` / unknown template ids → skipped;
    * ``translation_origin: "COM"`` → corrected to the base-link origin via the
      URDF root-link inertial offset;
    * non-upright rotations (residual tilt beyond ``tilt_tol`` after removing
      the catalog's standing orientation) → skipped with a warning.
    """
    name = scene_instance_file.name
    if name.endswith(".scene_instance.json"):
        name = name[: -len(".scene_instance.json")]
    with scene_instance_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    stage_id = Path(str(data["stage_instance"]["template_name"])).name
    shells_root = (shells_dir or source_assets) / "scenes" / "rooms" / "replicacad"
    glb = stage_shell_glb(shells_root, stage_id)
    room_size = _shell_room_frame(glb)

    catalog = _load_replicacad_catalog()
    kept: List[str] = []
    skipped: List[Tuple[str, str]] = []
    warnings_out: List[str] = []
    placements: List[FurniturePlacement] = []

    for instance in data.get("articulated_object_instances", []):
        template = str(instance["template_name"])
        info = catalog.get(template)
        if info is None:
            skipped.append((template, "not a ReplicaCAD catalog furniture id"))
            continue
        if not bool(instance.get("fixed_base", True)):
            skipped.append((template, "fixed_base is False (movable furniture)"))
            continue
        scale = float(instance.get("uniform_scale", 1.0) or 1.0)
        if abs(scale - 1.0) > _SCALE_TOL:
            skipped.append(
                (
                    template,
                    f"uniform_scale {scale:g}: FurniturePlacement has no scale "
                    "field and RoomBuilder loads catalog descriptors at scale 1.0",
                )
            )
            continue

        translation = np.array(
            [float(v) for v in instance["translation"]], dtype=np.float64
        )
        rotation = tuple(float(v) for v in instance["rotation"])
        origin_tag = str(instance.get("translation_origin", "COM"))
        if origin_tag == "COM":
            com_local = np.array(
                urdf_root_com_offset(
                    source_assets / REPLICACAD_ASSETS_ROOT / template / f"{template}.urdf"
                ),
                dtype=np.float64,
            )
        elif origin_tag == "origin":
            com_local = np.zeros(3)
        else:
            warnings_out.append(
                f"{name}: {template} carries unknown translation_origin "
                f"{origin_tag!r} — treated as the base-link origin."
            )
            com_local = np.zeros(3)

        # COM → base-link origin: t_base = t_com − R_stage · com_local.
        R_stage = quat_to_matrix(rotation)
        base = translation - R_stage @ com_local

        yaw, tilt = yaw_about_up_axis(rotation, info.standing_quaternion, tol=tilt_tol)
        if tilt > tilt_tol:
            warnings_out.append(
                f"{name}: {template} rotation is not upright (residual tilt "
                f"{tilt:.4g} > {tilt_tol:g}) — skipped (a yaw-only spec cannot "
                "carry roll/pitch)."
            )
            continue
        kept.append(template)
        placements.append(
            FurniturePlacement(
                object_id=template,
                source="replicacad",
                pos=(float(base[0]), float(-base[2]), float(base[1])),
                yaw=yaw,
            )
        )

    spec = RoomLayoutSpec(
        shell_id=f"replicacad/{stage_id}/{glb.name}",
        room_size=room_size,
        furniture=tuple(placements),
    )
    out_path = (
        out_dir / "scenes" / "rooms" / "replicacad" / "arrangements" / f"{name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(room_layout_spec_to_json_dict(spec), handle, indent=2)
        handle.write("\n")
    return ArrangementReport(
        name=name,
        spec=spec,
        kept=tuple(kept),
        skipped=tuple(skipped),
        warnings=tuple(warnings_out),
        baked_path=out_path,
    )


def bake_replicacad_arrangements(
    source_assets: Path,
    out_dir: Path,
    *,
    names: Optional[Sequence[str]] = None,
) -> List[ArrangementReport]:
    """Bake every (or the selected) ReplicaCAD arrangement."""
    arrangements_dir = (
        source_assets / "scenes" / "rooms" / "replicacad" / "arrangements"
    )
    files = sorted(arrangements_dir.glob("*.scene_instance.json"))
    if names is not None:
        wanted = {str(n) for n in names}
        files = [
            f for f in files if f.name[: -len(".scene_instance.json")] in wanted
        ]
    reports: List[ArrangementReport] = []
    for scene_file in files:
        report = bake_replicacad_arrangement(scene_file, source_assets, out_dir)
        reports.append(report)
        kept = f"{len(report.kept)} kept" if report.kept else "0 kept"
        skipped = f", {len(report.skipped)} skipped" if report.skipped else ""
        print(
            f"  {report.name}: shell={report.spec.shell_id} "
            f"room_size={tuple(round(v, 3) for v in report.spec.room_size)} — "
            f"{kept}{skipped} → {report.baked_path}"
        )
        for template, reason in report.skipped:
            print(f"    skipped {template}: {reason}")
        for message in report.warnings:
            print(f"    warning: {message}")
    return reports


# ---------------------------------------------------------------------------
# RoboCasa kitchen layout baking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CounterRun:
    """One counter run of a RoboCasa kitchen shell (axis-aligned, Z-up)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    top_z: float

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def back_is_positive_y(self) -> bool:
        """Whether the run's wall-side face lies on its ``+y`` side."""
        return abs(self.y_max) > abs(self.y_min)

    @property
    def y_back(self) -> float:
        """The run's wall-side face (flush against the wall behind it)."""
        return self.y_max if self.back_is_positive_y else self.y_min

    @property
    def y_front(self) -> float:
        """The run's room-side face."""
        return self.y_min if self.back_is_positive_y else self.y_max


def parse_robocasa_shell(
    model_xml: Path,
) -> Tuple[Tuple[float, float, float], List[_CounterRun]]:
    """Room size + counter runs of a RoboCasa kitchen shell MJCF (Z-up).

    The shell's floor must be centred on the origin with its top at ``z = 0``
    (the room-frame contract for a baked shell).  ``room_size`` is the floor's
    ``(width, depth)`` with the walls' top as the height; the counter runs come
    from the ``counter_base_*`` / ``counter_top_*`` geoms (the paired top slab
    gives the work-surface height).
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(str(model_xml)).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{model_xml}: no <worldbody>.")
    geoms = [geom for geom in worldbody.iter("geom") if geom.get("type") == "box"]

    def box_bounds(geom) -> Tuple[np.ndarray, np.ndarray]:
        size = np.array(
            [float(v) for v in str(geom.get("size")).replace(",", " ").split()],
            dtype=np.float64,
        )
        pos_raw = geom.get("pos")
        pos = (
            np.zeros(3)
            if pos_raw is None
            else np.array(
                [float(v) for v in str(pos_raw).replace(",", " ").split()],
                dtype=np.float64,
            )
        )
        return pos - size, pos + size

    floor = next((geom for geom in geoms if geom.get("name") == "floor"), None)
    if floor is None:
        raise ValueError(f"{model_xml}: no 'floor' geom.")
    floor_lo, floor_hi = box_bounds(floor)
    if (
        abs(float(floor_lo[0]) + float(floor_hi[0])) > 1e-6
        or abs(float(floor_lo[1]) + float(floor_hi[1])) > 1e-6
        or abs(float(floor_hi[2])) > 1e-6
    ):
        raise ValueError(
            f"{model_xml}: the shell floor must be centred on the origin with "
            f"its top at z=0, got lo={floor_lo.tolist()} hi={floor_hi.tolist()}."
        )
    room_size = (
        float(floor_hi[0] - floor_lo[0]),
        float(floor_hi[1] - floor_lo[1]),
        0.0,
    )

    wall_height = 0.0
    top_slabs: Dict[str, Tuple[float, float]] = {}
    for geom in geoms:
        label = str(geom.get("name", ""))
        g_lo, g_hi = box_bounds(geom)
        if label.startswith("wall"):
            wall_height = max(wall_height, float(g_hi[2]))
        if label.startswith("counter_top_"):
            top_slabs[label] = (float(g_hi[0]), float(g_hi[2]))
    room_size = (room_size[0], room_size[1], wall_height)

    runs: List[_CounterRun] = []
    for geom in geoms:
        label = str(geom.get("name", ""))
        if not label.startswith("counter_base"):
            continue
        g_lo, g_hi = box_bounds(geom)
        index = label.rsplit("_", 1)[-1]
        top_z = float(g_hi[2])
        slab = top_slabs.get(f"counter_top_{index}")
        if slab is not None:
            top_z = slab[1]
        runs.append(
            _CounterRun(
                x_min=float(g_lo[0]),
                x_max=float(g_hi[0]),
                y_min=float(g_lo[1]),
                y_max=float(g_hi[1]),
                top_z=top_z,
            )
        )
    runs.sort(key=lambda run: -abs(run.y_back))  # the wall-adjacent run first
    return room_size, runs


def load_fixture_geometry(
    fixture_ids: Sequence[str],
    source_assets: Path,
) -> Dict[str, Dict[str, float]]:
    """Offline-FK footprint per fixture id (``x_lo/x_hi/y_lo/y_hi/z_lo/z_hi``).

    Uses the collection's MJCF forward-kinematics walker — the same bbox
    ``RoboCasaFixture.get_local_bounding_box()`` exposes to the room builder —
    so the baked footprints match what the loader builds.
    """
    from unienv_genesis_collection.objects.articulated_object import (
        mjcf_link_meshes_at_config,
    )

    geometry: Dict[str, Dict[str, float]] = {}
    for fixture_id in fixture_ids:
        model_xml = source_assets / ROBOCASA_ASSETS_ROOT / fixture_id / "model.xml"
        meshes = mjcf_link_meshes_at_config(str(model_xml))
        lo = np.min([m.bounds[0] for m in meshes.values()], axis=0)
        hi = np.max([m.bounds[1] for m in meshes.values()], axis=0)
        geometry[fixture_id] = {
            "x_lo": float(lo[0]),
            "x_hi": float(hi[0]),
            "y_lo": float(lo[1]),
            "y_hi": float(hi[1]),
            "z_lo": float(lo[2]),
            "z_hi": float(hi[2]),
        }
    return geometry


def _layout_fixtures_along_run(
    run: _CounterRun,
    fixture_ids: Sequence[str],
    geometry: Mapping[str, Mapping[str, float]],
    *,
    gap: float = _FIXTURE_GAP,
) -> Tuple[List[FurniturePlacement], List[Tuple[str, str]]]:
    """Deterministic fixture row for one counter run.

    The fixtures stand on the floor in front of the run: back face flush with
    the run's room-side face, fronts (the fixture's local ``-y`` at yaw 0)
    facing the room interior, laid out along the run centred and
    non-overlapping.  Fixtures that do not fit the run's remaining length are
    skipped in id order.
    """
    widths = {
        fid: geometry[fid]["x_hi"] - geometry[fid]["x_lo"]
        for fid in fixture_ids
        if fid in geometry
    }
    ordered = [fid for fid in fixture_ids if fid in geometry]
    total = sum(widths[fid] for fid in ordered) + gap * max(len(ordered) - 1, 0)
    x_centre = (run.x_min + run.x_max) / 2.0
    yaw = 0.0 if run.back_is_positive_y else math.pi
    placements: List[FurniturePlacement] = []
    skipped: List[Tuple[str, str]] = []
    cursor = x_centre - total / 2.0
    if cursor < run.x_min:
        skipped.append(("run", f"row of {total:.3f} m exceeds run {run.length:.3f} m"))
        return [], skipped
    for fixture_id in ordered:
        geometry_i = geometry[fixture_id]
        width = geometry_i["x_hi"] - geometry_i["x_lo"]
        slot_centre = cursor + width / 2.0
        cx_local = (geometry_i["x_lo"] + geometry_i["x_hi"]) / 2.0
        pos_x = slot_centre - cx_local if yaw == 0.0 else slot_centre + cx_local
        # The fixture's back (its local +y at yaw 0) flush with the run's
        # room-side face; the front (the local -y) faces the room interior.
        pos_y = (
            run.y_front - geometry_i["y_hi"]
            if run.back_is_positive_y
            else run.y_front + geometry_i["y_hi"]
        )
        # Standing on the floor: the bbox bottom rests at z=0 (the loader's
        # `_floor_z` convention — the fixture's local origin sits at -z_lo).
        pos_z = -geometry_i["z_lo"]
        placements.append(
            FurniturePlacement(
                object_id=fixture_id,
                source="robocasa",
                pos=(pos_x, pos_y, pos_z),
                yaw=yaw,
            )
        )
        cursor += width + gap
    return placements, []


def bake_robocasa_layout(
    style: str,
    source_assets: Path,
    out_dir: Path,
    *,
    fixture_plan: Optional[Mapping[str, Sequence[Sequence[str]]]] = None,
) -> Tuple[RoomLayoutSpec, List[str]]:
    """Bake one deterministic RoboCasa kitchen layout (see module docstring)."""
    room_size, runs = parse_robocasa_shell(
        source_assets / "scenes" / "rooms" / "robocasa_kitchen" / style / "model.xml"
    )
    plan = dict(fixture_plan if fixture_plan is not None else ROBOCASA_FIXTURE_PLAN)
    run_plans = [list(run_ids) for run_ids in plan.get(style, [])]
    geometry = load_fixture_geometry(
        [fid for run_ids in run_plans for fid in run_ids], source_assets
    )

    placements: List[FurniturePlacement] = []
    warnings_out: List[str] = []
    for run_index, run in enumerate(runs):
        run_ids = run_plans[run_index] if run_index < len(run_plans) else []
        run_placements, run_warnings = _layout_fixtures_along_run(
            run, run_ids, geometry
        )
        placements.extend(run_placements)
        warnings_out.extend(
            f"{style} counter run {run_index + 1}: {message}"
            for _, message in run_warnings
        )

    spec = RoomLayoutSpec(
        shell_id=f"robocasa_kitchen/{style}",
        room_size=room_size,
        furniture=tuple(placements),
    )
    out_path = (
        out_dir / "scenes" / "rooms" / "robocasa_kitchen" / "layouts" / f"{style}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(room_layout_spec_to_json_dict(spec), handle, indent=2)
        handle.write("\n")
    for message in warnings_out:
        print(f"    warning: {message}")
    return spec, warnings_out


def bake_robocasa_layouts(
    source_assets: Path,
    out_dir: Path,
    *,
    styles: Optional[Sequence[str]] = None,
) -> List[Tuple[str, RoomLayoutSpec]]:
    """Bake every RoboCasa kitchen layout whose shell has a ``model.xml``."""
    shells_root = source_assets / "scenes" / "rooms" / "robocasa_kitchen"
    available = sorted(
        entry.name
        for entry in shells_root.iterdir()
        if entry.is_dir() and (entry / "model.xml").is_file()
    )
    wanted = list(styles) if styles is not None else available
    specs: List[Tuple[str, RoomLayoutSpec]] = []
    for style in wanted:
        if style not in available:
            raise FileNotFoundError(
                f"RoboCasa kitchen shell {style!r} has no model.xml under "
                f"{shells_root}."
            )
        spec, _ = bake_robocasa_layout(style, source_assets, out_dir)
        specs.append((style, spec))
        print(
            f"  {style}: shell={spec.shell_id} room_size={spec.room_size} "
            f"{len(spec.furniture)} fixtures → "
            f"{out_dir / 'scenes' / 'rooms' / 'robocasa_kitchen' / 'layouts' / (style + '.json')}"
        )
        for placement in spec.furniture:
            print(
                f"    {placement.object_id:32s} pos="
                f"{tuple(round(v, 3) for v in placement.pos)} yaw={placement.yaw:.3f}"
            )
    return specs


# ---------------------------------------------------------------------------
# CLI (mirrors scripts/prepare_*.py: argparse, --source optional, prints)
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Assets root holding the ReplicaCAD arrangements, stage shells and "
             "RoboCasa kitchen shells (default: the collection's own assets "
             "directory, read via unienv_genesis_collection.asset_path).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Assets root to write the baked specs into (default: --source, i.e. "
             "the baked files live next to their upstream inputs).",
    )
    parser.add_argument(
        "--families", default="replicacad,robocasa",
        help="Comma-separated families to bake: replicacad, robocasa. Default: both.",
    )
    parser.add_argument(
        "--arrangement", action="append", default=None,
        help="Only bake the named ReplicaCAD arrangement (repeatable).",
    )
    args = parser.parse_args(argv)

    source = (args.source or ASSETS_DIR).expanduser().resolve()
    out_dir = (args.output_dir or source).expanduser().resolve()
    families = {token.strip() for token in args.families.split(",") if token.strip()}
    unknown = families - {"replicacad", "robocasa"}
    if unknown:
        raise SystemExit(f"Unknown families {sorted(unknown)} (use replicacad, robocasa).")

    print(f"Source: {source}")
    print(f"Output: {out_dir}")

    if "replicacad" in families:
        print("ReplicaCAD arrangements (scenes/rooms/replicacad/arrangements/<name>.json):")
        reports = bake_replicacad_arrangements(
            source, out_dir, names=args.arrangement
        )
        n_kept = sum(len(report.kept) for report in reports)
        n_skipped = sum(len(report.skipped) for report in reports)
        print(
            f"  {len(reports)} arrangements baked, {n_kept} furniture placements, "
            f"{n_skipped} instances skipped (non-unit uniform_scale: "
            "kitchen_counter 0.98 / kitchenCupboard_01 0.38 / chestOfDrawers_01 0.4 "
            "— FurniturePlacement has no scale field —, unknown ids, movable "
            "bases or tilted rotations)."
        )
    if "robocasa" in families:
        print("RoboCasa kitchen layouts (scenes/rooms/robocasa_kitchen/layouts/<style>.json):")
        bake_robocasa_layouts(source, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
