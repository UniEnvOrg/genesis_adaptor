"""Room builder for procedural room generation (plan §3).

:class:`RoomBuilder` is the room counterpart of
:class:`~unienv_genesis_collection.scenes.table_builder.TableBuilder`: a
:class:`~unienv_interface.world.node.WorldNode` that builds a room — a shell
(walls/floor, procedural primitives or a baked shell asset) plus **fixed
articulated furniture** loaded once during ``reload`` via each descriptor's
``load_into_scene`` (the morph pose carries the placement; see plan §3.2).

Layout sources (mutually exclusive):

* ``layout=RoomLayoutSpec(...)`` — explicit spec.
* ``arrangement="replicacad/apt_0"`` — baked ReplicaCAD arrangement
  (``scenes/rooms/replicacad/arrangements/apt_0.json``).
* ``layout_yaml="robocasa_kitchen/open_kitchen"`` — baked RoboCasa kitchen
  layout (``scenes/rooms/robocasa_kitchen/layouts/open_kitchen.json``).
* ``procedural=True`` — sampled layout (plan §3.3): ``n ~ Uniform(n_furniture)``
  descriptors from ``furniture_pool``, placed wall-following (flush against a
  wall with :attr:`wall_clearance`, yaw facing inward) with an interior-grid
  fallback and circular footprint-overlap rejection.

Room frame convention (mirrors :class:`TableBuilder`): the **origin is the
room floor centre** — ``local (0, 0, 0)`` is the centre of the floor, ``z``
is up, the X axis runs along the room *width* and the Y axis along its
*depth*.  Furniture placements are expressed in this frame; the shell is
built around it, so the frame coincides with the world frame.

Conventions shared with ``table_builder.py``: ``WorldNode`` with a static
``reload_priorities`` set, a seedable :class:`random.Random` reseeded in
``reload(seed=...)``, ``(w, x, y, z)`` quaternions throughout and
batch-aware tensors at the API boundary (this builder keeps a *shared*
layout by default — per-env base poses of fixed entities are opt-in via
:meth:`set_furniture_pose_3d` and require ``batch_fixed_verts=True``; DoF
state is per-env regardless, so ``initial_open`` articulation needs no extra
morph flags).

Device convention (same as ``base_object.py`` / ``pickplace.py``): the
task-facing API (:meth:`RoomBuilder.get_receptacles`,
:meth:`RoomBuilder.get_interaction_points`,
:meth:`RoomBuilder.sample_free_floor_positions`) is **world-device**: every
returned tensor lives on ``world.device`` with ``world.dtype``, because the
regions are world-frame quantities a task consumes together with robot state.
Descriptor-local geometry (the offline bboxes the regions are derived from)
stays host-side, exactly like ``get_local_bounding_box``, and is moved
explicitly.  The device/dtype is therefore *pinned* here instead of being
inherited from torch's ambient default device — on genesis ≤1.4.2
``gs.init()`` installed the compute device as the default
(``torch.set_default_device``) and on genesis ≥1.4.3 it no longer does, so
a bare ``torch.tensor(...)`` would follow whatever the last initialized
Genesis context left behind (or plain CPU) rather than this builder's
world.

Baked-spec JSON schema (consumed here, produced by the asset-preparation
stage that converts ReplicaCAD/RoboCasa layouts — plan §3.4)::

    {
      "shell_id": "replicacad/stage_greyscale" | null,
      "room_size": [width, depth, wall_height],
      "furniture": [
        {
          "object_id": "cabinet_hinge_0",       # catalog id (source-specific)
          "source": "robocasa",                 # "robocasa"|"replicacad"|...
          "pos": [x, y, z],                     # room frame, metres
          "yaw": 0.0,                           # radians, Z-up
          "initial_open": {"top_drawer": 0.5}   # optional, joint -> fraction
        }
      ]
    }

``shell_id`` resolves against ``scenes/rooms/<shell_id>``: either a mesh /
MJCF *file* (``.glb`` / ``.obj`` / ``.ply`` / ``.xml``) or a *directory*
containing one recognised shell file (``shell.glb`` / ``shell.obj`` /
``model.xml``).  A baked shell's origin must coincide with the room frame
(floor centre at the origin).  A baked shell whose asset origin is *not*
the centre of its own floor rectangle (e.g. the ReplicaCAD stage GLBs)
therefore gets the largest ``room_size`` rectangle centred on the shell
origin that still lies inside the shell's bounding box, so
``sample_free_floor_positions`` never samples outside the shell.

Baked furniture poses carry the *upright* room-frame placement of each
item's base-link origin: the source frame (ReplicaCAD stage GLBs are
glTF — Y-up, so the stage's up axis is ``+y``) is rotated into the room
frame and ``translation_origin: "COM"`` translations are corrected to the
base-link origin using the URDF root-link inertial offset.  The baked
``yaw`` is the item's upright yaw; the descriptor's catalog
``standing_quaternion`` is composed on top of it by
:meth:`RoomBuilder.reload` (see :func:`placement_load_quat`).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch
from unienv_interface.space import BoxSpace
from unienv_interface.world.node import WorldNode
from xbarray.transformations.rotation_conversions.pytorch import quaternion_multiply

from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.objects.articulated_object import (
    ArticulatedObject,
    ArticulationError,
)
from unienv_genesis_collection.objects.base_object import (
    GraspCandidates,
    bbox_center,
    bbox_size,
    make_bbox,
)

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity as RigidEntity

    from unienv_genesis.world.genesis_world import GenesisWorld

__all__ = [
    "DEFAULT_ROOM_SIZE",
    "DEFAULT_N_FURNITURE",
    "DEFAULT_WALL_CLEARANCE",
    "DEFAULT_WALL_THICKNESS",
    "DEFAULT_INTERACTION_GRIPPER_WIDTH",
    "PLACEMENT_PADDING",
    "FurniturePlacement",
    "RoomLayoutSpec",
    "RoomBuilder",
    "placement_load_quat",
    "load_room_layout_spec_json",
    "room_layout_spec_to_json_dict",
    "baked_spec_relative_path",
    "load_baked_room_layout_spec",
]

# Default room interior (width, depth, wall height) in metres.
DEFAULT_ROOM_SIZE = (6.0, 5.0, 2.7)
# Default number of furniture pieces sampled in procedural mode (inclusive).
DEFAULT_N_FURNITURE = (3, 6)
# Gap kept between furniture and a wall's inner face, in metres.
DEFAULT_WALL_CLEARANCE = 0.05
# Procedural wall / floor slab thickness, in metres.
DEFAULT_WALL_THICKNESS = 0.1
# Thickness of the "top" receptacle slab cut from a furniture union bbox.
DEFAULT_RECEPTACLE_SLAB_THICKNESS = 0.04
# Shrink margin applied to a moving link's bbox for its interior region.
DEFAULT_INTERIOR_MARGIN = 0.02
# Default gripper opening used by :meth:`RoomBuilder.get_interaction_points`.
DEFAULT_INTERACTION_GRIPPER_WIDTH = 0.08
# Extra padding (metres) added to each footprint in overlap rejection.
PLACEMENT_PADDING = 0.10
# Wall-placement attempts before falling back to the interior grid.
MAX_WALL_ATTEMPTS = 16
# Rejection-sampling attempts (per requested sample) before giving up.
_FREE_POSITION_MAX_ATTEMPTS_PER_SAMPLE = 200

# Wall ids and their inward-facing yaw (local +x is the furniture front).
_WALL_INWARD_YAW: Tuple[float, float, float, float] = (
    -math.pi / 2,  # 0: north wall (+y), faces -y
    +math.pi / 2,  # 1: south wall (-y), faces +y
    math.pi,       # 2: east wall (+x), faces -x
    0.0,           # 3: west wall (-x), faces +x
)
_WALL_NAMES: Tuple[str, str, str, str] = ("north", "south", "east", "west")


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Convert a Z-axis yaw angle (radians) to a quaternion ``(w, x, y, z)``.

    Import-vs-inline decision (same rationale as
    ``unienv_genesis_collection.objects.articulated_object``): ``yaw_to_quat``
    already exists in ``table_builder.py``, but importing it from there would
    pull the whole ``unienv_genesis``/``genesis`` stack (which cannot even be
    imported before ``gs.init()``) plus a module-level asset lookup into this
    module — breaking offline use before any scene is built.  The two-line
    body is inlined instead and pinned to the ``TableBuilder`` behaviour by
    the room-builder tests.
    """
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# ---------------------------------------------------------------------------
# Layout spec dataclasses + JSON (de)serialisation (plan §3.2 / §3.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FurniturePlacement:
    """One fixed furniture instance inside a room layout.

    Attributes:
        object_id: Catalog id of the furniture (e.g. ``"cabinet_hinge_0"``).
            In procedural layouts this is the sampled descriptor's
            :attr:`name` / ``object_id``.
        source: Where the descriptor comes from — ``"robocasa"``,
            ``"replicacad"``, ``"procedural"``, or any other source tag an
            arrangement converter (plan §3.4) emits.  ``"procedural"``
            descriptors are resolved against the builder's
            ``furniture_pool``.
        pos: World position ``(x, y, z)`` in the room frame (origin at floor
            centre).  ``z`` places the entity origin on the floor.
        yaw: Rotation around the Z axis in radians (rooms are yaw-only —
            furniture stands upright).  Descriptors whose base-link frame is
            not upright (the ReplicaCAD URDF convention) have the catalog's
            ``standing_quaternion`` composed on top of it at load time (see
            :func:`placement_load_quat`); *yaw* itself is always the upright,
            source-independent yaw — this is what the baked arrangements
            (plan §3.4) carry.
        initial_open: Joint name → open fraction in ``[0, 1]`` applied at
            ``reset``.  Names are resolved at runtime against the built
            entity; unknown names raise listing the available joints.
    """

    object_id: str
    source: str
    pos: Tuple[float, float, float]
    yaw: float
    initial_open: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RoomLayoutSpec:
    """A fully resolved room layout: shell + furniture placements.

    Attributes:
        shell_id: Shell asset id under ``scenes/rooms/`` (e.g.
            ``"replicacad/stage_greyscale"``), or ``None`` for procedural
            walls/floor built from primitives around ``room_size``.
        room_size: ``(width, depth, wall_height)`` in metres.  The shell's
            inner floor spans ``width × depth`` centred on the room origin.
        furniture: Fixed furniture placements, in load order.
    """

    shell_id: Optional[str] = None
    room_size: Tuple[float, float, float] = DEFAULT_ROOM_SIZE
    furniture: Tuple[FurniturePlacement, ...] = ()


def _validate_room_size(room_size: Sequence[float]) -> Tuple[float, float, float]:
    try:
        size = tuple(float(v) for v in room_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"room_size must be numeric (width, depth, height): {room_size!r}") from exc
    if len(size) != 3 or any(v <= 0.0 for v in size):
        raise ValueError(
            f"room_size must be positive (width, depth, wall_height), got {size!r}"
        )
    return size


def _validate_spec(spec: RoomLayoutSpec) -> None:
    """Validate a resolved layout spec (sizes, placement tuples)."""
    _validate_room_size(spec.room_size)
    for placement in spec.furniture:
        if not placement.object_id:
            raise ValueError("FurniturePlacement.object_id must be a non-empty string.")
        if len(placement.pos) != 3:
            raise ValueError(
                f"FurniturePlacement {placement.object_id!r}: pos must be (x, y, z)."
            )


def _placement_from_mapping(data: Any) -> FurniturePlacement:
    """Parse one ``FurniturePlacement`` from baked-spec JSON."""
    if not isinstance(data, Mapping):
        raise ValueError(f"Furniture entry must be an object, got {data!r}.")
    try:
        object_id = str(data["object_id"])
        source = str(data["source"])
        pos = tuple(float(v) for v in data["pos"])
        yaw = float(data.get("yaw", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid furniture placement entry {data!r}: {exc}") from exc
    if len(pos) != 3:
        raise ValueError(f"Invalid furniture placement {object_id!r}: pos must be (x, y, z).")
    initial_open_raw = data.get("initial_open") or {}
    if not isinstance(initial_open_raw, Mapping):
        raise ValueError(
            f"Invalid furniture placement {object_id!r}: initial_open must be an "
            "object of joint name -> fraction."
        )
    initial_open: Dict[str, float] = {
        str(joint): float(fraction) for joint, fraction in initial_open_raw.items()
    }
    return FurniturePlacement(
        object_id=object_id,
        source=source,
        pos=pos,
        yaw=yaw,
        initial_open=initial_open,
    )


def _spec_from_mapping(data: Any) -> RoomLayoutSpec:
    """Parse a :class:`RoomLayoutSpec` from baked-spec JSON (see module docstring)."""
    if not isinstance(data, Mapping):
        raise ValueError(f"Room layout spec must be an object, got {data!r}.")
    if "room_size" in data:
        room_size = _validate_room_size(data["room_size"])
    else:
        room_size = DEFAULT_ROOM_SIZE
    shell_id = data.get("shell_id")
    furniture_raw = data.get("furniture") or []
    if not isinstance(furniture_raw, Sequence) or isinstance(furniture_raw, (str, bytes)):
        raise ValueError("Room layout 'furniture' must be a list of placement objects.")
    spec = RoomLayoutSpec(
        shell_id=None if shell_id is None else str(shell_id),
        room_size=room_size,
        furniture=tuple(_placement_from_mapping(entry) for entry in furniture_raw),
    )
    _validate_spec(spec)
    return spec


def load_room_layout_spec_json(path) -> RoomLayoutSpec:
    """Load a :class:`RoomLayoutSpec` from a baked-spec JSON file (plan §3.4)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _spec_from_mapping(data)


def room_layout_spec_to_json_dict(spec: RoomLayoutSpec) -> Dict[str, Any]:
    """Serialise a :class:`RoomLayoutSpec` to the baked-spec JSON structure.

    The inverse of :func:`load_room_layout_spec_json` — writers of baked
    specs (the asset-preparation stage) should emit exactly this structure.
    """
    return {
        "shell_id": spec.shell_id,
        "room_size": [float(v) for v in spec.room_size],
        "furniture": [
            {
                "object_id": placement.object_id,
                "source": placement.source,
                "pos": [float(v) for v in placement.pos],
                "yaw": float(placement.yaw),
                "initial_open": {k: float(v) for k, v in placement.initial_open.items()},
            }
            for placement in spec.furniture
        ],
    }


def baked_spec_relative_path(spec_id: str, *, kind: str) -> Tuple[str, ...]:
    """Asset-relative path components of a baked room layout spec.

    Args:
        spec_id: ``"<source>/<name>"`` — e.g. ``"replicacad/apt_0"`` or
            ``"robocasa_kitchen/open_kitchen"``.
        kind: ``"arrangement"`` (ReplicaCAD scene instances, baked under
            ``scenes/rooms/<source>/arrangements/``) or ``"layout"``
            (RoboCasa kitchen layouts, baked under
            ``scenes/rooms/<source>/layouts/``).

    Returns:
        Path components passed to :func:`get_asset_path`.
    """
    if kind not in ("arrangement", "layout"):
        raise ValueError(f"Unknown baked-spec kind {kind!r} (use 'arrangement' or 'layout').")
    spec_id = str(spec_id).strip("/")
    if spec_id.endswith(".json"):
        spec_id = spec_id[: -len(".json")]
    elif spec_id.endswith((".yaml", ".yml")):
        # A layout_yaml id may carry its (unbaked) YAML extension.
        spec_id = spec_id[: -len(spec_id.split(".", 1)[1]) - 1]
    if "/" not in spec_id:
        raise ValueError(
            f"Baked room spec id must be '<source>/<name>', got {spec_id!r} "
            f"(e.g. 'replicacad/apt_0' or 'robocasa_kitchen/open_kitchen')."
        )
    source, name = spec_id.split("/", 1)
    subdir = "arrangements" if kind == "arrangement" else "layouts"
    return ("scenes", "rooms", source, subdir, f"{name}.json")


def load_baked_room_layout_spec(spec_id: str, *, kind: str = "arrangement") -> RoomLayoutSpec:
    """Resolve a baked :class:`RoomLayoutSpec` by id (plan §3.4).

    Args:
        spec_id: ``"<source>/<name>"`` baked spec id.
        kind: ``"arrangement"`` or ``"layout"`` (see
            :func:`baked_spec_relative_path`).

    Returns:
        The parsed and validated spec.

    Raises:
        FileNotFoundError: If the baked spec is not present in the assets
            collection (baked specs are produced — and uploaded — by the
            asset-preparation stage, plan §3.4; this error names the
            expected path).
        ValueError: If the JSON does not match the baked-spec schema.
    """
    rel = baked_spec_relative_path(spec_id, kind=kind)
    try:
        path = get_asset_path(*rel)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Baked room layout spec {spec_id!r} (kind={kind!r}) is not available. "
            f"Expected asset '{'/'.join(rel)}' in the collection assets. Baked specs "
            "are produced by the asset-preparation stage that converts ReplicaCAD "
            "arrangements / RoboCasa layout YAMLs into RoomLayoutSpec JSON "
            "(plan §3.4); bake and upload them first."
        ) from exc
    return load_room_layout_spec_json(path)


# ---------------------------------------------------------------------------
# Geometry helpers (rotated AABB of a local box under a yaw)
# ---------------------------------------------------------------------------


def _floor_z(size: Sequence[float], center: Sequence[float]) -> float:
    """Placement Z that rests a descriptor's bbox bottom on the floor (z=0)."""
    bottom = float(center[2]) - float(size[2]) / 2.0
    return max(0.0, -bottom)


def _is_identity_quat(quat: Optional[Sequence[float]]) -> bool:
    """Whether *quat* is ``None`` or the identity (within FP tolerance)."""
    if quat is None:
        return True
    w, x, y, z = (float(v) for v in quat[:4])
    return math.isclose(w, 1.0, abs_tol=1e-9) and all(
        math.isclose(v, 0.0, abs_tol=1e-9) for v in (x, y, z)
    )


def placement_load_quat(
    yaw: float,
    standing_quaternion: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float, float]:
    """Entity orientation for a placement: ``yaw ⊗ standing_quaternion``.

    ``FurniturePlacement.yaw`` is the *upright* room-frame yaw (the spec
    contract: "rooms are yaw-only — furniture stands upright").  Some asset
    sources author their base-link frames in a non-Z-up convention (e.g. the
    ReplicaCAD URDFs, whose base-link frames lie the item down: height along
    ``+y``); their catalog carries a ``standing_quaternion`` — the upright
    placement orientation in a Z-up world — which any yaw must be composed
    on top of, exactly like ``ManipulatableObject._standing_quaternion``
    composes a random world yaw on top of it (``q = yaw ⊗ standing``).

    Descriptors without a ``standing_quaternion`` (or with the identity) load
    with the plain yaw quaternion — the historical behaviour.
    """
    quat = yaw_to_quat(yaw)
    if standing_quaternion is None or _is_identity_quat(standing_quaternion):
        return quat
    from xbarray.transformations.rotation_conversions.pytorch import (
        quaternion_multiply as _qm,
    )

    standing = torch.as_tensor(
        tuple(float(v) for v in standing_quaternion[:4]), dtype=torch.float64
    )
    composed = _qm(torch.as_tensor(quat, dtype=torch.float64), standing).tolist()
    # Re-normalise against accumulated FP error, keep the w >= 0 hemisphere
    # (matching yaw_to_quat's sign convention for |yaw| <= pi).
    norm = math.sqrt(sum(v * v for v in composed))
    composed = [v / norm for v in composed] if norm > 0.0 else list(quat)
    if composed[0] < 0.0:
        composed = [-v for v in composed]
    return tuple(composed)


def _rotated_aabb(
    size: Sequence[float],
    center: Sequence[float],
    yaw: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Axis-aligned XY extents of a box rotated by *yaw* about Z.

    Args:
        size: ``(sx, sy, sz)`` local box extents.
        center: ``(cx, cy, cz)`` local box centre (the Z component is
            returned untouched).

    Returns:
        ``(center_xy, half_extents_xy, lo_hi_z)``: the rotated footprint
        centre, the rotated AABB half extents on X/Y, and the Z range of
        the box as ``(z_min, z_max)``.
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    cx, cy, cz = (float(v) for v in center[:3])
    sx, sy, sz = (float(v) for v in size[:3])
    center_xy = (cx * c - cy * s, cx * s + cy * c)
    half_x = abs(c) * sx / 2.0 + abs(s) * sy / 2.0
    half_y = abs(s) * sx / 2.0 + abs(c) * sy / 2.0
    return center_xy, (half_x, half_y), (cz - sz / 2.0, cz + sz / 2.0)


# ---------------------------------------------------------------------------
# RoomBuilder
# ---------------------------------------------------------------------------


class RoomBuilder(WorldNode):
    """Builds a room: shell (walls/floor) + fixed articulated furniture.

    The room frame origin is the **floor centre** (mirroring
    :class:`TableBuilder`'s anchor convention): :meth:`xy_to_world` and
    :meth:`placement_world_pose` map room-local coordinates to world
    coordinates (identity here, but the composition math is kept so a future
    room origin/yaw stays correct).

    Modes (mutually exclusive — pass exactly one):

    * ``layout=RoomLayoutSpec(...)``: explicit spec.
    * ``arrangement="<source>/<name>"``: baked ReplicaCAD arrangement JSON.
    * ``layout_yaml="<source>/<name>"``: baked RoboCasa kitchen layout JSON.
    * ``procedural=True``: seeded layout sampled from ``furniture_pool``.

    With no mode, the builder creates only the procedural shell for
    ``room_size`` (empty furniture) — useful for shell-only rooms.

    Furniture is loaded **once** in :meth:`reload` through the descriptor's
    ``load_into_scene`` (plan §3.2): the morph pose carries the placement, so
    no post-load ``set_pos`` is needed.  Entities are stored on
    :attr:`furniture_entities` (keyed by furniture key) together with their
    descriptors on :attr:`furniture`.

    Batched worlds: ``batch_fixed_verts`` defaults to
    ``world.batch_size is not None`` (the exact check
    ``pickplace.py`` performs), which enables per-env *base poses* of fixed
    entities (opt-in per-env layouts through :meth:`set_furniture_pose_3d`)
    at a significant memory cost.  Without it all envs share one layout —
    DoF state is per-env regardless, so per-env ``initial_open``
    articulation randomisation needs no extra morph flags (plan §3.2).

    Args:
        world: :class:`~unienv_genesis.world.genesis_world.GenesisWorld` instance.
        name: Base name for the room's entities (prefixed per furniture key).
        layout: Explicit :class:`RoomLayoutSpec`.
        arrangement: Baked ReplicaCAD arrangement id (``"<source>/<name>"``).
        layout_yaml: Baked RoboCasa layout id (``"<source>/<name>"``).
        procedural: Sample the layout from ``furniture_pool`` (plan §3.3).
        room_size: ``(width, depth, wall_height)`` in metres (procedural /
            shell-only mode; an explicit spec carries its own).
        furniture_pool: Furniture descriptors.  In procedural mode this is
            the sampling pool; in the other modes it doubles as an explicit
            id → descriptor registry (matched by ``name`` or ``object_id``)
            that shadows hub-catalog resolution.
        n_furniture: Inclusive ``(min, max)`` number of pieces sampled in
            procedural mode.
        wall_clearance: Gap between furniture and a wall's inner face
            (metres); also the margin kept by
            :meth:`sample_free_floor_positions`.
        batch_fixed_verts: Forwarded to every ``load_into_scene`` morph.
            ``None`` defaults to ``world.batch_size is not None``.
        seed: Seed for the procedural sampler (also reseedable per
            ``reload(seed=...)``).

    Raises:
        ValueError: On mutually exclusive modes or invalid parameters.
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    # Rooms are the outermost structure: before TableBuilder (100), robots
    # (50) and clutter (25).
    reload_priorities = {110}
    # Apply each placement's initial_open joint state every episode.
    reset_priorities = {110}
    # Mirror pickplace: initialise joint states right after the first reload
    # (the reload phase alone does not run reset()).
    after_reload_priorities = {110}

    def __init__(
        self,
        world: "GenesisWorld",
        name: str = "room",
        *,
        layout: Optional[RoomLayoutSpec] = None,
        arrangement: Optional[str] = None,
        layout_yaml: Optional[str] = None,
        procedural: bool = False,
        room_size: Tuple[float, float, float] = DEFAULT_ROOM_SIZE,
        furniture_pool: Optional[Sequence[ArticulatedObject]] = None,
        n_furniture: Tuple[int, int] = DEFAULT_N_FURNITURE,
        wall_clearance: float = DEFAULT_WALL_CLEARANCE,
        batch_fixed_verts: Optional[bool] = None,
        seed: Optional[int] = None,
    ):
        self.world = world
        self._name = name

        modes = [
            ("layout", layout is not None),
            ("arrangement", arrangement is not None),
            ("layout_yaml", layout_yaml is not None),
            ("procedural", bool(procedural)),
        ]
        selected = [label for label, given in modes if given]
        if len(selected) > 1:
            raise ValueError(
                "RoomBuilder modes are mutually exclusive: pass only one of "
                f"layout=, arrangement=, layout_yaml=, procedural=True "
                f"(got {', '.join(selected)})."
            )
        self._mode = selected[0] if selected else "shell_only"

        self._layout = layout
        self._arrangement = arrangement
        self._layout_yaml = layout_yaml
        self._procedural = bool(procedural)

        self.default_room_size = _validate_room_size(room_size)
        if wall_clearance < 0.0:
            raise ValueError(f"wall_clearance must be non-negative, got {wall_clearance!r}")
        self.wall_clearance = float(wall_clearance)

        n_lo, n_hi = (int(v) for v in n_furniture)
        if n_lo < 0 or n_hi < n_lo:
            raise ValueError(
                f"n_furniture must be 0 <= min <= max, got {n_furniture!r}"
            )
        self._n_furniture = (n_lo, n_hi)

        self.furniture_pool: List[ArticulatedObject] = (
            list(furniture_pool) if furniture_pool is not None else []
        )
        if self._procedural and not self.furniture_pool:
            raise ValueError(
                "procedural=True requires a non-empty furniture_pool to sample from."
            )

        # Same defaulting as pickplace.py:132 — enables per-env base poses of
        # fixed entities (per-env layout diversity) in batched worlds.
        self._batch_fixed_verts = (
            world.batch_size is not None if batch_fixed_verts is None else bool(batch_fixed_verts)
        )

        # Per-instance RNG for the procedural sampler (TableBuilder._rng
        # pattern): seeded from `seed`, reseeded again on reload(seed=...).
        self._rng = random.Random(seed)

        # Resolved state (populated in reload()).
        self._spec: Optional[RoomLayoutSpec] = None
        self.placements: Dict[str, FurniturePlacement] = {}
        self.furniture: Dict[str, ArticulatedObject] = {}
        self.furniture_entities: Dict[str, "RigidEntity"] = {}
        self.shell_entities: List["RigidEntity"] = []
        self._occupied: List[Tuple[float, float, float]] = []  # (cx, cy, radius)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def layout(self) -> Optional[RoomLayoutSpec]:
        """The resolved :class:`RoomLayoutSpec` (set by :meth:`reload`)."""
        return self._spec

    @property
    def room_size(self) -> Tuple[float, float, float]:
        """``(width, depth, wall_height)`` of the resolved (or default) room."""
        if self._spec is not None:
            return self._spec.room_size
        return self.default_room_size

    @property
    def batch_fixed_verts(self) -> bool:
        """Whether furniture/shell morphs enable per-env pose overrides."""
        return self._batch_fixed_verts

    # ------------------------------------------------------------------
    # Room-frame helpers (origin at floor centre — TableBuilder pattern)
    # ------------------------------------------------------------------

    def _as_room_tensor(self, value) -> torch.Tensor:
        """Materialize a caller-supplied sequence on the world's device/dtype.

        Used by the room-frame helpers.  A :class:`torch.Tensor` input keeps
        its own device (callers broadcast these helpers against tensors they
        already hold, exactly as in :class:`TableBuilder`); a plain sequence
        has no device of its own, so it is created on ``world.device`` /
        ``world.dtype`` rather than on torch's *ambient* default device —
        ``gs.init()`` only keeps that default in sync with the backend it
        initialized last, so relying on it would make the returned device
        depend on process history instead of on this builder's world.
        """
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value, dtype=self.world.dtype, device=self.world.device)

    def xy_to_world(self, local_xy) -> torch.Tensor:
        """Convert room-local XY coordinates to world-frame XY.

        The room frame origin is the floor centre; the room carries no yaw
        of its own, so this is an affine map with the room origin offset
        (identity for the default placement).

        Args:
            local_xy: ``(..., 2)`` local XY coordinates (tensor or sequence).

        Returns:
            ``(..., 2)`` world-frame XY tensor — on ``local_xy``'s device for
            tensor input, on ``world.device`` for a plain sequence.
        """
        local_xy = self._as_room_tensor(local_xy)
        if local_xy.shape[-1] != 2:
            raise ValueError(
                f"local_xy must have last dimension 2, got {tuple(local_xy.shape)}"
            )
        origin = torch.tensor(self._room_origin_xy, dtype=local_xy.dtype, device=local_xy.device)
        c = math.cos(self._room_yaw)
        s = math.sin(self._room_yaw)
        wx = origin[0] + local_xy[..., 0] * c - local_xy[..., 1] * s
        wy = origin[1] + local_xy[..., 0] * s + local_xy[..., 1] * c
        return torch.stack([wx, wy], dim=-1)

    def placement_world_pose(
        self,
        local_pos,
        local_quat=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a room-local position/quaternion to the world frame.

        Mirrors :meth:`TableBuilder.placement_world_pose` with the room
        frame anchored at the floor centre.

        Args:
            local_pos: ``(..., 3)`` or ``(..., 2)`` position in the room
                frame.  A 2-D position gets ``z`` of the room origin (the
                floor, ``0.0``).
            local_quat: ``(..., 4)`` quaternion ``(w, x, y, z)`` in the room
                frame; ``None`` uses the room's own orientation (identity).

        Returns:
            ``(world_pos, world_quat)`` tensors — on ``local_pos``'s device
            for tensor input, on ``world.device`` for a plain sequence.
        """
        local_pos = self._as_room_tensor(local_pos)
        if local_pos.shape[-1] not in (2, 3):
            raise ValueError(
                f"local_pos must have last dimension 2 or 3, got {tuple(local_pos.shape)}"
            )

        world_xy = self.xy_to_world(local_pos[..., :2])
        room_z = torch.tensor(
            self._room_origin_z, dtype=world_xy.dtype, device=world_xy.device
        )
        if local_pos.shape[-1] == 2:
            world_z = torch.full_like(world_xy[..., 0], room_z)
        else:
            world_z = room_z + local_pos[..., 2]
        world_pos = torch.cat([world_xy, world_z.unsqueeze(-1)], dim=-1)

        c = math.cos(self._room_yaw)
        s = math.sin(self._room_yaw)
        room_quat = torch.tensor(
            (math.cos(self._room_yaw / 2.0), 0.0, 0.0, math.sin(self._room_yaw / 2.0)),
            dtype=world_pos.dtype,
            device=world_pos.device,
        )
        if local_quat is None:
            world_quat = room_quat.expand(world_pos.shape[:-1] + (4,))
        else:
            local_quat = torch.as_tensor(local_quat, device=world_pos.device, dtype=world_pos.dtype)
            world_quat = quaternion_multiply(room_quat, local_quat)
        # Keep the trailing batch dims of the input quaternion if broadcast.
        if world_quat.shape != world_pos.shape[:-1] + (4,):
            world_quat = world_quat.expand(world_pos.shape[:-1] + (4,))
        return world_pos, world_quat

    # Internal room-frame constants (the room is anchored at the world
    # origin, floor centre; mirroring TableBuilder's `pos`/`yaw` fields).
    _room_origin_xy: Tuple[float, float] = (0.0, 0.0)
    _room_origin_z: float = 0.0
    _room_yaw: float = 0.0

    # ------------------------------------------------------------------
    # Spec resolution
    # ------------------------------------------------------------------

    def _resolve_spec(self) -> RoomLayoutSpec:
        """Resolve this builder's :class:`RoomLayoutSpec` deterministically."""
        if self._mode == "layout":
            spec = self._layout
        elif self._mode == "arrangement":
            spec = load_baked_room_layout_spec(self._arrangement, kind="arrangement")
        elif self._mode == "layout_yaml":
            spec = load_baked_room_layout_spec(self._layout_yaml, kind="layout")
        elif self._mode == "procedural":
            spec = self._sample_procedural_layout(self._rng)
        else:
            spec = RoomLayoutSpec(
                shell_id=None,
                room_size=self.default_room_size,
                furniture=(),
            )
        _validate_spec(spec)
        return spec

    def resolve_layout_spec(self) -> RoomLayoutSpec:
        """Public spec resolution (re-runs the procedural sampler).

        Uses the builder's internal RNG state — call ``reload(seed=...)``
        (or construct with ``seed=``) for reproducibility.
        """
        self._spec = self._resolve_spec()
        return self._spec

    # ------------------------------------------------------------------
    # Procedural layout sampling (plan §3.3)
    # ------------------------------------------------------------------

    def _sample_procedural_layout(self, rng: random.Random) -> RoomLayoutSpec:
        """Sample a seeded layout: n descriptors, wall-following placement."""
        if not self.furniture_pool:
            raise ValueError(
                "procedural=True requires a non-empty furniture_pool to sample from."
            )
        n = rng.randint(self._n_furniture[0], self._n_furniture[1])
        chosen = [rng.choice(self.furniture_pool) for _ in range(n)]

        occupied: List[Tuple[float, float, float]] = []
        radius_max = max(d.get_footprint_radius() for d in self.furniture_pool)
        placements: List[FurniturePlacement] = []
        for descriptor in chosen:
            pos, yaw = self._place_descriptor(descriptor, occupied, radius_max, rng)
            center_xy, _, _ = self._rotated_aabb_of(descriptor, pos, yaw)
            occupied.append((center_xy[0], center_xy[1], descriptor.get_footprint_radius()))
            placements.append(
                FurniturePlacement(
                    object_id=descriptor.name,
                    source="procedural",
                    pos=pos,
                    yaw=yaw,
                )
            )
        return RoomLayoutSpec(
            shell_id=None,
            room_size=self.default_room_size,
            furniture=tuple(placements),
        )

    def _rotated_aabb_of(
        self,
        descriptor: ArticulatedObject,
        pos: Tuple[float, float, float],
        yaw: float,
        standing_quaternion: Optional[Sequence[float]] = None,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Rotated XY footprint geometry of *descriptor* placed at ``pos/yaw``.

        ``standing_quaternion`` (the descriptor's upright placement
        orientation, see :func:`placement_load_quat`) is composed with the
        yaw into the full entity orientation; the local bbox is transformed
        by that rotation rather than by the bare yaw.  The default ``None``
        keeps the historical yaw-only path (unit standing quaternions).

        Returns:
            ``(center_xy, half_extents_xy, z_range)`` — the footprint centre
            (which generally differs from ``pos[:2]``: the entity origin may
            be off-centre), the rotated AABB half extents, and the bbox Z
            range in world coordinates.
        """
        local = descriptor.get_local_bounding_box()
        size = bbox_size(local)
        center = bbox_center(local)
        if standing_quaternion is None or _is_identity_quat(standing_quaternion):
            center_xy, half_ext, (z_lo, z_hi) = _rotated_aabb(size, center, yaw)
        else:
            from xbarray.transformations.rotation_conversions.pytorch import (
                quaternion_apply,
            )

            corners = torch.stack(torch.meshgrid(
                torch.stack([local.low[0], local.high[0]]),
                torch.stack([local.low[1], local.high[1]]),
                torch.stack([local.low[2], local.high[2]]),
                indexing="ij",
            ), dim=-1).reshape(-1, 3).to(torch.float64)
            quat = torch.as_tensor(
                placement_load_quat(yaw, standing_quaternion), dtype=torch.float64
            )
            world = quaternion_apply(quat, corners)
            lo, hi = world.min(dim=0).values, world.max(dim=0).values
            center_xy = (
                float(lo[0] + hi[0]) / 2.0 + pos[0],
                float(lo[1] + hi[1]) / 2.0 + pos[1],
            )
            half_ext = (float(hi[0] - lo[0]) / 2.0, float(hi[1] - lo[1]) / 2.0)
            return (
                center_xy,
                half_ext,
                (float(lo[2]) + pos[2], float(hi[2]) + pos[2]),
            )
        return (
            (pos[0] + center_xy[0], pos[1] + center_xy[1]),
            half_ext,
            (z_lo + pos[2], z_hi + pos[2]),
        )

    def _footprint_free(
        self,
        center_xy: Tuple[float, float],
        radius: float,
        occupied: Sequence[Tuple[float, float, float]],
    ) -> bool:
        """Circular overlap check against already-placed footprints.

        Uses the descriptor's circumscribed footprint radius (the same
        primitive :class:`~unienv_genesis_collection.scenes.cluttered_table_manager.ClutteredTableManager`
        relies on) padded by :data:`PLACEMENT_PADDING`.
        """
        for (ox, oy, orad) in occupied:
            if math.hypot(center_xy[0] - ox, center_xy[1] - oy) <= (
                radius + orad + PLACEMENT_PADDING
            ):
                return False
        return True

    def _wall_candidate(
        self,
        wall: int,
        size: Sequence[float],
        center: Sequence[float],
        rng: random.Random,
    ) -> Optional[Tuple[Tuple[float, float, float], float]]:
        """Sample a wall-flush candidate: ``None`` if it cannot fit.

        The furniture stands flush against the wall's inner face at
        :attr:`wall_clearance` (exact rotated-AABB distance) with its yaw
        facing the room interior.
        """
        width, depth, _ = self.default_room_size
        clearance = self.wall_clearance
        yaw = _WALL_INWARD_YAW[wall]
        center_xy, (half_x, half_y), _ = _rotated_aabb(size, center, yaw)

        if wall in (0, 1):  # north (+y) / south (-y): flush along Y, slide along X
            x_min, x_max = -width / 2.0 + half_x + clearance, width / 2.0 - half_x - clearance
            if x_min > x_max:
                return None
            x = rng.uniform(x_min, x_max)
            if wall == 0:
                y = depth / 2.0 - clearance - half_y - center_xy[1]
            else:
                y = -depth / 2.0 + clearance + half_y - center_xy[1]
        else:  # east (+x) / west (-x): flush along X, slide along Y
            y_min, y_max = -depth / 2.0 + half_y + clearance, depth / 2.0 - half_y - clearance
            if y_min > y_max:
                return None
            y = rng.uniform(y_min, y_max)
            if wall == 2:
                x = width / 2.0 - clearance - half_x - center_xy[0]
            else:
                x = -width / 2.0 + clearance + half_x - center_xy[0]

        z = _floor_z(size, center)
        return (x, y, z), yaw

    def _interior_grid_candidates(
        self,
        size: Sequence[float],
        center: Sequence[float],
        radius: float,
        radius_max: float,
        rng: random.Random,
    ):
        """Yield interior-grid fallback candidates (shuffled, random yaw)."""
        width, depth, _ = self.default_room_size
        clearance = self.wall_clearance
        half_w = max(width / 2.0 - clearance - radius, 0.0)
        half_d = max(depth / 2.0 - clearance - radius, 0.0)
        spacing = max(2.0 * (radius_max + PLACEMENT_PADDING), 0.05)
        nx = max(int(math.floor(2.0 * half_w / spacing)) + 1, 1)
        ny = max(int(math.floor(2.0 * half_d / spacing)) + 1, 1)
        points = []
        for i in range(nx):
            for j in range(ny):
                x = -half_w if nx == 1 else -half_w + 2.0 * half_w * i / (nx - 1)
                y = -half_d if ny == 1 else -half_d + 2.0 * half_d * j / (ny - 1)
                points.append((x, y))
        rng.shuffle(points)
        z = _floor_z(size, center)
        for x, y in points:
            yield (x, y, z), rng.uniform(-math.pi, math.pi)

    def _place_descriptor(
        self,
        descriptor: ArticulatedObject,
        occupied: Sequence[Tuple[float, float, float]],
        radius_max: float,
        rng: random.Random,
    ) -> Tuple[Tuple[float, float, float], float]:
        """Choose a non-overlapping placement for one procedural piece.

        Tries wall-following candidates first (random wall + random
        along-wall offset), then the shuffled interior grid.
        """
        local = descriptor.get_local_bounding_box()
        size = bbox_size(local)
        center = bbox_center(local)
        radius = descriptor.get_footprint_radius()

        for _ in range(MAX_WALL_ATTEMPTS):
            candidate = self._wall_candidate(rng.randrange(4), size, center, rng)
            if candidate is None:
                continue
            pos, yaw = candidate
            center_xy, _, _ = self._rotated_aabb_of(descriptor, pos, yaw)
            if self._footprint_free(center_xy, radius, occupied):
                return pos, yaw

        for pos, yaw in self._interior_grid_candidates(size, center, radius, radius_max, rng):
            center_xy, _, _ = self._rotated_aabb_of(descriptor, pos, yaw)
            if self._footprint_free(center_xy, radius, occupied):
                return pos, yaw

        raise ValueError(
            f"Could not place furniture {descriptor.name!r} in a "
            f"{self.default_room_size} room without overlap "
            f"({len(occupied)} already placed, footprint radius {radius:.3f}). "
            "Increase room_size, reduce n_furniture, or use smaller furniture."
        )

    # ------------------------------------------------------------------
    # Descriptor resolution
    # ------------------------------------------------------------------

    def _resolve_descriptor(self, placement: FurniturePlacement) -> ArticulatedObject:
        """Map a placement's ``object_id``/``source`` to a descriptor.

        The ``furniture_pool`` acts as an explicit registry (matched by
        ``name`` or ``object_id``) and shadows hub resolution; otherwise the
        hub-family sources (``"robocasa"``/``"replicacad"``/``"artvip"``/
        ``"molmospaces"``/``"lightwheel"``/``"simready_warehouse"``)
        construct the matching hub descriptor lazily.
        """
        object_id = placement.object_id
        for descriptor in self.furniture_pool:
            if descriptor.name == object_id or getattr(descriptor, "object_id", None) == object_id:
                return descriptor

        if placement.source == "procedural":
            raise KeyError(
                f"Procedural furniture {object_id!r} is not in furniture_pool "
                f"({[d.name for d in self.furniture_pool]}); sampled placements "
                "resolve by descriptor name."
            )
        if placement.source == "robocasa":
            from unienv_genesis_collection.objects.hub_objects.articulated import RoboCasaFixture

            return RoboCasaFixture(object_id)
        if placement.source == "replicacad":
            from unienv_genesis_collection.objects.hub_objects.articulated import (
                ReplicaCADFurniture,
            )

            return ReplicaCADFurniture(object_id)
        if placement.source == "artvip":
            from unienv_genesis_collection.objects.hub_objects.articulated import (
                ArtVIPFurniture,
            )

            return ArtVIPFurniture(object_id)
        if placement.source == "molmospaces":
            from unienv_genesis_collection.objects.hub_objects.articulated import (
                MolmoSpacesFurniture,
            )

            return MolmoSpacesFurniture(object_id)
        if placement.source == "lightwheel":
            from unienv_genesis_collection.objects.hub_objects.articulated import (
                LightwheelFurniture,
            )

            return LightwheelFurniture(object_id)
        if placement.source == "simready_warehouse":
            from unienv_genesis_collection.objects.hub_objects.articulated import (
                SimReadyWarehouseProp,
            )

            return SimReadyWarehouseProp(object_id)
        raise ValueError(
            f"Unknown furniture source {placement.source!r} for {object_id!r}: "
            "expected 'robocasa', 'replicacad', 'artvip', 'molmospaces', "
            "'lightwheel', 'simready_warehouse' or 'procedural' (or provide "
            "the descriptor through furniture_pool)."
        )

    @staticmethod
    def _unique_furniture_key(object_id: str, used: Set[str]) -> str:
        """Deterministic unique key for a (possibly repeated) object id."""
        if object_id not in used:
            return object_id
        index = 1
        while f"{object_id}_{index}" in used:
            index += 1
        return f"{object_id}_{index}"

    # ------------------------------------------------------------------
    # Shell construction
    # ------------------------------------------------------------------

    def _resolve_shell_file(self, shell_id: str):
        """Resolve ``scenes/rooms/<shell_id>`` to a loadable shell file."""
        rel = str(shell_id).strip("/")
        path = get_asset_path("scenes", "rooms", *rel.split("/"))
        if path.is_dir():
            for candidate in ("shell.glb", "shell.gltf", "shell.obj", "shell.ply", "model.xml", "shell.xml"):
                shell_file = path / candidate
                if shell_file.is_file():
                    return shell_file
            available = sorted(p.name for p in path.iterdir())
            raise FileNotFoundError(
                f"Room shell {shell_id!r} directory {path} has no recognised shell "
                f"file (shell.glb/.gltf/.obj/.ply or model.xml); contents: {available}."
            )
        return path

    def _build_shell(self, spec: RoomLayoutSpec, batch_fixed_verts: bool) -> List["RigidEntity"]:
        """Load the room shell into the (unbuilt) scene.

        ``shell_id=None`` builds procedural floor + four walls from box
        primitives around ``room_size``; otherwise the baked shell asset is
        loaded as ``gs.morphs.Mesh(fixed=True)`` (GLB/OBJ/PLY) or
        ``gs.morphs.MJCF`` (``.xml`` — fixed via its jointless root).
        """
        import genesis as gs

        from unienv_genesis.utils.entity_group import EntityGroup

        world = self.world
        group = EntityGroup(pos=(0.0, 0.0, 0.0), quat=yaw_to_quat(0.0), dtype=world.dtype)

        if spec.shell_id is None:
            width, depth, height = spec.room_size
            thickness = DEFAULT_WALL_THICKNESS
            floor_surface = gs.surfaces.Plastic(color=(0.55, 0.55, 0.55, 1.0))
            wall_surface = gs.surfaces.Plastic(color=(0.85, 0.85, 0.82, 1.0))

            floor_local = (0.0, 0.0, -thickness / 2.0)
            floor = world.scene.add_entity(
                gs.morphs.Box(
                    size=(width + 2.0 * thickness, depth + 2.0 * thickness, thickness),
                    pos=floor_local,
                    fixed=True,
                    batch_fixed_verts=batch_fixed_verts,
                ),
                surface=floor_surface,
                name=f"{self._name}_floor",
            )
            group.add_member(floor, local_pos=floor_local)

            wall_specs = [
                # (name, size, local pos) — walls just outside the room rect.
                ("north", (width + 2.0 * thickness, thickness, height),
                 (0.0, depth / 2.0 + thickness / 2.0, height / 2.0)),
                ("south", (width + 2.0 * thickness, thickness, height),
                 (0.0, -(depth / 2.0 + thickness / 2.0), height / 2.0)),
                ("east", (thickness, depth, height),
                 (width / 2.0 + thickness / 2.0, 0.0, height / 2.0)),
                ("west", (thickness, depth, height),
                 (-(width / 2.0 + thickness / 2.0), 0.0, height / 2.0)),
            ]
            entities = [floor]
            for wall_name, wall_size, wall_local in wall_specs:
                wall = world.scene.add_entity(
                    gs.morphs.Box(
                        size=wall_size,
                        pos=wall_local,
                        fixed=True,
                        batch_fixed_verts=batch_fixed_verts,
                    ),
                    surface=wall_surface,
                    name=f"{self._name}_wall_{wall_name}",
                )
                group.add_member(wall, local_pos=wall_local)
                entities.append(wall)
        else:
            shell_file = self._resolve_shell_file(spec.shell_id)
            suffix = shell_file.suffix.lower()
            if suffix == ".xml":
                # MJCF shells are fixed through their jointless root body.
                morph = gs.morphs.MJCF(
                    file=str(shell_file),
                    pos=(0.0, 0.0, 0.0),
                    quat=yaw_to_quat(0.0),
                    align=False,
                    batch_fixed_verts=batch_fixed_verts,
                    requires_jac_and_IK=False,
                )
            else:
                morph = gs.morphs.Mesh(
                    file=str(shell_file),
                    scale=1.0,
                    pos=(0.0, 0.0, 0.0),
                    quat=yaw_to_quat(0.0),
                    fixed=True,
                    decimate=False,
                    batch_fixed_verts=batch_fixed_verts,
                )
            shell = world.scene.add_entity(morph, name=f"{self._name}_shell")
            group.add_member(shell, local_pos=(0.0, 0.0, 0.0))
            entities = [shell]

        self._shell_group = group
        return entities

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        """Build the room: resolve the spec, load the shell, load furniture."""
        if priority not in self.reload_priorities:
            return
        if seed is not None:
            self._rng.seed(seed)

        self._spec = self._resolve_spec()
        _validate_spec(self._spec)

        # Drop state from a previous reload (the world has destroyed the
        # scene before nodes add entities again).
        self.placements = {}
        self.furniture = {}
        self.furniture_entities = {}
        self.shell_entities = []
        self._occupied = []

        self.shell_entities = self._build_shell(self._spec, self._batch_fixed_verts)

        used_keys: Set[str] = set()
        for placement in self._spec.furniture:
            key = self._unique_furniture_key(placement.object_id, used_keys)
            used_keys.add(key)

            descriptor = self._resolve_descriptor(placement)
            # The placement's yaw is the *upright* room-frame yaw; compose the
            # descriptor's standing orientation (the catalog's upright
            # placement convention, e.g. the ReplicaCAD URDF base-link frames)
            # on top of it so the furniture stands upright (see
            # placement_load_quat).
            standing = getattr(descriptor, "standing_quaternion", None)
            entity = descriptor.load_into_scene(
                world=self.world,
                pos=placement.pos,
                quat=placement_load_quat(placement.yaw, standing),
                name_override=f"{self._name}_{key}",
                batch_fixed_verts=self._batch_fixed_verts,
            )
            self.placements[key] = placement
            self.furniture[key] = descriptor
            self.furniture_entities[key] = entity

            center_xy, _, _ = self._rotated_aabb_of(
                descriptor, placement.pos, placement.yaw, standing_quaternion=standing
            )
            self._occupied.append((center_xy[0], center_xy[1], descriptor.get_footprint_radius()))

    def after_reload(self, *, priority: int = 0, mask=None):
        """Initialise joint states right after the first (re)load.

        Mirrors ``pickplace.py``: the reload phase alone does not run
        ``reset()``, so the placements' ``initial_open`` would never be
        applied on the first build without this hook.
        """
        if priority not in self.after_reload_priorities:
            return
        self.reset(priority=min(self.reset_priorities), seed=None, mask=None)

    def reset(self, *, priority: int = 0, seed=None, mask=None, **kwargs):
        """Apply each placement's ``initial_open`` joint fractions.

        DoF state is per-env even for fixed entities (plan §3.2), so a
        ``mask`` randomises only the selected environments without needing
        ``batch_fixed_verts``.  Unknown joint names raise
        :class:`ArticulationError` listing the available joints (the
        robot-keyframe behaviour).
        """
        if priority not in self.reset_priorities:
            return
        for key, placement in self.placements.items():
            if not placement.initial_open:
                continue
            entity = self.furniture_entities[key]
            descriptor = self.furniture[key]
            for joint_name, fraction in placement.initial_open.items():
                try:
                    descriptor.set_open_fraction(
                        entity,
                        joint_name,
                        fraction,
                        envs_idx=mask,
                    )
                except ArticulationError as exc:
                    raise ArticulationError(
                        f"RoomBuilder furniture {key!r} (source "
                        f"{placement.source!r}): {exc}"
                    ) from exc

    # ------------------------------------------------------------------
    # Task-facing API (plan §3.1)
    # ------------------------------------------------------------------

    def _assert_resolved(self) -> None:
        if self._spec is None:
            raise RuntimeError(
                f"RoomBuilder {self._name!r} has no resolved layout yet — "
                "call reload() first."
            )

    def get_receptacles(
        self,
        *,
        slab_thickness: float = DEFAULT_RECEPTACLE_SLAB_THICKNESS,
        interior_margin: float = DEFAULT_INTERIOR_MARGIN,
    ) -> Dict[str, Dict[str, BoxSpace]]:
        """Named interior/surface regions per furniture instance (world frame).

        Derived from the descriptors' bboxes (plan §3.1):

        * ``"top"`` — the top slab of the *union* bbox (counter tops,
          shelf surfaces), from the world-frame bbox of the built entity so
          it tracks the current pose and joint configuration.
        * ``<link_name>`` — for every joint with DOFs on the built entity,
          the child link's bbox at the reference configuration shrunk by
          ``interior_margin`` on every axis (a drawer/door interior
          approximation), transformed to the world frame by the placement
          pose.

        Links without an offline visual mesh are skipped (the descriptor
        raises :class:`ArticulationError`, which is swallowed here).

        Returns:
            ``{furniture_key: {region_name: BoxSpace}}`` with world-frame
            ``low``/``high`` corners on **``world.device`` / ``world.dtype``**
            (see the module docstring's device convention).  The regions are
            derived from the built entity's runtime pose, so their device
            follows the world rather than the host-side descriptor bboxes;
            callers comparing against NumPy/Python scalars must ``.cpu()``
            them on a GPU world.
        """
        self._assert_resolved()
        # World-device API boundary: pin dtype/device explicitly instead of
        # letting ``torch.tensor``/``make_bbox`` follow torch's ambient
        # default device (which ``gs.init()`` sets to the compute device).
        device = self.world.device
        dtype = self.world.dtype
        receptacles: Dict[str, Dict[str, BoxSpace]] = {}
        for key, descriptor in self.furniture.items():
            entity = self.furniture_entities[key]
            regions: Dict[str, BoxSpace] = {}

            world_bbox = descriptor.get_bounding_box(entity)
            # Top slab: keep the union bbox's full XY extents and cut the
            # upper *slab_thickness* off the Z range (batch-aware indexing).
            low = world_bbox.low.to(device=device, dtype=dtype)
            high = world_bbox.high.to(device=device, dtype=dtype)
            slab_t = torch.as_tensor(float(slab_thickness), dtype=dtype, device=device)
            slab_z = torch.maximum(low[..., 2], high[..., 2] - slab_t)
            top_lo = torch.stack((low[..., 0], low[..., 1], slab_z), dim=-1)
            regions["top"] = make_bbox(top_lo, high, dtype=dtype, device=device)

            for joint in entity.joints:
                if not list(joint.dofs_idx_local):
                    continue  # fixed joint: its link is static structure
                link_name = joint.link.name
                try:
                    link_bbox = descriptor.get_link_local_bounding_box(link_name)
                except ArticulationError:
                    continue  # link without an offline visual mesh
                # ``link_bbox`` is host-side descriptor data; the shrink and
                # the pose transform below move it to the world device.
                interior = self._shrink_bbox(link_bbox, interior_margin)
                # The interior region is derived from the placement pose (the
                # shared layout); broadcast it to the batch dimension so the
                # region matches the batched world bbox above.
                descriptor = self.furniture[key]
                pos_t = torch.tensor(
                    self.placements[key].pos,
                    dtype=dtype,
                    device=device,
                )
                quat_t = torch.tensor(
                    placement_load_quat(
                        self.placements[key].yaw,
                        getattr(descriptor, "standing_quaternion", None),
                    ),
                    dtype=dtype,
                    device=device,
                )
                if world_bbox.low.ndim > 1:
                    n_envs = world_bbox.low.shape[0]
                    pos_t = pos_t.unsqueeze(0).expand(n_envs, 3)
                    quat_t = quat_t.unsqueeze(0).expand(n_envs, 4)
                regions[link_name] = self._world_bbox_from_local(interior, pos_t, quat_t)
            receptacles[key] = regions
        return receptacles

    @staticmethod
    def _shrink_bbox(bbox: BoxSpace, margin: float) -> BoxSpace:
        """Shrink a bbox by *margin* on every axis without inverting it.

        The result keeps *bbox*'s device and dtype (never torch's ambient
        default device — see :func:`~...objects.base_object.make_bbox`).
        """
        center = bbox_center(bbox)
        low = torch.minimum(bbox.low + margin, center)
        high = torch.maximum(bbox.high - margin, center)
        return make_bbox(low, high, device=center.device)

    @staticmethod
    def _world_bbox_from_local(
        local_bbox: BoxSpace,
        pos: torch.Tensor,
        quat: torch.Tensor,
    ) -> BoxSpace:
        """Transform a local AABB to the world frame by ``pos``/``quat``.

        Batch-aware: with batched ``pos``/``quat`` of shape ``(n, 3)`` /
        ``(n, 4)`` the returned ``low``/``high`` are ``(n, 3)`` (the same
        broadcasting as ``ManipulatableObject.get_bounding_box``).

        The host-side *local_bbox* is moved onto ``pos``'s device/dtype, so
        the result lives on the world device like the pose that produced it.
        """
        from xbarray.transformations.rotation_conversions.pytorch import quaternion_apply

        lo = local_bbox.low.to(device=pos.device, dtype=pos.dtype)
        hi = local_bbox.high.to(device=pos.device, dtype=pos.dtype)
        corners = torch.stack(torch.meshgrid(
            torch.stack([lo[0], hi[0]]),
            torch.stack([lo[1], hi[1]]),
            torch.stack([lo[2], hi[2]]),
            indexing="ij",
        ), dim=-1).reshape(-1, 3)

        if pos.ndim > 1:
            world_corners = quaternion_apply(
                quat.unsqueeze(-2), corners.unsqueeze(0),
            ) + pos.unsqueeze(-2)                          # (n, 8, 3)
            return make_bbox(
                world_corners.min(dim=-2).values,
                world_corners.max(dim=-2).values,
                dtype=pos.dtype,
                device=pos.device,
            )

        world_corners = quaternion_apply(quat, corners) + pos  # (8, 3)
        return make_bbox(
            world_corners.min(dim=0).values,
            world_corners.max(dim=0).values,
            dtype=pos.dtype,
            device=pos.device,
        )

    def get_interaction_points(
        self,
        link_aliases: Optional[Mapping[str, str]] = None,
        *,
        gripper_width: float = DEFAULT_INTERACTION_GRIPPER_WIDTH,
        gripper_depth: float = 0.01,
    ) -> Dict[str, Dict[str, GraspCandidates]]:
        """Part grasp candidates per furniture instance (plan §3.1).

        A thin wrapper over
        :meth:`~unienv_genesis_collection.objects.articulated_object.ArticulatedObject.list_part_grasping_candidates`
        for every link of the built entity that has an offline visual mesh;
        *link_aliases* is passed through unchanged (e.g.
        ``{"handle": "cabinet_door_handle"}``).

        Args:
            link_aliases: Optional alias map forwarded to the descriptor.
            gripper_width: Maximum gripper opening (metres).
            gripper_depth: Gripper finger depth (metres).

        Returns:
            ``{furniture_key: {link_name: GraspCandidates}}`` in world
            coordinates — one entry per built-entity link that has an
            offline visual mesh, plus one entry per *alias* name resolving
            to such a link (keyed by the alias).  Links whose offline mesh
            is missing are skipped.

        Raises:
            RuntimeError: If :meth:`reload` has not run yet.
        """
        self._assert_resolved()
        interaction: Dict[str, Dict[str, GraspCandidates]] = {}
        for key, descriptor in self.furniture.items():
            entity = self.furniture_entities[key]
            # Every built-entity link, plus caller-facing alias names (an
            # alias resolves to its real link inside the descriptor call).
            link_names = [link.name for link in entity.links]
            alias_names = list(link_aliases) if link_aliases else []
            requested = list(dict.fromkeys(link_names + alias_names))
            parts: Dict[str, GraspCandidates] = {}
            for name in requested:
                try:
                    parts[name] = descriptor.list_part_grasping_candidates(
                        entity,
                        name,
                        float(gripper_width),
                        gripper_depth=float(gripper_depth),
                        part_aliases=link_aliases,
                    )
                except ArticulationError:
                    continue  # link without an offline visual mesh
            interaction[key] = parts
        return interaction

    def sample_free_floor_positions(
        self,
        n: int,
        rng: Optional[torch.Generator],
        *,
        margin: float = DEFAULT_WALL_CLEARANCE,
    ) -> torch.Tensor:
        """Rejection-sample free floor positions in the room (plan §3.1).

        Uniformly samples inside the room rect (shrunk by *margin*), rejecting
        candidates whose footprint circle overlaps any furniture footprint
        (the same padded circular check the procedural placer uses).  For
        robot spawn / clutter placement.

        Args:
            n: Number of positions.
            rng: Optional :class:`torch.Generator` (pickplace convention);
                ``None`` uses the global torch RNG.  As in ``pickplace``, a
                generator must be on the world device (a CPU generator with
                a CUDA world is rejected by torch).
            margin: Keep-out distance from the walls (metres).

        Returns:
            ``(n, 3)`` world-frame positions on the floor (``z = 0``), on
            **``world.device`` / ``world.dtype``** (see the module docstring's
            device convention).  The sampling itself is host-side: each
            candidate is compared against the furniture footprints in Python
            and only the accepted scalars are written into the result, so the
            result's device never depends on torch's ambient default device.

        Raises:
            RuntimeError: If :meth:`reload` has not run yet.
            RuntimeError: If *n* positions cannot be found (room too
                crowded) — the requested count is never silently shrunk.
        """
        self._assert_resolved()
        device = self.world.device
        dtype = self.world.dtype
        width, depth, _ = self.room_size
        half_w = max(width / 2.0 - margin, 0.0)
        half_d = max(depth / 2.0 - margin, 0.0)
        occupied = [(cx, cy, r + PLACEMENT_PADDING) for (cx, cy, r) in self._occupied]

        samples = torch.empty((n, 3), dtype=dtype, device=device)
        filled = 0
        attempts = 0
        max_attempts = max(n, 1) * _FREE_POSITION_MAX_ATTEMPTS_PER_SAMPLE
        while filled < n and attempts < max_attempts:
            attempts += 1
            xy = torch.rand(2, generator=rng, dtype=dtype, device=device) * 2.0 - 1.0
            x = float(xy[0]) * half_w
            y = float(xy[1]) * half_d
            if all(
                math.hypot(x - cx, y - cy) > r for (cx, cy, r) in occupied
            ):
                samples[filled, 0] = x
                samples[filled, 1] = y
                samples[filled, 2] = 0.0
                filled += 1
        if filled < n:
            raise RuntimeError(
                f"Could only sample {filled}/{n} free floor positions after "
                f"{attempts} attempts — the room is too crowded "
                f"({len(occupied)} furniture footprints)."
            )
        return samples

    # ------------------------------------------------------------------
    # Per-env layout (batch_fixed_verts path, plan §3.2 / §7 risk #2)
    # ------------------------------------------------------------------

    def set_furniture_pose_3d(
        self,
        furniture_key: str,
        positions,
        envs_idx=None,
    ) -> None:
        """Move one furniture instance, optionally per environment.

        Only meaningful in batched worlds whose morphs were loaded with
        ``batch_fixed_verts=True`` (the default there): per-env base poses of
        fixed entities otherwise share one layout and Genesis raises.
        Orientation is kept (``random_quaternion=False`` — furniture keeps
        its authored yaw; see plan §3.2).

        Args:
            furniture_key: Key into :attr:`furniture` / :attr:`furniture_entities`.
            positions: World positions ``(3,)`` or ``(n_selected, 3)``.
            envs_idx: Optional ``(batch_size,)`` ``torch.bool`` mask.
        """
        self._assert_resolved()
        if furniture_key not in self.furniture:
            raise KeyError(
                f"Unknown furniture key {furniture_key!r}; available: {sorted(self.furniture)}"
            )
        # A tensor keeps the device the caller built it on (tasks build
        # these on ``world.device``); a plain sequence goes to the world.
        positions_t = self._as_room_tensor(positions)
        batched = (
            positions_t.ndim > 1 or (envs_idx is not None and self.world.batch_size is not None)
        )
        if batched and not self._batch_fixed_verts:
            raise RuntimeError(
                f"Per-env furniture poses require the morphs to have been loaded "
                f"with batch_fixed_verts=True (RoomBuilder {self._name!r} loaded "
                "with batch_fixed_verts=False — all envs share one layout; the "
                "flag trades significant GPU memory for layout diversity)."
            )
        self.furniture[furniture_key].set_pos_3d(
            self.furniture_entities[furniture_key],
            positions_t,
            random_quaternion=False,
            envs_idx=envs_idx,
        )
