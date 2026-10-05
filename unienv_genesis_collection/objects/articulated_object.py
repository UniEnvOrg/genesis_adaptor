"""Articulated furniture descriptors (``ArticulatedObject``).

Articulated objects (cabinets, drawers, appliances, …) are
:class:`~unienv_genesis_collection.objects.base_object.ManipulatableObject`
descriptors with **multiple visual meshes** (one per link) whose geometric
extent depends on the joint configuration.  This module implements the
descriptor, the offline reference-configuration forward kinematics used for
its bounding box, and the runtime name resolvers used to address joints and
links.

Design (see ``docs/research/articulated_room_implementation_plan.md``):

* **No joint registry.**  Joint and link addressing is *by name*, resolved
  statelessly against the built Genesis entity at call time —
  :func:`resolve_joint_dofs` / :func:`resolve_link_index` mirror
  ``unienv_genesis.nodes.entities.entity_controllers.base.resolve_joint_indexes``
  (the pattern the robot actor already uses).  Nothing per-asset is baked
  into descriptors or catalog metadata.
* **Offline union bbox.**  :meth:`ArticulatedObject.get_local_bounding_box`
  returns the union AABB over all links at a *reference configuration*
  (``reference_qpos``, default all-zero = the authored closed pose) in the
  base-link frame.  It is computed **offline** — no Genesis scene is
  required — via :func:`urdf_link_meshes_at_config` (URDF, using
  ``genesis.ext.urdfpy`` forward kinematics),
  :func:`mjcf_link_meshes_at_config` (MJCF, minimal FK walker) or
  :func:`usd_link_meshes_at_config` (USD, minimal UsdPhysics FK walker with
  box-proxy geometry built from ``UsdGeom`` bounds) in the concrete hub
  subclasses.  The inherited
  :meth:`~unienv_genesis_collection.objects.base_object.ManipulatableObject.get_bounding_box`
  contract (transform the local AABB by the entity's pose) then keeps
  working verbatim.
* **Part-aware grasps.**  :meth:`ArticulatedObject.list_part_grasping_candidates`
  enumerates grasps on a *named link*'s mesh **in the link's own local
  frame** (the FK helpers' ``link_local`` geometry — the visual mesh before
  ``base_T_link`` is applied) and transforms the candidates by that link's
  **current** pose, so they stay rigidly attached to the link's geometry and
  track a moving drawer or door.

Quaternion convention: ``(w, x, y, z)`` throughout, matching Genesis,
``EntityGroup`` and the rest of the collection.  All entity-facing methods
accept both unbatched (``(3,)`` / ``(4,)``) and batched (``(B, 3)`` /
``(B, 4)``) poses as well as an optional ``envs_idx`` mask, matching
``base_object.py`` / ``pickplace.py`` conventions.  ``trimesh`` and
``genesis.ext.urdfpy`` are imported lazily so the module stays offline-safe.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import torch

from unienv_genesis_collection.objects.base_object import (
    ManipulatableObject,
    EntityT,
    GraspCandidates,
    make_bbox,
    _enumerate_bbox_gripper_candidates,
    _local_to_world_candidates,
)

if TYPE_CHECKING:
    import trimesh
    from unienv_interface.space import BoxSpace

__all__ = [
    "ArticulationError",
    "resolve_joint_dofs",
    "resolve_link_index",
    "ArticulatedObject",
    "urdf_link_meshes_at_config",
    "mjcf_link_meshes_at_config",
    "usd_link_meshes_at_config",
]


class ArticulationError(RuntimeError):
    """Raised for articulation queries that cannot be satisfied.

    Typical causes: a joint or link name that does not exist on the built
    entity (the message lists the available names), unbounded joint limits
    (``±inf`` — an unbounded joint has no meaningful open fraction), or a
    geometry/description file that cannot be parsed by the offline FK
    helpers.
    """


# ---------------------------------------------------------------------------
# Stateless runtime resolvers (mirror entity_controllers/base.py:27)
# ---------------------------------------------------------------------------

# Import-vs-inline decision (plan §1.2 "Import note"): the ~10-line name-map
# logic below is **inlined** rather than importing
# ``unienv_genesis.nodes.entities.entity_controllers.base.resolve_joint_indexes``.
# Reason: the plan's stated premise — "the dependency direction is already
# collection→unienv_genesis" — does not hold in the code: no module under
# ``unienv_genesis_collection/`` imports ``unienv_genesis`` at runtime (the
# GenesisWorld references are all local ``TYPE_CHECKING``/duck-typed).  The
# robot controller module also drags the whole controller/Genesis-world stack
# into a descriptor module that is deliberately offline-safe.  The inlined
# map is behaviourally identical (same ``{j.name: j.dofs_idx_local …}`` build,
# same "available names" error UX); see the module tests that pin both.

def resolve_joint_dofs(entity: EntityT, joint_name: str) -> List[int]:
    """Resolve a joint name to its DOF indexes on the built entity.

    Builds the name → ``dofs_idx_local`` map on the fly (the same map as
    ``resolve_joint_indexes``) and returns the DOF indexes of the requested
    joint.  Fixed joints resolve to an empty list.

    Args:
        entity: The loaded Genesis rigid entity.
        joint_name: Name of the joint, as declared by the asset
            (e.g. ``"cabinet_door_joint"``).

    Returns:
        List of local DOF indexes (``entity.set_dofs_position``-compatible).

    Raises:
        ArticulationError: If *joint_name* is not found; the message lists
            the available joint names.
    """
    name_to_dofs: Dict[str, List[int]] = {
        joint.name: list(joint.dofs_idx_local) for joint in entity.joints
    }
    if joint_name not in name_to_dofs:
        raise ArticulationError(
            f"Joint name not found on entity: {joint_name!r}. "
            f"Available joints: {sorted(name_to_dofs.keys())}"
        )
    return name_to_dofs[joint_name]


def resolve_link_index(entity: EntityT, link_name: str) -> int:
    """Resolve a link name to its local link index on the built entity.

    Link-level analogue of :func:`resolve_joint_dofs`: builds
    ``{l.name: l.idx_local for l in entity.links}`` on the fly and returns
    the requested index.  Used for reference-frame queries (e.g.
    ``entity.get_links_pos([idx], relative=True)``).

    Args:
        entity: The loaded Genesis rigid entity.
        link_name: Name of the link, as declared by the asset
            (e.g. ``"drawer_handle"``).

    Returns:
        The link's ``idx_local``.

    Raises:
        ArticulationError: If *link_name* is not found; the message lists
            the available link names.
    """
    name_to_idx: Dict[str, int] = {link.name: link.idx_local for link in entity.links}
    if link_name not in name_to_idx:
        raise ArticulationError(
            f"Link name not found on entity: {link_name!r}. "
            f"Available links: {sorted(name_to_idx.keys())}"
        )
    return name_to_idx[link_name]


# ---------------------------------------------------------------------------
# Offline FK: union-bbox geometry at a reference configuration
# ---------------------------------------------------------------------------

def _transform_meshes(
    meshes: Sequence["trimesh.Trimesh"],
    matrix: np.ndarray,
) -> List["trimesh.Trimesh"]:
    """Return copies of *meshes* with the 4x4 *matrix* applied to their vertices."""
    transformed = []
    for mesh in meshes:
        copy = mesh.copy()
        copy.apply_transform(matrix)
        transformed.append(copy)
    return transformed


def _merge_link_meshes(
    parts: Sequence["trimesh.Trimesh"],
) -> "trimesh.Trimesh":
    """Concatenate the visual meshes of a single link into one mesh."""
    import trimesh

    if len(parts) == 1:
        return parts[0]
    return trimesh.util.concatenate(parts)


def urdf_link_meshes_at_config(
    urdf_file: str,
    reference_qpos: Optional[Mapping[str, float]] = None,
    link_local: bool = False,
) -> Dict[str, "trimesh.Trimesh"]:
    """Per-link visual meshes of a URDF at a reference joint configuration.

    Forward kinematics is computed offline with ``genesis.ext.urdfpy``
    (``URDF.load`` + ``robot.link_fk(cfg=…)``) — no Genesis scene is built.
    For every link, its visual meshes are loaded with
    ``trimesh.load(…, force="mesh")`` (GLB scenes are concatenated into a
    single mesh), scaled by the URDF ``<mesh scale=…>``, transformed by
    ``base_T_link · visual_origin`` and concatenated into **one mesh per
    link** expressed in the base-link frame.

    With ``link_local=True`` the meshes skip the FK transform: each visual
    mesh is transformed by its ``<visual><origin>`` only, i.e. expressed in
    the **link's own frame**.  That geometry is configuration-independent
    (``visual_origin`` is authored per link, not per joint configuration)
    and is what part-grasp enumeration needs — a point in the link frame
    maps to the world through the link's *current* ``world_T_link`` pose,
    so it tracks the articulation.  (Note ``reference_qpos`` is still
    validated and the FK is still computed, so error behaviour matches the
    base-frame mode.)

    Links without visual geometry are omitted from the result.

    Args:
        urdf_file: Path to the URDF (relative mesh references are resolved
            against its directory, matching URDF semantics).
        reference_qpos: Joint name → position at the reference
            configuration (radians for revolute joints, metres for
            prismatic joints — urdfpy/Genesis convention).  Omitted joints
            stay at their default (zero) position.  ``None`` / empty means
            the asset's authored closed pose.
        link_local: If ``True``, return meshes in each link's own frame
            (``visual_origin`` only) instead of the base-link frame.

    Returns:
        ``{link_name: trimesh.Trimesh}`` with one merged visual mesh per
        link that has visuals, in the base-link frame — or in the link's
        own frame when *link_local* is set.

    Raises:
        ArticulationError: If *reference_qpos* names a joint that does not
            exist in the URDF (the message lists the available joints).
    """
    from genesis.ext.urdfpy import URDF

    import trimesh

    path = Path(urdf_file)
    robot = URDF.load(str(path))

    reference_qpos = dict(reference_qpos) if reference_qpos else {}
    known_joints = {joint.name for joint in robot.joints}
    unknown = sorted(set(reference_qpos) - known_joints)
    if unknown:
        raise ArticulationError(
            f"Unknown joint name(s) in reference_qpos: {unknown}. "
            f"Available joints: {sorted(known_joints)}"
        )

    link_fk = robot.link_fk(cfg=reference_qpos)

    # One trimesh load per distinct file, shared across links/visuals.
    mesh_cache: Dict[str, "trimesh.Trimesh"] = {}

    def _visual_meshes(visual) -> List["trimesh.Trimesh"]:
        geometry = visual.geometry
        if geometry.mesh is None:
            # Primitive visual (box / cylinder / sphere) — urdfpy generates
            # the trimesh representation itself.
            return list(geometry.meshes)
        mesh_file = Path(geometry.mesh.filename)
        if not mesh_file.is_absolute():
            mesh_file = path.parent / mesh_file
        if not mesh_file.is_file():
            # Unresolvable path (e.g. ``package://`` handling): fall back to
            # the meshes urdfpy already loaded when parsing the URDF.
            return list(geometry.meshes)
        key = str(mesh_file)
        if key not in mesh_cache:
            mesh_cache[key] = trimesh.load(key, force="mesh")
        mesh = mesh_cache[key].copy()
        if geometry.mesh.scale is not None:
            mesh.apply_scale(geometry.mesh.scale)
        return [mesh]

    result: Dict[str, "trimesh.Trimesh"] = {}
    for link, base_to_link in link_fk.items():
        parts: List["trimesh.Trimesh"] = []
        for visual in link.visuals:
            visual_origin = np.asarray(visual.origin)
            # ``base_to_link`` maps link-frame points into the base frame at
            # the reference configuration; with ``link_local`` it is skipped
            # so the mesh stays in the link's own frame.
            matrix = (
                visual_origin if link_local else np.asarray(base_to_link) @ visual_origin
            )
            parts.extend(_transform_meshes(_visual_meshes(visual), matrix))
        if parts:
            result[link.name] = _merge_link_meshes(parts)
    return result


def _mjcf_vector(value: Optional[str], default: Sequence[float]) -> np.ndarray:
    """Parse a whitespace-separated MJCF vector attribute (``None`` → *default*)."""
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in value.split()], dtype=np.float64)


def _mjcf_body_transform(
    body: ET.Element,
    angle_scale: float,
) -> np.ndarray:
    """Rest transform of an MJCF ``<body>``/``<geom>`` element (pos/quat/euler).

    ``pos`` composes with the parent frame; ``quat`` (``w x y z``) or
    ``euler`` (intrinsic x-y-z, in the ``compiler angle`` unit) gives the
    orientation.  ``axisangle`` and defaults/inherited classes are **not**
    supported (see :func:`mjcf_link_meshes_at_config`).
    """
    from trimesh.transformations import quaternion_matrix, rotation_matrix, translation_matrix

    matrix = translation_matrix(_mjcf_vector(body.get("pos"), (0.0, 0.0, 0.0)))
    if body.get("quat") is not None:
        matrix = matrix @ quaternion_matrix(_mjcf_vector(body.get("quat"), (1.0, 0.0, 0.0, 0.0)))
    elif body.get("euler") is not None:
        rx, ry, rz = _mjcf_vector(body.get("euler"), (0.0, 0.0, 0.0)) * angle_scale
        # MuJoCo's default ``eulerseq="xyz"`` rotates about the (intrinsic)
        # local axes in that order: R = Rx · Ry · Rz.
        matrix = (
            matrix
            @ rotation_matrix(rx, (1.0, 0.0, 0.0))
            @ rotation_matrix(ry, (0.0, 1.0, 0.0))
            @ rotation_matrix(rz, (0.0, 0.0, 1.0))
        )
    return matrix


def _mjcf_joint_transform(
    joint: ET.Element,
    reference_qpos: Mapping[str, float],
) -> np.ndarray:
    """Joint transform of an MJCF ``<joint>`` at its reference position.

    Hinge: rotation by ``q`` (radians) about the joint axis through the
    joint anchor, both expressed in the body's rest frame —
    ``T(p) · R(axis, q) · T(-p)``.  Slide: translation by ``q · axis``
    (metres) in the body frame.

    Args:
        joint: ``<joint>`` element.
        reference_qpos: Joint name → position (radians / metres, Genesis
            convention).  Anonymous joints (MJCF auto-names them) stay at
            zero.

    Raises:
        ArticulationError: On an unsupported joint type (``free`` /
        ``ball``) or a degenerate (zero-length) axis.
    """
    from trimesh.transformations import rotation_matrix, translation_matrix

    joint_type = joint.get("type", "hinge")
    name = joint.get("name")
    q = float(reference_qpos.get(name, 0.0)) if name is not None else 0.0
    axis = _mjcf_vector(joint.get("axis"), (0.0, 0.0, 1.0))
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ArticulationError(
            f"Degenerate joint axis for joint {name!r}: {joint.get('axis')!r}"
        )
    axis = axis / norm

    if joint_type == "hinge":
        anchor = _mjcf_vector(joint.get("pos"), (0.0, 0.0, 0.0))
        return rotation_matrix(q, axis, point=anchor)
    if joint_type == "slide":
        return translation_matrix(axis * q)
    raise ArticulationError(
        f"Unsupported MJCF joint type {joint_type!r} for joint {name!r}: "
        "only 'hinge' and 'slide' joints are supported by the offline FK walker."
    )


def mjcf_link_meshes_at_config(
    mjcf_file: str,
    reference_qpos: Optional[Mapping[str, float]] = None,
    link_local: bool = False,
) -> Dict[str, "trimesh.Trimesh"]:
    """Per-body visual meshes of an MJCF at a reference joint configuration.

    Minimal offline forward-kinematics walker for MuJoCo scene files —
    Genesis does not expose an MJCF FK API.  It walks the ``<worldbody>``
    nesting, composing every body's ``pos`` / ``quat`` / ``euler``, applies
    hinge/slide joint transforms at the reference positions, and resolves
    ``<geom mesh=…>`` (any geom carrying a ``mesh`` attribute — ``type`` is
    not required to be ``"mesh"``) against ``<asset><mesh name=… file=…>``
    with ``compiler meshdir``.  Mesh geoms are loaded with
    ``trimesh.load(…, force="mesh")``, transformed into the world (base)
    frame and concatenated **per body** — the key in the returned mapping is
    the body name (bodies are links in Genesis).

    With ``link_local=True`` the walker skips the accumulated parent/body/
    joint chain: each mesh geom is transformed by its own ``pos``/``quat``/
    ``euler`` only, i.e. expressed in the **body's own frame** (the frame
    Genesis reports for that body through ``get_links_pos/quat``, which
    already includes the joint displacement).  That geometry is
    configuration-independent and is what part-grasp enumeration needs — a
    point in the link frame maps to the world through the link's *current*
    ``world_T_link`` pose, so it tracks the articulation.  (Note
    ``reference_qpos`` is still applied/validated while walking, so error
    behaviour matches the base-frame mode.)

    The fixture-shaped subset of MJCF that RoboCasa kitchen assets use is
    supported; intentionally **not** supported:

    * ``<include>`` files (imported bodies/geoms/assets are ignored),
    * defaults / ``childclass`` attribute inheritance,
    * ``free`` / ``ball`` joints (raise :class:`ArticulationError`; a
      ``<freejoint/>`` root is not part of a fixed fixture),
    * ``axisangle`` body orientations, ``fromto`` geoms and non-mesh geoms
      (``box`` / ``sphere`` / … are skipped),
    * visual/collision geom distinction (all mesh geoms are included),
    * ``compiler strippath`` handling.

    The asset-prepare pipelines cross-check the resulting bbox against
    ``entity.get_vAABB()`` at packaging time, so parser gaps are caught
    before assets ship.

    Args:
        mjcf_file: Path to the MJCF file.
        reference_qpos: Joint name → position at the reference
            configuration, in **radians / metres** (Genesis convention —
            unlike MJCF ``range`` and ``euler`` attributes which are in
            ``compiler angle`` units).  Anonymous joints stay at zero.
            ``None`` / empty means the asset's authored closed pose.
        link_local: If ``True``, return meshes in each body's own frame
            (the geom's own ``pos``/``quat``/``euler`` only) instead of the
            world frame.

    Returns:
        ``{body_name: trimesh.Trimesh}`` with one merged visual mesh per
        body that has at least one mesh geom, in the world frame — or in
        the body's own frame when *link_local* is set.

    Raises:
        ArticulationError: If the file is not an MJCF root, has no
            ``<worldbody>``, references an unknown mesh, declares body
            names twice, or uses an unsupported joint type.
    """
    import trimesh

    path = Path(mjcf_file)
    root = ET.parse(str(path)).getroot()
    if root.tag != "mujoco":
        raise ArticulationError(f"Not an MJCF file (root tag {root.tag!r}): {path}")

    reference_qpos = dict(reference_qpos) if reference_qpos else {}

    compiler = root.find("compiler")
    mesh_dir = path.parent
    angle_scale = math.pi / 180.0  # MJCF default: angles in degrees
    if compiler is not None:
        mesh_dir_attr = compiler.get("meshdir")
        if mesh_dir_attr:
            mesh_dir_attr = Path(mesh_dir_attr)
            mesh_dir = (
                mesh_dir_attr if mesh_dir_attr.is_absolute() else path.parent / mesh_dir_attr
            )
        if compiler.get("angle", "degree") == "radian":
            angle_scale = 1.0

    # <asset><mesh name=... file=.../> — name → resolved file path.
    mesh_files: Dict[str, Path] = {}
    for mesh_asset in root.findall("./asset/mesh"):
        name = mesh_asset.get("name")
        file_attr = mesh_asset.get("file")
        if name is None or file_attr is None:
            continue
        resolved = Path(file_attr)
        if not resolved.is_absolute():
            resolved = mesh_dir / resolved
        mesh_files[name] = resolved

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ArticulationError(f"MJCF file has no <worldbody>: {path}")

    mesh_cache: Dict[str, "trimesh.Trimesh"] = {}

    def _load_mesh(mesh_name: Optional[str]) -> "trimesh.Trimesh":
        if mesh_name is None or mesh_name not in mesh_files:
            raise ArticulationError(
                f"MJCF geom references unknown mesh asset {mesh_name!r}. "
                f"Available mesh assets: {sorted(mesh_files.keys())}"
            )
        key = str(mesh_files[mesh_name])
        if key not in mesh_cache:
            mesh_cache[key] = trimesh.load(key, force="mesh")
        return mesh_cache[key]

    result: Dict[str, "trimesh.Trimesh"] = {}
    anonymous_count = 0

    def _walk(body: ET.Element, parent_matrix: np.ndarray) -> None:
        nonlocal anonymous_count

        body_matrix = parent_matrix @ _mjcf_body_transform(body, angle_scale)
        for joint in body.findall("joint"):
            body_matrix = body_matrix @ _mjcf_joint_transform(joint, reference_qpos)

        parts: List["trimesh.Trimesh"] = []
        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None:
                continue  # non-mesh geoms are unsupported (see docstring)
            geom_local = _mjcf_body_transform(geom, angle_scale)
            # ``body_matrix`` is the body frame at the reference configuration
            # (rest transform + joint displacement, matching the frame Genesis
            # reports for the body); with ``link_local`` it is skipped so the
            # geom mesh stays in the body's own frame.
            geom_matrix = geom_local if link_local else body_matrix @ geom_local
            parts.extend(_transform_meshes([_load_mesh(mesh_name)], geom_matrix))

        if parts:
            name = body.get("name")
            if name is None:
                name = f"unnamed_body_{anonymous_count}"
                anonymous_count += 1
            if name in result:
                raise ArticulationError(
                    f"Duplicate body name {name!r} in MJCF file {path}: the "
                    "offline FK walker keys meshes by body name."
                )
            result[name] = _merge_link_meshes(parts)

        for child in body.findall("body"):
            _walk(child, body_matrix)

    for body in worldbody.findall("body"):
        # Top-level bodies compose directly with the world frame; the body's
        # own pos/quat/euler are applied inside ``_walk``.
        _walk(body, np.eye(4))

    return result


# ---------------------------------------------------------------------------
# USD (UsdPhysics) offline FK — box-proxy geometry
# ---------------------------------------------------------------------------

# Displacement direction enumerated by the UsdPhysics ``axis`` attribute,
# expressed in the *joint frame's* own coordinates.
_USD_AXIS_VECTORS = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

# ``genesis.utils.mesh.Y_UP_TRANSFORM`` (pxr row-vector world matrix,
# translation on the bottom row) already transposed to the column-vector
# convention: pre-multiplying a column-convention world matrix with it maps a
# Y-up stage onto the Z-up Genesis world, ((X, Y, Z) -> (X, -Z, Y)).
_USD_Y_UP_TRANSFORM = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _usd_matrix_to_numpy(matrix) -> np.ndarray:
    """pxr :class:`Gf.Matrix4d` (row-vector convention, translation on the
    bottom row) → numpy ``(4, 4)`` in the column-vector convention
    (translation in the last column), mirroring
    ``genesis.utils.usd.UsdContext.compute_transform``'s ``transpose()``."""
    return np.array(
        [[matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64
    ).T


def _usd_quat_to_matrix(quat) -> np.ndarray:
    """pxr quaternion (``.real`` + ``.imaginary`` xyz) → ``(4, 4)`` rotation matrix."""
    qi = quat.imaginary
    w = float(quat.real)
    return np.array(
        [
            [1 - 2 * (qi[1] ** 2 + qi[2] ** 2), 2 * (qi[0] * qi[1] - qi[2] * w), 2 * (qi[0] * qi[2] + qi[1] * w), 0.0],
            [2 * (qi[0] * qi[1] + qi[2] * w), 1 - 2 * (qi[0] ** 2 + qi[2] ** 2), 2 * (qi[1] * qi[2] - qi[0] * w), 0.0],
            [2 * (qi[0] * qi[2] - qi[1] * w), 2 * (qi[1] * qi[2] + qi[0] * w), 1 - 2 * (qi[0] ** 2 + qi[1] ** 2), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _usd_joint_frame(pos, rot_quat) -> np.ndarray:
    """UsdPhysics joint frame ``T(localPos) · R(localRot)`` (column convention).

    ``localPos0``/``localRot0`` are authored relative to the ``body0``
    relationship target's frame, ``localPos1``/``localRot1`` relative to
    ``body1``'s frame.
    """
    frame = _usd_quat_to_matrix(rot_quat) if rot_quat is not None else np.eye(4)
    if pos is not None:
        frame[:3, 3] = np.asarray(pos, dtype=np.float64)
    return frame


def _usd_rotation_about_axis(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation about *axis* through the origin, ``(4, 4)``."""
    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = (float(v) / float(np.linalg.norm(axis)) for v in axis)
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return matrix


_USD_EXTRACT_SCALE = None  # memoised ``genesis.utils.usd.usd_utils.extract_scale``


def _ensure_genesis_eps() -> None:
    """Make ``gs.EPS`` a float for offline (pre-``gs.init``) FK runs.

    ``genesis.utils.usd.usd_utils.extract_scale`` → ``gu.polar`` →
    ``_np_polar_core_single`` is a numba-jitted kernel that reads the module
    global ``gs.EPS``.  Genesis declares it ``None`` at import and only gives
    it a float inside :func:`gs.init`, so an offline FK call in a process that
    has not initialized Genesis (or ran ``gs.destroy``) makes numba type
    ``float64 > none`` and die with a ``TypingError``.  Whether that ever
    surfaces depends on numba's *disk cache*: a cache entry compiled in a
    process where ``EPS`` was a float is happily reused even when the current
    process has ``EPS=None``, which is why this only ever looked broken after
    a genesis re-install invalidated the cache.  Install the exact value
    ``gs.init(precision="32")`` would (its default ``eps=1e-15`` bounded below
    by float32 machine epsilon) so pre- and post-init numerics are identical;
    a later ``gs.init`` overwrites it again.
    """
    try:
        import genesis as gs
    except Exception:  # noqa: BLE001 — pxr-only context; nothing to fix up
        return
    if getattr(gs, "EPS", None) is None:
        gs.EPS = float(max(1e-15, np.finfo(np.float32).eps))


def _usd_extract_scale(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Polar-decompose a (column-convention) transform into ``(rigid, scale)``.

    Genesis's own USD importer splits every link transform this way
    (``UsdContext.compute_gs_transform`` → ``usd_utils.extract_scale``:
    ``gu.polar`` on the linear block, a reflection folded into the scale so
    the rigid part stays a proper rotation).  The rigid part is what the
    built entity's link frame uses — the scale rides on the geometry
    (vgeoms keep the full transform) — so the offline FK walker must split
    the very same way to mirror it.

    Unit note: a non-1.0 ``metersPerUnit`` never leaks into the *rigid*
    part.  Genesis multiplies the whole linear block by ``meter_scale``
    (``compute_transform``), a positive uniform factor which the polar
    decomposition folds into the scale — ``polar(m · A) = (polar(A)[0],
    m · polar(A)[1])`` — so the extracted rotations (and therefore every
    rigid link frame) are identical for any meter scale.  This walker keeps
    its stage-unit FK (positions unscaled) and converts the final boxes to
    metres once at the end, exactly like before.

    Falls back to column normalisation when ``genesis`` is not importable
    (pxr-only contexts); the fallback is exact for the pure-rotation /
    uniform-scale cases and a conservative approximation otherwise.
    """
    global _USD_EXTRACT_SCALE
    if _USD_EXTRACT_SCALE is None:
        try:
            from genesis.utils.usd.usd_utils import extract_scale

            _USD_EXTRACT_SCALE = extract_scale
        except Exception:  # noqa: BLE001 — pxr-only context; approximating

            def _column_normalised(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                rigid = np.eye(4, dtype=np.float64)
                rigid[:3, :3] = np.array(T[:3, :3], dtype=np.float64, copy=True)
                for column in range(3):
                    norm = np.linalg.norm(rigid[:3, column])
                    if norm > 0.0:
                        rigid[:3, column] /= norm
                rigid[:3, 3] = T[:3, 3]
                return rigid, np.array(T, dtype=np.float64, copy=True)

            _USD_EXTRACT_SCALE = _column_normalised
    _ensure_genesis_eps()
    return _USD_EXTRACT_SCALE(matrix)


def _usd_rigid_transform(matrix: np.ndarray) -> np.ndarray:
    """The rigid (scale-extracted) part of a column-convention world matrix."""
    rigid, _ = _usd_extract_scale(matrix)
    return rigid


def usd_clean_names(prims: Sequence) -> List[str]:
    """Leaf-name sanitisation for USD link/joint prims (plan D2).

    Genesis names USD joints and links by their **full prim path**, but the
    task-facing clean name is the leaf token (``prim.GetName()`` — e.g.
    ``E_door1_1``).  Leaf names are not guaranteed unique across a stage, so
    colliding prims are disambiguated with the parent-chain suffix, growing
    deterministically (shortest suffix first) for *all* prims of a colliding
    group until the whole group is unique (``E_door1_1/RevoluteJoint``).
    Stable for a given asset; the ArtVIP spike leaves are already unique
    (the suffix fallback is belt and braces).

    Args:
        prims: ``Usd.Prim`` sequence (one link/joint set).

    Returns:
        One clean name per input prim (same order).
    """
    chains = [str(prim.GetPath()).split("/") for prim in prims]

    def _suffix(chain: List[str], depth: int) -> str:
        taken = chain[-(depth + 1):] if depth < len(chain) - 1 else list(chain)
        return "/".join(taken)

    depth = 0
    names = [_suffix(chain, depth) for chain in chains]
    max_depth = max((len(chain) for chain in chains), default=1)
    while depth < max_depth and any(names.count(name) > 1 for name in names):
        colliding = {name for name in names if names.count(name) > 1}
        depth += 1
        for idx, chain in enumerate(chains):
            if names[idx] in colliding:
                names[idx] = _suffix(chain, depth)
    return names


def usd_link_meshes_at_config(
    usd_file: str,
    reference_qpos: Optional[Mapping[str, float]] = None,
    link_local: bool = False,
) -> Dict[str, "trimesh.Trimesh"]:
    """Per-link geometry of a UsdPhysics USD stage at a reference configuration.

    Minimal offline forward-kinematics walker for USD articulations,
    mirroring Genesis's own USD split
    (``genesis/utils/usd/usd_rigid_entity.py``) so the enumerated link set
    matches the built entity 1:1:

    * **Links** are the prims with ``UsdPhysics.RigidBodyAPI`` applied.
      Genesis names them by **full prim path** — the descriptor sanitises
      them to these *clean* (leaf) names, see :func:`usd_clean_names`.
    * **Joints** are the ``UsdPhysics.RevoluteJoint`` / ``PrismaticJoint``
      prims (the moving DOFs Genesis grants) plus ``FixedJoint`` welds — the
      welds are only used for the link graph (no DOF of their own).
    * **FK walks RIGID link frames** (RC-1).  Every link's frame is its
      *authored world transform with the linear map's scale polar-decomposed
      out* — exactly Genesis's ``compute_gs_transform`` /
      ``usd_utils.extract_scale`` (the built entity's link ``pos``/``quat``
      is ``inv(Q_parent) @ Q_child`` over those rigid frames) — and the DOF
      acts about the **body1-side** anchor:

      ``Q_rel(child) = inv(Q(parent)) @ Q(child)`` at ``q = 0`` (the authored
      rigid relative pose — the built entity's rest configuration),
      then, for ``q ≠ 0``,

      ``R' = R · Rot(axis, q)``,  ``t' = t + R·a − R'·a``  (revolute)
      ``t' = t + R·(axis·q)``                          (prismatic)

      with ``a`` / ``axis`` the joint anchor and unit axis expressed in the
      child's *rigid* link frame — ``localPos1``/``localRot1`` (authored in
      the body1 relationship-target prim's frame, which may be a nested
      wrapper carrying its own scale) mapped through that prim's authored
      world matrix and folded by the child's rigid frame, mirroring
      Genesis's ``_parse_joint_axis_pos``.  Two defects of the previous
      joint-frame-chain walker are fixed by this:

      - the chain composed every ancestor's authored *scale* into the child
        frame (SUNDVIK's ×1000 scope, table_1/drawers_chest's ×0.03937
        inch-to-metre group, door001's non-uniform scope) — the rigid
        decomposition of each link's own world transform cannot inherit any
        of it;
      - the child's rest rotation was substituted with ``R(localRot0)`` and
        anchored on the body0 frame, so a stage whose link is authored
        rotated away from its joint frame (a 180° door hinge) FK'd to the
        wrong pose — the authored rigid relative pose keeps it.

      At ``q = 0`` the composition reproduces the authored pose exactly
      (the relative-pose chain telescopes back to the authored rigid world
      frames) — ArtVIP stages author the *closed* pose, like the URDF/MJCF
      assets.

    * **Geometry is a box proxy over point-accurate bounds**: the AABB of the
      link's *visible* visual subtree, expressed in the link's own **rigid**
      frame by folding the link's rigid (scale-extracted) world transform out
      of the world points (so the authored scale rides in the geometry, like
      Genesis's vgeoms — *not* ``ComputeUntransformedBound``, which would
      drop it entirely), converted to an 8-corner box mesh.  The world points
      are the meshes' actual ``points`` transformed by their local-to-world
      matrices (xform + non-uniform scale), so rotated thin/diagonal parts
      are not over-approximated the way a rigidly-transformed authored
      ``extent`` box would be; non-mesh drawables (implicit gprims, curves,
      points) fall back to the bound cache's box.  In base-frame mode the
      points are carried into the reference configuration by the FK
      reference frame.  Every consumer only reads ``mesh.bounds`` —
      vertex-level mesh fidelity would be dead weight, and bounding boxes
      keep the walker independent of any mesh-format support.

      Drawables whose computed visibility resolves to ``invisible`` are
      skipped (RC-2) — Genesis's visual import drops them from both the
      visual and the collision set (``usd_geometry.parse_prim_geoms``'s
      ``is_visible`` gate), so an invisible authored site/receptacle gprim
      must not widen the offline box; a link whose drawables are all
      invisible has no Genesis geometry at all and is omitted.

    Units: outputs are in **metres** — the stage's ``metersPerUnit`` scales
    positions *and* extents uniformly (mirroring
    ``genesis.utils.usd.UsdContext``); revolute ``reference_qpos`` values
    are radians (Genesis convention — USD revolute limits are degrees),
    prismatic ones metres.  The FK itself is computed in stage units
    (per-transform scale factors do not telescope through the joint
    composition) and the resulting boxes are scaled once at the end; Y-up
    stages are mapped onto the Z-up Genesis world with the pre-multiplied
    ``genesis.utils.mesh.Y_UP_TRANSFORM``.

    Frames: results are expressed in the **stage's world frame at the
    reference configuration** — empirically the frame Genesis reports
    ``entity.get_vAABB()`` in for a USD entity loaded at the identity morph
    pose (USD stages are *not* re-centred onto the root body, unlike the
    URDF/MJCF assets, whose base link sits at the origin).  With
    ``link_local=True`` the FK transform is skipped and each mesh is the
    link's **rigid own-frame** box (configuration-independent) — the same
    frame the built entity's link pose (``get_link_pose``) maps back to the
    world, which is what makes the part-grasp enumeration frame sound for
    links carrying an authored scale (the scale sits in the geometry, not
    in the frame).

    Args:
        usd_file: Path to the USD stage (``.usd``/``.usda``/``.usdc``);
            composition arcs resolve sublayers/references relative to it.
        reference_qpos: **Clean** joint name → position at the reference
            configuration (radians for revolute, metres for prismatic).
            Omitted joints stay at the default (zero) displacement.  ``None``
            / empty means the asset's authored closed pose.
        link_local: If ``True``, return geometry in each link's own frame
            instead of the stage's world frame.

    Returns:
        ``{clean_link_name: trimesh.Trimesh}`` with one box-proxy mesh per
        link that has drawable bounds, in the stage's world frame at the
        reference configuration — or in the link's own frame when
        *link_local* is set.

    Raises:
        ArticulationError: If *reference_qpos* names a joint that does not
            exist on the stage (the message lists the clean joint names).
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    import trimesh

    stage = Usd.Stage.Open(str(usd_file))
    reference_qpos = dict(reference_qpos) if reference_qpos else {}

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    y_up = UsdGeom.GetStageUpAxis(stage) == "Y"

    # ---- links: RigidBodyAPI prims + clean names --------------------------
    body_prims = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    path_to_clean = {
        str(prim.GetPath()): clean
        for prim, clean in zip(body_prims, usd_clean_names(body_prims))
    }

    def _resolve_body(path_str: Optional[str]) -> Optional["Usd.Prim"]:
        """Joint ``body0``/``body1`` target path → nearest RigidBodyAPI prim."""
        if not path_str:
            return None
        walked = stage.GetPrimAtPath(path_str)
        while walked.IsValid() and str(walked.GetPath()) != "/":
            if walked.HasAPI(UsdPhysics.RigidBodyAPI):
                return walked
            walked = walked.GetParent()
        return None

    # ---- joints: parent joint keyed by child body path --------------------
    moving: Dict[str, "Usd.Prim"] = {  # full prim path -> joint prim
        str(prim.GetPath()): prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    }
    other_joints = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint) and str(prim.GetPath()) not in moving
    ]
    child_to_joint: Dict[str, "Usd.Prim"] = {}
    # The child link's parentage: child path -> (joint prim, parent body
    # path), plus the *literal* body1 relationship target per child link —
    # localPos1/localRot1 are authored in ITS frame, which may be a nested
    # wrapper prim carrying its own scale (never assumed to be the link
    # itself; Genesis resolves the same literal ``frame_path``).
    child_links: Dict[str, Tuple["Usd.Prim", str]] = {}
    child_frame_prim: Dict[str, "Usd.Prim"] = {}
    for joint_prim in list(moving.values()) + other_joints:
        joint = UsdPhysics.Joint(joint_prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        body0 = _resolve_body(str(body0_targets[0]) if body0_targets else None)
        body1 = _resolve_body(str(body1_targets[0]) if body1_targets else None)
        # body1-less joints weld their body0 to the world (a jointless root).
        if body1 is None or body0 is None or str(body0.GetPath()) == str(body1.GetPath()):
            continue
        child_path = str(body1.GetPath())
        if child_path in child_to_joint and child_to_joint[child_path] != joint_prim:
            import logging

            logging.getLogger(__name__).warning(
                "USD stage %s: body %s is jointed to multiple parents; the "
                "offline FK walker keeps %s and drops %s.",
                usd_file,
                child_path,
                str(child_to_joint[child_path].GetPath()),
                str(joint_prim.GetPath()),
            )
            continue
        child_to_joint[child_path] = joint_prim
        child_links[child_path] = (joint_prim, str(body0.GetPath()))
        frame_prim = (
            stage.GetPrimAtPath(str(body1_targets[0])) if body1_targets else None
        )
        child_frame_prim[child_path] = (
            frame_prim if frame_prim is not None and frame_prim.IsValid() else body1
        )

    # Clean joint names (leaf names; the entity's moving joints, sanitised).
    path_to_clean_joint = {
        str(prim.GetPath()): clean
        for prim, clean in zip(moving.values(), usd_clean_names(moving.values()))
    }
    unknown = sorted(set(reference_qpos) - set(path_to_clean_joint.values()))
    if unknown:
        raise ArticulationError(
            f"Unknown joint name(s) in reference_qpos: {unknown}. "
            f"Available joints: {sorted(path_to_clean_joint.values())}"
        )

    # ---- FK: walk the joint graph root -> leaves --------------------------
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def _authored_world(prim) -> np.ndarray:
        """Authored local-to-world transform (column convention, stage units)."""
        return _usd_matrix_to_numpy(xform_cache.GetLocalToWorldTransform(prim))

    def _joint_anchor_axis(
        joint_prim, child_body, frame_prim
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """``(anchor, axis)`` of a moving joint in the CHILD's rigid link frame.

        Mirrors Genesis's ``_parse_joint_axis_pos``: ``localPos1``/``localRot1``
        (the child is always the joint's ``body1`` here) are authored in the
        *body1 relationship-target prim's* frame; the anchor is the joint
        frame ``T(localPos1)·R(localRot1)`` carried through that prim's
        AUTHORED world matrix (so the target's own scale/rotation counts)
        and folded into the child's rigid link frame.  The axis is the
        ``UsdPhysics.axis`` token's direction rotated by ``localRot1`` and
        carried through the same two frames, normalised (the target frame
        may carry scale).
        """
        joint = UsdPhysics.Joint(joint_prim)
        pos = joint.GetLocalPos1Attr().Get() if joint.GetLocalPos1Attr().HasValue() else None
        rot = joint.GetLocalRot1Attr().Get() if joint.GetLocalRot1Attr().HasValue() else None
        joint_frame = _usd_joint_frame(pos, rot)
        anchor_world = _authored_world(frame_prim) @ joint_frame
        child_rigid = rigid_worlds[str(child_body.GetPath())]
        anchor = np.linalg.inv(child_rigid) @ anchor_world

        axis = None
        if joint_prim.IsA(UsdPhysics.RevoluteJoint):
            axis_str = UsdPhysics.RevoluteJoint(joint_prim).GetAxisAttr().Get() or "X"
        elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
            axis_str = UsdPhysics.PrismaticJoint(joint_prim).GetAxisAttr().Get() or "X"
        else:
            axis_str = None
        if axis_str is not None:
            direction = np.asarray(_USD_AXIS_VECTORS[axis_str], dtype=np.float64)
            if rot is not None:
                direction = _usd_quat_to_matrix(rot)[:3, :3] @ direction
            direction = _authored_world(frame_prim)[:3, :3] @ direction
            direction = np.linalg.inv(child_rigid)[:3, :3] @ direction
            norm = np.linalg.norm(direction)
            if norm > 0.0:
                axis = direction / norm
        return anchor[:3, 3], axis

    # ---- rigid link frames (the FK's state) --------------------------------
    # Each link's authored world transform with the linear map's scale
    # polar-decomposed out — Genesis's ``compute_gs_transform`` split, i.e.
    # the frame the built entity's link actually occupies (pos/quat) while
    # the scale rides on the geometry.  Composing joint frames inside these
    # (the previous walker) inherited every ancestor's authored scale into
    # the child frame; the per-link decomposition cannot.
    rigid_worlds: Dict[str, np.ndarray] = {
        str(body.GetPath()): _usd_rigid_transform(_authored_world(body))
        for body in body_prims
    }
    # q = 0 relative poses: the authored rigid relative pose
    # ``inv(Q_parent) @ Q_child`` — Genesis's ``compute_gs_transform(link,
    # parent_link)``, the built entity's rest configuration (NOT a
    # joint-frame chain: a link authored rotated away from its joint frame —
    # a 180° hinge — keeps that rotation).
    rigid_rels: Dict[str, np.ndarray] = {}
    for child_path, (joint_prim, parent_path) in child_links.items():
        try:
            rigid_rels[child_path] = (
                np.linalg.inv(rigid_worlds[parent_path]) @ rigid_worlds[child_path]
            )
        except np.linalg.LinAlgError:
            rigid_rels[child_path] = rigid_worlds[child_path]

    # FK pass(es): the rest frames (q = 0) ARE the rigid world frames above
    # (the relative-pose chain telescopes), so only the reference
    # configuration needs the walk — and the geometry folds the rest frame
    # out of the points and carries it into the reference configuration by
    # the displaced frame ``frame(q) ∘ frame(0)⁻¹``.
    def _run_fk(q_for_joints) -> Dict[str, np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        unprocessed: Dict[str, "Usd.Prim"] = {
            str(body.GetPath()): body for body in body_prims
        }
        while unprocessed:
            progressed = False
            for body_path, body in list(unprocessed.items()):
                entry = child_links.get(body_path)
                if entry is None:
                    # Root (no joint's body1 — or welded to the world by a
                    # body0-less joint, which the graph skipped): its rigid
                    # authored pose, exactly Genesis's
                    # ``compute_gs_transform(link, None)``.
                    frames[body_path] = rigid_worlds[body_path]
                    progressed = True
                    unprocessed.pop(body_path)
                    continue
                joint_prim, parent_path = entry
                if parent_path not in frames:
                    continue  # parent's frame not computed yet — next sweep
                # q = 0 baseline: the authored rigid relative pose.
                frame = frames[parent_path] @ rigid_rels[body_path]
                if joint_prim in moving.values():
                    q = float(
                        q_for_joints.get(path_to_clean_joint[str(joint_prim.GetPath())], 0.0)
                    )
                    if q != 0.0:
                        anchor, axis = _joint_anchor_axis(
                            joint_prim, body, child_frame_prim[body_path]
                        )
                        if axis is not None:
                            rest_t = frame[:3, 3].copy()
                            rest_R = frame[:3, :3].copy()
                            frame = np.eye(4)
                            if joint_prim.IsA(UsdPhysics.RevoluteJoint):
                                # reference_qpos is in radians (Genesis
                                # convention); the DOF swings the child about
                                # the anchor through its own rigid-frame axis:
                                #   R' = R · Rot(axis, q);  t' = t + R·a − R'·a.
                                new_rotation = rest_R @ _usd_rotation_about_axis(axis, q)[:3, :3]
                                frame[:3, :3] = new_rotation
                                frame[:3, 3] = rest_t + rest_R @ anchor - new_rotation @ anchor
                            else:
                                # Prismatic: translate along the axis, in stage
                                # units (reference_qpos is metres → / metersPerUnit).
                                frame[:3, :3] = rest_R
                                frame[:3, 3] = rest_t + rest_R @ (axis * (q / meters_per_unit))
                        # axis is None only for a degenerate (zero-norm) axis —
                        # leave the child at its q = 0 pose rather than
                        # NaN-ing the whole walk.
                frames[body_path] = frame
                progressed = True
                unprocessed.pop(body_path)
            if not progressed and unprocessed:
                # Cycle / unresolvable chain (broken stage): fall back to the
                # authored rigid poses for the remainder so the walker still
                # returns.
                for body_path in unprocessed:
                    frames.setdefault(body_path, rigid_worlds[body_path])
                unprocessed.clear()
        return frames

    # Under the rigid FK the q = 0 rest frame of every link IS its authored
    # rigid world frame (the relative-pose chain telescopes), so the geometry
    # fold uses it directly — one FK pass, only for the reference config.
    rest_frames = rigid_worlds
    link_frames = _run_fk(reference_qpos) if reference_qpos else dict(rest_frames)

    # ---- geometry: per-link box proxy over point-accurate bounds ----------
    # The link-local bound is computed as (world points) ∘ (inverse of the
    # link's RIGID world frame) — NOT ``ComputeUntransformedBound``, which
    # drops the link's authored *scale* entirely (ArtVIP links can carry a
    # non-unit authored scale: raw points span ±12 while the world bounds sit
    # at ~0.5 m).  Folding the RIGID frame out leaves the scale riding in the
    # geometry — the same split Genesis's vgeoms keep (the link frame is the
    # rigid part, the vgeom carries the full transform) — while still
    # excluding the joint displacement.
    #
    # The world points are *point-accurate* for ``UsdGeom.Mesh`` subtrees: the
    # bound cache builds each sub-prim's box from its authored ``extent`` and
    # then transforms that box rigidly, so a rotated thin/diagonal part under
    # a non-uniform scale can be over-approximated by 10-100x (ArtVIP's
    # desk_lamp002 / cardboard_box_carton_7 / basket_1 shapes — false
    # bbox-gate drops).  Meshes therefore contribute their actual ``points``
    # transformed by their local-to-world matrix (xform + scale); the bound
    # cache remains the fallback for non-mesh drawables (implicit gprims,
    # curves, points) and for links whose points cannot be read.
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True
    )
    bound_purposes = ("default", "render")

    def _is_drawable(prim) -> bool:
        return (
            prim.IsA(UsdGeom.Mesh)
            or prim.IsA(UsdGeom.BasisCurves)
            or prim.IsA(UsdGeom.Points)
            or prim.IsA(UsdGeom.Gprim)
        )

    def _purpose_of(prim) -> str:
        try:
            return str(UsdGeom.Imageable(prim).ComputePurpose())
        except Exception:  # noqa: BLE001 — older pxr without ComputePurpose
            return str(UsdGeom.Imageable(prim).GetPurpose())

    def _is_visible(prim) -> bool:
        """Genesis's visibility gate (``usd_geometry.parse_prim_geoms``).

        Genesis drops every gprim whose computed visibility resolves to
        ``invisible`` from BOTH the visual and the collision set
        (``is_visible = str(gprim.ComputeVisibility()) != "invisible"``), so
        an invisible authored drawable (Lightwheel's ``Sites/*`` burner
        rings, place sites, receptacles — purpose ``default``, visibility
        ``invisible``) must not contribute to the offline box either.
        ``ComputeVisibility`` resolves *inherited* visibility, so a drawable
        under an invisible ancestor scope is caught too.
        """
        try:
            return str(UsdGeom.Imageable(prim).ComputeVisibility()) != "invisible"
        except Exception:  # noqa: BLE001 — older pxr without ComputeVisibility
            return True

    def _mesh_world_points(prim) -> Optional[np.ndarray]:
        """World-space points of a mesh (xform + non-uniform scale applied)."""
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points is None or len(points) == 0:
            return None
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(array).all():
            return None
        matrix = _usd_matrix_to_numpy(xform_cache.GetLocalToWorldTransform(prim))
        world = array @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.isfinite(world).all():
            return None
        return world

    def _bound_box_points(bound) -> np.ndarray:
        """The 8 corners of a bound-cache range (the non-mesh fallback)."""
        aligned = bound.ComputeAlignedRange()
        low = np.asarray(aligned.GetMin(), dtype=np.float64)
        high = np.asarray(aligned.GetMax(), dtype=np.float64)
        return np.array([
            [sx, sy, sz]
            for sx in (low[0], high[0])
            for sy in (low[1], high[1])
            for sz in (low[2], high[2])
        ])

    def _link_world_points(body) -> Optional[np.ndarray]:
        """Point-accurate world points of a link's drawable geometry.

        Meshes contribute their actual ``points`` (transformed by their
        local-to-world matrix, i.e. including non-uniform scale and every
        intermediate rotation), so a rotated thin/diagonal part is *not*
        over-approximated the way a rigidly-transformed authored ``extent``
        box would be.  Non-mesh drawables (implicit gprims, curves, points)
        and links whose points cannot be read fall back to the bound cache's
        8-corner box — the pre-existing conservative path.
        """
        mesh_prims: List = []
        other_drawables: List = []
        stack = [body]
        while stack:
            prim = stack.pop()
            if _is_drawable(prim):
                if prim.IsA(UsdGeom.Mesh):
                    mesh_prims.append(prim)
                else:
                    other_drawables.append(prim)
            stack.extend(prim.GetChildren())

        chunks: List[np.ndarray] = []
        mesh_covered_paths = set()
        for prim in mesh_prims:
            if _purpose_of(prim) not in bound_purposes:
                continue
            if not _is_visible(prim):
                continue  # Genesis's visibility gate — no visual, no collision
            points = _mesh_world_points(prim)
            if points is None:
                continue
            chunks.append(points)
            walked = prim
            while walked.IsValid() and str(walked.GetPath()) != str(body.GetPath()):
                mesh_covered_paths.add(str(walked.GetPath()))
                walked = walked.GetParent()

        if chunks:
            # Non-mesh drawables whose geometry no mesh already accounts for
            # (implicit gprims / curves) keep the bound cache's box corners.
            for prim in other_drawables:
                if str(prim.GetPath()) in mesh_covered_paths:
                    continue
                if _purpose_of(prim) not in bound_purposes:
                    continue
                if not _is_visible(prim):
                    continue  # Genesis's visibility gate
                bound = bbox_cache.ComputeWorldBound(prim)
                if bound.ComputeAlignedRange().IsEmpty():
                    continue
                chunks.append(_bound_box_points(bound))
            return np.vstack(chunks)

        # Fallback: the whole-subtree bound cache (non-mesh links, unreadable
        # points) — the pre-existing behaviour.  The bound cache ignores
        # visibility, so a link whose drawables are ALL invisible (no Genesis
        # visual at all) must be omitted instead of falling through to it.
        if not any(_is_visible(prim) for prim in mesh_prims + other_drawables):
            return None
        bound = bbox_cache.ComputeWorldBound(body)
        if bound.ComputeAlignedRange().IsEmpty():
            return None
        return _bound_box_points(bound)

    result: Dict[str, "trimesh.Trimesh"] = {}
    for body in body_prims:
        body_path = str(body.GetPath())
        world_points = _link_world_points(body)
        if world_points is None or len(world_points) == 0:
            continue  # link without drawable geometry (visuals omitted)
        if not np.isfinite(world_points).all():
            continue
        # The rest frame at q = 0: the link's RIGID world frame (the authored
        # transform with its linear scale polar-decomposed out — the frame
        # the built entity's link occupies; Genesis's vgeoms keep the authored
        # full transform, i.e. the scale rides in the geometry).
        rest_frame = rest_frames[body_path]
        try:
            rest_frame_inv = np.linalg.inv(rest_frame)
        except np.linalg.LinAlgError:
            continue
        # Geometry in the link's own (rigid rest) frame: fold the rigid frame
        # out of the *points* — exact at q = 0 by construction (the rigid FK's
        # baseline is the authored rigid relative pose, so the chain telescopes
        # back onto the authored world frames).
        local_points = (
            rest_frame_inv[:3, :3] @ world_points.T
        ).T + rest_frame_inv[:3, 3]
        if link_local:
            # The link's RIGID own frame — the frame ``get_link_pose`` maps
            # back to the world (the built entity's link pos/quat), which is
            # what makes this the sound part-grasp enumeration frame: the
            # authored scale already rides in the points, so mapping the
            # candidates by the runtime link pose (rigid) neither loses nor
            # double-applies it.  (Re-applying the authored linear map here —
            # the pre-rigid-FK behaviour — would bake scale+rotation into the
            # box that the runtime link pose then applies AGAIN.)
            low, high = local_points.min(axis=0), local_points.max(axis=0)
            mesh = trimesh.creation.box(extents=np.maximum(high - low, 1e-9))
            mesh.apply_translation((low + high) / 2.0)
        else:
            # World frame at the reference configuration: the link's points
            # carried by the FK reference frame (identical to the rest frame
            # when *reference_qpos* is empty, so the closed pose is exact).
            frame = link_frames[body_path]
            if y_up:
                frame = _USD_Y_UP_TRANSFORM @ frame
            placed = (frame[:3, :3] @ local_points.T).T + frame[:3, 3]
            low, high = placed.min(axis=0), placed.max(axis=0)
            mesh = trimesh.creation.box(extents=np.maximum(high - low, 1e-9))
            mesh.apply_translation((low + high) / 2.0)
        if meters_per_unit != 1.0:
            mesh.apply_scale(meters_per_unit)
        clean = path_to_clean[body_path]
        if clean not in result:
            result[clean] = mesh
    return result

# ---------------------------------------------------------------------------
# ArticulatedObject
# ---------------------------------------------------------------------------

class ArticulatedObject(ManipulatableObject[EntityT]):
    """Descriptor for articulated furniture (cabinets, drawers, appliances).

    A sibling of :class:`~unienv_genesis_collection.objects.mesh_object.MeshObject`
    under :class:`ManipulatableObject`: furniture has one visual mesh *per
    link*, so the single-``mesh_file`` assumptions of ``MeshObject`` /
    ``HubAssetMeshObject`` do not apply.  Subclasses (the URDF / MJCF hub
    classes) implement :meth:`_load_link_meshes_at_reference` (base-link
    frame, for the union bbox),
    :meth:`_load_link_local_meshes_at_reference` (link-local frame, for
    part grasps) and :meth:`load_into_scene`.

    * **Fixed by default** — furniture is placed once at load via the morph
      pose.  :meth:`set_pos_3d` is overridden only to default
      ``random_quaternion=False`` (furniture keeps its authored yaw).
    * :meth:`get_local_bounding_box` returns the **union AABB over all
      links at the reference configuration** (``reference_qpos``, default
      all-zero = authored closed pose), in the base-link frame.  Computed
      offline — no Genesis scene required.
    * Joint/link addressing is **by name** against the built entity via
      :func:`resolve_joint_dofs` / :func:`resolve_link_index` — no registry,
      no baked metadata.
    * Per-link queries (pose, bbox, part grasps) work on the *current*
      runtime state, so they track a moving drawer or door.

    Args:
        name: Technical identifier used when loading into the scene.
        fixed: Whether the object is immovable.  Defaults to ``True``:
            furniture is placed once at load.  URDF loads pass this to the
            morph; MJCF assets encode it with a jointless root body.
            Note that DoF *state* is per-env even for fixed entities, so
            per-env articulation randomisation needs no extra morph flags.
            Per-env *base poses* of fixed entities require
            ``batch_fixed_verts=True`` on the morph (significant memory
            cost in batched worlds).
        reference_qpos: Joint name → position at the reference
            configuration used by the offline union bbox (FK keys only —
            not a registry).  Defaults to all-zero.
        language_annotations: Optional English-language descriptions
            (e.g. ``["a wooden cabinet", "a kitchen cabinet"]``).

    Quaternion convention: ``(w, x, y, z)``.
    """

    def __init__(
        self,
        name: str = "articulated_object",
        fixed: bool = True,
        reference_qpos: Optional[Mapping[str, float]] = None,
        language_annotations: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            name=name,
            fixed=fixed,
            language_annotations=language_annotations,
        )
        # FK keys only: joint names resolvable by the offline FK helpers.
        self.reference_qpos: Dict[str, float] = (
            dict(reference_qpos) if reference_qpos is not None else {}
        )
        self._link_meshes_cache: Optional[Dict[str, "trimesh.Trimesh"]] = None
        self._link_local_meshes_cache: Optional[Dict[str, "trimesh.Trimesh"]] = None
        self._link_bbox_cache: Dict[str, "BoxSpace"] = {}
        self._part_grasp_bbox_cache: Dict[str, "BoxSpace"] = {}
        self._local_bbox_cache: Optional["BoxSpace"] = None
        self._part_grasp_cache: Dict[Tuple[str, float, float], GraspCandidates] = {}

    # ------------------------------------------------------------------
    # Per-link geometry (offline; subclass provides the meshes)
    # ------------------------------------------------------------------

    def _load_link_meshes_at_reference(self) -> Dict[str, "trimesh.Trimesh"]:
        """Link name → visual mesh in the base-link frame at the reference
        configuration (default all-zero / ``reference_qpos`` keys).

        Implemented by the concrete hub subclasses via
        :func:`urdf_link_meshes_at_config` (ReplicaCAD, URDF+GLB) or
        :func:`mjcf_link_meshes_at_config` (RoboCasa, MJCF).  The result is
        cached by :meth:`_link_meshes_at_reference`.

        Raises:
            NotImplementedError: In the abstract base — subclasses must
                override it.
        """
        raise NotImplementedError(
            "ArticulatedObject subclasses must implement "
            "_load_link_meshes_at_reference() (link visual meshes in the "
            "base-link frame at reference_qpos); use "
            "urdf_link_meshes_at_config / mjcf_link_meshes_at_config / "
            "usd_link_meshes_at_config."
        )

    def _link_meshes_at_reference(self) -> Dict[str, "trimesh.Trimesh"]:
        """Cached :meth:`_load_link_meshes_at_reference` result."""
        if self._link_meshes_cache is None:
            self._link_meshes_cache = dict(self._load_link_meshes_at_reference())
        return self._link_meshes_cache

    def _load_link_local_meshes_at_reference(self) -> Dict[str, "trimesh.Trimesh"]:
        """Link name → visual mesh in the **link's own local frame**.

        The enumeration frame for part-grasp candidates: a point expressed
        here maps to the world through the link's *current*
        ``world_T_link`` pose (:meth:`get_link_pose`), so candidates stay
        rigidly attached to the link's geometry as the articulation moves.
        The geometry is configuration-independent (a URDF visual is its
        mesh with ``<visual><origin>`` applied; an MJCF geom mesh carries
        its own ``pos``/``quat``/``euler`` — both without any
        ``base_T_link`` FK transform).

        Implemented by the concrete hub subclasses via
        :func:`urdf_link_meshes_at_config` /
        :func:`mjcf_link_meshes_at_config` /
        :func:`usd_link_meshes_at_config` with ``link_local=True``.  The
        result is cached by :meth:`_link_local_meshes_at_reference`.

        Raises:
            NotImplementedError: In the abstract base — subclasses must
                override it.
        """
        raise NotImplementedError(
            "ArticulatedObject subclasses must implement "
            "_load_link_local_meshes_at_reference() (link visual meshes in "
            "the link's own local frame); use "
            "urdf_link_meshes_at_config / mjcf_link_meshes_at_config / "
            "usd_link_meshes_at_config with link_local=True."
        )

    def _link_local_meshes_at_reference(self) -> Dict[str, "trimesh.Trimesh"]:
        """Cached :meth:`_load_link_local_meshes_at_reference` result."""
        if self._link_local_meshes_cache is None:
            self._link_local_meshes_cache = dict(
                self._load_link_local_meshes_at_reference()
            )
        return self._link_local_meshes_cache

    def get_link_local_bounding_box(self, link_name: str) -> "BoxSpace":
        """AABB of one link's visual mesh in the base frame (reference config).

        The result is cached per link name — the bbox is
        configuration-independent (computed offline at ``reference_qpos``).

        Args:
            link_name: Link whose *visual* geometry defines the box.

        Returns:
            ``BoxSpace`` (shape ``(3,)``) in the base-link frame.

        Raises:
            ArticulationError: If the link has no offline visual mesh (the
                message lists the links that do).
        """
        if link_name not in self._link_bbox_cache:
            meshes = self._link_meshes_at_reference()
            if link_name not in meshes:
                raise ArticulationError(
                    f"Link {link_name!r} has no offline visual mesh. "
                    f"Links with meshes: {sorted(meshes.keys())}"
                )
            bounds = np.array(meshes[link_name].bounds, dtype=np.float32)
            self._link_bbox_cache[link_name] = make_bbox(bounds[0], bounds[1])
        return self._link_bbox_cache[link_name]

    def get_local_bounding_box(self) -> "BoxSpace":
        """Union AABB over all links at the reference configuration.

        The union is taken over the per-link offline visual meshes (base-link
        frame), i.e. it is the AABB of the asset with every joint at
        ``reference_qpos`` (default all-zero — the authored closed pose).
        Cached after the first call; entity-independent and therefore
        offline-safe (matching the descriptor contract used by tasks, which
        call it right after reload).

        Returns:
            ``BoxSpace`` (shape ``(3,)``) in the base-link frame.

        Raises:
            ArticulationError: If the descriptor has no link meshes at all.
        """
        if self._local_bbox_cache is None:
            meshes = self._link_meshes_at_reference()
            if not meshes:
                raise ArticulationError(
                    f"ArticulatedObject {self.name!r} has no link meshes; "
                    "cannot compute a local bounding box."
                )
            lows = []
            highs = []
            for mesh in meshes.values():
                bounds = np.array(mesh.bounds, dtype=np.float32)
                lows.append(bounds[0])
                highs.append(bounds[1])
            self._local_bbox_cache = make_bbox(
                np.min(np.stack(lows), axis=0),
                np.max(np.stack(highs), axis=0),
            )
        return self._local_bbox_cache

    # ------------------------------------------------------------------
    # Joint state (runtime, entity-backed; thin pass-throughs by name)
    # ------------------------------------------------------------------

    def get_dofs_position(
        self,
        entity: EntityT,
        joint_name: str,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Current position of a named joint's DOFs, resolved at runtime.

        Args:
            entity: The loaded Genesis entity.
            joint_name: Joint name (resolved via :func:`resolve_joint_dofs`).
            envs_idx: Optional environment selector (a ``(batch_size,)``
                ``torch.bool`` mask is accepted, as in ``pickplace.py``).

        Returns:
            Tensor of shape ``(n_dofs_of_joint,)`` or
            ``(n_selected_envs, n_dofs_of_joint)`` as returned by Genesis —
            the index dimension is **not** squeezed.

        Raises:
            ArticulationError: If *joint_name* is unknown.
        """
        dofs_idx = resolve_joint_dofs(entity, joint_name)
        return entity.get_dofs_position(dofs_idx_local=dofs_idx, envs_idx=envs_idx)

    def set_dofs_position(
        self,
        entity: EntityT,
        position: torch.Tensor,
        joint_name: str,
        envs_idx: Optional[torch.Tensor] = None,
        zero_velocity: bool = True,
    ) -> None:
        """Set a named joint's DOFs, resolved at runtime.

        Args:
            entity: The loaded Genesis entity.
            position: Target positions for the joint's DOFs — shape
                ``(n_dofs_of_joint,)`` / ``(n_selected_envs,
                n_dofs_of_joint)``.
            joint_name: Joint name (resolved via :func:`resolve_joint_dofs`).
            envs_idx: Optional environment selector (``torch.bool`` mask
                accepted); only the selected environments are written.
            zero_velocity: Zero the joint velocities after the write
                (Genesis ``set_dofs_position`` flag).

        Raises:
            ArticulationError: If *joint_name* is unknown.
        """
        dofs_idx = resolve_joint_dofs(entity, joint_name)
        entity.set_dofs_position(
            position,
            dofs_idx_local=dofs_idx,
            envs_idx=envs_idx,
            zero_velocity=zero_velocity,
        )

    def get_joint_limits(
        self,
        entity: EntityT,
        joint_name: str,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runtime position limits ``(q_min, q_max)`` of a named joint.

        Limits are read from the built entity via
        ``entity.get_dofs_limit(dofs_idx_local=…)`` — never baked metadata.
        The read deliberately does **not** forward *envs_idx*: with the
        default ``RigidOptions(batch_dofs_info=False)`` Genesis raises
        ``GenesisException`` when ``envs_idx`` is given, because joint limits
        are a **per-joint constant** shared by all environments.  The
        unmasked read therefore returns the canonical
        ``(n_dofs_of_joint,)`` limits, which broadcast against a masked
        position read.  Only when the scene explicitly opts into
        ``batch_dofs_info=True`` does the unmasked read return per-env limits
        ``(n_envs, n_dofs_of_joint)``; the caller's *envs_idx* mask is then
        applied to those locally so the result lines up with the masked
        position read (no ``envs_idx`` is ever forwarded to Genesis).

        Hinge (rad) and slide (m) joints are uniform in fraction space.

        Args:
            entity: The loaded Genesis entity.
            joint_name: Joint name (resolved via :func:`resolve_joint_dofs`).
            envs_idx: Optional environment selector (mask or index tensor);
                applied to the limits only in ``batch_dofs_info=True``
                scenes (per-env limits), otherwise they are scene-wide
                constants and the selector is irrelevant.

        Returns:
            ``(q_min, q_max)``, each ``(n_dofs_of_joint,)`` /
            ``(n_selected_envs, n_dofs_of_joint)``.

        Raises:
            ArticulationError: If *joint_name* is unknown, the limits are
                ``±inf`` (an unbounded joint has no meaningful open
                fraction), or the range is degenerate (``q_max <= q_min``).
        """
        dofs_idx = resolve_joint_dofs(entity, joint_name)
        # Unmasked read: works for both batch_dofs_info settings (see above).
        q_min, q_max = entity.get_dofs_limit(dofs_idx_local=dofs_idx)
        if envs_idx is not None and q_min.ndim >= 2:
            # batch_dofs_info=True scene: the unmasked read returned per-env
            # limits (n_envs, n_dofs); mask them to match the position read.
            q_min = q_min[envs_idx]
            q_max = q_max[envs_idx]
        if bool(torch.isinf(q_min).any()) or bool(torch.isinf(q_max).any()):
            raise ArticulationError(
                f"Joint {joint_name!r} has unbounded (±inf) position limits "
                f"(q_min={q_min}, q_max={q_max}); open fraction is undefined. "
                "Use get_dofs_position/set_dofs_position for unbounded joints."
            )
        if bool((q_max <= q_min).any()):
            raise ArticulationError(
                f"Joint {joint_name!r} has a degenerate position range "
                f"(q_min={q_min}, q_max={q_max})."
            )
        return q_min, q_max

    def get_open_fraction(
        self,
        entity: EntityT,
        joint_name: str,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalised opening ``(q - q_min) / (q_max - q_min)`` clamped to [0, 1].

        0 = fully closed (at the lower limit), 1 = fully open (at the upper
        limit).  ``q_min`` / ``q_max`` come from
        ``entity.get_dofs_limit(dofs_idx_local=…)`` at runtime — **not** from
        baked metadata — so hinge (rad) and slide (m) are uniform in
        fraction space.  Limits are read **without** ``envs_idx`` (joint
        limits are scene-wide constants unless the scene opts into
        ``batch_dofs_info=True``, where they are per-env and get masked
        locally); only the position read is masked by *envs_idx*.

        Args:
            entity: The loaded Genesis entity.
            joint_name: Joint name (resolved via :func:`resolve_joint_dofs`).
            envs_idx: Optional environment selector (applied to the position
                read — and to per-env limits in ``batch_dofs_info=True``
                scenes — see :meth:`get_joint_limits`).

        Returns:
            Open fraction, shape ``(n_dofs_of_joint,)`` /
            ``(n_selected_envs, n_dofs_of_joint)``, each entry in ``[0, 1]``.

        Raises:
            ArticulationError: If *joint_name* is unknown or its limits are
                unbounded (see :meth:`get_joint_limits`).
        """
        q_min, q_max = self.get_joint_limits(entity, joint_name, envs_idx=envs_idx)
        q = self.get_dofs_position(entity, joint_name, envs_idx=envs_idx)
        fraction = (q - q_min) / (q_max - q_min)
        return torch.clamp(fraction, 0.0, 1.0)

    def set_open_fraction(
        self,
        entity: EntityT,
        joint_name: str,
        fraction: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None,
        zero_velocity: bool = True,
    ) -> None:
        """Drive a named joint to an opening *fraction* (clamped to [0, 1]).

        ``q = q_min + fraction · (q_max - q_min)`` with runtime limits (see
        :meth:`get_joint_limits`), applied through :meth:`set_dofs_position`.
        Limits are read **unmasked** (scene-wide constants unless the scene
        opts into ``batch_dofs_info=True``, where they are per-env and get
        masked locally) — Genesis rejects ``envs_idx`` for the default
        non-batched DOF info, and limits do not depend on the env.  A
        *fraction* may be a scalar or carry one fewer dimension than the
        limits (e.g. ``(n_selected_envs,)`` for a single-DOF joint in a
        batched world); it is broadcast to the joint's DOFs.

        Args:
            entity: The loaded Genesis entity.
            joint_name: Joint name (resolved via :func:`resolve_joint_dofs`).
            fraction: Target open fraction(s), clamped to ``[0, 1]``.
            envs_idx: Optional environment selector (``torch.bool`` mask
                accepted); only the selected environments are written
                (per-env limits, when present, are selected the same way).
            zero_velocity: Zero the joint velocities after the write.

        Raises:
            ArticulationError: If *joint_name* is unknown or its limits are
                unbounded (see :meth:`get_joint_limits`).
        """
        q_min, q_max = self.get_joint_limits(entity, joint_name, envs_idx=envs_idx)
        clamped = torch.clamp(
            torch.as_tensor(fraction, dtype=q_min.dtype, device=q_min.device),
            0.0,
            1.0,
        )
        # Broadcast a per-env fraction (B,) to a (B, n_dofs) limits tensor.
        if clamped.ndim == q_min.ndim - 1:
            clamped = clamped.unsqueeze(-1)
        q = q_min + clamped * (q_max - q_min)
        self.set_dofs_position(
            entity,
            q.to(dtype=q_min.dtype),
            joint_name,
            envs_idx=envs_idx,
            zero_velocity=zero_velocity,
        )

    # ------------------------------------------------------------------
    # Per-link runtime pose
    # ------------------------------------------------------------------

    def get_link_pose(
        self,
        entity: EntityT,
        link_name: str,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """World-frame pose of a named link at the current configuration.

        The link index is resolved off the built entity
        (:func:`resolve_link_index`) and queried with
        ``entity.get_links_pos/get_links_quat(links_idx_local=[idx],
        relative=True)`` — the same user-frame convention as
        ``entity.get_pos()/get_quat()``, so the result composes with
        :meth:`get_bounding_box` and ``pickplace``-style placement.

        Args:
            entity: The loaded Genesis entity.
            link_name: Link name (resolved via :func:`resolve_link_index`).
            envs_idx: Optional environment selector.

        Returns:
            ``(position, quaternion)`` with shapes ``(3,)`` / ``(4,)`` for an
            unbatched world, or ``(n_selected_envs, 3)`` /
            ``(n_selected_envs, 4)`` when batched — the single-link index
            dimension is squeezed.  Quaternion convention ``(w, x, y, z)``.

        Raises:
            ArticulationError: If *link_name* is unknown.
        """
        link_idx = resolve_link_index(entity, link_name)
        position = entity.get_links_pos(
            links_idx_local=[link_idx], envs_idx=envs_idx, relative=True,
        ).detach().squeeze(-2)
        quaternion = entity.get_links_quat(
            links_idx_local=[link_idx], envs_idx=envs_idx, relative=True,
        ).detach().squeeze(-2)
        return position, quaternion

    # ------------------------------------------------------------------
    # Part-aware grasps
    # ------------------------------------------------------------------

    def _part_grasp_bounding_box(self, link_name: str) -> "BoxSpace":
        """AABB of one link's visual mesh in the **link's own frame**.

        The enumeration frame of :meth:`list_part_grasping_candidates`:
        candidates are enumerated here and then transformed by the link's
        *current* world pose, so they sit on the link's geometry wherever
        the articulation takes it.  This is deliberately **not**
        :meth:`get_link_local_bounding_box` — that method returns the
        base-link frame AABB (at the reference configuration), which feeds
        the union bbox / receptacles and must not be combined with
        ``world_T_link`` (the ``base_T_link`` transform would be applied
        twice, shifting every candidate by the link's offset/rotation from
        the base).

        The result is cached per link name — the link-local bbox is
        configuration-independent.

        Args:
            link_name: Link whose *visual* geometry defines the box.

        Returns:
            ``BoxSpace`` (shape ``(3,)``) in the link's own frame.

        Raises:
            ArticulationError: If the link has no offline visual mesh (the
                message lists the links that do).
        """
        if link_name not in self._part_grasp_bbox_cache:
            meshes = self._link_local_meshes_at_reference()
            if link_name not in meshes:
                raise ArticulationError(
                    f"Link {link_name!r} has no offline visual mesh. "
                    f"Links with meshes: {sorted(meshes.keys())}"
                )
            bounds = np.array(meshes[link_name].bounds, dtype=np.float32)
            self._part_grasp_bbox_cache[link_name] = make_bbox(bounds[0], bounds[1])
        return self._part_grasp_bbox_cache[link_name]

    def list_part_grasping_candidates(
        self,
        entity: EntityT,
        link_name: str,
        gripper_width: float,
        gripper_depth: float = 0.01,
        part_aliases: Optional[Mapping[str, str]] = None,
        rng: Optional[torch.Generator] = None,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> GraspCandidates:
        """Candidate grasps on a named link, in world coordinates.

        The link's visual geometry is enumerated **offline in the link's
        own local frame** — the AABB of
        :meth:`_part_grasp_bounding_box` (the mesh *before* ``base_T_link``
        is applied; cached per ``(link, gripper_width, gripper_depth)``) —
        with the same algorithm as
        :meth:`ManipulatableObject.list_box_gripper_grasping_candidates_local`.
        The candidates are then transformed by the link's **current** world
        pose via :meth:`get_link_pose`:

        ``world = world_T_link(current) · link_local``

        so they are rigidly attached to the link's geometry and track a
        drawer as it slides or a door as it swings.

        Frame note: the per-link bbox from
        :meth:`get_link_local_bounding_box` is expressed in the **base-link
        frame** at the reference configuration instead — it feeds the union
        bbox / receptacles and must **not** be combined with
        ``world_T_link`` here (that would multiply already-base-frame
        candidates by ``base_T_link`` again, shifting them by the link's
        offset/rotation from the base).

        Args:
            entity: The loaded Genesis entity.
            link_name: Link name — or a key of *part_aliases* — resolved
                against the entity at runtime (no baked metadata).
            gripper_width: Maximum gripper opening in metres.
            gripper_depth: Depth of the gripper fingers in metres.
            part_aliases: Optional caller-supplied alias map, e.g.
                ``{"handle": "cabinet_door_handle"}``.  A plain runtime
                argument, not baked per-asset metadata.
            rng: Optional :class:`torch.Generator`, accepted for API
                symmetry with the bbox grasp method.  The enumeration is
                deterministic and does not consume it.
            envs_idx: Optional environment selector (``torch.bool`` mask
                accepted) forwarded to :meth:`get_link_pose`; only the
                selected environments contribute poses, so the returned
                candidates are masked to ``(n_selected_envs, N, 3)``.

        Returns:
            :class:`GraspCandidates` in world coordinates, shape
            ``(N, 3)`` for an unbatched world, ``(n_envs, N, 3)`` for a
            batched world, or ``(n_selected_envs, N, 3)`` when *envs_idx*
            selects a subset of environments.

        Raises:
            ArticulationError: If the resolved name is not a link of the
                entity (the message lists the available links) or has no
                offline visual mesh (the message lists the links that do).
        """
        resolved_name = (
            part_aliases.get(link_name, link_name) if part_aliases is not None else link_name
        )
        # Fail loud on the entity first: it is the runtime authority for
        # link names, and its error lists every link of the built asset.
        resolve_link_index(entity, resolved_name)

        key = (resolved_name, float(gripper_width), float(gripper_depth))
        local = self._part_grasp_cache.get(key)
        if local is None:
            local = _enumerate_bbox_gripper_candidates(
                self._part_grasp_bounding_box(resolved_name),
                gripper_width,
                gripper_depth,
            )
            self._part_grasp_cache[key] = local

        position, quaternion = self.get_link_pose(entity, resolved_name, envs_idx=envs_idx)
        return _local_to_world_candidates(local.clone(), position, quaternion)

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def set_pos_3d(
        self,
        entity: EntityT,
        positions: torch.Tensor,
        random_quaternion: bool = False,
        rng: Optional[torch.Generator] = None,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Place the entity, keeping its authored orientation by default.

        Delegates to :meth:`ManipulatableObject.set_pos_3d` (which handles
        unbatched ``(3,)`` and batched ``(n, 3)`` positions as well as the
        ``envs_idx`` boolean mask); only the default of
        *random_quaternion* differs — furniture is yaw-sensitive and keeps
        its authored orientation unless randomisation is requested.

        Note for batched worlds: per-env poses of a **fixed** entity require
        the morph to have been loaded with ``batch_fixed_verts=True``;
        without it, all environments share one layout (documented Genesis
        limitation).  DoF state is per-env regardless.

        Args:
            entity: The loaded Genesis entity.
            positions: World positions, shape ``(3,)`` or ``(n, 3)``.
            random_quaternion: If ``True``, apply a random yaw on top of the
                standing orientation; default ``False`` (authored yaw).
            rng: Optional :class:`torch.Generator` for reproducibility.
            envs_idx: Optional environment selector for partial (masked)
                updates — a ``(batch_size,)`` ``torch.bool`` mask.
        """
        super().set_pos_3d(
            entity,
            positions,
            random_quaternion=random_quaternion,
            rng=rng,
            envs_idx=envs_idx,
        )
