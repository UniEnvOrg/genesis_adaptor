"""Mesh-based ManipulatableObject.

Uses **trimesh** (lazy import) for offline bounding-box computation and
grasp-candidate analysis.  The mesh file is loaded on first call to
:meth:`get_local_bounding_box` or :meth:`list_box_gripper_grasping_candidates_local`
and cached for subsequent calls.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, List, Union, Sequence, TYPE_CHECKING

import numpy as np
import torch
from genesis.engine import entities as gs_entities

from unienv_genesis_collection.objects.base_object import (
    ManipulatableObject,
    GraspCandidates,
    make_bbox,
)

if TYPE_CHECKING:
    import trimesh
    from unienv_interface.space import BoxSpace
    from unienv_genesis.world.genesis_world import GenesisWorld

__all__ = ["MeshObject"]


class MeshObject(ManipulatableObject):
    """A mesh-based manipulatable object (``gs.morphs.Mesh``).

    The mesh file is loaded lazily via ``trimesh`` on first geometry query.

    Args:
        mesh_file: Path to mesh file (``.obj``, ``.stl``, ``.ply``, …).
        scale: Uniform scalar **or** per-axis ``(sx, sy, sz)`` scale factor.
        decimate: Whether Genesis should decimate the mesh for physics.
        color: RGBA colour tuple.
        name: Technical identifier forwarded to :class:`ManipulatableObject`.
        fixed: Whether the object is immovable; forwarded to
            :class:`ManipulatableObject`.
        language_annotations: Optional sequence of English-language
            descriptions.  Unlike the primitive subclasses, meshes have no
            default semantic annotation — ``None`` yields an empty tuple.
        visual_texture_path: Optional path to an image file (e.g. a PNG)
            to be bound as the diffuse texture of the loaded mesh.  When
            set, callers that load this descriptor as a bare visual mesh
            (e.g. :class:`FloatingVisualPropsManager`) build a
            textured :class:`genesis.surfaces.Plastic` surface from it so
            the polygon's intrinsic texture survives the loss of its
            native URDF/MJCF material binding.  ``None`` (the default)
            keeps the Genesis default surface — a plain mesh with no
            explicit texture.
        mass: Optional total mass of the object in **kg**.  When set, the
            loaded entity's mass is forced to this value by
            :meth:`apply_mass` (which scene-assembly nodes call once the
            scene is built) instead of the mass Genesis derives from the
            geometry and the default material density.  ``None`` (the
            default) keeps Genesis's derived mass.
    """

    def __init__(
        self,
        mesh_file: Union[str, os.PathLike],
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        decimate: bool = True,
        color: Optional[Tuple[float, float, float, float]] = None,
        name: str = "object",
        fixed: bool = False,
        language_annotations: Optional[Sequence[str]] = None,
        visual_texture_path: Optional[Union[str, os.PathLike]] = None,
        standing_quaternion: Optional[Sequence[float]] = None,
        mass: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            fixed=fixed,
            language_annotations=language_annotations,
            standing_quaternion=standing_quaternion,
        )
        self.mesh_file = str(mesh_file)
        self.scale = scale if isinstance(scale, tuple) else (scale, scale, scale)
        self.decimate = decimate
        self.color = color
        self.visual_texture_path: Optional[str] = (
            str(visual_texture_path) if visual_texture_path is not None else None
        )
        if mass is not None and not mass > 0.0:
            raise ValueError(f"mass must be a positive mass in kg, got {mass!r}")
        self.mass: Optional[float] = float(mass) if mass is not None else None
        self._trimesh_cache: Optional["trimesh.Trimesh"] = None

    # ------------------------------------------------------------------
    # Trimesh loading
    # ------------------------------------------------------------------

    def _load_trimesh(self) -> "trimesh.Trimesh":
        """Load and cache the trimesh geometry."""
        if self._trimesh_cache is None:
            import trimesh

            mesh = trimesh.load(self.mesh_file, force="mesh")
            mesh.apply_scale(self.scale)
            self._trimesh_cache = mesh
        return self._trimesh_cache

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def get_local_bounding_box(self) -> "BoxSpace":
        mesh = self._load_trimesh()
        return make_bbox(
            torch.from_numpy(mesh.bounds[0]),
            torch.from_numpy(mesh.bounds[1]),
        )

    # ------------------------------------------------------------------
    # Grasp analysis (mesh-based antipodal sampling)
    # ------------------------------------------------------------------

    def list_box_gripper_grasping_candidates_local(
        self,
        entity: gs_entities.RigidEntity,
        gripper_width: float,
        gripper_depth: float = 0.01,
        n_samples: int = 1000,
        rng: Optional[torch.Generator] = None,
        *,
        friction_cone_deg: float = 15.0,
        pair_normal_deg: float = 20.0,
        pair_axis_deg: float = 30.0,
        pair_lateral_tol: float = 0.003,
        min_grasp_width: float = 0.0005,
        width_margin: float = 0.01,
        max_pairs: int = 128,
        pair_voxel_size: float = 0.01,
        approach_tilts_deg: Sequence[float] = (-60.0, -30.0, 0.0, 30.0, 60.0),
        max_approach_tilt_deg: float = 60.0,
        contact_inset: float = 0.01,
        finger_width: float = 0.022,
        finger_thickness: float = 0.01,
        n_clearance_samples: int = 4000,
    ) -> GraspCandidates:
        """Generate antipodal grasp candidates via mesh surface sampling (local frame).

        1. **Pairs.**  Samples ``n_samples`` surface points and keeps the pairs
           a parallel-jaw gripper can pinch square-on: ``min_grasp_width`` ..
           ``gripper_width - width_margin`` apart (the slack leaves the open
           jaws room to clear both faces despite pose error), outward normals opposed within
           ``pair_normal_deg``, pointing away from each other, and each within
           ``pair_axis_deg`` of the pair axis *or* offset sideways by at most
           ``pair_lateral_tol`` (thin plates such as a box's cardboard hang
           tab, where an angular cone would demand sub-millimetre alignment).
           The jaws close along the *averaged contact normal*
           (``opening_direction``), so they meet both faces square-on; step 4
           checks the actual contacts against the tighter
           ``friction_cone_deg``.  Pairs whose closing axis leaves no approach
           within ``max_approach_tilt_deg`` of down are dropped here.
        2. **Coverage.**  Pairs are bucketed by contact midpoint
           (``pair_voxel_size`` voxels) and closing direction (~30° bins), one
           pair per bucket, and up to ``max_pairs`` buckets are kept — spread
           over the object instead of concentrated where its surface area is.
        3. **Approaches.**  Each pair gets approaches perpendicular to the
           closing axis, ``approach_tilts_deg`` about it from the most downward
           one, where "down" is the object's upright down
           (:attr:`standing_quaternion`); approaches tilted more than
           ``max_approach_tilt_deg`` from it are dropped.  The grasp position
           is the EEF point (the fingertip), ``contact_inset`` past the contact
           midpoint, so the contacts land on the finger pads.
        4. **Clearance.**  Checked against a dense surface sample
           (``n_clearance_samples`` points with normals): no material in a
           finger slab (``finger_width`` × ``finger_thickness`` just outside
           the open jaws) behind the fingertip, nor between the jaws more than
           ``gripper_depth`` behind it (the palm).  When only the palm is
           blocked, the grasp slides back along the approach until it clears.
           Every kept grasp (slid or not) must close on surface: the outermost
           sample near the closing line on each side faces outward within the
           friction cone, and the two are ``min_grasp_width`` ..
           ``gripper_width - width_margin`` apart.  Clearance from the table and other objects
           is left to world-frame candidate filters.

        Falls back to the bounding-box method of the base class if no
        candidate survives.  Results are cached per argument set and seed.

        Args:
            rng: Optional :class:`torch.Generator` for reproducible surface
                sampling and pair selection.  When ``None``, trimesh's and
                numpy's default RNG behaviour is used.
        """
        if not hasattr(self, "_local_grasp_cache"):
            self._local_grasp_cache: dict[tuple, GraspCandidates] = {}

        # Derive a trimesh-compatible int seed at the boundary.  trimesh's
        # ``sample_surface`` accepts an int seed (not a torch.Generator), so
        # draw one from the generator's stream when provided.  The derived
        # int is also used as the cache key so caching stays consistent.
        # The output device is pinned to the generator's device to avoid a
        # device mismatch under a CUDA default device.
        seed = (
            int(torch.randint(2**31 - 1, (), generator=rng, device=rng.device))
            if rng is not None
            else None
        )

        key = (
            gripper_width, gripper_depth, n_samples, seed, friction_cone_deg,
            pair_normal_deg, pair_axis_deg, pair_lateral_tol, min_grasp_width,
            width_margin, max_pairs, pair_voxel_size, tuple(approach_tilts_deg),
            max_approach_tilt_deg, contact_inset, finger_width, finger_thickness,
            n_clearance_samples,
        )
        if key in self._local_grasp_cache:
            return self._local_grasp_cache[key].clone()

        mesh = self._load_trimesh()
        # ``Trimesh.sample`` does not forward a ``seed`` to ``sample_surface``,
        # so call ``trimesh.sample.sample_surface`` directly to obtain
        # deterministic results when a seed is supplied.
        import trimesh

        max_grasp_width = gripper_width - width_margin
        np_rng = np.random.default_rng(seed)
        cos_cone = float(np.cos(np.radians(friction_cone_deg)))
        points, face_indices = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
        normals = mesh.face_normals[face_indices]
        pair_mid, pair_axis = _antipodal_pairs(
            points, normals,
            cos_normal=float(np.cos(np.radians(pair_normal_deg))),
            cos_axis=float(np.cos(np.radians(pair_axis_deg))),
            lateral_tol=pair_lateral_tol,
            min_width=min_grasp_width, max_width=max_grasp_width,
        )
        # An approach is perpendicular to the closing axis, so it can come
        # within max_approach_tilt_deg of down only if |axis · down| allows it.
        down = _local_down(self.standing_quaternion)
        feasible = np.abs(pair_axis @ down) <= np.sin(np.radians(max_approach_tilt_deg)) + 1e-9
        pair_mid, pair_axis = pair_mid[feasible], pair_axis[feasible]
        keep = _spread_pairs(pair_mid, pair_axis, pair_voxel_size, max_pairs, np_rng)
        pair_mid, pair_axis = pair_mid[keep], pair_axis[keep]

        local_positions = np.zeros((0, 3))
        local_approaches = np.zeros((0, 3))
        local_openings = np.zeros((0, 3))
        if len(pair_mid):
            approach, valid_tilt = _approaches(
                pair_axis, down, approach_tilts_deg, max_approach_tilt_deg,
            )                                                             # (P, K, 3)
            pair_idx, angle_idx = np.nonzero(valid_tilt)
            approach = approach[pair_idx, angle_idx]                      # (C, 3)
            opening = pair_axis[pair_idx]                                 # (C, 3)
            fingertip = pair_mid[pair_idx] + contact_inset * approach

            cloud, cloud_faces = trimesh.sample.sample_surface(
                mesh, n_clearance_samples,
                seed=None if seed is None else seed + 1,
            )
            cloud_normals = mesh.face_normals[cloud_faces]
            # Contact search radius ~1.5× the cloud's mean point spacing.
            radius = 1.5 * float(np.sqrt(mesh.area / max(n_clearance_samples, 1)))
            gripper = dict(
                gripper_width=gripper_width, gripper_depth=gripper_depth,
                finger_width=finger_width, finger_thickness=finger_thickness,
            )
            fingertip, ok = _clear_or_slide(
                cloud, fingertip, approach, opening, **gripper,
            )
            ok &= _closes_on_surface(
                cloud, cloud_normals, fingertip - contact_inset * approach,
                approach, opening, radius, cos_cone, min_grasp_width, max_grasp_width,
            )
            # Both opening signs (a 180° roll about the approach); the
            # world-frame filter / IK ranking picks the reachable one.
            local_positions = np.repeat(fingertip[ok], 2, axis=0)
            local_approaches = np.repeat(approach[ok], 2, axis=0)
            local_openings = np.repeat(opening[ok], 2, axis=0)
            local_openings[1::2] *= -1.0

        if not len(local_positions):
            # Fall back to bounding-box-based generation
            result = super().list_box_gripper_grasping_candidates_local(
                entity, max_grasp_width, gripper_depth=gripper_depth,
            )
            self._local_grasp_cache[key] = result
            return result.clone()

        result = GraspCandidates(
            position=torch.from_numpy(local_positions),
            approach_direction=torch.from_numpy(local_approaches),
            opening_direction=torch.from_numpy(local_openings),
        )
        self._local_grasp_cache[key] = result
        return result.clone()

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def apply_mass(self, entity: gs_entities.RigidEntity) -> None:
        """Force the entity's total mass to :attr:`mass` (kg), when set.

        Genesis derives a morph's mass from its geometry and the default
        material density, which has no relation to the real object for scanned
        meshes; ``entity.set_mass()`` overrides it exactly (rescaling the
        inertia of every link by the same factor, i.e. how the same shape made
        heavier would behave).

        Must be called **after** the scene is built — Genesis only accepts the
        override on a built scene, and it must happen before the first physics
        step so the object never moves with the derived mass.

        Args:
            entity: The entity returned by :meth:`load_into_scene`.
        """
        if self.mass is None or self.fixed:
            # A fixed object's mass is ill-defined (its links move no mass),
            # so ``set_mass`` would raise; nothing to override either.
            return
        entity.set_mass(self.mass)

    def load_into_scene(
        self,
        world: "GenesisWorld",
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        name_override=None,
        batch_fixed_verts=False,
    ):
        import genesis as gs

        morph = gs.morphs.Mesh(
            file=self.mesh_file,
            scale=self.scale,
            pos=pos,
            quat=quat,
            fixed=self.fixed,
            decimate=self.decimate,
            batch_fixed_verts=batch_fixed_verts,
        )
        if self.visual_texture_path is not None:
            surface = gs.surfaces.Plastic(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path=self.visual_texture_path,
                ),
            )
        elif self.color is not None:
            surface = gs.surfaces.Plastic(color=self.color)
        else:
            surface = None
        return world.scene.add_entity(
            morph,
            surface=surface,
            name=name_override or self.name,
        )


# ----------------------------------------------------------------------
# Antipodal grasp sampling helpers (numpy, object-local frame)
# ----------------------------------------------------------------------


def _antipodal_pairs(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    cos_normal: float,
    cos_axis: float,
    lateral_tol: float,
    min_width: float,
    max_width: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Antipodal surface-point pairs → ``(midpoints, closing axes)``, each ``(P, 3)``.

    A pair ``(i, j)`` is kept when the points are ``min_width`` .. ``max_width``
    apart, their outward normals are opposed within ``acos(cos_normal)`` and
    point away from each other, and either each normal lies within
    ``acos(cos_axis)`` of the pair axis or the points are offset at most
    ``lateral_tol`` across the normal (thin plates).  The closing axis is the averaged contact normal
    ``n_j - n_i`` (the jaws meet both faces square-on).
    """
    diff = (points[None, :, :] - points[:, None, :]).astype(np.float32)
    dist = np.linalg.norm(diff, axis=-1)
    axis = diff / np.maximum(dist, 1e-12)[..., None]
    normals32 = normals.astype(np.float32)
    # axis[i, j] points from p_i to p_j: n_i must face along -axis, n_j along +axis.
    cos_i = -np.einsum("ijk,ik->ij", axis, normals32)
    cos_j = np.einsum("ijk,jk->ij", axis, normals32)
    opposed = -(normals32 @ normals32.T)
    # Offset of p_j from p_i along n_j (positive: facing away) and across it.
    along = np.einsum("ijk,jk->ij", diff, normals32)
    lateral = np.sqrt(np.maximum(dist ** 2 - along ** 2, 0.0))
    aligned = ((cos_i >= cos_axis) & (cos_j >= cos_axis)) | (
        (along > 0.0) & (lateral <= lateral_tol)
    )
    valid = (
        np.triu(np.ones(dist.shape, dtype=bool), k=1)
        & (dist >= min_width) & (dist <= max_width)
        & aligned & (opposed >= cos_normal)
    )
    pair_i, pair_j = np.nonzero(valid)
    closing = normals[pair_j] - normals[pair_i]
    closing /= np.linalg.norm(closing, axis=-1, keepdims=True)
    return (points[pair_i] + points[pair_j]) / 2.0, closing


def _spread_pairs(
    midpoints: np.ndarray,
    axes: np.ndarray,
    voxel_size: float,
    max_pairs: int,
    np_rng: np.random.Generator,
) -> np.ndarray:
    """Indices of at most ``max_pairs`` pairs spread over the object.

    One pair per (midpoint voxel, ~30° closing-direction bin) bucket, chosen at
    random within the bucket; then a random subset of the buckets.
    """
    if len(midpoints) == 0:
        return np.zeros(0, dtype=np.int64)
    # Sign-canonical closing direction: its largest component positive.
    dominant = np.abs(axes).argmax(axis=-1)
    sign = np.sign(axes[np.arange(len(axes)), dominant])
    direction_bin = np.round(axes * sign[:, None] * 2.0).astype(np.int64)
    voxel = np.floor(midpoints / voxel_size).astype(np.int64)
    buckets = np.concatenate([voxel, direction_bin], axis=-1)
    order = np_rng.permutation(len(midpoints))
    _, first = np.unique(buckets[order], axis=0, return_index=True)
    chosen = order[first]
    if len(chosen) > max_pairs:
        chosen = np_rng.choice(chosen, size=max_pairs, replace=False)
    return np.sort(chosen)


def _local_down(standing_quaternion: Optional[Sequence[float]]) -> np.ndarray:
    """World down ``(0, 0, -1)`` in the object frame of its upright pose."""
    down = np.array([0.0, 0.0, -1.0])
    if standing_quaternion is None:
        return down
    w, x, y, z = (float(c) for c in standing_quaternion)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    # The placement yaw is about world z, so it leaves "down" unchanged.
    return rotation.T @ down


def _approaches(
    closing: np.ndarray,
    down: np.ndarray,
    tilts_deg: Sequence[float],
    max_tilt_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approach directions ``(P, K, 3)`` perpendicular to each closing axis.

    Rotated by ``tilts_deg`` about the closing axis from its most downward
    perpendicular (any perpendicular when the axis is vertical); the mask
    ``(P, K)`` marks the ones within ``max_tilt_deg`` of ``down``.
    """
    u = down[None, :] - (closing @ down)[:, None] * closing
    vertical = np.linalg.norm(u, axis=-1) < 1e-3
    if vertical.any():
        helper = np.where(np.abs(closing[vertical, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        u[vertical] = helper - (closing[vertical] * helper).sum(-1, keepdims=True) * closing[vertical]
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(closing, u)
    tilts = np.radians(np.asarray(tilts_deg, dtype=float))
    approach = (
        np.cos(tilts)[None, :, None] * u[:, None, :]
        + np.sin(tilts)[None, :, None] * v[:, None, :]
    )
    within = approach @ down >= np.cos(np.radians(max_tilt_deg)) - 1e-9
    return approach, within


def _gripper_frame(cloud, origin, approach, opening):
    """Cloud coordinates in each grasp frame: ``(s_a, |s_o|, |s_b|)``, each ``(C, M)``."""
    rel = cloud[None, :, :] - origin[:, None, :]
    lateral = np.cross(approach, opening)
    s_a = np.einsum("cmi,ci->cm", rel, approach)
    s_o = np.einsum("cmi,ci->cm", rel, opening)
    s_b = np.einsum("cmi,ci->cm", rel, lateral)
    return s_a, s_o, s_b


def _clear_or_slide(
    cloud: np.ndarray,
    fingertip: np.ndarray,
    approach: np.ndarray,
    opening: np.ndarray,
    *,
    gripper_width: float,
    gripper_depth: float,
    finger_width: float,
    finger_thickness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Self-collision check of the open gripper, sliding back to clear the palm.

    Returns the (possibly slid) fingertips and a mask of the collision-free
    grasps.  In each grasp frame, ``s_a`` runs along the approach (0 at the
    fingertip, negative toward the palm), ``s_o`` along the closing axis and
    ``s_b`` across the finger width.
    """
    half_open = gripper_width / 2.0
    reach = half_open + finger_thickness

    def blocked(tip):
        s_a, s_o, s_b = _gripper_frame(cloud, tip, approach, opening)
        s_o, s_b = np.abs(s_o), np.abs(s_b)
        behind = (s_b <= finger_width / 2.0) & (s_a <= 0.0)
        in_finger = behind & (s_o > half_open) & (s_o <= reach)
        in_palm = behind & (s_a < -gripper_depth) & (s_o <= reach)
        return in_finger, in_palm, s_a

    in_finger, in_palm, s_a = blocked(fingertip)
    # Slide back just past the deepest palm intrusion (+1 mm).
    intrusion = np.where(in_palm, -gripper_depth - s_a, 0.0).max(axis=-1)
    slide = np.where(intrusion > 0.0, intrusion + 1e-3, 0.0)
    slid = fingertip - slide[:, None] * approach
    in_finger, in_palm, _ = blocked(slid)
    clear = ~(in_finger | in_palm).any(axis=-1)
    return slid, clear


def _closes_on_surface(
    cloud: np.ndarray,
    cloud_normals: np.ndarray,
    contact_mid: np.ndarray,
    approach: np.ndarray,
    opening: np.ndarray,
    radius: float,
    cos_cone: float,
    min_width: float,
    max_width: float,
) -> np.ndarray:
    """Whether the jaws, closing along ``opening`` through ``contact_mid``, pinch surface.

    On each side the outermost cloud point within ``radius`` of the closing
    line (where that jaw touches first) must face outward within the friction
    cone, and the two contacts must be ``min_width`` .. ``max_width`` apart.
    """
    s_a, s_o, s_b = _gripper_frame(cloud, contact_mid, approach, opening)
    near = (np.abs(s_a) <= radius) & (np.abs(s_b) <= radius) & (np.abs(s_o) <= max_width / 2.0 + radius)
    facing = cloud_normals @ opening.T                                 # (M, C)
    ok = np.ones(len(contact_mid), dtype=bool)
    outer = np.zeros((2, len(contact_mid)))
    for side, sign in enumerate((1.0, -1.0)):
        on_side = near & (sign * s_o > 0.0)
        depth = np.where(on_side, sign * s_o, -np.inf)
        first = depth.argmax(axis=-1)
        rows = np.arange(len(contact_mid))
        found = on_side[rows, first]
        outward = sign * facing[first, rows] >= cos_cone
        ok &= found & outward
        outer[side] = depth[rows, first]
    width = outer[0] + outer[1]
    return ok & (width >= min_width) & (width <= max_width)
