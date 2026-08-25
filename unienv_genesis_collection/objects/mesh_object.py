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
        n_samples: int = 100,
        rng: Optional[torch.Generator] = None,
    ) -> GraspCandidates:
        """Generate grasp candidates via mesh surface sampling (local frame).

        Samples surface points, finds antipodal pairs whose distance is
        within ``[gripper_depth, gripper_width]``, and returns midpoint
        grasps in the **object's local frame**.  Results are cached.

        Falls back to the bounding-box-based method from the base class if
        no antipodal pairs are found.

        Args:
            rng: Optional :class:`torch.Generator` for reproducible surface
                sampling.  When ``None``, trimesh's default sampling RNG
                behavior is used, matching the previous behavior.
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

        key = (gripper_width, gripper_depth, n_samples, seed)
        if key in self._local_grasp_cache:
            return self._local_grasp_cache[key].clone()

        mesh = self._load_trimesh()

        # Sample surface points and their normals (numpy).  ``Trimesh.sample``
        # does not forward a ``seed`` to ``sample_surface``, so call
        # ``trimesh.sample.sample_surface`` directly to obtain deterministic
        # results when a seed is supplied.
        import trimesh

        points, face_indices = trimesh.sample.sample_surface(
            mesh, n_samples, seed=seed,
        )
        normals = mesh.face_normals[face_indices]

        local_positions: List[np.ndarray] = []
        local_approaches: List[np.ndarray] = []
        local_openings: List[np.ndarray] = []

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = float(np.linalg.norm(points[i] - points[j]))
                if dist > gripper_width:
                    continue
                # Check roughly antipodal normals
                dot = float(np.dot(normals[i], normals[j]))
                if dot > -0.5:
                    continue

                midpoint = (points[i] + points[j]) / 2.0
                opening_dir = points[j] - points[i]
                opening_dir = opening_dir / np.linalg.norm(opening_dir)

                # Approach direction perpendicular to opening
                approach_dir = np.cross(opening_dir, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(approach_dir) < 1e-6:
                    approach_dir = np.cross(opening_dir, np.array([0.0, 1.0, 0.0]))
                approach_dir = approach_dir / np.linalg.norm(approach_dir)

                # Both opening directions (flipped = 180° rotation around approach)
                for sign_o in (+1, -1):
                    local_positions.append(midpoint)
                    local_approaches.append(approach_dir)
                    local_openings.append(sign_o * opening_dir)

        if not local_positions:
            # Fall back to bounding-box-based generation
            result = super().list_box_gripper_grasping_candidates_local(
                entity, gripper_width, gripper_depth=gripper_depth,
            )
            self._local_grasp_cache[key] = result
            return result.clone()

        result = GraspCandidates(
            position=torch.from_numpy(np.stack(local_positions)),
            approach_direction=torch.from_numpy(np.stack(local_approaches)),
            opening_direction=torch.from_numpy(np.stack(local_openings)),
        )
        self._local_grasp_cache[key] = result
        return result.clone()

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

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
