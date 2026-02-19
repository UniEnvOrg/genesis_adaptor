"""Mesh-based ManipulatableObject.

Uses **trimesh** (lazy import) for offline bounding-box computation and
grasp-candidate analysis.  The mesh file is loaded on first call to
:meth:`get_local_bounding_box` or :meth:`list_box_gripper_grasping_candidates`
and cached for subsequent calls.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, List, Union, TYPE_CHECKING

import numpy as np
import torch

from unienv_genesis_collection.objects.base_object import (
    ManipulatableObject,
    GraspCandidates,
    make_bbox,
    _local_to_world_candidates,
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
        **kwargs: Forwarded to :class:`ManipulatableObject`.
    """

    def __init__(
        self,
        mesh_file: Union[str, os.PathLike],
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        decimate: bool = True,
        color: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mesh_file = str(mesh_file)
        self.scale = scale if isinstance(scale, tuple) else (scale, scale, scale)
        self.decimate = decimate
        self.color = color
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

    def get_local_bounding_box(self, entity) -> "BoxSpace":
        mesh = self._load_trimesh()
        return make_bbox(
            torch.from_numpy(mesh.bounds[0].astype(np.float32)),
            torch.from_numpy(mesh.bounds[1].astype(np.float32)),
        )

    # ------------------------------------------------------------------
    # Grasp analysis (mesh-based antipodal sampling)
    # ------------------------------------------------------------------

    def list_box_gripper_grasping_candidates(
        self,
        entity,
        gripper_width: float,
        gripper_depth: float = 0.01,
        n_samples: int = 100,
    ) -> GraspCandidates:
        """Generate grasp candidates via mesh surface sampling.

        Samples surface points, finds antipodal pairs whose distance is
        within ``[gripper_depth, gripper_width]``, and returns midpoint
        grasps in **world coordinates**.

        Falls back to the bounding-box-based method from the base class if
        no antipodal pairs are found.
        """
        mesh = self._load_trimesh()

        # Sample surface points and their normals (numpy)
        points, face_indices = mesh.sample(n_samples, return_index=True)
        normals = mesh.face_normals[face_indices]

        local_positions: List[np.ndarray] = []
        local_approaches: List[np.ndarray] = []
        local_openings: List[np.ndarray] = []

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = float(np.linalg.norm(points[i] - points[j]))
                if dist > gripper_width or dist < gripper_depth:
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

                local_positions.append(midpoint)
                local_approaches.append(approach_dir)
                local_openings.append(opening_dir)

        if not local_positions:
            # Fall back to bounding-box-based generation
            return super().list_box_gripper_grasping_candidates(
                entity, gripper_width, gripper_depth=gripper_depth,
            )

        local = GraspCandidates(
            position=torch.from_numpy(np.stack(local_positions).astype(np.float32)),
            approach_direction=torch.from_numpy(np.stack(local_approaches).astype(np.float32)),
            opening_direction=torch.from_numpy(np.stack(local_openings).astype(np.float32)),
        )

        # Transform to world coordinates (batched-env aware)
        pos = entity.get_pos().detach().float()
        quat = entity.get_quat().detach().float()
        return _local_to_world_candidates(local, pos, quat)

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
        surface = gs.surfaces.Plastic(color=self.color) if self.color else None
        return world.scene.add_entity(
            morph,
            surface=surface,
            name=name_override or self.name,
        )
