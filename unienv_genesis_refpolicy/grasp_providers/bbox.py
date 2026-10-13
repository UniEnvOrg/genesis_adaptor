"""Axis-aligned bounding-box grasp candidates for box-like objects.

The mesh-antipodal sampler (the stock :class:`ObjectGeometryGraspCandidateProvider`)
can miss the *clean top-down* grasp on a boxy object: it emits candidates at the
sampled contact-patch orientation, which for some geometries never produces a
steep-enough (``-z``) approach, so a top-down filter falls back to a marginal
side grasp on a thin edge.  For a genuinely box-like object the axis-aligned
bounding box is a faithful stand-in for the surface, so a grasp across an
opposing pair of bbox faces closes on real material.

:class:`BboxGraspCandidateProvider` adds those bbox-face grasps as a supplement
to a base provider, **gated on box-likeness** so it does not regress the
non-convex case (where bbox-face midpoints sit in empty space — e.g. a toy
plane's wings or a hammer's head/handle — and the gripper would close on air).

The bbox grasps are enumerated with the object's own
:func:`~unienv_genesis_collection.objects.base_object._enumerate_bbox_gripper_candidates`
(correct fingertip depth convention) and transformed to world coordinates with
:func:`~unienv_genesis_collection.objects.base_object._local_to_world_candidates`,
matching the frame and batching of the base provider's candidates.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch

from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider

__all__ = ["BboxGraspCandidateProvider"]


class BboxGraspCandidateProvider(GraspCandidateProvider):
    """Axis-aligned bbox-face grasps, supplementing a base provider.

    Candidates are generated only when the object is **box-like**, measured by
    the mesh-volume / bbox-volume fill ratio (``1.0`` for a perfect box; a toy
    plane / hammer scores far lower and is rejected).  For a box-like object the
    bbox faces coincide with real surface, so closing across an opposing face
    pair grasps material.

    The bbox candidates are concatenated after the base provider's, so the
    antipodal proposals remain the primary source.

    Args:
        target_object: The :class:`ManipulatableObject` descriptor.
        target_entity: The loaded Genesis entity (fixed binding).  May be
            ``None`` when ``target_entity_fn`` is given.
        target_entity_fn: Zero-argument callable returning the current entity;
            re-read on every :meth:`propose` so a scene reload re-targets the
            provider without a rebuild.  Takes precedence over ``target_entity``.
        gripper_width: Maximum gripper opening in metres.
        gripper_depth: Gripper finger depth in metres.
        base_provider: Provider supplying the primary (antipodal) candidates.
            When ``None``, an
            :class:`~unienv_genesis_refpolicy.grasp_providers.object_geometry.ObjectGeometryGraspCandidateProvider`
            sharing this provider's entity source is built.
        min_fill_ratio: Minimum mesh-volume / bbox-volume ratio for the object
            to count as box-like (and thus receive bbox-face grasps).
        width_margin: Slack (m) below ``gripper_width``: bbox-face grasps are
            only enumerated across box axes narrower than
            ``gripper_width - width_margin``, so the open jaws clear both faces
            despite pose error.
    """

    def __init__(
        self,
        *,
        target_object: Any,
        target_entity: Any = None,
        target_entity_fn: Optional[Callable[[], Any]] = None,
        gripper_width: float,
        gripper_depth: float,
        base_provider: Optional[GraspCandidateProvider] = None,
        min_fill_ratio: float = 0.6,
        width_margin: float = 0.01,
    ) -> None:
        self.target_object = target_object
        self.target_entity = target_entity
        self.target_entity_fn = target_entity_fn
        self.gripper_width = float(gripper_width)
        self.gripper_depth = float(gripper_depth)
        self.min_fill_ratio = float(min_fill_ratio)
        self.width_margin = float(width_margin)
        self._fill_ratio: Optional[float] = None
        if base_provider is None:
            from unienv_genesis_refpolicy.grasp_providers.object_geometry import (
                ObjectGeometryGraspCandidateProvider,
            )

            base_provider = ObjectGeometryGraspCandidateProvider(
                target_object=target_object,
                target_entity=target_entity,
                target_entity_fn=target_entity_fn,
                gripper_width=self.gripper_width,
                gripper_depth=self.gripper_depth,
            )
        self.base_provider = base_provider

    # ------------------------------------------------------------------
    def _entity(self) -> Any:
        return (
            self.target_entity_fn()
            if self.target_entity_fn is not None
            else self.target_entity
        )

    def _box_fill_ratio(self) -> float:
        """Mesh-volume / bbox-volume fill ratio (cached; geometry is static).

        ``mesh.extents`` / ``mesh.volume`` come from trimesh as numpy, so the
        product is taken in numpy (not ``torch.prod``) to stay dtype-agnostic.
        """
        if self._fill_ratio is None:
            ratio = 0.0
            load = getattr(self.target_object, "_load_trimesh", None)
            if callable(load):
                try:
                    mesh = load()
                    extents = np.asarray(mesh.extents, dtype=np.float64).reshape(-1)
                    bbox_volume = float(np.prod(extents))
                    if bbox_volume > 0.0:
                        ratio = float(abs(float(mesh.volume))) / bbox_volume
                except Exception:
                    ratio = 0.0
            self._fill_ratio = ratio
        return self._fill_ratio

    def is_box_like(self) -> bool:
        """Whether the bbox is a faithful stand-in for the object's surface."""
        return self._box_fill_ratio() >= self.min_fill_ratio

    # ------------------------------------------------------------------
    def propose(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> "GraspCandidates":
        del mask
        from unienv_genesis_collection.objects.base_object import (
            GraspCandidates,
            _enumerate_bbox_gripper_candidates,
            _local_to_world_candidates,
        )

        base = self.base_provider.propose(seed=seed)
        if not self.is_box_like():
            return base

        entity = self._entity()
        if entity is None:
            return base

        # Enumerate bbox-face grasps in the object's local frame, then transform
        # to world (matching the base candidates' frame and batch dims).
        bbox_local = _enumerate_bbox_gripper_candidates(
            self.target_object.get_local_bounding_box(),
            gripper_width=self.gripper_width - self.width_margin,
            gripper_depth=self.gripper_depth,
        )
        if bbox_local.position.numel() == 0:
            return base

        # The enumerator's depth convention places every grasp a finger-depth in
        # from the face being approached.  That is correct for the horizontal
        # (side) approaches, but for the *downward* approach it puts the grasp
        # point a finger-depth above the object's BOTTOM face — so on a tall box
        # the pre-grasp point (grasp + approach_distance up) sits inside the
        # object and the arm stalls.  Move the downward-approach grasps up to a
        # finger-depth below the TOP face (clamped to stay above the bottom) so
        # the fingers close on material near the top and the pre-grasp is clear.
        bbox = self.target_object.get_local_bounding_box()
        z_low = bbox.low[..., 2].to(bbox_local.position.device)
        z_high = bbox.high[..., 2].to(bbox_local.position.device)
        downward = bbox_local.approach_direction[..., 2] < -0.5
        if bool(downward.any()):
            target_z = torch.maximum(
                z_high - self.gripper_depth, z_low + self.gripper_depth
            )
            position = bbox_local.position.clone()
            position[..., 2] = torch.where(
                downward, target_z.expand_as(position[..., 2]), position[..., 2]
            )
            bbox_local = GraspCandidates(
                position=position,
                approach_direction=bbox_local.approach_direction,
                opening_direction=bbox_local.opening_direction,
            )

        bbox_world = _local_to_world_candidates(
            bbox_local,
            entity.get_pos().detach(),
            entity.get_quat().detach(),
        )

        return GraspCandidates(
            position=torch.cat([base.position, bbox_world.position], dim=-2),
            approach_direction=torch.cat(
                [base.approach_direction, bbox_world.approach_direction], dim=-2
            ),
            opening_direction=torch.cat(
                [base.opening_direction, bbox_world.opening_direction], dim=-2
            ),
        )
