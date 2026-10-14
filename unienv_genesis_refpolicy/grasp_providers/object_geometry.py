from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch

from unienv_genesis_collection.objects.base_object import (
    GraspCandidates,
    ManipulatableObject,
    _enumerate_bbox_gripper_candidates,
    _local_to_world_candidates,
)
from unienv_genesis_collection.objects.mesh_object import CONTACT_INSET
from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider

__all__ = ["ObjectGeometryGraspCandidateProvider", "lowest_finger_z"]

#: Robotiq 2F-85 finger geometry (m): pad width across the finger, and finger
#: thickness along the closing axis.
FINGER_WIDTH = 0.022
FINGER_THICKNESS = 0.01


def lowest_finger_z(
    candidates: GraspCandidates,
    *,
    gripper_width: float,
    gripper_depth: float,
    finger_width: float = FINGER_WIDTH,
    finger_thickness: float = FINGER_THICKNESS,
) -> torch.Tensor:
    """World height of the lowest point of each candidate's open fingers.

    The fingertip corners are the grasp point ± ``(gripper_width / 2 +
    finger_thickness)`` along the closing axis ± ``finger_width / 2`` across it;
    the fingers then run ``gripper_depth`` back along the approach (lower than
    the tips only for an upward approach).
    """
    approach = torch.nn.functional.normalize(candidates.approach_direction, dim=-1)
    opening = candidates.opening_direction
    opening = opening - (opening * approach).sum(-1, keepdim=True) * approach
    opening = torch.nn.functional.normalize(opening, dim=-1)
    lateral = torch.cross(approach, opening, dim=-1)
    drop = (
        (float(gripper_width) / 2.0 + float(finger_thickness)) * opening[..., 2].abs()
        + float(finger_width) / 2.0 * lateral[..., 2].abs()
        + torch.clamp(float(gripper_depth) * approach[..., 2], min=0.0)
    )
    return candidates.position[..., 2] - drop


def _any_or_broadcast(mask: torch.Tensor) -> torch.Tensor:
    """``mask.any`` over the candidate dimension, keeping a broadcastable shape."""
    return mask.any(dim=-1, keepdim=True)


class ObjectGeometryGraspCandidateProvider(GraspCandidateProvider):
    """Default provider that delegates to the target object's geometry helpers.

    The loaded entity may be supplied either as a fixed ``target_entity`` or,
    preferably, as a zero-argument ``target_entity_fn`` returning the *current*
    entity: the callable is re-read on every :meth:`propose`, so a scene reload
    (which swaps the entity id) is picked up without rebuilding the provider.

    **Bounding-box grasps** (opt-in, ``bbox_grasps``): for a *box-like* object
    — mesh volume at least ``bbox_min_fill_ratio`` of its bounding box, so the
    bbox faces coincide with real surface — axis-aligned grasps across opposing
    bbox faces narrower than ``gripper_width - width_margin`` are appended to
    the object's own candidates.  Non-box-like objects (a toy plane, a hammer)
    get none, as their bbox-face midpoints sit in empty space.

    **Selection** (opt-in, for a table-top scene):

    * :meth:`filter` — with ``table_z`` set, prefers candidates whose lowest
      open-finger point (:func:`lowest_finger_z`) is at least
      ``min_table_clearance`` above the table; with ``prefer_top_down``, also
      candidates approaching within 60° of straight down.  Falls back, per
      environment, to the top-down set and then to every candidate, so a
      feasible-or-not grasp is still attempted.
    * :meth:`cost` — with ``quality_cost``, ranks reachable candidates by
      approach tilt (``1 + approach_z``), contacts less than
      ``min_contact_depth`` below the object's top (normalised to ``[0, 1]``),
      and horizontal distance of the contacts from the object's AABB centre (in
      units of ``centering_scale``).  Tilted approaches and contacts right
      under the top edge lift the object as a lever about the fingertips.

    Args:
        target_object: The :class:`ManipulatableObject` descriptor.
        target_entity: The loaded Genesis entity (fixed binding).  May be
            ``None`` when ``target_entity_fn`` is given.
        target_entity_fn: Zero-argument callable returning the current
            (single, batched) entity.  Takes precedence over ``target_entity``
            when provided.
        gripper_width: Maximum gripper opening in metres.
        gripper_depth: Gripper finger depth in metres.
        table_z: Table surface height (m); enables the clearance filter.
        min_table_clearance: Preferred minimum height (m) of the lowest open
            finger point above the table.
        prefer_top_down: Prefer approaches within 60° of straight down.
        quality_cost: Enable :meth:`cost`.
        min_contact_depth: Contact depth (m) below the object's top under which
            :meth:`cost` penalises a grasp.
        centering_scale: Distance (m) from the object's centre that costs as
            much as a horizontal approach.
        bbox_grasps: Append bounding-box grasps for box-like objects.
        bbox_min_fill_ratio: Minimum mesh-volume / bbox-volume ratio for the
            object to count as box-like.
        width_margin: Slack (m) below ``gripper_width`` for the bbox grasps, so
            the open jaws clear both faces despite pose error.
    """

    def __init__(
        self,
        *,
        target_object: ManipulatableObject,
        target_entity: Any = None,
        target_entity_fn: Optional[Callable[[], Any]] = None,
        gripper_width: float,
        gripper_depth: float,
        table_z: Optional[float] = None,
        min_table_clearance: float = 0.0,
        prefer_top_down: bool = False,
        quality_cost: bool = False,
        min_contact_depth: float = 0.02,
        centering_scale: float = 0.05,
        bbox_grasps: bool = False,
        bbox_min_fill_ratio: float = 0.6,
        width_margin: float = 0.01,
    ) -> None:
        self.target_object = target_object
        self.target_entity = target_entity
        self.target_entity_fn = target_entity_fn
        self.gripper_width = gripper_width
        self.gripper_depth = gripper_depth
        self.table_z = table_z
        self.min_table_clearance = float(min_table_clearance)
        self.prefer_top_down = bool(prefer_top_down)
        self.quality_cost = bool(quality_cost)
        self.min_contact_depth = float(min_contact_depth)
        self.centering_scale = float(centering_scale)
        self.bbox_grasps = bool(bbox_grasps)
        self.bbox_min_fill_ratio = float(bbox_min_fill_ratio)
        self.width_margin = float(width_margin)
        self._fill_ratio: Optional[float] = None

    def _entity(self) -> Any:
        # Re-read every call: a scene reload swaps the entity id, and resolving
        # it here re-targets the provider without a rebuild.
        return (
            self.target_entity_fn()
            if self.target_entity_fn is not None
            else self.target_entity
        )

    def propose(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> GraspCandidates:
        del mask
        rng = None
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
        entity = self._entity()
        candidates = self.target_object.list_box_gripper_grasping_candidates(
            entity,
            self.gripper_width,
            gripper_depth=self.gripper_depth,
            rng=rng,
        )
        if not self.bbox_grasps or entity is None or not self.is_box_like():
            return candidates
        bbox = self._bbox_candidates(entity)
        if bbox is None:
            return candidates
        return GraspCandidates(
            position=torch.cat([candidates.position, bbox.position], dim=-2),
            approach_direction=torch.cat(
                [candidates.approach_direction, bbox.approach_direction], dim=-2
            ),
            opening_direction=torch.cat(
                [candidates.opening_direction, bbox.opening_direction], dim=-2
            ),
        )

    def is_box_like(self) -> bool:
        """Whether the object's bounding box is a faithful stand-in for its surface."""
        if self._fill_ratio is None:
            ratio = 0.0
            load = getattr(self.target_object, "_load_trimesh", None)
            if callable(load):
                try:
                    mesh = load()
                    bbox_volume = float(np.prod(np.asarray(mesh.extents, dtype=np.float64)))
                    if bbox_volume > 0.0:
                        ratio = abs(float(mesh.volume)) / bbox_volume
                except Exception:
                    ratio = 0.0
            self._fill_ratio = ratio
        return self._fill_ratio >= self.bbox_min_fill_ratio

    def _bbox_candidates(self, entity: Any) -> Optional[GraspCandidates]:
        """World-frame grasps across opposing faces of the object's local bbox."""
        bbox = self.target_object.get_local_bounding_box()
        local = _enumerate_bbox_gripper_candidates(
            bbox,
            gripper_width=self.gripper_width - self.width_margin,
            gripper_depth=self.gripper_depth,
        )
        if local.position.numel() == 0:
            return None
        # The enumerator puts a downward grasp a finger-depth above the bbox's
        # BOTTOM face, which on a tall box leaves the pre-grasp point inside the
        # object.  Move it to a finger-depth below the TOP face (but still above
        # the bottom) so the fingers close near the top with a clear approach.
        position = local.position.clone()
        z_low = bbox.low[..., 2].to(position.device)
        z_high = bbox.high[..., 2].to(position.device)
        downward = local.approach_direction[..., 2] < -0.5
        top_z = torch.maximum(z_high - self.gripper_depth, z_low + self.gripper_depth)
        position[..., 2] = torch.where(downward, top_z.expand_as(position[..., 2]), position[..., 2])
        return _local_to_world_candidates(
            GraspCandidates(
                position=position,
                approach_direction=local.approach_direction,
                opening_direction=local.opening_direction,
            ),
            entity.get_pos().detach(),
            entity.get_quat().detach(),
        )

    def filter(self, candidates: GraspCandidates) -> Optional[torch.Tensor]:
        if self.table_z is None and not self.prefer_top_down:
            return None
        approach = candidates.approach_direction
        down = torch.ones_like(approach[..., 2], dtype=torch.bool)
        if self.prefer_top_down:
            down = approach[..., 2] <= -0.5
            down = down | ~_any_or_broadcast(down)
        if self.table_z is None:
            return down
        clear = lowest_finger_z(
            candidates,
            gripper_width=self.gripper_width,
            gripper_depth=self.gripper_depth,
        ) >= float(self.table_z) + self.min_table_clearance
        preferred = clear & down
        return torch.where(_any_or_broadcast(preferred), preferred, down)

    def cost(self, candidates: GraspCandidates) -> Optional[torch.Tensor]:
        if not self.quality_cost:
            return None
        aabb = self._entity().get_AABB().detach()  # ([B], 2, 3)
        top_z = aabb[..., 1, 2].unsqueeze(-1)
        center_xy = aabb.mean(dim=-2)[..., :2].unsqueeze(-2)
        approach = candidates.approach_direction
        contact = candidates.position - CONTACT_INSET * approach
        tilt = 1.0 + approach[..., 2]
        shallow = torch.clamp(
            (self.min_contact_depth - (top_z - contact[..., 2])) / self.min_contact_depth,
            min=0.0, max=1.0,
        )
        off_center = torch.linalg.vector_norm(
            contact[..., :2] - center_xy, dim=-1
        ) / self.centering_scale
        return (tilt + shallow + off_center).reshape(candidates.position.shape[:-1])
