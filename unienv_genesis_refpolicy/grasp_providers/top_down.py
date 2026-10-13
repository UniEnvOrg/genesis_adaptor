"""Top-down grasp candidates built from the target object's own grasp points.

The provider here re-frames the candidates found by the object's own
(mesh- or bbox-based) enumeration into the grasps a parallel jaw can actually
execute on a table.  See :class:`TopDownGraspCandidateProvider` for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import torch

from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider

if TYPE_CHECKING:  # pragma: no cover - typing only (kept out of the import path)
    from unienv_genesis_collection.objects.base_object import GraspCandidates

__all__ = ["TopDownGraspCandidateProvider"]


class TopDownGraspCandidateProvider(GraspCandidateProvider):
    """**Top-down** grasp candidates built from the object's own grasp points.

    The catalog (GSO / YCB) descriptors sample their candidates on the mesh, and
    that sampler always derives the approach direction perpendicular to the
    object's local z axis — i.e. every candidate is a **side** grasp at the
    object's mid-height.  For an object lying on a table a side grasp means the
    gripper has to slide in a couple of centimetres above the surface, where the
    parallel jaw cannot close without its lower finger colliding with the table
    (measured: the arm stalls 5-10 cm above such grasp poses, while the top-down
    poses below are reached to within millimetres).

    This provider keeps the *grasp points* found by the object's own
    enumeration — those sit on actual material — and re-frames each one into the
    grasp a parallel jaw **can** execute on a table:

    * approach straight down (``-z``),
    * closing along the object's narrowest horizontal bounding-box axis (whose
      extent is provably below the gripper opening, so the fingers meet material
      instead of each other),
    * keeping the candidate's own height (a mesh-sampled grasp point sits on
      material; lifting it into empty space would make the fingers miss),

    with both opening signs.  Duplicate grasp points are collapsed in XY and the
    candidate count is capped (``max_candidates``) to keep the per-candidate IK
    evaluation cheap.

    Args:
        target_object: The :class:`ManipulatableObject` descriptor.
        target_entity: The loaded Genesis entity (fixed binding).  May be
            ``None`` when ``target_entity_fn`` is given.
        target_entity_fn: Zero-argument callable returning the current
            (single, batched) entity.  Preferred: it is re-read on every
            :meth:`propose`, so a scene reload re-targets the provider without
            rebuilding it.  Takes precedence over ``target_entity``.
        gripper_width: Maximum gripper opening in metres.
        gripper_depth: Gripper finger depth in metres.
        base_provider: Optional provider supplying the underlying grasp points.
            When ``None``, an
            :class:`~unienv_genesis_refpolicy.grasp_providers.object_geometry.ObjectGeometryGraspCandidateProvider`
            is built, sharing this provider's entity source.
        max_candidates: Upper bound on the (pre-re-framing) grasp points.
        height_range: Reserved for API compatibility; the current implementation
            keeps each candidate's own material-sampled height instead of
            clamping it into this range.
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
        max_candidates: int = 32,
        height_range: Tuple[float, float] = (0.5, 0.85),
    ) -> None:
        self.target_object = target_object
        self.target_entity = target_entity
        self.target_entity_fn = target_entity_fn
        self.gripper_width = float(gripper_width)
        self.gripper_depth = float(gripper_depth)
        self.max_candidates = int(max_candidates)
        self.height_range = (float(height_range[0]), float(height_range[1]))
        if base_provider is None:
            # Function-local: keeps importing this module light.
            from unienv_genesis_refpolicy.grasp_providers.object_geometry import (
                ObjectGeometryGraspCandidateProvider,
            )

            base_provider = ObjectGeometryGraspCandidateProvider(
                target_object=target_object,
                target_entity=target_entity,
                # Share the entity source so the base provider re-resolves the
                # entity after a scene reload too.
                target_entity_fn=target_entity_fn,
                gripper_width=self.gripper_width,
                gripper_depth=self.gripper_depth,
            )
        self.base_provider = base_provider

    def propose(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> "GraspCandidates":
        """Return the re-framed top-down candidates, ``(B, N, ...)``."""
        from unienv_genesis_collection.objects.base_object import (
            GraspCandidates,
            bbox_center,
            bbox_size,
        )

        base = self.base_provider.propose(mask=mask, seed=seed)
        positions = base.position
        if positions.ndim == 2:  # unbatched: (N, 3) -> (1, N, 3)
            positions = positions.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        n_candidates = int(positions.shape[-2])
        if n_candidates == 0:
            return GraspCandidates(
                position=positions,
                approach_direction=positions.clone(),
                opening_direction=positions.clone(),
            )
        if n_candidates > self.max_candidates:
            index = torch.linspace(
                0, n_candidates - 1, self.max_candidates, device=positions.device
            ).round().to(dtype=torch.int64)
            positions = positions.index_select(-2, index)
            n_candidates = self.max_candidates

        # Re-read the entity: a scene reload swaps the entity id.
        entity = (
            self.target_entity_fn()
            if self.target_entity_fn is not None
            else self.target_entity
        )
        bbox = self.target_object.get_bounding_box(entity)
        centre = bbox_center(bbox)
        size = bbox_size(bbox)

        # Narrowest horizontal axis per environment (closing axis).
        narrow_x = (size[..., 0] <= size[..., 1]).to(dtype=positions.dtype)
        # Keep the candidate's own height: a mesh-sampled grasp point sits on
        # actual material, and lifting it (clamping z upwards) would move the
        # fingers into empty space next to low parts of the object.  Height is
        # only used for *ranking* below; the candidate filter prefers grasps that
        # clear the table.
        z = positions[..., 2]

        approach = torch.stack(
            [torch.zeros_like(z), torch.zeros_like(z), -torch.ones_like(z)], dim=-1
        )
        openings = []
        for sign in (1.0, -1.0):
            opening = torch.stack(
                [
                    sign * narrow_x.unsqueeze(-1) * torch.ones_like(z),
                    sign * (1.0 - narrow_x).unsqueeze(-1) * torch.ones_like(z),
                    torch.zeros_like(z),
                ],
                dim=-1,
            )
            openings.append(opening)

        # Rank the grasp points: prefer grasps that sit high on the object (so
        # the fingers clear the table) and near its bounding-box centre (least
        # torque on the two-pad grip).  Deduplicate in XY so a long object is not
        # represented by dozens of near-identical grasps.
        centre_distance = torch.linalg.vector_norm(
            positions[..., :2] - centre[..., :2].unsqueeze(-2), dim=-1
        )
        score = z - 0.1 * centre_distance
        order = torch.argsort(score, dim=-1, descending=True)
        order = _dedupe_xy_order(positions[..., :2], order, min_spacing=0.02)
        positions = positions.gather(-2, order.unsqueeze(-1).expand_as(positions))
        positions = torch.cat(
            [
                positions[..., 0:1],
                positions[..., 1:2],
                z.gather(-1, order.squeeze(-1)).unsqueeze(-1),
            ],
            dim=-1,
        )
        approach = approach.gather(-2, order.unsqueeze(-1).expand_as(approach))
        openings = [o.gather(-2, order.unsqueeze(-1).expand_as(o)) for o in openings]

        result = GraspCandidates(
            position=torch.cat([positions, positions], dim=-2),
            approach_direction=torch.cat([approach, approach], dim=-2),
            opening_direction=torch.cat(openings, dim=-2),
        )
        if squeeze:
            return GraspCandidates(
                position=result.position[0],
                approach_direction=result.approach_direction[0],
                opening_direction=result.opening_direction[0],
            )
        return result


def _dedupe_xy_order(
    positions_xy: torch.Tensor,
    order: torch.Tensor,
    *,
    min_spacing: float,
) -> torch.Tensor:
    """Drop candidates whose XY lies within *min_spacing* of a kept one.

    ``positions_xy`` is ``(B, N, 2)`` (batched worlds only) and *order* is the
    ``(B, N)`` ranking to filter.  Dropped entries are pushed to the end of the
    order, so the caller can simply keep the first *k* columns.
    """
    if positions_xy.dim() != 3:  # pragma: no cover - batched worlds only
        return order
    import numpy as np

    kept_rows: List[np.ndarray] = []
    for env_index in range(positions_xy.shape[0]):
        xy = positions_xy[env_index].detach().cpu().numpy()
        ranked = order[env_index].detach().cpu().numpy()
        kept: List[int] = []
        for candidate in ranked.tolist():
            point = xy[candidate]
            if all(
                float(np.linalg.norm(point - xy[other])) >= min_spacing for other in kept
            ):
                kept.append(candidate)
        remaining = [index for index in ranked.tolist() if index not in set(kept)]
        kept_rows.append(np.asarray(kept + remaining, dtype=np.int64))
    return torch.as_tensor(
        np.stack(kept_rows), dtype=order.dtype, device=order.device
    )
