"""Point proposers for keypoint initialisation.

A :class:`PointProposer` selects keypoints from batched camera observations
at reset/reload time.  The selected keypoints are then mapped to rigid
entities and stored as local offsets by the :class:`KeypointTrackerNode`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple, Union

import torch

__all__ = [
    "FocusedSegmentationPointProposer",
    "PointProposer",
    "UniformSegmentationPointProposer",
    "farthest_point_sample",
    "interior_mask",
]


class PointProposer(ABC):
    """Base class for selecting keypoints from camera observations."""

    @abstractmethod
    def propose(
        self,
        *,
        camera_obs: Dict[str, torch.Tensor],
        seg_to_entity_idx: Mapping[int, int],
        entity_names: Mapping[int, str],
        max_points: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select keypoints from batched camera observations.

        The caller is responsible for pre-filtering ``camera_obs`` to only
        contain the environments that need proposals.  That is, if only
        ``B'`` out of ``B`` total environments need proposals, the caller
        should slice all tensors in ``camera_obs`` down to ``B'`` before
        calling this method, so the batch dimension of every value in
        ``camera_obs`` is ``B'``.

        Args:
            camera_obs: Pre-filtered camera observation dictionary with
                batch size ``B'`` (only active environments).  Guaranteed to
                contain at least ``"seg"`` (``(B', H, W)`` int segmentation
                map) and ``"position"`` (``(B', H, W, 3)`` float32 world
                positions).  May also contain ``"rgb"``, ``"depth"``,
                ``"normal"``, ``"intrinsics_cv"``, ``"extrinsics_cv"``.
            seg_to_entity_idx: Maps segmentation IDs to entity indices.
            entity_names: Mapping from entity index to entity name string.
                Proposers can use this to filter which entities to track.
            max_points: Maximum number of keypoints to return per environment.
            generator: Optional torch RNG for reproducible randomization.

        Returns:
            ``(points_world, seg_ids, valid_mask)`` where ``points_world``
            is a ``(B', max_points, 3)`` float32 tensor, ``seg_ids`` is a
            ``(B', max_points)`` int64 tensor, and ``valid_mask`` is a
            ``(B', max_points)`` bool tensor indicating which slots hold
            proposed points.
        """


class UniformSegmentationPointProposer(PointProposer):
    """Simple point proposer using valid segmented pixels.

    Args:
        max_points_per_entity: Optional cap for points selected from each
            segmentation id.
        randomize: If ``True``, randomly sub-sample candidate pixels.
            If ``False``, keep deterministic raster-scan order.
        tracked_entity_names: If provided, only track entities whose names
            are in this set.
        tracked_entity_name_prefixes: If provided, also track entities
            whose names start with any of these prefixes.
    """

    def __init__(
        self,
        *,
        max_points_per_entity: Optional[int] = None,
        randomize: bool = False,
        tracked_entity_names: Optional[Sequence[str]] = None,
        tracked_entity_name_prefixes: Optional[Sequence[str]] = None,
    ):
        self.max_points_per_entity = max_points_per_entity
        self.randomize = randomize
        self.tracked_entity_names: Optional[Set[str]] = (
            set(tracked_entity_names) if tracked_entity_names is not None else None
        )
        self.tracked_entity_name_prefixes: Optional[Tuple[str, ...]] = (
            tuple(tracked_entity_name_prefixes)
            if tracked_entity_name_prefixes is not None
            else None
        )

    def _name_allowed(self, entity_name: str) -> bool:
        if self.tracked_entity_names is None and self.tracked_entity_name_prefixes is None:
            return True
        exact_ok = (
            self.tracked_entity_names is not None
            and entity_name in self.tracked_entity_names
        )
        prefix_ok = (
            self.tracked_entity_name_prefixes is not None
            and any(entity_name.startswith(p) for p in self.tracked_entity_name_prefixes)
        )
        return exact_ok or prefix_ok

    def _choose(
        self,
        pixels: torch.Tensor,
        n: int,
        *,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if pixels.shape[0] <= n:
            return pixels
        if self.randomize:
            perm = torch.randperm(pixels.shape[0], generator=generator, device=pixels.device)
            return pixels[perm[:n]]
        return pixels[:n]

    def _propose_single(
        self,
        segmentation: torch.Tensor,
        position: torch.Tensor,
        filtered_seg_to_entity_idx: Mapping[int, int],
        max_points: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propose keypoints for a single environment.

        Returns ``(points_world, seg_ids)`` with variable length ``N``.
        """
        valid_mask = torch.isfinite(position).all(dim=-1)  # (H, W)
        device = segmentation.device

        _empty = (
            torch.empty((0, 3), dtype=position.dtype, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
        )

        if max_points <= 0 or len(filtered_seg_to_entity_idx) == 0:
            return _empty

        allowed_seg_ids = torch.tensor(
            list(filtered_seg_to_entity_idx.keys()),
            dtype=segmentation.dtype,
            device=device,
        )
        candidate_mask = valid_mask & torch.isin(segmentation, allowed_seg_ids)
        candidate_pixels = torch.nonzero(candidate_mask, as_tuple=False)  # (K, 2)
        if candidate_pixels.numel() == 0:
            return _empty

        if self.max_points_per_entity is None:
            chosen_pixels = self._choose(candidate_pixels, max_points, generator=generator)
        else:
            seg_values = segmentation[candidate_pixels[:, 0], candidate_pixels[:, 1]]
            entity_values = torch.full_like(seg_values, -1, dtype=torch.int64)
            for seg_id, entity_idx in filtered_seg_to_entity_idx.items():
                entity_values[seg_values == int(seg_id)] = int(entity_idx)

            valid_entity = entity_values >= 0
            if not torch.any(valid_entity):
                return _empty
            candidate_pixels = candidate_pixels[valid_entity]
            entity_values = entity_values[valid_entity]

            selected: list[torch.Tensor] = []
            remaining = max_points
            for entity_idx in torch.unique(entity_values):
                if remaining <= 0:
                    break
                entity_pixels = candidate_pixels[entity_values == entity_idx]
                take = min(self.max_points_per_entity, entity_pixels.shape[0], remaining)
                if take <= 0:
                    continue
                selected.append(self._choose(entity_pixels, take, generator=generator))
                remaining -= take

            if len(selected) == 0:
                return _empty
            chosen_pixels = torch.cat(selected, dim=0)

        # Look up 3D positions and seg IDs at selected pixels.
        y = chosen_pixels[:, 0].long()
        x = chosen_pixels[:, 1].long()
        points_world = position[y, x].to(position.dtype)  # (N, 3)
        seg_ids = segmentation[y, x].to(torch.int64)      # (N,)
        return points_world, seg_ids

    def propose(
        self,
        *,
        camera_obs: Dict[str, torch.Tensor],
        seg_to_entity_idx: Mapping[int, int],
        entity_names: Mapping[int, str],
        max_points: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        segmentation = camera_obs["seg"]       # (B, H, W)
        position = camera_obs["position"]      # (B, H, W, 3)
        B = segmentation.shape[0]
        device = segmentation.device

        filtered_seg_to_entity_idx = {
            seg_id: entity_idx
            for seg_id, entity_idx in seg_to_entity_idx.items()
            if self._name_allowed(entity_names.get(entity_idx, ""))
        }

        out_points = torch.zeros((B, max_points, 3), dtype=position.dtype, device=device)
        out_seg = torch.zeros((B, max_points), dtype=torch.int64, device=device)
        out_valid = torch.zeros((B, max_points), dtype=torch.bool, device=device)

        for b in range(B):
            pts, sids = self._propose_single(
                segmentation[b], position[b],
                filtered_seg_to_entity_idx, max_points, generator,
            )
            n = pts.shape[0]
            if n > 0:
                out_points[b, :n] = pts
                out_seg[b, :n] = sids
                out_valid[b, :n] = True

        return out_points, out_seg, out_valid


def farthest_point_sample(
    points: torch.Tensor,
    n: int,
    *,
    generator: Optional[torch.Generator] = None,
    max_candidates: Optional[int] = 4096,
) -> torch.Tensor:
    """Indices of ``n`` of ``points`` ``(K, 3)`` spread by farthest-point sampling.

    Starts from a random point (``generator``), then repeatedly takes the
    point farthest from those taken. With more than ``max_candidates`` points,
    a random subset of that size is sampled from (bounds the cost). Returns
    every index when ``K <= n``.
    """
    K = points.shape[0]
    device = points.device
    if K <= n:
        return torch.arange(K, device=device)
    candidates = torch.arange(K, device=device)
    if max_candidates is not None and K > max_candidates:
        candidates = torch.randperm(K, generator=generator, device=device)[:max_candidates]
    pts = points[candidates].to(torch.float32)
    chosen = torch.empty(n, dtype=torch.long, device=device)
    current = torch.randint(pts.shape[0], (1,), generator=generator, device=device)[0]
    distance = torch.full((pts.shape[0],), float("inf"), device=device)
    for i in range(n):
        chosen[i] = current
        distance = torch.minimum(distance, torch.linalg.vector_norm(pts - pts[current], dim=-1))
        current = torch.argmax(distance)
    return candidates[chosen]


def interior_mask(segmentation: torch.Tensor, margin: int = 1) -> torch.Tensor:
    """``(H, W)`` pixels whose neighbours within ``margin`` (4-connected steps) share their segment id.

    Drops silhouette pixels, whose rendered depth may belong to what lies
    behind the object (they back-project far off the object).
    """
    inside = torch.ones_like(segmentation, dtype=torch.bool)
    for step in range(1, int(margin) + 1):
        for dim, shift in ((0, step), (0, -step), (1, step), (1, -step)):
            neighbour = torch.roll(segmentation, shifts=shift, dims=dim)
            same = neighbour == segmentation
            # Pixels whose neighbour would wrap around the image edge.
            if dim == 0:
                (same[:step] if shift > 0 else same[-step:]).fill_(False)
            else:
                (same[:, :step] if shift > 0 else same[:, -step:]).fill_(False)
            inside &= same
    return inside


class FocusedSegmentationPointProposer(UniformSegmentationPointProposer):
    """Most points on the *focus* entities, the rest over the other tracked ones.

    Mirrors a moving / non-moving split: ``n_focus_points`` points go to the
    entities named by ``focus_entity_names`` (e.g. the objects a task moves),
    the remaining ``max_points - n_focus_points`` to every other tracked
    entity (clutter, table, ...). When one group has too few candidate pixels
    the other fills the budget. Within each group the points are spread over
    the candidates' 3D positions by farthest-point sampling from a random start
    (``sampling="fps"``) or drawn uniformly at random (``"random"``), so they
    cover the objects rather than one image region. Candidates are the
    pixels ``edge_margin`` steps inside their segment (:func:`interior_mask`):
    silhouette pixels can carry the depth of the background, and
    farthest-point sampling would pick exactly those outliers; a group with no
    interior pixel falls back to all of its pixels.

    Args:
        focus_entity_names: Names of the focus entities, or a zero-argument
            callable returning them (resolved at every proposal, so it can
            follow entities created or chosen per scene build / episode).
        n_focus_points: Point budget of the focus entities.
        sampling: ``"fps"`` or ``"random"``.
        max_candidates: Candidate cap per group for farthest-point sampling.
        edge_margin: Pixels dropped along every segment boundary (``0`` keeps
            all pixels).
        tracked_entity_names / tracked_entity_name_prefixes: Which entities
            are tracked at all (see :class:`UniformSegmentationPointProposer`);
            focus entities must be among them.
    """

    def __init__(
        self,
        *,
        focus_entity_names: Union[Callable[[], Iterable[str]], Iterable[str]] = (),
        n_focus_points: int = 48,
        sampling: str = "fps",
        max_candidates: Optional[int] = 4096,
        edge_margin: int = 1,
        tracked_entity_names: Optional[Sequence[str]] = None,
        tracked_entity_name_prefixes: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            tracked_entity_names=tracked_entity_names,
            tracked_entity_name_prefixes=tracked_entity_name_prefixes,
        )
        if sampling not in ("fps", "random"):
            raise ValueError(f"sampling must be 'fps' or 'random', got {sampling!r}")
        if int(n_focus_points) < 0:
            raise ValueError(f"n_focus_points must be >= 0, got {n_focus_points}")
        self.focus_entity_names = focus_entity_names
        self.n_focus_points = int(n_focus_points)
        self.sampling = sampling
        self.max_candidates = max_candidates
        self.edge_margin = int(edge_margin)

    def _focus_names(self) -> Set[str]:
        names = self.focus_entity_names() if callable(self.focus_entity_names) else self.focus_entity_names
        return {str(name) for name in names}

    def _sample(self, positions: torch.Tensor, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        if n <= 0:
            return torch.empty(0, dtype=torch.long, device=positions.device)
        if self.sampling == "fps":
            return farthest_point_sample(positions, n, generator=generator, max_candidates=self.max_candidates)
        if positions.shape[0] <= n:
            return torch.arange(positions.shape[0], device=positions.device)
        return torch.randperm(positions.shape[0], generator=generator, device=positions.device)[:n]

    def propose(
        self,
        *,
        camera_obs: Dict[str, torch.Tensor],
        seg_to_entity_idx: Mapping[int, int],
        entity_names: Mapping[int, str],
        max_points: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        segmentation = camera_obs["seg"]       # (B, H, W)
        position = camera_obs["position"]      # (B, H, W, 3)
        B = segmentation.shape[0]
        device = segmentation.device

        focus = self._focus_names()
        focus_seg, other_seg = [], []
        for seg_id, entity_idx in seg_to_entity_idx.items():
            name = entity_names.get(entity_idx, "")
            if name in focus:
                focus_seg.append(int(seg_id))
            elif self._name_allowed(name):
                other_seg.append(int(seg_id))
        focus_seg_t = torch.tensor(focus_seg, dtype=segmentation.dtype, device=device)
        other_seg_t = torch.tensor(other_seg, dtype=segmentation.dtype, device=device)

        out_points = torch.zeros((B, max_points, 3), dtype=position.dtype, device=device)
        out_seg = torch.zeros((B, max_points), dtype=torch.int64, device=device)
        out_valid = torch.zeros((B, max_points), dtype=torch.bool, device=device)

        for b in range(B):
            valid = torch.isfinite(position[b]).all(dim=-1)
            inside = interior_mask(segmentation[b], self.edge_margin) if self.edge_margin > 0 else None
            groups = []
            for seg_t in (focus_seg_t, other_seg_t):
                member = valid & torch.isin(segmentation[b], seg_t)
                if inside is not None and bool((member & inside).any()):
                    member = member & inside
                groups.append(torch.nonzero(member, as_tuple=False))
            focus_px, other_px = groups
            n_focus = min(self.n_focus_points, max_points, focus_px.shape[0])
            n_other = min(max_points - n_focus, other_px.shape[0])
            n_focus = min(max_points - n_other, focus_px.shape[0])  # focus fills a short scene budget
            chosen = []
            for pixels, n in ((focus_px, n_focus), (other_px, n_other)):
                if n > 0:
                    points = position[b][pixels[:, 0], pixels[:, 1]]
                    chosen.append(pixels[self._sample(points, n, generator)])
            if not chosen:
                continue
            pixels = torch.cat(chosen, dim=0)
            n = pixels.shape[0]
            out_points[b, :n] = position[b][pixels[:, 0], pixels[:, 1]].to(position.dtype)
            out_seg[b, :n] = segmentation[b][pixels[:, 0], pixels[:, 1]].to(torch.int64)
            out_valid[b, :n] = True

        return out_points, out_seg, out_valid
