"""Point proposers for keypoint initialisation.

A :class:`PointProposer` selects keypoints from batched camera observations
at reset/reload time.  The selected keypoints are then mapped to rigid
entities and stored as local offsets by the :class:`KeypointTrackerNode`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

import torch

__all__ = [
    "PointProposer",
    "UniformSegmentationPointProposer",
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
