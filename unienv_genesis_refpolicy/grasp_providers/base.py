from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from unienv_genesis_collection.objects.base_object import GraspCandidates


class GraspCandidateProvider(ABC):
    """Interface for grasp candidate proposal sources.

    Providers can use object geometry, learned models, camera observations, or
    any other source to produce world-frame :class:`GraspCandidates`.
    """

    @abstractmethod
    def propose(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> GraspCandidates:
        """Return grasp candidates in world coordinates."""
        raise NotImplementedError
