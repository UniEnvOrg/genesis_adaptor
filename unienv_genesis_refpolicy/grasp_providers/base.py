from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from unienv_genesis_collection.objects.base_object import GraspCandidates


class GraspCandidateProvider(ABC):
    """Interface for grasp candidate proposal sources.

    Providers can use object geometry, learned models, camera observations, or
    any other source to produce world-frame :class:`GraspCandidates`.

    **Target binding**: a provider that proposes candidates for a loaded Genesis
    entity may be bound to its target either way:

    * ``target_entity_fn: Callable[[], Any]`` — *preferred*.  A zero-argument
      callable returning the current (single, batched) entity.  It is re-read on
      every :meth:`propose`, so a scene reload — which swaps the entity id — is
      picked up without rebuilding the provider (and thus without rebuilding the
      policy graph that owns it).
    * a fixed ``target_entity`` — the entity used for the provider's whole
      lifetime (the historical behavior; still fully supported).

    Binding is an implementation detail of each provider: the :meth:`propose`
    signature is deliberately unchanged, so fixed-entity providers remain valid
    drop-in implementations.
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
