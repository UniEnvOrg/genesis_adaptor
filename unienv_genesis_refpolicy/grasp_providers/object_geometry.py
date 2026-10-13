from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from unienv_genesis_collection.objects.base_object import (
    GraspCandidates,
    ManipulatableObject,
)
from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider


class ObjectGeometryGraspCandidateProvider(GraspCandidateProvider):
    """Default provider that delegates to the target object's geometry helpers.

    The loaded entity may be supplied either as a fixed ``target_entity`` or,
    preferably, as a zero-argument ``target_entity_fn`` returning the *current*
    entity: the callable is re-read on every :meth:`propose`, so a scene reload
    (which swaps the entity id) is picked up without rebuilding the provider.

    Args:
        target_object: The :class:`ManipulatableObject` descriptor.
        target_entity: The loaded Genesis entity (fixed binding).  May be
            ``None`` when ``target_entity_fn`` is given.
        target_entity_fn: Zero-argument callable returning the current
            (single, batched) entity.  Takes precedence over ``target_entity``
            when provided.
        gripper_width: Maximum gripper opening in metres.
        gripper_depth: Gripper finger depth in metres.
    """

    def __init__(
        self,
        *,
        target_object: ManipulatableObject,
        target_entity: Any = None,
        target_entity_fn: Optional[Callable[[], Any]] = None,
        gripper_width: float,
        gripper_depth: float,
    ) -> None:
        self.target_object = target_object
        self.target_entity = target_entity
        self.target_entity_fn = target_entity_fn
        self.gripper_width = gripper_width
        self.gripper_depth = gripper_depth

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
        # Re-read the entity every call: a scene reload swaps the entity id, and
        # resolving it here re-targets the provider without a rebuild.
        entity = (
            self.target_entity_fn()
            if self.target_entity_fn is not None
            else self.target_entity
        )
        return self.target_object.list_box_gripper_grasping_candidates(
            entity,
            self.gripper_width,
            gripper_depth=self.gripper_depth,
            rng=rng,
        )
