from __future__ import annotations

from typing import Optional

import torch

from unienv_genesis_collection.objects.base_object import (
    GraspCandidates,
    ManipulatableObject,
)
from unienv_genesis_refpolicy.grasp_providers.base import GraspCandidateProvider


class ObjectGeometryGraspCandidateProvider(GraspCandidateProvider):
    """Default provider that delegates to the target object's geometry helpers."""

    def __init__(
        self,
        *,
        target_object: ManipulatableObject,
        target_entity,
        gripper_width: float,
        gripper_depth: float,
    ) -> None:
        self.target_object = target_object
        self.target_entity = target_entity
        self.gripper_width = gripper_width
        self.gripper_depth = gripper_depth

    def propose(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> GraspCandidates:
        del mask, seed
        return self.target_object.list_box_gripper_grasping_candidates(
            self.target_entity,
            self.gripper_width,
            gripper_depth=self.gripper_depth,
        )
