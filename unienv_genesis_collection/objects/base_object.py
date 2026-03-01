"""Abstract base class for manipulatable objects in Genesis scenes.

A ``ManipulatableObject`` is a **standalone descriptor** — it is *not* a
``WorldNode``.  It is passed to a ``WorldNode`` such as
``ClutteredTableManager`` that manages the lifecycle (reload / reset).
Most geometry queries (bounding box, graspability) take the loaded Genesis
entity as a parameter so they can reflect runtime state (e.g. articulated
joints).

Bounding boxes are represented as ``BoxSpace`` instances from
``unienv_interface.space`` (PyTorch backend, float32, shape ``(3,)``).  The
``low`` / ``high`` of the box correspond to the min / max corners of the
axis-aligned bounding box in the object's local frame.

Grasp candidates are computed in the **object's local frame** (and cached)
via ``list_box_gripper_grasping_candidates_local``.  The world-frame
accessor ``list_box_gripper_grasping_candidates`` applies the entity's
current pose transform on every call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, List, Tuple, TypeVar, Union, Sequence, TYPE_CHECKING

import torch
from unienv_interface.space import BoxSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity as GenesisEntity
    from unienv_genesis.world.genesis_world import GenesisWorld

    EntityT = TypeVar("EntityT", bound=GenesisEntity)
else:
    EntityT = TypeVar("EntityT")

__all__ = [
    "ManipulatableObject",
    "EntityT",
    "GraspCandidates",
    "make_bbox",
    "bbox_size",
    "bbox_center",
    "bbox_half_extents",
]


# ---------------------------------------------------------------------------
# Bounding-box helpers (operate on BoxSpace instances)
# ---------------------------------------------------------------------------

def make_bbox(
    min_point: Union[torch.Tensor, Sequence[float]],
    max_point: Union[torch.Tensor, Sequence[float]],
) -> BoxSpace:
    """Create a ``BoxSpace`` representing an axis-aligned bounding box.

    Args:
        min_point: Minimum corner ``(x, y, z)``.
        max_point: Maximum corner ``(x, y, z)``.

    Returns:
        A ``BoxSpace`` with PyTorch backend, float32, shape ``(3,)``.
    """
    low = torch.as_tensor(min_point, dtype=torch.float32)
    high = torch.as_tensor(max_point, dtype=torch.float32)
    return BoxSpace(
        backend=PyTorchComputeBackend,
        low=low,
        high=high,
        dtype=torch.float32,
    )


def bbox_size(bbox: BoxSpace) -> torch.Tensor:
    """Extent along each axis ``(width, depth, height)``, shape ``(3,)``."""
    return bbox.high - bbox.low


def bbox_center(bbox: BoxSpace) -> torch.Tensor:
    """Centre point of the AABB, shape ``(3,)``."""
    return (bbox.low + bbox.high) / 2.0


def bbox_half_extents(bbox: BoxSpace) -> torch.Tensor:
    """Half-extents along each axis, shape ``(3,)``."""
    return bbox_size(bbox) / 2.0


# ---------------------------------------------------------------------------
# Grasp candidates (batched)
# ---------------------------------------------------------------------------

@dataclass
class GraspCandidates:
    """A batched collection of candidate grasp poses for a parallel-jaw gripper.

    Positions and directions may be in either the **object's local frame**
    or **world coordinates**, depending on context.

    Every field shares the same ``*batch_shape`` prefix.  The most common
    layout is ``batch_shape = (N,)`` where *N* is the number of candidates.

    Attributes:
        position: Grasp centre positions, shape ``(*batch_shape, 3)``.
        approach_direction: Unit vectors for gripper approach,
            shape ``(*batch_shape, 3)``.
        opening_direction: Unit vectors for gripper opening axis,
            shape ``(*batch_shape, 3)``.
    """

    position: torch.Tensor  # (*batch_shape, 3)
    approach_direction: torch.Tensor  # (*batch_shape, 3)
    opening_direction: torch.Tensor  # (*batch_shape, 3)

    @property
    def batch_shape(self) -> torch.Size:
        """The leading batch dimensions (everything before the last axis)."""
        return self.position.shape[:-1]

    def __len__(self) -> int:
        """Number of elements along the first batch dimension."""
        return self.position.shape[0]

    @classmethod
    def empty(cls) -> GraspCandidates:
        """Return an empty ``GraspCandidates`` with zero candidates."""
        return cls(
            position=torch.empty(0, 3),
            approach_direction=torch.empty(0, 3),
            opening_direction=torch.empty(0, 3),
        )


# ---------------------------------------------------------------------------
# Local → world transform helper
# ---------------------------------------------------------------------------

def _local_to_world_candidates(
    candidates: GraspCandidates,
    pos: torch.Tensor,
    quat: torch.Tensor,
) -> GraspCandidates:
    """Transform local-frame :class:`GraspCandidates` to world coordinates.

    Uses the ``quaternion_apply`` utility from xbarray to rotate vectors
    (broadcasts over ``*batch_shape``), then translates positions by *pos*.

    Supports batched environments: when *pos* has shape ``(n_envs, 3)``
    and *quat* has shape ``(n_envs, 4)``, the output gains a leading
    ``n_envs`` dimension — e.g. local candidates of shape ``(N, 3)``
    become ``(n_envs, N, 3)`` in the result.

    Args:
        candidates: Grasp candidates in the object's local frame,
            shape ``(N, 3)`` / ``(N,)``.
        pos: Object world position, shape ``(3,)`` or ``(n_envs, 3)``.
        quat: Object world quaternion ``(w, x, y, z)``, shape ``(4,)``
            or ``(n_envs, 4)``.

    Returns:
        New :class:`GraspCandidates` with world-coordinate fields.
    """
    from xbarray.transformations.rotation_conversions.pytorch import quaternion_apply

    if pos.ndim > 1:
        # Batched: pos (n_envs, 3), quat (n_envs, 4)
        # candidates fields are (N, 3)
        # Result fields become (n_envs, N, 3)
        q = quat.unsqueeze(-2)   # (n_envs, 1, 4)
        p = pos.unsqueeze(-2)    # (n_envs, 1, 3)
        return GraspCandidates(
            position=quaternion_apply(q, candidates.position) + p,
            approach_direction=quaternion_apply(q, candidates.approach_direction),
            opening_direction=quaternion_apply(q, candidates.opening_direction),
        )

    return GraspCandidates(
        position=quaternion_apply(quat, candidates.position) + pos,
        approach_direction=quaternion_apply(quat, candidates.approach_direction),
        opening_direction=quaternion_apply(quat, candidates.opening_direction),
    )


# ---------------------------------------------------------------------------
# ManipulatableObject
# ---------------------------------------------------------------------------

class ManipulatableObject(ABC, Generic[EntityT]):
    """Abstract descriptor for an object that can be loaded into a Genesis
    scene and queried for geometric properties (bounding box, graspability).

    This is NOT a ``WorldNode``.  It is a standalone descriptor that can be
    used offline for planning or passed to a manager node for scene loading.

    Subclasses must implement:
        - :meth:`load_into_scene` — add the object to a Genesis scene.
        - :meth:`get_local_bounding_box` — return the AABB in the local frame.

    Quaternion convention: ``(w, x, y, z)`` throughout, matching Genesis and
    ``EntityGroup``.
    """

    def __init__(
        self,
        name: str = "object",
        fixed: bool = False,
    ):
        """
        Args:
            name: Default entity name used when loading into the scene.
            fixed: If ``True`` the object is immovable (no dynamics).
        """
        self.name = name
        self.fixed = fixed

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def load_into_scene(
        self,
        world: "GenesisWorld",
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        name_override: Optional[str] = None,
        batch_fixed_verts: bool = False,
    ) -> EntityT:
        """Load this object into a Genesis scene.

        Args:
            world: ``GenesisWorld`` with an unbuilt scene.
            pos: World position ``(x, y, z)``.
            quat: World orientation ``(w, x, y, z)``.
            name_override: Override the default ``self.name``.
            batch_fixed_verts: Enable per-env position overrides for
                fixed entities in batched worlds.

        Returns:
            A Genesis entity reference.
        """
        ...

    @abstractmethod
    def get_local_bounding_box(self, entity: EntityT) -> BoxSpace:
        """Return the axis-aligned bounding box in the object's local frame.

        Args:
            entity: The loaded Genesis entity.

        Returns:
            ``BoxSpace`` (PyTorch backend, float32, shape ``(3,)``) whose
            ``low`` / ``high`` are the min / max corners.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete geometry helpers
    # ------------------------------------------------------------------

    def get_bounding_box(self, entity: EntityT) -> BoxSpace:
        """Return the world-frame axis-aligned bounding box.

        Transforms the local-frame AABB corners by the entity's current
        pose (position + orientation) and recomputes the AABB.

        Supports batched environments: when the entity has batched
        ``get_pos()`` / ``get_quat()`` of shape ``(n_envs, 3)`` /
        ``(n_envs, 4)``, the returned ``BoxSpace`` has ``low`` / ``high``
        of shape ``(n_envs, 3)``.

        Args:
            entity: The loaded Genesis entity.

        Returns:
            ``BoxSpace`` with ``low`` / ``high`` in world coordinates.
        """
        from xbarray.transformations.rotation_conversions.pytorch import quaternion_apply

        local_bbox = self.get_local_bounding_box(entity)
        pos = entity.get_pos().detach().float()
        quat = entity.get_quat().detach().float()

        lo, hi = local_bbox.low, local_bbox.high
        # All 8 corners of the local AABB: (8, 3)
        corners = torch.stack([
            torch.stack([lo[0], lo[1], lo[2]]),
            torch.stack([lo[0], lo[1], hi[2]]),
            torch.stack([lo[0], hi[1], lo[2]]),
            torch.stack([lo[0], hi[1], hi[2]]),
            torch.stack([hi[0], lo[1], lo[2]]),
            torch.stack([hi[0], lo[1], hi[2]]),
            torch.stack([hi[0], hi[1], lo[2]]),
            torch.stack([hi[0], hi[1], hi[2]]),
        ])

        if pos.ndim > 1:
            # Batched: pos (n_envs, 3), quat (n_envs, 4)
            world_corners = quaternion_apply(
                quat.unsqueeze(-2), corners.unsqueeze(0),
            ) + pos.unsqueeze(-2)                          # (n_envs, 8, 3)
            world_lo = world_corners.min(dim=-2).values    # (n_envs, 3)
            world_hi = world_corners.max(dim=-2).values    # (n_envs, 3)
        else:
            world_corners = quaternion_apply(quat, corners) + pos  # (8, 3)
            world_lo = world_corners.min(dim=0).values             # (3,)
            world_hi = world_corners.max(dim=0).values             # (3,)

        return make_bbox(world_lo, world_hi)

    def get_footprint_radius(self, entity: EntityT) -> float:
        """Return the 2D footprint radius on the XY plane (metres).

        Used by placement algorithms for quick circular overlap checks.
        Default: half of the local bounding-box XY diagonal.

        Args:
            entity: The loaded Genesis entity.
        """
        bbox = self.get_local_bounding_box(entity)
        size_xy = bbox_size(bbox)[:2]
        return float(torch.linalg.norm(size_xy) / 2.0)

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def set_pos_3d(
        self,
        entity: EntityT,
        positions: torch.Tensor,
        random_quaternion: bool = True,
        rng: Optional[torch.Generator] = None,
        envs_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Set the entity position and orientation.

        When *random_quaternion* is ``True`` (the default) the object
        receives a random yaw-only rotation so that it stands upright in
        its canonical pose.  Subclasses may override this method to
        provide a different randomisation strategy (e.g. full SO(3) for
        spherical objects, or identity for asymmetric objects that should
        always face the same direction).

        Args:
            entity: The loaded Genesis entity.
            positions: World positions.  Shape ``(3,)`` for unbatched or
                ``(n, 3)`` for *n* environments.
            random_quaternion: If ``True`` generate random orientations;
                if ``False`` use the identity quaternion ``(1, 0, 0, 0)``.
            rng: Optional :class:`torch.Generator` for reproducibility.
            envs_idx: Optional ``(n,)`` :class:`~torch.LongTensor` of
                environment indices for partial (masked) updates.  When
                provided, *positions* has shape ``(n, 3)`` and only the
                listed environments are overwritten; the rest keep their
                current state.
        """
        if positions.ndim == 1:
            # Unbatched — single environment.
            quat = self._standing_quaternion(1, random_quaternion, rng).squeeze(0)
            entity.set_pos(positions)
            entity.set_quat(quat)
            return

        n = positions.shape[0]
        quat = self._standing_quaternion(n, random_quaternion, rng)

        if envs_idx is None:
            entity.set_pos(positions)
            entity.set_quat(quat)
        else:
            cur_pos = entity.get_pos().clone()
            cur_quat = entity.get_quat().clone()
            cur_pos[envs_idx] = positions
            cur_quat[envs_idx] = quat
            entity.set_pos(cur_pos)
            entity.set_quat(cur_quat)

    def _standing_quaternion(
        self,
        n: int,
        randomise: bool,
        rng: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Return *n* orientation quaternions ``(n, 4)`` for upright placement.

        The default implementation uses a random yaw (Z-rotation) when
        *randomise* is ``True`` and the identity otherwise.  Subclasses
        can override for custom behaviour (e.g. full SO(3), fixed pose).

        Args:
            n: Number of quaternions to generate.
            randomise: Whether to randomise.
            rng: Optional :class:`torch.Generator`.
        """
        if randomise:
            import math
            yaw = torch.rand(n, generator=rng) * (2 * math.pi)
            half = yaw / 2.0
            quat = torch.zeros(n, 4)
            quat[:, 0] = torch.cos(half)
            quat[:, 3] = torch.sin(half)
            return quat
        quat = torch.zeros(n, 4)
        quat[:, 0] = 1.0
        return quat

    # ------------------------------------------------------------------
    # Grasp analysis
    # ------------------------------------------------------------------

    def is_box_gripper_graspable(
        self,
        entity: EntityT,
        gripper_width: float,
        gripper_depth: float = 0.01,
    ) -> bool:
        """Check if the object can be grasped by a parallel-jaw gripper.

        Default implementation delegates to
        :meth:`list_box_gripper_grasping_candidates_local` and returns
        ``True`` if any candidates exist.  No world transform is needed.

        Args:
            entity: The loaded Genesis entity (used to query current pose).
            gripper_width: Maximum gripper opening in metres.
            gripper_depth: Depth of the gripper fingers in metres.

        Returns:
            ``True`` if at least one grasp candidate exists.
        """
        return len(self.list_box_gripper_grasping_candidates_local(
            entity, gripper_width, gripper_depth=gripper_depth,
        )) > 0

    def list_box_gripper_grasping_candidates_local(
        self,
        entity: EntityT,
        gripper_width: float,
        gripper_depth: float = 0.01,
    ) -> GraspCandidates:
        """Return candidate grasp poses in the **object's local frame**.

        Results are cached on first call (keyed by ``gripper_width`` and
        ``gripper_depth``).  Subsequent calls with the same parameters
        return the cached value immediately.

        The default implementation generates axis-aligned grasps from the
        local bounding box.  Subclasses can override for more sophisticated
        analysis (e.g. mesh-based antipodal sampling).

        Args:
            entity: The loaded Genesis entity.
            gripper_width: Maximum gripper opening in metres.
            gripper_depth: Depth of the gripper fingers in metres.

        Returns:
            :class:`GraspCandidates` in the object's local frame,
            shape ``(N, 3)`` where *N* is the number of candidates.
        """
        if not hasattr(self, "_local_grasp_cache"):
            self._local_grasp_cache: dict[tuple, GraspCandidates] = {}

        key = (gripper_width, gripper_depth)
        if key in self._local_grasp_cache:
            return self._local_grasp_cache[key]

        bbox = self.get_local_bounding_box(entity)
        center = bbox_center(bbox)
        dims = bbox_size(bbox)

        positions: List[torch.Tensor] = []
        approach_dirs: List[torch.Tensor] = []
        opening_dirs: List[torch.Tensor] = []

        axes = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]
        for i, opening_dir in enumerate(axes):
            if dims[i] >= gripper_width or dims[i] < gripper_depth:
                continue
            for j in range(3):
                if j == i:
                    continue
                approach_dir = axes[j]
                # Forward approach
                positions.append(center)
                approach_dirs.append(approach_dir)
                opening_dirs.append(opening_dir)
                # Reverse approach
                positions.append(center)
                approach_dirs.append(-approach_dir)
                opening_dirs.append(opening_dir)

        if not positions:
            result = GraspCandidates.empty()
        else:
            result = GraspCandidates(
                position=torch.stack(positions),
                approach_direction=torch.stack(approach_dirs),
                opening_direction=torch.stack(opening_dirs),
            )

        self._local_grasp_cache[key] = result
        return result

    def list_box_gripper_grasping_candidates(
        self,
        entity: EntityT,
        gripper_width: float,
        gripper_depth: float = 0.01,
    ) -> GraspCandidates:
        """Return candidate grasp poses for a parallel-jaw gripper.

        Delegates to :meth:`list_box_gripper_grasping_candidates_local`
        for the local-frame candidates (cached), then transforms them to
        **world coordinates** using the entity's current pose.

        Args:
            entity: The loaded Genesis entity (``entity.get_pos()`` and
                ``entity.get_quat()`` are called to obtain the current
                world pose).
            gripper_width: Maximum gripper opening in metres.
            gripper_depth: Depth of the gripper fingers in metres.

        Returns:
            :class:`GraspCandidates` in world coordinates,
            shape ``([B], N, ...)`` where *N* is the number of candidates and *B* is the batch size (if the world is batched).
        """
        local = self.list_box_gripper_grasping_candidates_local(
            entity, gripper_width, gripper_depth=gripper_depth,
        )
        if len(local) == 0:
            return local

        pos = entity.get_pos().detach().float()
        quat = entity.get_quat().detach().float()
        return _local_to_world_candidates(local, pos, quat)

