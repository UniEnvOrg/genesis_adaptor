"""Articulated-furniture manipulation task (plan §4 / stage S6).

:class:`OpenArticulatedJointTask` is the canonical "open the top drawer of
cabinet X" task: a :class:`~unienv_interface.world.node.WorldNode` that drives
**one named joint of one room furniture instance** to a randomly sampled target
open fraction, modelled directly on
:class:`~unienv_genesis_collection.tasks.tabletop.pickplace.ObjectRelocateTask`
(priority convention, masked-reset semantics, dense reward in
``post_environment_step``, batch-aware ``DictSpace`` observations).

The furniture itself is owned by
:class:`~unienv_genesis_collection.scenes.room_builder.RoomBuilder` (which loads
every fixture once during ``reload``), so this task loads **no** entities of its
own — it binds to a ``(furniture_key, joint_name)`` pair.

Joint and link addressing is **by name at runtime** (plan decision log: no joint
registry, nothing baked per asset):

* DOF indexes come from
  :func:`~unienv_genesis_collection.objects.articulated_object.resolve_joint_dofs`
  — the same on-the-fly ``{joint.name: joint.dofs_idx_local}`` map the robot
  actor uses (``unienv_genesis/nodes/entities/entity_controllers/base.py``).
* The open fraction derives ``q_min`` / ``q_max`` from
  ``entity.get_dofs_limit()`` at call time through
  :meth:`ArticulatedObject.get_open_fraction`, so hinge (rad) and slide (m)
  joints are uniform in ``[0, 1]`` fraction space.
* The handle link defaults to the joint's **child link** as reported by the
  built entity (``joint.link.name``), overridable through ``grasp_link_name``.

Unknown joint/link names raise :class:`ArticulationError` listing the names the
built entity actually has — the same fail-loud UX as robot keyframes.

Contract:

* **observation** (``DictSpace``, batch-aware shapes)::

      joint_position   ([B],)   current open fraction in [0, 1]
      target_fraction  ([B],)   goal fraction
      handle_position  ([B], 3) world position of the handle link's bbox centre

* **reward**: ``-|current_fraction - target_fraction|`` (dense, in ``[-1, 0]``)
* **termination**: ``|current_fraction - target_fraction| < success_threshold``

Initial-state randomisation (:meth:`reset`): every reset environment starts
**strictly below its own target** — ``f_init ~ U(0, target - success_threshold)``
(0 when the target is already within the success band of closed) — so an episode
always begins outside the success region and requires an *opening* motion.  Both
the goal and the initial fraction are sampled for the whole batch and then
sliced by the reset mask, so RNG consumption is identical for masked and
unmasked resets while unmasked envs keep their joint state, goal, reward and
termination flag (``pickplace.py`` semantics).

Lifecycle priorities default to ``max(room_builder.<phase>_priorities) + 1`` so
the task binds and resets **after** the room has (re)built its furniture — the
same "one above the manager's maximum" convention as ``pickplace``.  Because
phases dispatch in descending priority order, the task's ``reset`` then runs
*before* :meth:`RoomBuilder.reset`: a layout that pins ``initial_open`` on this
task's own joint overwrites the sampled start state.  Leave ``initial_open``
empty for the joint the task owns (or pass ``reset_priority`` below the room's)
when that is not wanted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world.node import WorldNode
from unienv_interface.space.space_utils.metadata_utils import metadata
from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
from unienv_genesis.utils.space_metadata import position_metadata
from unienv_genesis_collection.objects.articulated_object import (
    ArticulatedObject,
    ArticulationError,
    resolve_joint_dofs,
    resolve_link_index,
)
from unienv_genesis_collection.objects.base_object import GraspCandidates, bbox_center
from unienv_genesis_collection.scenes.room_builder import (
    DEFAULT_INTERACTION_GRIPPER_WIDTH,
    RoomBuilder,
)

if TYPE_CHECKING:
    from unienv_genesis.world import GenesisWorld

__all__ = ["OpenArticulatedJointTask"]


class OpenArticulatedJointTask(WorldNode):
    """Open a named furniture joint of a room instance to a target open fraction.

    Args:
        world: :class:`GenesisWorld` the room belongs to.
        room_builder: The :class:`RoomBuilder` owning the furniture; its
            :attr:`~RoomBuilder.furniture` /
            :attr:`~RoomBuilder.furniture_entities` mappings are bound in
            :meth:`after_reload`.
        furniture_key: Key into ``room_builder.furniture`` (the layout's
            furniture key, e.g. ``"drawer"`` or ``"cabinet_hinge_1"``).
        joint_name: Name of the joint to drive; addressed **by name** against
            the built entity at runtime.
        grasp_link_name: Optional handle link name for the ``handle_position``
            observation and :meth:`get_handle_grasp_candidates`.  ``None``
            (default) uses the joint's child link.
        target_fraction_range: Inclusive ``(low, high)`` range in ``[0, 1]`` the
            per-env goal fraction is sampled from.
        success_threshold: Fraction distance below which the episode terminates
            (also the guaranteed initial gap).
        seed: Seed for the task's :class:`torch.Generator` (reseedable through
            ``reload(seed=...)`` / ``reset(seed=...)``, pickplace convention).
        name: Node name.
        **priorities: ``reload_priority`` / ``reset_priority`` /
            ``after_reload_priority`` / ``post_environment_step_priority``
            overrides; the first three default to
            ``max(room_builder.<phase>_priorities) + 1``, the last to 0.

    Raises:
        ValueError: On an invalid ``target_fraction_range`` /
            ``success_threshold`` or an unknown priority keyword.
        KeyError: If ``furniture_key`` is not a furniture instance of a room
            whose layout is already resolved, and again in
            :meth:`after_reload` for rooms built later).
    """

    # Fallback when the room exposes no priority for a phase (objects/tasks range).
    DEFAULT_RESET_PRIORITY = 25
    DEFAULT_POSTSTEP_PRIORITY = 0

    has_reward = True
    has_termination_signal = True
    has_truncation_signal = False

    def __init__(
        self,
        world: "GenesisWorld",
        room_builder: RoomBuilder,
        furniture_key: str,
        joint_name: str,
        grasp_link_name: Optional[str] = None,
        target_fraction_range: Tuple[float, float] = (0.6, 1.0),
        success_threshold: float = 0.05,
        seed: Optional[int] = None,
        name: str = "open_fixture_task",
        **priorities: Optional[int],
    ):
        self.world = world
        self._name = name
        self.room_builder = room_builder
        self.furniture_key = furniture_key
        self.joint_name = joint_name
        self.grasp_link_name = grasp_link_name

        # Task parameters (validated eagerly — fraction space is [0, 1]).
        if len(target_fraction_range) != 2:
            raise ValueError(
                "target_fraction_range must be (low, high), got "
                f"{target_fraction_range!r}"
            )
        low, high = float(target_fraction_range[0]), float(target_fraction_range[1])
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError(
                "target_fraction_range must satisfy 0 <= low <= high <= 1, got "
                f"{(low, high)!r} (the open fraction is normalised to [0, 1])"
            )
        self.target_fraction_range: Tuple[float, float] = (low, high)
        self.success_threshold = float(success_threshold)
        if not self.success_threshold > 0.0:
            raise ValueError(
                f"success_threshold must be positive, got {success_threshold!r}"
            )

        # Priorities: one above the room builder's MAXIMUM per phase.  Phases
        # dispatch in DESCENDING priority order, so the task binds and resets
        # after every room callback (explicit values, including 0, always win —
        # same rule as ObjectRelocateTask).
        self.reload_priorities = {
            self._phase_priority("reload_priorities", priorities, "reload_priority")
        }
        self.after_reload_priorities = {
            self._phase_priority(
                "after_reload_priorities", priorities, "after_reload_priority"
            )
        }
        self.reset_priorities = {
            self._phase_priority("reset_priorities", priorities, "reset_priority")
        }
        post_step = priorities.pop(
            "post_environment_step_priority", self.DEFAULT_POSTSTEP_PRIORITY
        )
        if priorities:
            raise ValueError(
                f"Unknown priority keyword(s): {sorted(priorities)}. Expected "
                "reload_priority / reset_priority / after_reload_priority / "
                "post_environment_step_priority."
            )
        self.post_environment_step_priorities = {int(post_step)}

        # Random state (pickplace convention: a generator on the world device).
        self.rng: torch.Generator = torch.Generator(device=world.device)
        if seed is None:
            self.rng.seed()
        else:
            self.rng.manual_seed(seed)

        # Bound furniture (resolved by name in after_reload).
        self.descriptor: Optional[ArticulatedObject] = None
        self.entity: Any = None
        self._dofs_idx: List[int] = []
        self._handle_link_name: Optional[str] = None
        # Handle bbox centre expressed in the grasp link's own frame (cached at
        # the reference configuration) — see _cache_handle_offset().
        self._handle_offset: Optional[torch.Tensor] = None

        # Per-episode state.
        self._target_fraction: Optional[torch.Tensor] = None  # ([B],)
        self._reward: Union[float, torch.Tensor] = 0.0  # scalar or ([B],)
        self._terminated: Union[bool, torch.Tensor] = False  # bool or ([B],)

        # Fail fast when the room has already resolved its layout (a room built
        # later is validated in reload()).
        resolved = getattr(room_builder, "furniture", None)
        if resolved and furniture_key not in resolved:
            raise self._unknown_furniture_error(resolved)

        # Observation space — known at construction time (batch-aware shapes,
        # world device), like pickplace / reach.  The fractions are one scalar
        # per environment, so the feature tuple is empty: ([B],) batched, ()
        # unbatched (``handle_position`` is the only vectorised entry).
        self._fraction_shape = get_batch_aware_shape(self.world.batch_size, ())
        fraction_shape = self._fraction_shape
        # ``joint_position`` is NOT the joint coordinate (rad / m) but the open
        # fraction of the joint's range, like ``target_fraction``.
        fraction_md = dict(
            units="normalized",
            normalized=True,
            furniture=furniture_key,
            joint=joint_name,
            fraction_of="joint range: 0 = closed (lower limit), 1 = open (upper limit)",
        )
        self.observation_space = DictSpace(
            PyTorchComputeBackend,
            {
                "joint_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=0.0,
                    high=1.0,
                    shape=fraction_shape,
                    dtype=self.world.dtype,
                    device=self.world.device,
                    metadata=metadata("task", label="current open fraction of the joint", **fraction_md),
                ),
                "target_fraction": BoxSpace(
                    PyTorchComputeBackend,
                    low=0.0,
                    high=1.0,
                    shape=fraction_shape,
                    dtype=self.world.dtype,
                    device=self.world.device,
                    metadata=metadata("task", label="goal open fraction of the joint", **fraction_md),
                ),
                "handle_position": BoxSpace(
                    PyTorchComputeBackend,
                    low=-float("inf"),
                    high=float("inf"),
                    shape=get_batch_aware_shape(self.world.batch_size, 3),
                    dtype=self.world.dtype,
                    device=self.world.device,
                    metadata=position_metadata(
                        "task", frame="world", label="handle link bounding-box centre",
                        furniture=furniture_key, joint=joint_name,
                    ),
                ),
            },
            device=self.world.device,
        )

    # ------------------------------------------------------------------
    # Identity / construction helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def handle_link_name(self) -> Optional[str]:
        """Resolved grasp/handle link name (``None`` before binding)."""
        return self._handle_link_name

    def _phase_priority(
        self,
        room_attr: str,
        priorities: Dict[str, Optional[int]],
        key: str,
    ) -> int:
        """Explicit priority if given, else ``max(room_builder.<attr>) + 1``."""
        explicit = priorities.pop(key, None)
        if explicit is not None:
            return int(explicit)
        room_priorities = set(getattr(self.room_builder, room_attr, set()) or set())
        if not room_priorities:
            return self.DEFAULT_RESET_PRIORITY
        return max(room_priorities) + 1

    def _unknown_furniture_error(self, resolved: Dict[str, Any]) -> KeyError:
        return KeyError(
            f"Unknown furniture key {self.furniture_key!r} for room "
            f"{self.room_builder.name!r}; available furniture: {sorted(resolved)}"
        )

    def _reseed(self, seed: int) -> None:
        self.rng.manual_seed(seed)

    def _assert_bound(self) -> None:
        if self.descriptor is None or self.entity is None or not self._dofs_idx:
            raise self._not_bound_error()

    def _not_bound_error(self) -> RuntimeError:
        return RuntimeError(
            f"{type(self).__name__} {self._name!r} is not bound to a built "
            f"furniture joint yet — reload the environment first "
            "(RoomBuilder.reload at its priority, then this task's reload / "
            "after_reload)."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        """Invalidate the previous binding and reseed for this (re)build.

        Nothing is resolved here: this phase runs while nodes are still adding
        entities (and, at a priority above the room's, *before*
        :class:`RoomBuilder` has rebuilt the layout), so the descriptor, the
        entity and every joint/link name are bound in :meth:`after_reload`,
        once the world has compiled the scene.
        """
        if priority not in self.reload_priorities:
            return
        if seed is not None:
            self._reseed(seed)

        # The previous reload's entities are gone (the world rebuilt the scene).
        self.descriptor = None
        self.entity = None
        self._dofs_idx = []
        self._handle_link_name = None
        self._handle_offset = None

    def after_reload(self, *, priority: int = 0, mask=None):
        """Resolve joint/link names on the compiled entity, then reset."""
        if priority not in self.after_reload_priorities:
            return
        self._bind_runtime()
        # Initialise the first episode (reload does not run reset by itself).
        self.reset(priority=min(self.reset_priorities), seed=None, mask=None)

    def _bind_runtime(self) -> None:
        """Address the joint and the handle link BY NAME on the built entity."""
        furniture = self.room_builder.furniture
        if self.furniture_key not in furniture:
            raise self._unknown_furniture_error(furniture)
        self.descriptor = furniture[self.furniture_key]

        entities = self.room_builder.furniture_entities
        if self.furniture_key not in entities:
            raise KeyError(
                f"Room {self.room_builder.name!r} has no built entity for "
                f"furniture key {self.furniture_key!r}; available: "
                f"{sorted(entities)}"
            )
        self.entity = entities[self.furniture_key]

        # Joint: DOF indexes resolved off the built entity (the resulting
        # ArticulationError lists every available joint name).
        self._dofs_idx = list(resolve_joint_dofs(self.entity, self.joint_name))
        if not self._dofs_idx:
            raise ArticulationError(
                f"Joint {self.joint_name!r} of furniture "
                f"{self.furniture_key!r} has no DOFs (fixed joint) — it cannot "
                "be driven to an open fraction."
            )

        # Handle link: the caller's name, else the joint's child link as
        # declared by the built entity (again resolved by name, not baked).
        if self.grasp_link_name is not None:
            link_name = self.grasp_link_name
        else:
            link_name = next(
                joint.link.name
                for joint in self.entity.joints
                if joint.name == self.joint_name
            )
        resolve_link_index(self.entity, link_name)  # fails loud on unknown links
        self._handle_link_name = link_name

        self._cache_handle_offset()

    def _cache_handle_offset(self) -> None:
        """Cache the handle bbox centre as a point rigidly attached to the link.

        The descriptor's per-link bbox is the **offline** AABB of that link's
        visual mesh in the base-link frame at the reference configuration
        (joints at ``reference_qpos``, zero by default), so its centre maps to
        one fixed point of the grasp link's own frame.  Measuring it here — in
        ``after_reload``, before any reset touches the joints, i.e. at the
        reference configuration — and expressing it in the link frame lets
        :meth:`_handle_position` track a swinging door / sliding drawer with a
        single rigid transform: ``handle = link_pos + R_link · offset``.

        Links without an offline visual mesh (e.g. an intermediate hinge body
        carrying only sites) fall back to the link frame origin (zero offset).
        """
        from xbarray.transformations.rotation_conversions.pytorch import (
            quaternion_apply,
            quaternion_invert,
        )

        link_name = self._handle_link_name
        p_link, q_link = self.descriptor.get_link_pose(self.entity, link_name)

        try:
            center_local = bbox_center(
                self.descriptor.get_link_local_bounding_box(link_name)
            )
        except ArticulationError:
            center_local = torch.zeros(3)
        # Local-frame descriptor data is host-side; move it to the world device.
        center_local = torch.as_tensor(
            center_local, dtype=self.world.dtype, device=self.world.device
        )

        # Bring the base-frame centre into the world frame with the entity's
        # (base-link) pose — the same convention as get_bounding_box().
        base_pos = torch.as_tensor(self.entity.get_pos()).to(
            dtype=self.world.dtype, device=self.world.device
        )
        base_quat = torch.as_tensor(self.entity.get_quat()).to(
            dtype=self.world.dtype, device=self.world.device
        )
        center_world = base_pos + quaternion_apply(base_quat, center_local)
        offset = quaternion_apply(quaternion_invert(q_link), center_world - p_link)
        self._handle_offset = offset.reshape(p_link.shape)

    # ------------------------------------------------------------------
    # Fraction-space / handle reads
    # ------------------------------------------------------------------

    def _current_fraction(self, envs_idx=None) -> torch.Tensor:
        """Open fraction of the driven joint, shape ``([B],)`` (``()`` unbatched).

        ``ArticulatedObject.get_open_fraction`` returns ``([B], n_dofs)``; the
        DOF axis is reduced by the mean — the identity for the single-DOF
        hinge/slide joints furniture uses, and consistent with
        :meth:`ArticulatedObject.set_open_fraction`, which broadcasts one
        fraction across all of the joint's DOFs.
        """
        self._assert_bound()
        fraction = self.descriptor.get_open_fraction(
            self.entity, self.joint_name, envs_idx=envs_idx
        )
        mean = fraction.reshape(-1, len(self._dofs_idx)).mean(dim=-1)
        if self.world.batch_size is None:
            # Unbatched worlds carry no environment axis at all.
            mean = mean.reshape(())
        return mean

    def _handle_position(self) -> torch.Tensor:
        """World position of the handle link's bbox centre, shape ``([B], 3)``."""
        self._assert_bound()
        from xbarray.transformations.rotation_conversions.pytorch import quaternion_apply

        p_link, q_link = self.descriptor.get_link_pose(
            self.entity, self._handle_link_name
        )
        return p_link + quaternion_apply(q_link, self._handle_offset)

    def get_handle_joint_name(self) -> str:
        """Name of the driven joint (the runtime addressing key)."""
        return self.joint_name

    def get_handle_link_name(self) -> Optional[str]:
        """Resolved handle link name (``None`` before the task is bound)."""
        return self._handle_link_name

    def get_target_fraction(self) -> Optional[torch.Tensor]:
        """Current per-env goal fraction, shape ``([B],)``."""
        return self._target_fraction

    # ------------------------------------------------------------------
    # Reference-policy hook (plan §4)
    # ------------------------------------------------------------------

    def get_handle_grasp_candidates(
        self,
        gripper_width: float = DEFAULT_INTERACTION_GRIPPER_WIDTH,
        gripper_depth: float = 0.01,
        envs_idx=None,
    ) -> GraspCandidates:
        """World-frame grasp candidates on the handle link, for a reference policy.

        A thin pass-through to
        :meth:`ArticulatedObject.list_part_grasping_candidates` on the resolved
        handle link (``grasp_link_name`` or the joint's child link), so an
        existing ``GraspPolicy`` / ``GotoPolicy`` from
        ``unienv_genesis_refpolicy`` can reach the handle.  The candidates are
        transformed by the link's **current** pose, so they track the part as
        the joint moves.

        Args:
            gripper_width: Maximum gripper opening (metres).
            gripper_depth: Gripper finger depth (metres).
            envs_idx: Optional ``(batch_size,)`` ``torch.bool`` mask; only the
                selected environments contribute poses.

        Returns:
            :class:`GraspCandidates` of shape ``(N, 3)`` unbatched,
            ``([B], N, 3)`` batched (``(n_selected, N, 3)`` when masked).
        """
        self._assert_bound()
        return self.descriptor.list_part_grasping_candidates(
            self.entity,
            self._handle_link_name,
            float(gripper_width),
            gripper_depth=float(gripper_depth),
            envs_idx=envs_idx,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_unit(self) -> torch.Tensor:
        """Uniform ``[0, 1)`` samples with the batch-aware fraction shape."""
        return torch.rand(
            self._fraction_shape,
            generator=self.rng,
            dtype=self.world.dtype,
            device=self.world.device,
        )

    def sample_goal_and_initial(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample per-env goal fractions and the matching initial fractions.

        Returns:
            ``(target_fraction, initial_fraction)``, each of the batch-aware
            fraction shape ``([B],)``.  ``initial`` is drawn uniformly from
            ``[0, target - success_threshold]`` so every episode starts outside
            the success band and asks for an *opening* motion (0 when the
            target is already closer to closed than the threshold).
        """
        low, high = self.target_fraction_range
        target = low + self._sample_unit() * (high - low)
        headroom = torch.clamp(target - self.success_threshold, min=0.0, max=1.0)
        initial = self._sample_unit() * headroom
        return target, initial

    # ------------------------------------------------------------------
    # Reset (masked semantics identical to pickplace)
    # ------------------------------------------------------------------

    def reset(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        """Sample goal fractions and randomise the joint's initial state."""
        if priority not in self.reset_priorities:
            return
        self._assert_bound()

        if seed is not None:
            self._reseed(seed)

        target, initial = self.sample_goal_and_initial()

        if mask is None:
            self._target_fraction = target
            self.descriptor.set_open_fraction(
                self.entity, self.joint_name, initial, envs_idx=None
            )
        else:
            # Reset masks are strictly ``(batch_size,)`` torch.bool tensors on
            # the world's device — validated here at the reset boundary.
            n_envs = self.world.batch_size
            assert n_envs is not None, "Masking only supported in batched environments"
            assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool, (
                f"Reset mask must be a torch.bool tensor, got "
                f"{type(mask)} with dtype {getattr(mask, 'dtype', None)}"
            )
            assert mask.shape == (n_envs,), (
                f"Reset mask must have shape ({n_envs},), got {tuple(mask.shape)}"
            )
            assert mask.device == self.world.device, (
                f"Reset mask must live on the world device "
                f"{self.world.device}, got {mask.device}"
            )
            # The boolean mask is used consistently: for task-state indexing,
            # for selecting the sampled rows, and as the Genesis envs_idx.
            if mask.any():
                # Only masked envs get a new goal / joint state; unmasked envs
                # keep their current goal and articulation.
                self._target_fraction[mask] = target[mask]
                self.descriptor.set_open_fraction(
                    self.entity, self.joint_name, initial[mask], envs_idx=mask
                )

        if mask is None:
            self._reward = (
                0.0
                if self.world.batch_size is None
                else torch.zeros(
                    self.world.batch_size, dtype=self.world.dtype, device=self.world.device
                )
            )
            self._terminated = (
                False
                if self.world.batch_size is None
                else torch.zeros(
                    self.world.batch_size, dtype=torch.bool, device=self.world.device
                )
            )
        else:
            # Preserve reward / termination outside the mask.
            self._reward.masked_fill_(mask, 0.0)
            self._terminated.masked_fill_(mask, False)

    # ------------------------------------------------------------------
    # Step / signals
    # ------------------------------------------------------------------

    def post_environment_step(self, dt, *, priority: int = 0):
        """Dense fraction-tracking reward + success termination."""
        if priority not in self.post_environment_step_priorities:
            return

        error = torch.abs(self._current_fraction() - self._target_fraction)
        if self.world.batch_size is None:
            error = float(error.item())

        self._reward = -error
        self._terminated = error < self.success_threshold

    def get_observation(self):
        return {
            "joint_position": self._current_fraction(),
            "target_fraction": self._target_fraction,
            "handle_position": self._handle_position(),
        }

    def get_reward(self) -> Union[float, torch.Tensor]:
        return self._reward

    def get_termination(self) -> Union[bool, torch.Tensor]:
        return self._terminated

    def get_info(self) -> Dict[str, Any]:
        """Auxiliary episode information (pickplace's ``get_info`` contract).

        Returns:
            A dictionary with keys:

            - ``"furniture_key"`` / ``"furniture_object_id"`` — the room's
              instance key and the catalog id placed under it.
            - ``"furniture_name"`` — technical identifier of the descriptor
              (:attr:`ArticulatedObject.name`).
            - ``"joint_name"`` / ``"handle_link_name"`` — runtime addressing
              names of the driven joint and the handle link.
            - ``"furniture_annotation"`` — the first language annotation of the
              furniture descriptor, or ``None`` when it has none.
            - ``"furniture_annotations"`` — the full
              :attr:`ArticulatedObject.language_annotations` tuple.
            - ``"target_fraction_range"`` / ``"success_threshold"`` — task
              parameters.
        """
        descriptor = self.descriptor
        annotations = tuple(descriptor.language_annotations) if descriptor is not None else ()
        placement = self.room_builder.placements.get(self.furniture_key)
        return {
            "furniture_key": self.furniture_key,
            "furniture_object_id": getattr(placement, "object_id", None),
            "furniture_name": descriptor.name if descriptor is not None else None,
            "joint_name": self.joint_name,
            "handle_link_name": self._handle_link_name,
            "furniture_annotation": annotations[0] if annotations else None,
            "furniture_annotations": annotations,
            "target_fraction_range": self.target_fraction_range,
            "success_threshold": self.success_threshold,
        }
