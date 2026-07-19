"""PickPlaceRefPolicy — pick-and-place an object to a goal position.

A companion reference policy for :class:`ObjectRelocateTask`.  Executes a
multi-phase pick-and-place sequence using a :class:`GraphRefPolicy`:

    grasp → lift → move_to_goal → lower → release → retreat → (success)

Each motion phase resolves its target from callables at :meth:`reset`
time, so targets depending on the current state or the task's
per-episode goal are resolved when the node is entered — no factory
callables needed. The transport phases (``lift`` / ``move_to_goal`` /
``lower``) use :class:`MoveObjectLambdaPolicy` to preserve the current
object→EEF grasp transform, while ``release`` and ``retreat`` use
:class:`GoToLambdaRefPolicy`.

Phase details
-------------

1. **grasp** (:class:`BoxGripperGraspPolicy`) — approach, grasp the target
   object, and close the gripper.
2. **lift** (:class:`MoveObjectLambdaPolicy`) — lift the grasped
   object straight up by ``lift_height`` metres from its current
   position while preserving the current object→EEF grasp transform.
3. **move_to_goal** (:class:`MoveObjectLambdaPolicy`) — move
   horizontally to a point directly above the goal, at ``lift_height``
   above the table surface, while preserving the grasp transform.
4. **lower** (:class:`MoveObjectLambdaPolicy`) — lower the object onto
   the goal position on the table surface while preserving the grasp
   transform.
5. **release** (:class:`RepeatActionRefPolicy`) — open the gripper for
   ``release_seconds`` to drop the object.
6. **retreat** (:class:`GoToLambdaRefPolicy`) — retreat straight up by
   ``retreat_height`` metres from the goal position.

Failure from any motion phase terminates the graph as a failure.  The
``grasp`` phase has its own internal failure detection (stall, unreachable
IK, no contact).

Example::

    from unienv_genesis_refpolicy.implementations.pickplace_policy import PickPlaceRefPolicy

    task = ObjectRelocateTask(world, table_builder=table, ...)
    policy = PickPlaceRefPolicy(
        world,
        eef_controller=arm,
        gripper_controller=gripper,
        task=task,
        target_object=target_cube,
        target_entity=task.target_object_entity,
    )
    env.reset(seed=0, reload=True)
    policy.reset(seed=0)
    for _ in range(max_steps):
        action = policy.get_action()
        env.step(action)
        policy.step()
        if policy.get_success() or policy.get_failure():
            break
"""

from typing import Any, Dict, List, Optional, Union

import torch

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers.end_effector_pose import (
    EndEffectorPoseController,
)
from unienv_genesis.nodes.entities.entity_controllers.base import EntityController
from unienv_genesis_collection.objects.base_object import ManipulatableObject
from unienv_genesis_refpolicy.base.policy import BaseRefPolicy
from unienv_genesis_refpolicy.implementations.goto_lambda_policy import GoToLambdaRefPolicy
from unienv_genesis_refpolicy.implementations.grasp_policy import BoxGripperGraspPolicy
from unienv_genesis_refpolicy.implementations.move_object_lambda_policy import (
    MoveObjectLambdaPolicy,
)
from unienv_genesis_refpolicy.implementations.repeat_action_policy import RepeatActionRefPolicy
from unienv_genesis_refpolicy.implementations.combined_policy import CombinedRefPolicy
from unienv_genesis_refpolicy.implementations.graph_policy import GraphRefPolicy


class PickPlaceRefPolicy(BaseRefPolicy[Dict[str, torch.Tensor]]):
    """Pick-and-place an object to a goal position read from a task.

    Wraps a :class:`GraphRefPolicy` with the phase sequence
    ``grasp → lift → move_to_goal → lower → release → retreat``.

    Args:
        world: Genesis world instance.
        eef_controller: :class:`EndEffectorPoseController` for the arm.
        gripper_controller: Controller for the gripper.
        task: :class:`ObjectRelocateTask` (or any object exposing
            ``_goal_pos`` and ``target_object_entity``).
        target_object: :class:`ManipulatableObject` descriptor for the
            object to grasp.
        target_entity: The loaded Genesis entity for *target_object*
            (typically ``task.target_object_entity``).
        eef_name: Key for the arm action in the returned dict.
        gripper_name: Key for the gripper action in the returned dict.
        lift_height: Height (m) to lift the object above its initial
            position after grasping.
        retreat_height: Height (m) to retreat upward after releasing.
        release_seconds: Seconds to hold the gripper open during release.
        grasp_kwargs: Optional extra kwargs forwarded to
            :class:`BoxGripperGraspPolicy`.
        goto_kwargs: Optional shared baseline kwargs forwarded to *every*
            motion phase (:class:`GoToLambdaRefPolicy` and
            :class:`MoveObjectLambdaPolicy`). Use this for options that
            should apply uniformly, such as ``position_threshold``,
            ``orientation_threshold``, ``stall_seconds``, ``max_seconds``.
        transport_kwargs: Optional kwargs applied *only* to the transport
            :class:`MoveObjectLambdaPolicy` phases (``lift``,
            ``move_to_goal``, ``lower``). These override ``goto_kwargs``
            when the same key is present in both. Use this for
            MoveObject-only options such as ``slip_position_threshold`` /
            ``slip_orientation_threshold`` (which ``GoToLambdaRefPolicy``
            does not accept).
        release_kwargs: Optional kwargs applied *only* to the ``release``
            phase's arm :class:`GoToLambdaRefPolicy`, overriding
            ``goto_kwargs``.
        retreat_kwargs: Optional kwargs applied *only* to the ``retreat``
            phase's arm :class:`GoToLambdaRefPolicy`, overriding
            ``goto_kwargs``.
        transport_slip_detection: Whether slip detection is enabled on the
            transport (``lift`` / ``move_to_goal`` / ``lower``)
            :class:`MoveObjectLambdaPolicy` phases at construction time.
            Defaults to ``True`` (current behavior). When ``False``, the
            transport phases start with slip detection disabled (thresholds
            ``0.0`` and ``track_relative_pose=False``), exactly as if
            :meth:`set_transport_slip_detection` had been called with
            ``enabled=False`` right after construction. The
            ``release`` / ``retreat`` phases are never affected, and the
            construction-time slip defaults are still recorded so the
            runtime setter can re-enable detection later.
    """

    def __init__(
        self,
        world: GenesisWorld,
        eef_controller: EndEffectorPoseController,
        gripper_controller: EntityController,
        task: Any,
        target_object: ManipulatableObject,
        target_entity: Any = None,
        eef_name: str = "arm",
        gripper_name: str = "gripper",
        lift_height: float = 0.15,
        retreat_height: float = 0.15,
        release_seconds: float = 1.0,
        grasp_kwargs: Optional[Dict[str, Any]] = None,
        goto_kwargs: Optional[Dict[str, Any]] = None,
        transport_kwargs: Optional[Dict[str, Any]] = None,
        release_kwargs: Optional[Dict[str, Any]] = None,
        retreat_kwargs: Optional[Dict[str, Any]] = None,
        transport_slip_detection: bool = True,
    ) -> None:
        super().__init__(world)
        self.controller = eef_controller
        self.gripper_controller = gripper_controller
        self.task = task
        self.target_object = target_object
        self.target_entity = target_entity
        self._eef_name = eef_name
        self._gripper_name = gripper_name
        self.lift_height = lift_height
        self.retreat_height = retreat_height
        self.release_seconds = release_seconds

        grasp_kwargs = dict(grasp_kwargs or {})
        goto_kwargs = dict(goto_kwargs or {})
        # Phase-specific kwargs override the shared goto_kwargs baseline.
        transport_merged = {**goto_kwargs, **(transport_kwargs or {})}
        release_merged = {**goto_kwargs, **(release_kwargs or {})}
        retreat_merged = {**goto_kwargs, **(retreat_kwargs or {})}

        # Direct references to the transport MoveObjectLambdaPolicy instances
        # and their construction-time slip defaults, used by
        # :meth:`set_transport_slip_detection`.
        self._transport_policies: List[MoveObjectLambdaPolicy] = []
        self._transport_slip_defaults: List[Dict[str, Any]] = []

        # Resolve gripper open action for the release phase.
        gripper_open_action = grasp_kwargs.pop(
            "gripper_open_action",
            getattr(gripper_controller, "gripper_open_action", -1.0),
        )

        # Device-correct offset tensors for lambda targets.
        device = world.device
        lift_offset = torch.tensor([0.0, 0.0, lift_height], dtype=torch.float32, device=device)
        retreat_offset = torch.tensor([0.0, 0.0, retreat_height], dtype=torch.float32, device=device)

        # Gripper hold actions for transport / release phases.
        gripper_close = torch.tensor(
            [getattr(gripper_controller, "gripper_close_action", 1.0)],
            dtype=torch.float32, device=device,
        )
        gripper_open = torch.tensor(
            [gripper_open_action], dtype=torch.float32, device=device,
        )

        # --- Build the graph ---
        nodes: Dict[str, Any] = {}

        def _get_target_entity():
            return target_entity if target_entity is not None else task.target_object_entity

        def _require_target_entity():
            entity = _get_target_entity()
            if entity is None:
                raise RuntimeError(
                    "PickPlaceRefPolicy: target_entity is None. Call "
                    "env.reset(reload=True) before policy.reset() so the "
                    "task can load the target object into the scene."
                )
            return entity

        # 1. Grasp (factory — target_entity may be None until env.reset loads it)
        def make_grasp():
            return BoxGripperGraspPolicy(
                world=world,
                eef_controller=eef_controller,
                gripper_controller=gripper_controller,
                target_object=target_object,
                target_entity=_require_target_entity(),
                eef_name=eef_name,
                gripper_name=gripper_name,
                **grasp_kwargs,
            )

        nodes["grasp"] = make_grasp

        # Helper: wrap an arm GoToLambdaRefPolicy with a gripper hold action
        # so the combined controller receives both "arm" and "gripper" keys.
        def _motion_node(goto_policy, gripper_action):
            return CombinedRefPolicy(
                world,
                {
                    eef_name: goto_policy,
                    gripper_name: RepeatActionRefPolicy(
                        world=world,
                        action=gripper_action,
                        steps=0,  # never signals success on its own
                        controller=gripper_controller,
                    ),
                },
                success_keys=[eef_name],
            )

        # Helper: build a transport (MoveObject) phase node, register the
        # underlying policy for later slip configuration, and capture its
        # construction-time slip defaults.
        def _transport_node(target_pos_fn):
            move_object_policy = MoveObjectLambdaPolicy(
                world=world,
                controller=eef_controller,
                object_entity_fn=_get_target_entity,
                target_pos_fn=target_pos_fn,
                gripper_controller=gripper_controller,
                fail_on_gripper_contact_loss=True,
                **transport_merged,
            )
            self._transport_policies.append(move_object_policy)
            self._transport_slip_defaults.append({
                "slip_position_threshold": move_object_policy.slip_position_threshold,
                "slip_orientation_threshold": move_object_policy.slip_orientation_threshold,
                "track_relative_pose": move_object_policy.track_relative_pose,
            })
            return _motion_node(move_object_policy, gripper_close)

        # 2. Lift — move the grasped object upward while preserving the current
        # grasp transform (gripper closed).
        nodes["lift"] = _transport_node(
            lambda: _require_target_entity().get_pos().detach().float() + lift_offset,
        )

        # 3. Move to goal — move the grasped object above the goal while
        # preserving the grasp transform (gripper closed).
        nodes["move_to_goal"] = _transport_node(
            lambda: task._goal_pos.detach().float() + lift_offset,
        )

        # 4. Lower — move the grasped object to the goal position while
        # preserving the grasp transform (gripper closed).
        nodes["lower"] = _transport_node(
            lambda: task._goal_pos.detach().float(),
        )

        # 5. Release (hold arm at its current EEF pose, open gripper for N steps).
        # The arm just holds the pose it reached at the end of `lower`; it must
        # NOT depend on object-contact tracking, since the gripper is opening
        # and the object is being dropped. Orientation is preserved by relying
        # on GoToLambdaRefPolicy's default current-orientation behavior.
        release_steps = max(1, int(release_seconds / float(world.world_timestep)))
        nodes["release"] = CombinedRefPolicy(
            world,
            {
                eef_name: GoToLambdaRefPolicy(
                    world=world,
                    controller=eef_controller,
                    target_pos_fn=lambda: eef_controller.get_eef_pose_absolute()[0].detach().float(),
                    **release_merged,
                ),
                gripper_name: RepeatActionRefPolicy(
                    world=world,
                    action=gripper_open,
                    steps=release_steps,
                    controller=gripper_controller,
                ),
            },
            success_keys=[gripper_name],
        )

        # 6. Retreat — after release, retreat the EEF upward from its current
        # pose (gripper open). This is EEF-centric, not object-centric.
        nodes["retreat"] = _motion_node(
            GoToLambdaRefPolicy(
                world=world,
                controller=eef_controller,
                target_pos_fn=lambda: eef_controller.get_eef_pose_absolute()[0].detach().float() + retreat_offset,
                **retreat_merged,
            ),
            gripper_open,
        )

        # Success edges: grasp → lift → move_to_goal → lower → release → retreat → (terminal)
        success_edges = [
            ("grasp", "lift"),
            ("lift", "move_to_goal"),
            ("move_to_goal", "lower"),
            ("lower", "release"),
            ("release", "retreat"),
        ]

        # No explicit fail edges — any failure terminates the graph as failure.
        fail_edges = []

        self._graph = GraphRefPolicy(
            world=world,
            nodes=nodes,
            success_edges=success_edges,
            fail_edges=fail_edges,
            initial="grasp",
        )

        # Apply the requested initial transport slip-detection state. This
        # goes through the same code path as the runtime setter so both stay
        # consistent; the recorded construction-time defaults are preserved
        # either way, and release/retreat are never touched.
        if not transport_slip_detection:
            self.set_transport_slip_detection(False)

    # --- Lifecycle delegation ---

    def set_transport_slip_detection(
        self,
        enabled: bool,
        *,
        position_threshold: Optional[float] = None,
        orientation_threshold: Optional[float] = None,
        track_relative_pose: Optional[bool] = None,
    ) -> None:
        """Configure slip detection on the transport (MoveObject) phases only.

        This is the public, supported way to toggle slip detection on the
        ``lift`` / ``move_to_goal`` / ``lower`` transport phases without
        reaching into the underlying graph.  The ``release`` and ``retreat``
        (GoToLambdaRefPolicy) phases are *not* affected — they do not perform
        object slip tracking.

        Args:
            enabled: When ``False``, slip detection is disabled on the
                transport phases by defaulting the thresholds to ``0.0`` and
                ``track_relative_pose`` to ``False`` (unless explicitly
                overridden below). When ``True`` with no explicit overrides,
                the transport phases are restored to their construction-time
                slip defaults (i.e. the values ultimately resolved from
                ``goto_kwargs`` / ``transport_kwargs`` at ``__init__`` time).
            position_threshold: Optional override for
                ``slip_position_threshold``. ``None`` means use the disabling
                default (``0.0``) when disabling, or the construction-time
                default when enabling.
            orientation_threshold: Optional override for
                ``slip_orientation_threshold`` (same ``None`` semantics).
            track_relative_pose: Optional override for
                ``track_relative_pose`` (same ``None`` semantics; disabling
                default is ``False``).
        """
        for policy, defaults in zip(self._transport_policies, self._transport_slip_defaults):
            if enabled:
                pos = (
                    defaults["slip_position_threshold"]
                    if position_threshold is None
                    else float(position_threshold)
                )
                orn = (
                    defaults["slip_orientation_threshold"]
                    if orientation_threshold is None
                    else float(orientation_threshold)
                )
                trk = (
                    defaults["track_relative_pose"]
                    if track_relative_pose is None
                    else bool(track_relative_pose)
                )
            else:
                pos = 0.0 if position_threshold is None else float(position_threshold)
                orn = 0.0 if orientation_threshold is None else float(orientation_threshold)
                trk = False if track_relative_pose is None else bool(track_relative_pose)
            policy.slip_position_threshold = pos
            policy.slip_orientation_threshold = orn
            policy.track_relative_pose = trk

    def reset(self, *, mask=None, seed=None) -> None:
        self._graph.reset(mask=mask, seed=seed)

    def reload(self, *, mask=None, seed=None) -> None:
        self._graph.reload(mask=mask, seed=seed)

    def step(self, *, mask=None) -> None:
        self._graph.step(mask=mask)

    def get_action(self) -> Dict[str, torch.Tensor]:
        return self._graph.get_action()

    def convert_action_dict(
        self, action: Dict[str, torch.Tensor], dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._graph.convert_action_dict(action, dt=dt)

    def get_action_dict(self, dt: Optional[float] = None) -> Dict[str, Any]:
        return self._graph.get_action_dict(dt=dt)

    def get_action_dict_space(self):
        return self._graph.get_action_dict_space()

    def get_success(self) -> Union[torch.Tensor, bool]:
        return self._graph.get_success()

    def get_failure(self) -> Union[torch.Tensor, bool]:
        return self._graph.get_failure()

    def get_timeout(self) -> Union[torch.Tensor, bool]:
        return self._graph.get_timeout()
