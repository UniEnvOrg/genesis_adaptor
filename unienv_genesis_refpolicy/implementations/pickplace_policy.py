"""PickPlaceRefPolicy — pick-and-place an object to a goal position.

A companion reference policy for :class:`ObjectRelocateTask`.  Executes a
multi-phase pick-and-place sequence using a :class:`GraphRefPolicy`:

    grasp → lift → move_to_goal → lower → release → retreat → (success)

Each motion phase uses :class:`GoToLambdaRefPolicy` with callables that
resolve the target at :meth:`reset` time, so targets depending on the
current state or the task's per-episode goal are resolved when the node
is entered — no factory callables needed.

Phase details
-------------

1. **grasp** (:class:`BoxGripperGraspPolicy`) — approach, grasp the target
   object, and close the gripper.
2. **lift** (:class:`GoToLambdaRefPolicy`) — lift the grasped object
   straight up by ``lift_height`` metres from its current position.
3. **move_to_goal** (:class:`GoToLambdaRefPolicy`) — move horizontally
   to a point directly above the goal, at ``lift_height`` above the table
   surface.
4. **lower** (:class:`GoToLambdaRefPolicy`) — lower the object onto the
   goal position on the table surface.
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

from typing import Any, Dict, Optional, Union

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
        goto_kwargs: Optional extra kwargs forwarded to each
            :class:`GoToLambdaRefPolicy` motion phase
            (``position_threshold``, ``orientation_threshold``,
            ``stall_seconds``, ``max_seconds``, …).
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

        # 1. Grasp (factory — target_entity may be None until env.reset loads it)
        def make_grasp():
            entity = target_entity if target_entity is not None else task.target_object_entity
            if entity is None:
                raise RuntimeError(
                    "PickPlaceRefPolicy: target_entity is None. Call "
                    "env.reset(reload=True) before policy.reset() so the "
                    "task can load the target object into the scene."
                )
            return BoxGripperGraspPolicy(
                world=world,
                eef_controller=eef_controller,
                gripper_controller=gripper_controller,
                target_object=target_object,
                target_entity=entity,
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

        # 2. Lift — target is object position + lift_offset (gripper closed).
        nodes["lift"] = _motion_node(
            GoToLambdaRefPolicy(
                world=world,
                controller=eef_controller,
                target_pos_fn=lambda: task.target_object_entity.get_pos().detach().float() + lift_offset,
                **goto_kwargs,
            ),
            gripper_close,
        )

        # 3. Move to goal — target is goal_pos + lift_offset (gripper closed).
        nodes["move_to_goal"] = _motion_node(
            GoToLambdaRefPolicy(
                world=world,
                controller=eef_controller,
                target_pos_fn=lambda: task._goal_pos.detach().float() + lift_offset,
                **goto_kwargs,
            ),
            gripper_close,
        )

        # 4. Lower — target is the goal position (gripper closed).
        nodes["lower"] = _motion_node(
            GoToLambdaRefPolicy(
                world=world,
                controller=eef_controller,
                target_pos_fn=lambda: task._goal_pos.detach().float(),
                **goto_kwargs,
            ),
            gripper_close,
        )

        # 5. Release (hold arm at goal position, open gripper for N steps)
        release_steps = max(1, int(release_seconds / float(world.world_timestep)))
        nodes["release"] = CombinedRefPolicy(
            world,
            {
                eef_name: GoToLambdaRefPolicy(
                    world=world,
                    controller=eef_controller,
                    target_pos_fn=lambda: task._goal_pos.detach().float(),
                    **goto_kwargs,
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

        # 6. Retreat — target is goal_pos + retreat_offset (gripper open).
        nodes["retreat"] = _motion_node(
            GoToLambdaRefPolicy(
                world=world,
                controller=eef_controller,
                target_pos_fn=lambda: task._goal_pos.detach().float() + retreat_offset,
                **goto_kwargs,
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

    # --- Lifecycle delegation ---

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
