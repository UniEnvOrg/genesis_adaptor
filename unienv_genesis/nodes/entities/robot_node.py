"""Robot entity node for loading URDF/MJCF/USD files."""

from typing import Literal, Optional, Dict, Any, List, Tuple, Union, Sequence
import genesis as gs
from genesis.engine.entities.base_entity import Entity as BaseEntity
from genesis.engine import entities as gs_entities
import torch
import logging
import os
import abc
from dataclasses import dataclass
from unienv_interface.space import Space, DictSpace, BoxSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_interface.world.nodes.flat_combined_node import FlatCombinedWorldNode
from .base import EntityNode, ContextType, ObsType, ActType
from .entity_controllers.base import EntityController
from .entity_controllers import CONTROL_MODE_MAP, CONTROL_MODE_T

__all__ = [
    "RobotKeyframe",
    "RobotDescriptionNode",
]

LOGGER = logging.getLogger(__name__)

@dataclass
class RobotKeyframe:
    """Keyframe for robot pose."""
    pos: Optional[PyTorchArrayType] = None # Shape (3,)
    quat: Optional[PyTorchArrayType] = None # (w, x, y, z)
    joint_position: Optional[PyTorchArrayType] = None # Shape (n_dofs,)
    joint_velocity: Optional[PyTorchArrayType] = None # Shape (n_dofs,)

class RobotDescriptionNode(EntityNode[
    BaseEntity,
    ContextType,
    ObsType,
    ActType,
], abc.ABC):
    """Node for loading robot models (URDF/MJCF/USD)."""
    reload_priorities = {50}
    reset_priorities = {50}

    after_reload_priorities = {50, 0}
    after_reset_priorities = {50, 0}
    # Triggers agent `qpos` reset before any normal nodes are being reset

    pre_environment_step_priorities = {0}
    post_environment_step_priorities = {0}

    def initialize_entity_from_description_file(self, description_file: Union[str, os.PathLike]) -> BaseEntity:
        """Load a Genesis Morph from a description file."""
        # Build optional pos/quat kwargs for the morph
        pose_kwargs: Dict[str, Any] = {}
        if self.pos is not None:
            pose_kwargs["pos"] = self.pos
        if self.quat is not None:
            pose_kwargs["quat"] = self.quat

        description_base = os.path.basename(description_file)
        if description_base.endswith('.urdf'):
            morph = gs.morphs.URDF(file=description_file, fixed=self.fix_root, **pose_kwargs)
        elif description_base.endswith('.xml'):
            with open(description_file, "r") as f:
                f_content = f.read()
                has_frejoint = f_content.find("freejoint") != -1
                if has_frejoint and self.fix_root:
                    logging.warning("`self.fix_root` is set to True, but the MJCF has `freejoint` attributes. By default Genesis will always follow the MJCF specification, which may cause unexpected behaviors")
                elif not has_frejoint and not self.fix_root:
                    logging.warning("`self.fix_root` is set to False, but the MJCF does not have `freejoint` attributes. By default Genesis will always follow the MJCF specification, which may cause unexpected behaviors")

            morph = gs.morphs.MJCF(file=description_file, **pose_kwargs)
        elif description_base.endswith('.usd') or description_base.endswith('.usda') or description_base.endswith('.usdc'):
            morph = gs.morphs.USD(file=description_file, fixed=self.fix_root, requires_jac_and_IK=True, **pose_kwargs)
        else:
            raise ValueError(f"Unsupported robot description file format: {description_file}")
        entity = self.world.scene.add_entity(morph, name=self._name)

        return entity

    @property
    def observation_space(self) -> Optional[Space[ObsType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]]:
        return self.controller.observation_space if self.controller is not None else None

    @property
    def action_space(self) -> Optional[Space[ActType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]]:
        return self.controller.action_space if self.controller is not None else None

    @property
    def max_joint_pos_vel(self) -> Optional[Any]:
        """Effective position rate limit for this robot's controller.

        Returns a tensor of shape ``([batch_size], n_dofs)`` representing the maximum allowed change in joint position per control step. Note that `batch_size` dimension is always consistent with the batch size of the world, if applicable.  Unconstrained joints have ``inf`` entries.

        Returns ``None`` only when the controller does not expose a usable
        rate limit (e.g. a sub-controller without ``max_joint_pos_vel``) or
        no value was configured.
        """
        return getattr(self.controller, 'max_joint_pos_vel', None)

    @property
    def max_joint_vel(self) -> Optional[Any]:
        """Effective joint velocity limit for this robot's controller.

        Returns a tensor of shape ``([batch_size], n_dofs)`` representing the maximum allowed joint velocity. Note that `batch_size` dimension is always consistent with the batch size of the world, if applicable.  Unconstrained joints have ``inf`` entries.

        Returns ``None`` if the controller does not support this value (e.g.
        combined controllers) or no value was configured.
        """
        return getattr(self.controller, 'max_joint_vel', None)
    
    def __init__(
        self,
        world: GenesisWorld,
        description_file: str,
        name: Optional[str] = None,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
        keyframes : Optional[Dict[Union[Literal['reset'], str], RobotKeyframe]] = None,
        fix_root : bool = True,
        pos: Optional[tuple] = None,
        quat: Optional[tuple] = None,
        controller_type: CONTROL_MODE_T = "joint_position_and_velocity",
        **controller_kwargs,
    ):
        super().__init__(world, name, control_timestep, update_timestep)
        self.world : GenesisWorld = world
        self.description_file = description_file
        self.fix_root = fix_root
        self.pos = pos
        self.quat = quat
        self.keyframes = dict(keyframes) if keyframes is not None else {}
        self.controller_type = controller_type
        self.controller_kwargs = controller_kwargs

        self.entity : Optional[BaseEntity] = None
        self.controller : Optional[EntityController] = None

        # ------------------------------------------------------------------
        # Temporary build / reload sequence (intentional side-effect).
        #
        # We build a throwaway single-robot scene here so that the controller
        # can be initialised immediately, giving real (non-None) action and
        # observation spaces *before* the enclosing WorldEnv is constructed.
        # The controller needs a compiled scene (and therefore a built entity)
        # in order to discover the robot's DOFs and set up its action space
        # and related controller spaces.
        #
        # The scene is rebuilt properly (with all nodes) during the first
        # WorldEnv.reload() call, so this temporary build is purely for
        # early space discovery and does not affect normal operation.
        # ------------------------------------------------------------------
        self.world.reload(disable_viewer=True) # fresh scene (destroys any prior built scene)
        self.reload(priority=50)               # add this entity to the fresh scene
        self.world.after_reload(temporary=True)              # compile scene
        self.after_reload(priority=50)         # init controller — real spaces now available
        self.world.reload()                    # restore fresh unbuilt scene for subsequent nodes

        self._last_observation = None
        self._next_action = None

    def init_controller(self) -> EntityController:
        """Initialize the controller based on controller_type and controller_kwargs.
        
        This method creates a controller instance using the controller_type
        specified in __init__ (default: "joint_position_and_velocity").
        
        Subclasses can override this method to provide custom controller logic.
        
        Returns:
            EntityController instance
            
        Raises:
            ValueError: If controller_type is not recognized
        """
        if self.controller_type not in CONTROL_MODE_MAP:
            raise ValueError(
                f"Unknown controller_type: {self.controller_type}. "
                f"Available types: {list(CONTROL_MODE_MAP.keys())}"
            )
        
        controller_class = CONTROL_MODE_MAP[self.controller_type]
        return controller_class(
            world=self.world,
            entity=self.entity,
            **self.controller_kwargs
        )

    def update_observation(self):
        self._last_observation = self.controller.get_observation()

    def reload(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask: Optional[PyTorchArrayType] = None,
        **kwargs
    ) -> None:
        if priority == 50:
            self.entity = self.initialize_entity_from_description_file(self.description_file)
    
    def reset(self, *, priority = 0, seed = None, mask = None, **kwargs):
        if priority == 50:
            # Apply the rest/keyframe pose FIRST so that the controller reset
            # below observes the final rest state rather than the world's
            # default post-`scene.reset()` state.  This runs during the
            # node-reset phase, i.e. before `world.after_reset` refreshes the
            # sensor caches, so sensors also reflect the rest state.
            self._apply_rest(mask=mask)
            if self.controller is not None:
                self.controller.reset(mask=mask)

    def _apply_rest(self, mask : Optional[torch.Tensor] = None) -> None:
        if 'reset' in self.keyframes and isinstance(self.entity, gs_entities.RigidEntity):
            reset_keyframe = self.keyframes['reset']
            # Genesis accepts boolean masks directly for envs_idx.
            env_idx = mask
            if reset_keyframe.pos is not None:
                self.entity.set_pos(
                    reset_keyframe.pos,
                    envs_idx=env_idx
                )
            if reset_keyframe.quat is not None:
                self.entity.set_quat(
                    reset_keyframe.quat,
                    envs_idx=env_idx
                )
            if reset_keyframe.joint_position is not None:
                self.entity.set_dofs_position(
                    reset_keyframe.joint_position,
                    envs_idx=env_idx
                )
            if reset_keyframe.joint_velocity is not None:
                self.entity.set_dofs_velocity(
                    reset_keyframe.joint_velocity,
                    envs_idx=env_idx,
                )

    def after_reset(self, *, priority = 0, mask = None):
        if priority == 50:
            # The rest/keyframe mutation is owned by `reset(priority=50)`;
            # here we only consume the final state (controller bookkeeping).
            if self.controller is not None:
                self.controller.after_reset(mask=mask)
        elif priority == 0:
            self.update_observation()
    
    def after_reload(self, *, priority = 0, mask = None):
        if priority == 50:
            if self.controller is None:
                self.controller = self.init_controller()
            else:
                self.controller.entity = self.entity
            self.controller.reload()
            # Reload-time rest application is intentional: the freshly built
            # scene starts from default entity state, so the rest pose must be
            # applied here (controllers require a built scene to exist first).
            self._apply_rest(mask=mask)
            self.controller.after_reload()
            self.update_observation()
        elif priority == 0:
            if self.controller is not None:
                self.update_observation()
    
    def pre_environment_step(self, dt, *, priority = 0):
        if priority == 0:
            if self._next_action is not None:
                self.controller.apply_action(self._next_action, dt=dt)

    def post_environment_step(
        self,
        dt: Union[float, PyTorchArrayType],
        *,
        priority: int = 0
    ) -> None:
        if priority == 0:
            self.update_observation()

    def get_observation(self) -> Optional[Dict[str, PyTorchArrayType]]:
        return self._last_observation

    def set_next_action(self, action):
        self._next_action = action

    def get_action_dict(self, action: Any, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return a structured dict describing the commanded target.

        Delegates to the underlying controller's :meth:`get_action_dict`.
        This is an introspection helper — it does **not** apply the action.

        Args:
            action: The raw action tensor or dict in the format expected by
                the controller's ``apply_action``.
            dt: Optional control timestep in seconds used for rate-limit
                clipping.  Defaults to ``self.control_timestep`` when not
                supplied.

        Returns:
            A dict with semantically meaningful keys such as
            ``target_joint_position``, ``target_joint_velocity``,
            ``target_eef_position``, ``target_eef_quaternion``.

        Raises:
            RuntimeError: If the controller has not been initialised yet.
        """
        if self.controller is None:
            raise RuntimeError(
                "Controller has not been initialised yet. "
                "Call reload()/after_reload() first."
            )
        if dt is None:
            dt = self.control_timestep
        return self.controller.get_action_dict(action, dt=dt)

    # ------------------------------------------------------------------
    # Low-level robot kinematics / planning primitives
    #
    # The methods below are thin proxies over Genesis RigidEntity APIs.
    # They operate on raw link frames and joint coordinates — they do
    # **not** know about controller-owned EEF semantics (virtual EEF
    # offsets, base-frame conventions, rotation representations, etc.).
    #
    # ``forward_kinematics`` is the **sole** robot-node FK primitive.
    # For EEF-aware FK (target_link + base_frame + virtual offset), use
    # ``DiscreteEntityController.compute_eef_pose_from_qpos`` on the
    # controller — that is the single canonical entry point for EEF
    # semantics.  The controller is the single source of truth for
    # target_link, base_frame, EEF offsets, and action-space conventions.
    # ------------------------------------------------------------------

    def _require_rigid_entity(self) -> gs_entities.RigidEntity:
        """Return the built entity, raising if it is not a ``RigidEntity``."""
        if self.entity is None:
            raise RuntimeError(
                "Entity has not been built yet. Call reload()/after_reload() first."
            )
        if not isinstance(self.entity, gs_entities.RigidEntity):
            raise TypeError(
                f"Expected a RigidEntity but got {type(self.entity).__name__}. "
                f"forward_kinematics / plan_path are only available on rigid robots."
            )
        return self.entity

    def forward_kinematics(
        self,
        qpos: PyTorchArrayType,
        qs_idx_local: Optional[Sequence[int]] = None,
        links_idx_local: Optional[Sequence[int]] = None,
        envs_idx: Optional[PyTorchArrayType] = None,
    ) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Compute forward kinematics for the given joint positions.

        .. note:: **Low-level robot primitive — sole FK entry point on the robot node.**
            This method returns poses in the **physical link frame** and
            knows nothing about virtual EEF offsets or base-frame
            conventions.  Reference policies that need EEF-aware FK
            should use
            :meth:`DiscreteEntityController.compute_eef_pose_from_qpos`
            on the controller instead, which is the single canonical
            entry point for ``target_link`` / ``base_frame`` / virtual
            EEF offset semantics.

        Proxy to :meth:`genesis.engine.entities.RigidEntity.forward_kinematics`.

        Args:
            qpos: Joint positions — shape ``(n_qs,)`` or ``(batch_size, n_qs)``.
            qs_idx_local: Optional subset of generalized coordinates to use.
            links_idx_local: Optional subset of links whose poses to return.
            envs_idx: Optional batch indices to evaluate.

        Returns:
            ``(links_pos, links_quat)`` — positions and quaternions of the
            requested links.
        """
        entity = self._require_rigid_entity()
        return entity.forward_kinematics(
            qpos,
            qs_idx_local=qs_idx_local,
            links_idx_local=links_idx_local,
            envs_idx=envs_idx,
        )

    # ------------------------------------------------------------------
    # Public kinematics API (IK)
    #
    # Low-level robot primitives.  They operate on physical link frames
    # and require the caller to supply target_link and base-frame
    # information explicitly.  Reference policies should prefer the
    # controller's own IK / FK methods, which handle these semantics
    # automatically.
    #
    # For EEF-aware FK from joint positions, use
    # ``DiscreteEntityController.compute_eef_pose_from_qpos`` on the
    # controller — that is the single canonical entry point for
    # target_link / base_frame / virtual-EEF-offset semantics.
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        target_link: Union[str, Any],
        target_position: PyTorchArrayType,
        target_quaternion: PyTorchArrayType,
        *,
        dofs_idx_local: Optional[Sequence[int]] = None,
        max_samples: int = 1,
        **ik_kwargs,
    ) -> Tuple[PyTorchArrayType, PyTorchArrayType]:
        """Solve inverse kinematics for a target link pose.

        .. note:: **Low-level robot primitive.**
            This method operates on the **physical** link frame and
            requires the caller to specify ``target_link`` explicitly.
            It does **not** apply any virtual EEF offset or base-frame
            convention that a controller may configure.

            **Reference policies should not call this method.**  Instead,
            use :meth:`EndEffectorPoseController.solve_ik_absolute` (or
            ``solve_ik_base``) on the controller, which handles the
            virtual-EEF offset, base-frame transform, and rotation
            representation automatically.  The controller is the single
            source of truth for ``target_link``, ``base_frame``, EEF
            offsets, and action-space semantics.

            This method is intended for advanced use cases that need raw
            IK on an arbitrary link (e.g. whole-body planning, custom
            multi-link objectives) outside the controller's EEF model.

        Public proxy to :meth:`genesis.engine.entities.RigidEntity.inverse_kinematics`.

        Args:
            target_link: Link name (str) or ``RigidLink`` instance.
            target_position: Target position — shape ``(3,)`` or
                ``(batch_size, 3)``.
            target_quaternion: Target quaternion ``(w, x, y, z)`` — shape
                ``(4,)`` or ``(batch_size, 4)``.
            dofs_idx_local: Optional subset of DOFs the solver may use.
            max_samples: Maximum IK solver attempts (default 1 = no random
                restarts, for solution continuity).
            **ik_kwargs: Forwarded to the Genesis IK solver.

        Returns:
            ``(target_joint_positions, achieved_error)`` — solved joint
            positions and the 6-D pose error.
        """
        entity = self._require_rigid_entity()
        if isinstance(target_link, str):
            target_link = entity.get_link(target_link)
        full_qpos, error = entity.inverse_kinematics(
            target_link,
            target_position,
            target_quaternion,
            return_error=True,
            dofs_idx_local=dofs_idx_local,
            max_samples=max_samples,
            **ik_kwargs,
        )
        return full_qpos, error

    def plan_path(
        self,
        qpos_goal: PyTorchArrayType,
        qpos_start: Optional[PyTorchArrayType] = None,
        max_nodes: int = 2000,
        resolution: float = 0.05,
        timeout: Optional[float] = None,
        max_retry: int = 1,
        smooth_path: bool = True,
        num_waypoints: Optional[int] = None,
        ignore_collision: bool = False,
        planner: str = "RRTConnect",
        envs_idx: Optional[PyTorchArrayType] = None,
        return_valid_mask: bool = False,
        *,
        ee_link_name: Optional[str] = None,
        with_entity=None,
        **kwargs,
    ):
        """Plan a collision-free joint-space path to ``qpos_goal``.

        .. note:: **Low-level robot primitive.**
            This method plans in joint space using the Genesis RRT planner.
            It does not know about controller-owned EEF semantics.  For
            Cartesian goal planning, convert the goal to joint positions
            via the controller's IK method first (e.g.
            :meth:`EndEffectorPoseController.solve_ik_absolute`).

        Proxy to :meth:`genesis.engine.entities.RigidEntity.plan_path`.

        Args:
            qpos_goal: Target joint positions — shape ``(n_qs,)`` or
                ``(batch_size, n_qs)``.
            qpos_start: Start joint positions (defaults to current qpos).
            max_nodes: Maximum RRT nodes before failure.
            resolution: Joint-space resolution for the planner.
            timeout: Per-attempt timeout in seconds (``None`` = no timeout).
            max_retry: Number of retry attempts.
            smooth_path: Whether to smooth the resulting path.
            num_waypoints: Number of waypoints after interpolation.  When
                ``None`` (the default), the method attempts to infer a
                minimum waypoint count from the controller's
                ``max_joint_pos_vel`` rate limit and the robot's
                ``control_timestep`` so that adjacent waypoints obey
                ``|Δq| <= max_joint_pos_vel * dt``.  Joints with ``inf``
                rate limits (unconstrained) do not increase the count.
                If inference is not possible (no rate limit configured at
                all), falls back to **300**.
            ignore_collision: If ``True``, ignore collision checks.
            planner: Planner name (default ``"RRTConnect"``).
            envs_idx: Optional batch indices to plan for.
            return_valid_mask: If ``True``, return ``(path, valid_mask)``
                where ``valid_mask=True`` indicates success.
            ee_link_name: Optional end-effector link name for task-space
                planning.
            with_entity: Optional entity for collision-aware planning.
            **kwargs: Forwarded to the Genesis planner.

        Returns:
            Planned path tensor, or ``(path, valid_mask)`` when
            ``return_valid_mask=True``.
        """
        entity = self._require_rigid_entity()
        if num_waypoints is None:
            num_waypoints = self._infer_path_waypoint_count(qpos_goal, qpos_start)
        return entity.plan_path(
            qpos_goal,
            qpos_start=qpos_start,
            max_nodes=max_nodes,
            resolution=resolution,
            timeout=timeout,
            max_retry=max_retry,
            smooth_path=smooth_path,
            num_waypoints=num_waypoints,
            ignore_collision=ignore_collision,
            planner=planner,
            envs_idx=envs_idx,
            return_valid_mask=return_valid_mask,
            ee_link_name=ee_link_name,
            with_entity=with_entity,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Motion-planning helpers (low-level, robot-node-internal)
    # ------------------------------------------------------------------

    def _get_controller_max_joint_pos_vel(self) -> Optional[torch.Tensor]:
        """Return the controller's ``max_joint_pos_vel`` as a 1-D ``(n_dofs,)`` tensor.

        Reads the public ``max_joint_pos_vel`` property on the controller and
        squeezes any batch dimension (the planner works on a single path).
        Unconstrained joints retain ``inf`` entries, which are handled
        correctly by :meth:`_infer_path_waypoint_count` (they contribute 0
        steps and don't limit the waypoint count).

        Returns ``None`` when the controller does not expose a rate limit or
        has not been initialised.
        """
        if self.controller is None:
            return None
        v = getattr(self.controller, 'max_joint_pos_vel', None)
        if v is None:
            return None
        # Squeeze batch dim if present → (n_dofs,)
        if v.dim() == 2:
            v = v[0]
        return v

    def _infer_path_waypoint_count(
        self,
        qpos_goal: PyTorchArrayType,
        qpos_start: Optional[PyTorchArrayType] = None,
    ) -> int:
        """Infer a minimum waypoint count from ``max_joint_pos_vel`` and ``dt``.

        The inferred count ensures that adjacent waypoints satisfy
        ``|Δq_i| <= max_joint_pos_vel_i * dt`` for every joint *i*.
        Joints with ``inf`` rate limits (unconstrained) contribute zero
        steps and therefore do not increase the waypoint count.

        Returns:
            Inferred waypoint count (≥ 2), or the default fallback of **300**
            when inference is not possible.
        """
        _DEFAULT_WAYPOINTS = 300
        max_pos_vel = self._get_controller_max_joint_pos_vel()
        if max_pos_vel is None:
            return _DEFAULT_WAYPOINTS

        dt = self.control_timestep
        max_delta = max_pos_vel * float(dt)  # (n_dofs,)
        if torch.any(max_delta <= 0):
            return _DEFAULT_WAYPOINTS

        # Resolve start qpos
        if qpos_start is None:
            try:
                qpos_start = self.entity.get_dofs_position()
            except Exception:
                return _DEFAULT_WAYPOINTS

        # Flatten to 1-D for the computation (handle batched case)
        goal = qpos_goal.detach()
        start = qpos_start.detach()
        if goal.dim() == 2:
            goal = goal[0]
        if start.dim() == 2:
            start = start[0]

        # Ensure max_delta matches the goal dimensionality
        if max_delta.shape[0] != goal.shape[0]:
            # max_joint_pos_vel may only cover a subset of DOFs; fall back
            return _DEFAULT_WAYPOINTS

        delta = (goal - start).abs()
        steps_per_joint = torch.ceil(delta / max_delta)
        num_waypoints = int(steps_per_joint.max().item()) + 1  # +1 for start frame
        return max(num_waypoints, 2)