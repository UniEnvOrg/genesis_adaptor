from typing import Optional, Dict, Any
import torch
from genesis.engine import entities as gs_entities
from unienv_interface.space import DictSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
from .base import EntityController


class CombinedController(EntityController[gs_entities.RigidEntity, dict, dict]):
    """Controls different joint indices with different sub-controllers.

    Args:
        world: The Genesis world.
        entity: The rigid entity to control.
        sub_controllers: Dict mapping controller name to a config dict.
            Each config must include ``"controller_type"`` and may include
            ``"joint_indexes"`` plus any controller-specific kwargs (``Kp``,
            ``Kd``, etc.).

    Example::

        controller = CombinedController(
            world=world,
            entity=robot,
            sub_controllers={
                "arm": {
                    "controller_type": "joint_position",
                    "joint_indexes": [0, 1, 2, 3, 4, 5],
                    "Kp": 100.0,
                    "Kd": 2.0,
                },
                "gripper": {
                    "controller_type": "joint_velocity",
                    "joint_indexes": [6, 7],
                },
            },
        )
        # action: {"arm": arm_pos_tensor, "gripper": gripper_vel_tensor}
        # observation: {"arm": {"joint_position": ..., ...}, "gripper": {...}}
    """

    controller_type = "combined"

    @property
    def entity(self) -> gs_entities.RigidEntity:
        return self._entity

    @entity.setter  
    def entity(self, value: gs_entities.RigidEntity) -> None:
        self._entity = value
        # Also update entity reference for all sub-controllers
        for ctrl in self._sub_controllers.values():
            ctrl.entity = value

    def __init__(
        self,
        world: GenesisWorld,
        entity: gs_entities.RigidEntity,
        sub_controllers: dict,
        **shared_kwargs,
    ) -> None:
        from . import CONTROL_MODE_MAP

        self._sub_controllers: dict[str, EntityController] = {}
        for name, cfg in sub_controllers.items():
            cfg_applied = dict(shared_kwargs)
            cfg = dict(cfg)  # shallow copy to avoid mutating caller's dict
            ctrl_type = cfg.pop("controller_type")
            ctrl_class = CONTROL_MODE_MAP[ctrl_type]
            cfg_applied.update(cfg)
            self._sub_controllers[name] = ctrl_class(world=world, entity=entity, **cfg_applied)

        # Action space: nested DictSpace {name: sub_ctrl.action_space}
        action_space = DictSpace(
            PyTorchComputeBackend,
            {name: ctrl.action_space for name, ctrl in self._sub_controllers.items()},
            device=world.device,
        )

        # Observation space: nested DictSpace {name: sub_ctrl.observation_space}
        # Only include sub-controllers that expose an observation_space
        obs_spaces = {
            name: ctrl.observation_space
            for name, ctrl in self._sub_controllers.items()
            if ctrl.observation_space is not None
        }
        observation_space = (
            DictSpace(PyTorchComputeBackend, obs_spaces, device=world.device)
            if obs_spaces
            else None
        )

        super().__init__(world, entity, observation_space, action_space)

    # ------------------------------------------------------------------
    # Public rate-limit / velocity clipping accessors
    # ------------------------------------------------------------------
    # CombinedController does not expose a single scalar for these because
    # each sub-controller may have its own limits over a disjoint joint
    # subset.  Callers that need per-sub-controller values should iterate
    # ``combined_controller._sub_controllers.values()`` (or use
    # ``combined_controller[name]``) and read the properties directly.
    #
    # ``max_joint_pos_vel`` below aggregates the position-rate limits from
    # every sub-controller into a full-DOF tensor suitable for motion
    # planning.  Uncovered DOFs retain ``inf`` (unconstrained) rather than
    # collapsing the result to ``None``.

    @property
    def max_joint_pos_vel(self) -> Optional[torch.Tensor]:
        """Aggregate ``max_joint_pos_vel`` from all sub-controllers.

        Returns a tensor of shape ``(batch_size, n_dofs)`` for batched
        environments or ``(n_dofs,)`` for unbatched ones, assembled by
        scattering each sub-controller's rate limit over the full DOF
        vector.

        Joints not covered by any sub-controller retain ``inf``
        (unconstrained).  ``inf`` entries are **preserved** rather than
        collapsing the result to ``None`` — callers that need finite limits
        can check with :func:`torch.isinf`.

        Returns ``None`` only when aggregation is genuinely impossible,
        i.e. a sub-controller is not a :class:`DiscreteEntityController` or
        its ``max_joint_pos_vel`` property is ``None``.
        """
        from .base import DiscreteEntityController
        n_dofs = self.entity.n_dofs
        device = self.world.device
        batch_size = self.world.batch_size
        batched = batch_size is not None and batch_size > 0

        full_limit = torch.full(get_batch_aware_shape(batch_size, n_dofs), float('inf'), device=device)

        for ctrl in self._sub_controllers.values():
            if not isinstance(ctrl, DiscreteEntityController):
                return None
            v = ctrl.max_joint_pos_vel
            if v is None:
                return None
            idx = ctrl.joint_indexes
            if idx is None:
                idx = list(range(n_dofs))
            # Sub-controller returns (batch_size, n_ctrl_dofs) or
            # (n_ctrl_dofs,).  Scatter into the full tensor, matching the
            # batched / unbatched convention.
            if batched:
                full_limit[:, idx] = v.to(device)
            else:
                full_limit[idx] = v.to(device)
        return full_limit

    @property
    def max_joint_vel(self) -> Optional[torch.Tensor]:
        """Not directly available on a combined controller.

        Returns ``None`` because each sub-controller may define its own
        velocity limit over a disjoint joint subset.  Iterate over
        sub-controllers to inspect individual limits.
        """
        return None

    def __getitem__(self, name: str) -> EntityController:
        """Access a sub-controller by name.

        Example::

            arm_ctrl = combined_controller["arm"]
            action = arm_ctrl.get_action_world(target_pos, target_quat)
        """
        return self._sub_controllers[name]

    def get_observation(self) -> Optional[dict]:
        result = {}
        for name, ctrl in self._sub_controllers.items():
            obs = ctrl.get_observation()
            if obs is not None:
                result[name] = obs
        return result if result else None

    def apply_action(self, action: dict, dt: Optional[float] = None) -> None:
        for name, ctrl in self._sub_controllers.items():
            ctrl.apply_action(action[name], dt=dt)

    def get_action_dict(self, action: dict, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return a nested structured action dict by recursing into sub-controllers.

        ``action`` is expected to be a dict mapping sub-controller names to
        their respective action tensors/dicts.  The returned dict has the same
        top-level keys, with each value being the structured action dict
        produced by the corresponding sub-controller.
        """
        return {
            name: ctrl.get_action_dict(action[name], dt=dt)
            for name, ctrl in self._sub_controllers.items()
        }

    def reload(self) -> None:
        for ctrl in self._sub_controllers.values():
            ctrl.reload()

    def reset(self, mask: Optional[PyTorchArrayType] = None) -> None:
        for ctrl in self._sub_controllers.values():
            ctrl.reset(mask)

    def after_reload(self) -> None:
        for ctrl in self._sub_controllers.values():
            ctrl.after_reload()

    def after_reset(self, mask: Optional[PyTorchArrayType] = None) -> None:
        for ctrl in self._sub_controllers.values():
            ctrl.after_reset(mask)
