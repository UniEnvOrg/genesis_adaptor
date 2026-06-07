from typing import Optional, Dict, Any
from genesis.engine import entities as gs_entities
from unienv_interface.space import DictSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType
from unienv_genesis.world.genesis_world import GenesisWorld
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

    def apply_action(self, action: dict) -> None:
        for name, ctrl in self._sub_controllers.items():
            ctrl.apply_action(action[name])

    def get_action_dict(self, action: dict) -> Dict[str, Any]:
        """Return a nested structured action dict by recursing into sub-controllers.

        ``action`` is expected to be a dict mapping sub-controller names to
        their respective action tensors/dicts.  The returned dict has the same
        top-level keys, with each value being the structured action dict
        produced by the corresponding sub-controller.
        """
        return {
            name: ctrl.get_action_dict(action[name])
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
