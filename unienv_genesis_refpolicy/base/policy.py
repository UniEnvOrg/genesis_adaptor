from typing import Any, Dict, Optional, Union, Generic, TypeVar

import torch
import abc

from unienv_interface.space import DictSpace, BoxSpace
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_genesis.world.genesis_world import GenesisWorld

ActT = TypeVar('ActT')


def require_eef_controller(controller: Any, *, policy_name: str = "This policy") -> None:
    """Fail fast if *controller* does not expose end-effector semantics.

    Cartesian / EEF-based reference policies should call this at the top of
    ``__init__`` to ensure the controller owns the necessary EEF metadata
    (``target_link``, ``base_frame``, EEF offsets, rotation representation).

    The controller is the single source of truth for these semantics; ref
    policies must **not** re-specify ``target_link`` or manually plumb offset
    semantics.  If the controller lacks EEF support, this function raises a
    :class:`RuntimeError` with a clear diagnostic message.

    Args:
        controller: The controller instance to validate.  Must expose a
            ``has_eef`` boolean property (as provided by
            :class:`DiscreteEntityController`).
        policy_name: Human-readable policy name for the error message.

    Raises:
        RuntimeError: If ``controller.has_eef`` is ``False`` or the
            controller does not expose the ``has_eef`` attribute.
    """
    if not getattr(controller, "has_eef", False):
        raise RuntimeError(
            f"{policy_name} requires a controller with end-effector semantics "
            f"(has_eef=True), but the provided "
            f"{type(controller).__name__} does not have a target_link "
            f"configured.  The controller — not the reference policy — owns "
            f"the EEF metadata (target_link, base_frame, eef_offset_pos/quat, "
            f"rotation_representation).  Please construct the controller with "
            f"target_link to enable EEF semantics."
        )

class BaseRefPolicy(abc.ABC, Generic[ActT]):
    def __init__(
        self,
        world : GenesisWorld,
    ):
        self.world = world

    # ========== Lifecycle methods ==========

    def reset(
        self,
        *,
        mask : Optional[torch.Tensor] = None,
        seed : Optional[int] = None,
    ) -> None:
        pass

    def reload(
        self,
        *,
        mask : Optional[torch.Tensor] = None,
        seed : Optional[int] = None,
    ) -> None:
        self.reset(mask=mask, seed=seed)
    
    @abc.abstractmethod
    def step(
        self,
        *,
        mask : Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError
    
    # ========== Methods for querying policy state ==========
    @abc.abstractmethod
    def get_action(self) -> ActT:
        raise NotImplementedError

    def get_action_dict(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return a controller-native action dict for the current action.

        The default implementation wraps :meth:`get_action` under the key
        ``"action"``, matching the base :class:`EntityController` convention.
        Concrete subclasses that hold a reference to a controller should
        override this to delegate to the controller's own
        :meth:`get_action_dict`, ensuring the returned dict uses the
        controller-native schema (e.g. ``target_joint_position``,
        ``target_eef_position``, …) rather than a ref-policy-specific one.

        This is an introspection helper — it does **not** apply the action.

        Args:
            dt: Optional control timestep in seconds, forwarded to the
                controller for rate-limit clipping.
        """
        return {"action": self.get_action()}

    def get_action_dict_space(self) -> DictSpace:
        """Return a :class:`DictSpace` describing the output of :meth:`get_action_dict`.

        The default implementation mirrors the base :meth:`get_action_dict`
        fallback, which wraps the action under the key ``"action"``.
        Concrete subclasses should override this to match their own
        :meth:`get_action_dict` schema, preserving the principle that
        ref policies expose controller-native action-dict structure
        rather than inventing a separate ref-policy schema.

        This is a structural introspection helper — it does **not** depend
        on any particular action value.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_action_dict_space(). "
            f"Subclasses that override get_action_dict() should also override "
            f"get_action_dict_space() to describe the returned dict schema."
        )

    @abc.abstractmethod
    def get_success(self) -> Union[torch.Tensor, bool]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_failure(self) -> Union[torch.Tensor, bool]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_timeout(self) -> Union[torch.Tensor, bool]:
        raise NotImplementedError