"""Reach-target task: move the end-effector to a random target pose.

A simple reaching task that does not involve any manipulatable object.  The
robot's end-effector must reach a randomly sampled target position in the
workspace.  Useful as a baseline or for evaluating EEF control.

Provides:
- **observation**: ``{"eef_position": (3,), "target_position": (3,)}``
- **reward**: ``-distance(eef, target)`` (dense)
- **termination**: ``True`` when ``distance < success_threshold``

The task does not load any entities — it relies on a robot node (with an
``EndEffectorPoseController``) being present in the environment.  The EEF
controller reference is captured at construction time (the robot node
initialises its controller during ``__init__`` via a temporary build, so the
controller is available before the first ``env.reset()``).

The reset lifecycle runs at priority 25 (the objects / props range), i.e.
**after** the robot (priority 50) has been set up.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch

from unienv_interface.backends.pytorch import (
    PyTorchComputeBackend,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world.node import WorldNode
from unienv_genesis.world import GenesisWorld

__all__ = ["ReachTargetTask"]


class ReachTargetTask(WorldNode):
    """Reach a randomly-sampled target end-effector position.

    Args:
        world: :class:`GenesisWorld` instance.
        eef_controller: An object exposing
            ``get_eef_pose_absolute() -> (pos, quat)`` — typically an
            :class:`EndEffectorPoseController`.
        target_workspace: ``(low, high)`` corner tuples of shape ``(3,)``
            defining the axis-aligned box from which target positions are
            sampled (world frame).  For example
            ``((0.2, -0.3, 0.3), (0.6, 0.3, 0.8))``.
        success_threshold: Distance (m) below which the episode is considered
            successful.
        name: Node name.
        target_orientation: Optional fixed target quaternion ``(w, x, y, z)``
            for orientation-aware reaching.  When ``None`` (default) only
            position is evaluated — orientation is ignored in the reward and
            success criterion.
        reset_priority / after_reload_priority: Override the lifecycle
            priorities (default 25).
        post_environment_step_priority: Priority for the per-step update.
    """

    DEFAULT_RESET_PRIORITY = 25
    DEFAULT_POSTSTEP_PRIORITY = 0

    has_reward = True
    has_termination_signal = True
    has_truncation_signal = False

    # No entities to load — reload is a no-op, so no reload priority.
    reload_priorities = set()
    after_reload_priorities = {DEFAULT_RESET_PRIORITY}
    reset_priorities = {DEFAULT_RESET_PRIORITY}

    def __init__(
        self,
        world: GenesisWorld,
        eef_controller: Any,
        target_workspace: Tuple[
            Tuple[float, float, float], Tuple[float, float, float]
        ],
        success_threshold: float = 0.03,
        name: str = "reach_task",
        target_orientation: Optional[Tuple[float, float, float, float]] = None,
        reset_priority: Optional[int] = None,
        after_reload_priority: Optional[int] = None,
        post_environment_step_priority: int = DEFAULT_POSTSTEP_PRIORITY,
    ):
        self.world = world
        self._name = name
        self.eef_controller = eef_controller
        self.success_threshold = success_threshold

        low = torch.tensor(target_workspace[0], dtype=world.dtype)
        high = torch.tensor(target_workspace[1], dtype=world.dtype)
        assert low.shape == (3,) and high.shape == (3,), (
            "target_workspace corners must be 3-tuples"
        )
        assert torch.all(high >= low), (
            "target_workspace high must be >= low on every axis"
        )
        self._workspace_low = low.to(device=world.device)
        self._workspace_high = high.to(device=world.device)

        self.target_orientation = target_orientation
        if target_orientation is not None:
            self._target_quat_t = torch.tensor(
                target_orientation, dtype=world.dtype, device=world.device,
            )
        else:
            self._target_quat_t = None

        # Priorities.
        if reset_priority is not None:
            self.reset_priorities = {reset_priority}
        if after_reload_priority is not None:
            self.after_reload_priorities = {after_reload_priority}
        self.post_environment_step_priorities = {post_environment_step_priority}

        # Random state.
        self.rng: torch.Generator = torch.Generator(device=world.device)
        self.rng.seed()

        # Per-episode state.
        self._target_pos: Optional[torch.Tensor] = None  # ([B], 3) or (3,)
        self._reward: Union[float, torch.Tensor] = 0.0
        self._terminated: Union[bool, torch.Tensor] = False

        # Observation space.
        obs_spaces = {
            "eef_position": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=(3,),
                dtype=self.world.dtype,
            ),
            "target_position": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=(3,),
                dtype=self.world.dtype,
            ),
        }
        if self.target_orientation is not None:
            obs_spaces["target_orientation"] = BoxSpace(
                PyTorchComputeBackend,
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=self.world.dtype,
            )
        self.observation_space = DictSpace(
            PyTorchComputeBackend,
            obs_spaces,
        )

    @property
    def name(self) -> str:
        return self._name

    def _reseed(self, seed: int) -> None:
        self.rng.manual_seed(seed)

    def after_reload(self, *, priority=0, mask=None):
        if priority not in self.after_reload_priorities:
            return
        self.reset(priority=next(iter(self.reset_priorities)), seed=None, mask=None)

    def _sample_target_position(self, n: int) -> torch.Tensor:
        """Sample *n* target positions uniformly in the workspace box.

        Returns shape ``(n, 3)`` for unbatched or ``(B, n, 3)`` for batched.
        """
        low = self._workspace_low
        high = self._workspace_high
        batched = self.world.batch_size is not None
        shape = (
            (n, 3) if not batched else (self.world.batch_size, n, 3)
        )
        u = torch.rand(
            shape, generator=self.rng, dtype=self.world.dtype, device=self.world.device,
        )
        return low + u * (high - low)

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority not in self.reset_priorities:
            return

        if seed is not None:
            self._reseed(seed)

        self._target_pos = self._sample_target_position(1).squeeze(-2)  # ([B], 3) or (3,)

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
            assert self.world.batch_size is not None, (
                "Masking only supported in batched environments"
            )
            # Sample for all envs (shape (B, 3)), then select only masked.
            new_pos = self._sample_target_position(1).squeeze(-2)  # (B, 3)
            self._target_pos[mask] = new_pos[mask]
            self._reward.masked_fill_(mask, 0.0)
            self._terminated.masked_fill_(mask, False)

    def post_environment_step(self, dt, *, priority=0):
        if priority not in self.post_environment_step_priorities:
            return

        eef_pos, _eef_quat = self.eef_controller.get_eef_pose_absolute()  # ([B], 3)
        dist = torch.linalg.vector_norm(eef_pos - self._target_pos, dim=-1)
        if self.world.batch_size is None:
            dist = float(dist.item())

        self._reward = -dist
        self._terminated = dist < self.success_threshold

    def get_observation(self):
        eef_pos, _ = self.eef_controller.get_eef_pose_absolute()
        obs = {
            "eef_position": eef_pos,
            "target_position": self._target_pos,
        }
        if self._target_quat_t is not None:
            obs["target_orientation"] = self._target_quat_t
        return obs

    def get_reward(self):
        return self._reward

    def get_termination(self):
        return self._terminated

    def get_info(self) -> Dict[str, Any]:
        info = {
            "target_position": self._target_pos,
            "success_threshold": self.success_threshold,
        }
        if self._target_quat_t is not None:
            info["target_orientation"] = self._target_quat_t
        return info
