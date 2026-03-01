"""GraphRefPolicy — chain reference policies via a success/failure transition graph."""

from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch

from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import (
    is_batched,
    get_batch_aware_shape,
)
from unienv_genesis_refpolicy.base.policy import BaseRefPolicy

ActT = TypeVar("ActT")


class GraphRefPolicy(BaseRefPolicy[ActT]):
    """State-machine policy that chains sub-policies via a directed graph.

    Each **node** is either a pre-built :class:`BaseRefPolicy` instance or a
    zero-argument **factory callable** that returns one.  Factories are called
    lazily the first time the node is entered, allowing state-dependent
    construction (e.g. reading the current EEF pose to build a lift target).

    Edges are specified as two lists of ``(source, destination)`` pairs:

    * ``success_edges`` — followed when the active node signals success.
    * ``fail_edges`` — followed when the active node signals failure **or**
      timeout.

    When a node signals a transition and **no matching edge exists**, the
    graph enters a **terminal** state (success or failure for the whole
    graph).

    An optional ``static_actions`` dict is merged into every action produced
    by the current node (useful for holding idle actuators at a fixed
    command).

    Args:
        world: Genesis world instance.
        nodes: Mapping from name to policy instance **or** factory callable.
        success_edges: ``[(from, to), ...]`` — taken on node success.
        fail_edges: ``[(from, to), ...]`` — taken on node failure/timeout.
        initial: Name of the starting node.
        static_actions: Constant ``Dict[str, Tensor]`` entries merged into
            every ``get_action`` result.  The node's own keys take
            precedence on conflict.
    """

    def __init__(
        self,
        world: GenesisWorld,
        nodes: Dict[str, Union[BaseRefPolicy[ActT], Callable[[], BaseRefPolicy[ActT]]]],
        success_edges: List[Tuple[str, str]],
        fail_edges: List[Tuple[str, str]],
        initial: str,
        static_actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__(world)

        self._node_specs = dict(nodes)
        self._success_map: Dict[str, str] = {src: dst for src, dst in success_edges}
        self._fail_map: Dict[str, str] = {src: dst for src, dst in fail_edges}
        self._initial = initial
        self._static_actions = static_actions

        # Build name ↔ index mapping for batched env support.
        self._node_names: List[str] = list(nodes.keys())
        self._name_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self._node_names)
        }

        # Eagerly resolve non-callable nodes; leave factories as None.
        self._node_policies: Dict[str, Optional[BaseRefPolicy[ActT]]] = {}
        for name, spec in self._node_specs.items():
            if isinstance(spec, BaseRefPolicy):
                self._node_policies[name] = spec
            else:
                self._node_policies[name] = None

        # Per-env mutable state.
        batch_size = world.batch_size
        device = world.device
        scalar_shape = get_batch_aware_shape(batch_size, ())

        self._current_idx = torch.full(
            scalar_shape, self._name_to_idx[initial], dtype=torch.long, device=device,
        )
        self._terminal_success = torch.zeros(scalar_shape, dtype=torch.bool, device=device)
        self._terminal_failure = torch.zeros(scalar_shape, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # Node resolution
    # ------------------------------------------------------------------

    def _get_or_create_node(self, name: str) -> BaseRefPolicy[ActT]:
        """Return the policy for *name*, calling the factory if needed."""
        policy = self._node_policies[name]
        if policy is None:
            factory = self._node_specs[name]
            policy = factory()
            self._node_policies[name] = policy
        return policy

    def _clear_factory_nodes(self) -> None:
        """Reset lazily-created nodes so factories are re-invoked next episode."""
        for name, spec in self._node_specs.items():
            if not isinstance(spec, BaseRefPolicy):
                self._node_policies[name] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._clear_factory_nodes()
        initial_idx = self._name_to_idx[self._initial]

        if mask is not None:
            self._current_idx[mask] = initial_idx
            self._terminal_success[mask] = False
            self._terminal_failure[mask] = False
        else:
            self._current_idx.fill_(initial_idx)
            self._terminal_success.zero_()
            self._terminal_failure.zero_()

        initial_policy = self._get_or_create_node(self._initial)
        initial_policy.reset(mask=mask, seed=seed)

    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        batched = is_batched(self.world.batch_size)

        if batched:
            self._step_batched(mask)
        else:
            self._step_unbatched()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_action(self, observation: torch.Tensor) -> ActT:
        batched = is_batched(self.world.batch_size)

        if batched:
            action = self._get_action_batched(observation)
        else:
            node_name = self._node_names[self._current_idx.item()]
            policy = self._get_or_create_node(node_name)
            action = policy.get_action(observation)

        if self._static_actions is not None and isinstance(action, dict):
            action = {**self._static_actions, **action}

        return action

    def get_success(self) -> Union[torch.Tensor, bool]:
        return self._unwrap(self._terminal_success)

    def get_failure(self) -> Union[torch.Tensor, bool]:
        return self._unwrap(self._terminal_failure)

    def get_timeout(self) -> Union[torch.Tensor, bool]:
        # Timeout is handled per-node via fail_edges; graph itself doesn't timeout.
        if is_batched(self.world.batch_size):
            return torch.zeros_like(self._terminal_success)
        return False

    # ------------------------------------------------------------------
    # Unbatched implementation
    # ------------------------------------------------------------------

    def _step_unbatched(self) -> None:
        if self._terminal_success.item() or self._terminal_failure.item():
            return

        node_name = self._node_names[self._current_idx.item()]
        policy = self._get_or_create_node(node_name)
        policy.step()

        # Check success.
        if policy.get_success():
            next_name = self._success_map.get(node_name)
            if next_name is not None:
                self._transition_to_unbatched(next_name)
            else:
                self._terminal_success.fill_(True)
            return

        # Check failure or timeout.
        failed = policy.get_failure()
        timed_out = policy.get_timeout()
        if failed or timed_out:
            reason = "failed" if failed else "timed out"
            next_name = self._fail_map.get(node_name)
            if next_name is not None:
                self._transition_to_unbatched(next_name)
            else:
                self._terminal_failure.fill_(True)

    def _transition_to_unbatched(self, next_name: str) -> None:
        self._current_idx.fill_(self._name_to_idx[next_name])
        next_policy = self._get_or_create_node(next_name)
        next_policy.reset()

    # ------------------------------------------------------------------
    # Batched implementation
    # ------------------------------------------------------------------

    def _step_batched(self, mask: Optional[torch.Tensor]) -> None:
        alive = mask if mask is not None else torch.ones_like(self._terminal_success)
        alive = alive & ~self._terminal_success & ~self._terminal_failure

        for node_idx, node_name in enumerate(self._node_names):
            node_mask = alive & (self._current_idx == node_idx)
            if not node_mask.any():
                continue

            policy = self._get_or_create_node(node_name)
            policy.step(mask=node_mask)

            success = _to_tensor(policy.get_success(), node_mask)
            failure = _to_tensor(policy.get_failure(), node_mask)
            timeout = _to_tensor(policy.get_timeout(), node_mask)

            # Success transitions.
            success_envs = node_mask & success
            if success_envs.any():
                next_name = self._success_map.get(node_name)
                if next_name is not None:
                    self._transition_to_batched(next_name, success_envs)
                else:
                    self._terminal_success[success_envs] = True

            # Failure / timeout transitions.
            fail_envs = node_mask & (failure | timeout) & ~success
            if fail_envs.any():
                next_name = self._fail_map.get(node_name)
                if next_name is not None:
                    self._transition_to_batched(next_name, fail_envs)
                else:
                    self._terminal_failure[fail_envs] = True

    def _transition_to_batched(self, next_name: str, env_mask: torch.Tensor) -> None:
        self._current_idx[env_mask] = self._name_to_idx[next_name]
        next_policy = self._get_or_create_node(next_name)
        next_policy.reset(mask=env_mask)

    def _get_action_batched(self, observation: torch.Tensor) -> ActT:
        """Compute actions from all active nodes and combine."""
        # Find which nodes are active.
        active_nodes = self._current_idx.unique().tolist()

        if len(active_nodes) == 1:
            # Fast path: all envs at the same node.
            node_name = self._node_names[active_nodes[0]]
            policy = self._get_or_create_node(node_name)
            return policy.get_action(observation)

        # Slow path: multiple active nodes — get action from each and merge.
        # This assumes ActT is torch.Tensor or Dict[str, torch.Tensor].
        first_name = self._node_names[active_nodes[0]]
        first_policy = self._get_or_create_node(first_name)
        result = first_policy.get_action(observation)

        for node_idx in active_nodes[1:]:
            node_name = self._node_names[node_idx]
            node_mask = self._current_idx == node_idx
            policy = self._get_or_create_node(node_name)
            node_action = policy.get_action(observation)

            if isinstance(result, dict):
                for key in result:
                    if key in node_action:
                        result[key][node_mask] = node_action[key][node_mask]
            elif isinstance(result, torch.Tensor):
                result[node_mask] = node_action[node_mask]

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unwrap(self, t: torch.Tensor) -> Union[torch.Tensor, bool]:
        if is_batched(self.world.batch_size):
            return t
        return bool(t.item())


def _to_tensor(value: Union[torch.Tensor, bool], like: torch.Tensor) -> torch.Tensor:
    """Coerce a bool or Tensor to a Tensor matching *like*'s shape/device."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.full_like(like, value, dtype=torch.bool)
