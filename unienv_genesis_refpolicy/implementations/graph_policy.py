"""GraphRefPolicy — chain reference policies via a success/failure transition graph."""

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch

from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis.utils.batch_dim_utils import (
    is_batched,
    get_batch_aware_shape,
    try_scalar
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

        # Per-env bitmask of the graph nodes visited in the *current* episode
        # (bit i set => env entered node ``self._node_names[i]``).  Leaf phase
        # nodes keep sticky flags across episodes (the graph only resets the
        # node being entered), so a diagnostic that dumps every node's flags
        # would attribute an episode to phases it never ran.  This mask records
        # actual visits so such diagnostics can report only entered nodes.
        self._visited_node_mask = torch.zeros(
            scalar_shape + (len(self._node_names),), dtype=torch.bool, device=device,
        )
        self._visited_node_mask[..., self._name_to_idx[initial]] = True

        # CPU-side mirror of the set of node indices currently occupied by at
        # least one env.  Kept as a (possibly slightly stale superset) list so
        # that get_action/get_action_dict don't need a GPU→CPU sync
        # (``unique().tolist()``) on every call.  Refreshed exactly once per
        # ``_step_batched`` and on ``reset``; transitions append eagerly.
        self._active_node_indices: List[int] = [self._name_to_idx[initial]]

    # ------------------------------------------------------------------
    # Node resolution
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _get_or_create_node(self, name: str) -> BaseRefPolicy[ActT]:
        """Return the policy for *name*, calling the factory if needed."""
        policy = self._node_policies[name]
        if policy is None:
            factory = self._node_specs[name]
            policy = factory()
            self._node_policies[name] = policy
        return policy

    @torch.inference_mode()
    def _clear_factory_nodes(self) -> None:
        """Reset lazily-created nodes so factories are re-invoked next episode."""
        for name, spec in self._node_specs.items():
            if not isinstance(spec, BaseRefPolicy):
                self._node_policies[name] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        initial_idx = self._name_to_idx[self._initial]

        if mask is not None:
            # Partial reset: do NOT clear factory nodes — other envs may be
            # mid-episode inside a lazily-created node, and discarding the
            # policy object would destroy their state.  The masked envs are
            # re-targeted via the node's own reset(mask) on (re-)entry.
            self._current_idx[mask] = initial_idx
            self._terminal_success[mask] = False
            self._terminal_failure[mask] = False
            self._visited_node_mask[mask] = False
            self._visited_node_mask[mask, initial_idx] = True
            # Exact refresh of the active-node mirror (single sync; resets
            # are infrequent).  Keeping it exact here avoids spuriously
            # instantiating factory nodes that are no longer occupied.
            self._active_node_indices = torch.unique(self._current_idx).tolist()
        else:
            self._clear_factory_nodes()
            self._current_idx.fill_(initial_idx)
            self._terminal_success.zero_()
            self._terminal_failure.zero_()
            self._visited_node_mask.zero_()
            self._visited_node_mask[..., initial_idx] = True
            self._active_node_indices = [initial_idx]

        initial_policy = self._get_or_create_node(self._initial)
        initial_policy.reset(mask=mask, seed=seed)

    @torch.inference_mode()
    def step(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        dt: float,
    ) -> None:
        batched = is_batched(self.world.batch_size)

        if batched:
            self._step_batched(mask, dt)
        else:
            self._step_unbatched(dt)

    def get_action(self) -> ActT:
        batched = is_batched(self.world.batch_size)

        if batched:
            action = self._get_action_batched()
        else:
            node_name = self._node_names[self._current_idx.item()]
            policy = self._get_or_create_node(node_name)
            action = policy.get_action()

        if self._static_actions is not None and isinstance(action, dict):
            action = {**self._static_actions, **action}

        return action

    def get_action_dict(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Return the action dict from the currently active node(s).

        For unbatched worlds, delegates to the single active node's
        :meth:`get_action_dict`.  For batched worlds with a single active
        node, delegates to that node.  For batched worlds with multiple
        active nodes, the per-node **raw** actions are merged first (cheap
        elementwise masking) and the controller-native conversion — which
        may involve an IK solve — runs **once** on the merged action via
        :meth:`convert_action_dict`, instead of once per active node.

        Static actions are merged into the final result.  The node's own
        keys take precedence over static actions on conflict (consistent
        with the unbatched behavior).  Static actions are wrapped as
        ``{key: {"action": value}}`` to match the action-dict structure.
        """
        batched = is_batched(self.world.batch_size)

        if not batched:
            node_name = self._node_names[self._current_idx.item()]
            policy = self._get_or_create_node(node_name)
            result = policy.get_action_dict(dt=dt)
            if self._static_actions is not None and isinstance(result, dict):
                result = {**{k: {"action": v} for k, v in self._static_actions.items()}, **result}
            return result

        active_nodes = self._active_node_indices
        if len(active_nodes) == 1:
            node_name = self._node_names[active_nodes[0]]
            policy = self._get_or_create_node(node_name)
            result = policy.get_action_dict(dt=dt)
            if self._static_actions is not None and isinstance(result, dict):
                result = {**{k: {"action": v} for k, v in self._static_actions.items()}, **result}
            return result

        # Multiple active nodes: merge raw actions, then convert once.
        # All nodes share the same action/action-dict schema (homogeneity
        # assumption), so any active node's converter is valid for the
        # merged action.
        action = self._get_action_batched()
        policy = self._get_or_create_node(self._node_names[active_nodes[0]])
        result = policy.convert_action_dict(action, dt=dt)

        if self._static_actions is not None and isinstance(result, dict):
            result = {**{k: {"action": v} for k, v in self._static_actions.items()}, **result}

        return result

    def convert_action_dict(
        self, action: ActT, dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a raw action by delegating to an active node's converter.

        Valid because all graph nodes share the same action schema
        (homogeneity assumption).  Prefers an already-resolved active node
        to avoid lazily instantiating a factory policy.
        """
        for node_idx in self._active_node_indices:
            policy = self._node_policies.get(self._node_names[node_idx])
            if policy is not None:
                return policy.convert_action_dict(action, dt=dt)
        first_policy = self._get_or_create_node(self._node_names[0])
        return first_policy.convert_action_dict(action, dt=dt)

    def get_action_dict_space(self):
        """Return a :class:`DictSpace` describing :meth:`get_action_dict` output.

        Assumes all graph nodes produce the **same** action-dict schema
        (homogeneous graph).

        To avoid unnecessarily instantiating a lazy factory just to query its
        space, this method prefers the first **already-initialized** policy
        among the currently active nodes.  If no active node has been resolved
        yet, it falls back to the first declared node (which may trigger lazy
        instantiation via :meth:`_get_or_create_node`).

        When ``static_actions`` are configured, the merged output includes
        those keys alongside the node's own keys; the space reflects the
        node's native schema (static actions are expected to be compatible
        with or a subset of this schema).
        """
        # Prefer an already-resolved active node to avoid lazy-instantiating
        # a factory policy solely for the space query.
        for node_idx in self._active_node_indices:
            name = self._node_names[node_idx]
            policy = self._node_policies.get(name)
            if policy is not None:
                return policy.get_action_dict_space()

        # Fallback: first declared node (may trigger lazy creation).
        first_policy = self._get_or_create_node(self._node_names[0])
        return first_policy.get_action_dict_space()

    @torch.inference_mode()
    def get_success(self) -> Union[torch.Tensor, bool]:
        return try_scalar(self._terminal_success)

    @torch.inference_mode()
    def get_failure(self) -> Union[torch.Tensor, bool]:
        return try_scalar(self._terminal_failure)

    @torch.inference_mode()
    def get_timeout(self) -> Union[torch.Tensor, bool]:
        # Timeout is handled per-node via fail_edges; graph itself doesn't timeout.
        if is_batched(self.world.batch_size):
            return torch.zeros_like(self._terminal_success)
        return False

    def get_visited_node_names(self) -> Union[List[str], List[List[str]]]:
        """Names of the graph nodes each env has entered in the current episode.

        Returns a flat list of names for an unbatched world, or a per-env list
        of name-lists for a batched world.  Diagnostics should use this to
        report only the phases an episode actually ran (leaf nodes keep sticky
        flags across episodes, so unvisited nodes carry stale state).
        """
        mask = self._visited_node_mask
        names = self._node_names
        if is_batched(self.world.batch_size):
            mask2 = mask.detach().cpu()
            return [
                [names[i] for i in range(len(names)) if bool(mask2[e, i])]
                for e in range(mask2.shape[0])
            ]
        flat = mask.reshape(-1).detach().cpu()
        return [names[i] for i in range(len(names)) if bool(flat[i])]

    # ------------------------------------------------------------------
    # Unbatched implementation
    # ------------------------------------------------------------------

    def _step_unbatched(self, dt: float) -> None:
        if self._terminal_success.item() or self._terminal_failure.item():
            return

        node_name = self._node_names[self._current_idx.item()]
        policy = self._get_or_create_node(node_name)
        policy.step(dt=dt)

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
        next_idx = self._name_to_idx[next_name]
        self._current_idx.fill_(next_idx)
        self._active_node_indices = [next_idx]
        self._visited_node_mask[..., next_idx] = True
        next_policy = self._get_or_create_node(next_name)
        next_policy.reset()
    
    def _step_batched(
        self,
        mask: Optional[torch.Tensor],
        dt: float,
    ) -> None:
        alive = mask if mask is not None else torch.ones_like(self._terminal_success)
        alive = alive & ~self._terminal_success & ~self._terminal_failure

        # Refresh the CPU-side active-node mirror — the single intentional
        # GPU→CPU sync of this step.  The unfiltered unique keeps nodes that
        # only hold terminal envs: get_action still needs those.
        active_nodes = torch.unique(self._current_idx).tolist()
        self._active_node_indices = active_nodes

        # Snapshot node assignment so that an env transitioning this step is
        # NOT stepped again by its destination node in the same tick.  This
        # matches the unbatched semantics (one transition per step); without
        # it, an env moving to a node visited later in this loop would be
        # reset and immediately stepped, and could even cascade through
        # several nodes in a single tick.
        current_idx_snapshot = self._current_idx.clone()

        for node_idx in active_nodes:
            node_name = self._node_names[node_idx]
            node_mask = alive & (current_idx_snapshot == node_idx)
            if not node_mask.any():
                continue

            policy = self._get_or_create_node(node_name)
            policy.step(mask=node_mask, dt=dt)

            success = policy.get_success()
            failure = policy.get_failure()
            timeout = policy.get_timeout()

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
        next_idx = self._name_to_idx[next_name]
        self._current_idx.masked_fill_(env_mask, next_idx)
        self._visited_node_mask[..., next_idx] |= env_mask
        if next_idx not in self._active_node_indices:
            self._active_node_indices.append(next_idx)
        next_policy = self._get_or_create_node(next_name)
        next_policy.reset(mask=env_mask)

    def _get_action_batched(self) -> ActT:
        """Compute actions from all active nodes and combine."""
        # CPU-side mirror — no GPU sync.  May be a slight superset right
        # after transitions; extra nodes only contribute no-op overlays.
        active_nodes = self._active_node_indices

        if len(active_nodes) == 1:
            # Fast path: all envs at the same node.
            node_name = self._node_names[active_nodes[0]]
            policy = self._get_or_create_node(node_name)
            return policy.get_action()

        # Slow path: multiple active nodes — get action from each and merge.
        first_name = self._node_names[active_nodes[0]]
        first_policy = self._get_or_create_node(first_name)
        result = first_policy.get_action()

        def mask_fn(r, a):
            node_mask_expanded = node_mask.reshape((-1,) + (1,) * (r.ndim - 1))
            return torch.where(node_mask_expanded, a, r)

        for node_idx in active_nodes[1:]:
            node_name = self._node_names[node_idx]
            node_mask = self._current_idx == node_idx
            policy = self._get_or_create_node(node_name)
            node_action = policy.get_action()

            result = PyTorchComputeBackend.map_fn_over_arrays_ex(
                result,
                node_action,
                func=mask_fn
            )
        return result
