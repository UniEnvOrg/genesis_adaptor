"""Forward kinematics across Genesis versions.

Genesis removed ``RigidEntity.forward_kinematics`` (genesis-world#3233); its
replacement, ``KinematicSolver.forward_kinematics_query(entity, qpos)``, takes
the entity's full configuration and returns the poses of all its links.
:func:`forward_kinematics` keeps the old call shape on top of either API.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch

__all__ = ["forward_kinematics"]


def forward_kinematics(
    entity: Any,
    qpos: torch.Tensor,
    qs_idx_local: Optional[Sequence[int]] = None,
    links_idx_local: Optional[Sequence[int]] = None,
    envs_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """World-frame link poses ``entity`` would have at ``qpos``; the live state is left as it is.

    Args:
        entity: A built ``RigidEntity``.
        qpos: The coordinates ``qs_idx_local`` (all of them when ``None``),
            ``(n_qs,)`` or ``(n_envs, n_qs)``; the other coordinates keep
            their current values.
        qs_idx_local: Entity-local coordinates ``qpos`` sets.
        links_idx_local: Entity-local links to return (all when ``None``).
        envs_idx: Environments to evaluate (all when ``None``).

    Returns:
        ``(links_pos, links_quat)``: ``(n_envs, n_links, 3)`` and
        ``(n_envs, n_links, 4)`` (wxyz), without the environment dimension in
        an unbatched scene.
    """
    query = getattr(entity.solver, "forward_kinematics_query", None)
    if query is None:  # Genesis before genesis-world#3233
        return entity.forward_kinematics(
            qpos, qs_idx_local=qs_idx_local, links_idx_local=links_idx_local, envs_idx=envs_idx,
        )

    if qs_idx_local is None:
        full = qpos
    else:
        full = entity.get_qpos(envs_idx=envs_idx).clone()
        idx = torch.as_tensor(qs_idx_local, dtype=torch.long, device=full.device)
        full[..., idx] = torch.as_tensor(qpos, dtype=full.dtype, device=full.device)
    links_pos, links_quat = query(entity, full, envs_idx=envs_idx)
    if links_idx_local is not None:
        idx = torch.as_tensor(links_idx_local, dtype=torch.long, device=links_pos.device)
        links_pos = links_pos[..., idx, :]
        links_quat = links_quat[..., idx, :]
    return links_pos, links_quat
