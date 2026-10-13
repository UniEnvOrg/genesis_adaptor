from typing import Optional, Sequence
import genesis as gs
from genesis.engine import entities as gs_entities
import torch


class GravityFeedforward:
    """Gravity-compensation feedforward torque for a subset of an entity's DOFs.

    Genesis' position-mode actuator force is::

        force = Kp (q* - q) - Kd (q_dot - q_dot*) + act_bias[0]

    evaluated every substep.  ``act_bias[0]`` is a constant feedforward slot
    (``set_dofs_kp`` resets it to 0), so writing the gravity torque ``g(q)``
    there turns the joint impedance into PD + gravity compensation without
    removing the gravity load from the joints (the motor supplies it).

    ``g(q)`` is computed from the current link COMs / masses and joint anchors /
    axes, batched over envs with no per-joint Python loop.  For controlled DOF
    ``j`` with subtree links ``l``::

        revolute:  g_j = -g . (a_j x sum_l m_l (c_l - p_j))
        prismatic: g_j = -(sum_l m_l) (a_j . g)

    Only the robot's own links are compensated; held / grasped objects are not.
    """

    def __init__(
        self,
        entity: gs_entities.RigidEntity,
        dofs_idx_local: Optional[Sequence[int]] = None,
    ) -> None:
        self.dofs_idx_local = (
            list(range(entity.n_dofs)) if dofs_idx_local is None else list(dofs_idx_local)
        )
        self.rebuild(entity)

    def rebuild(self, entity: gs_entities.RigidEntity) -> None:
        """(Re)build the per-DOF joint table and subtree mask for *entity*.

        Must be called again when the entity object is replaced (world reload).

        Raises:
            ValueError: If a controlled DOF belongs to a joint that is neither
                revolute nor prismatic.
        """
        self.entity = entity
        solver = entity.solver
        owner = {}
        for link in entity.links:
            for joint in link.joints:
                for d in joint.dofs_idx_local:
                    owner[d] = (link, joint)

        joints_idx, is_revolute = [], []
        mask = torch.zeros(len(self.dofs_idx_local), entity.n_links, dtype=gs.tc_float, device=gs.device)
        for row, d in enumerate(self.dofs_idx_local):
            link, joint = owner[d]
            if joint.type not in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                raise ValueError(
                    f"Gravity compensation supports revolute and prismatic joints only, "
                    f"but DOF {d} belongs to joint '{joint.name}' of type {joint.type}."
                )
            joints_idx.append(joint.idx)
            is_revolute.append(joint.type == gs.JOINT_TYPE.REVOLUTE)
            # link l is in the subtree when the joint's link is one of its ancestors (or itself)
            for other in entity.links:
                i = other.idx
                while i >= 0:
                    if i == link.idx:
                        mask[row, other.idx_local] = 1.0
                        break
                    i = solver.links[i].parent_idx
        self._joints_idx = joints_idx
        self._is_revolute = torch.tensor(is_revolute, dtype=torch.bool, device=gs.device)
        self._subtree_mask = mask

    def torque(self) -> torch.Tensor:
        """Gravity torque ``g(q)``: ``(n_dofs,)``, or ``(n_envs, n_dofs)`` in a batched scene."""
        entity, solver = self.entity, self.entity.solver
        com = entity.get_links_pos(ref=gs.link_ref_frame.link_COM)         # (..., L, 3)
        mass = entity.get_links_mass().to(com)                               # (L,) or (B, L)
        gravity = solver.get_gravity().to(com)[..., None, :]                 # (..., 1, 3)
        anchor = solver.get_joints_anchor_pos(self._joints_idx).to(com)      # (..., n, 3)
        axis = solver.get_joints_anchor_axis(self._joints_idx).to(com)       # (..., n, 3)

        subtree_mass = mass @ self._subtree_mask.T                           # (..., n)
        moment = self._subtree_mask @ (mass[..., None] * com)                # (..., n, 3)
        moment = moment - subtree_mass[..., None] * anchor
        revolute = -(torch.cross(axis, moment, dim=-1) * gravity).sum(-1)
        prismatic = -subtree_mass * (axis * gravity).sum(-1)
        return torch.where(self._is_revolute, revolute, prismatic)

    def apply(self) -> None:
        """Write ``g(q)`` into ``act_bias[0]``, keeping the current Kp / Kd terms."""
        _, bias1, bias2 = self.entity.get_dofs_act_bias(self.dofs_idx_local)
        self.entity.set_dofs_act_bias(self.torque(), bias1, bias2, self.dofs_idx_local)
