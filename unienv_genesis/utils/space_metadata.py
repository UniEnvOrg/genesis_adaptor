"""Space metadata shared by the Genesis nodes and controllers.

Builds the ``metadata`` of the spaces the adaptor declares, using the unienv
vocabulary (``unienv_interface.space.space_utils.metadata_utils``). The
helpers here cover what several modules describe the same way: per-DOF
units and names of a rigid entity's joints, and end-effector / root poses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import genesis as gs

from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata,
    metadata,
    proprio_metadata,
)

__all__ = [
    "DOF_QUANTITY_UNITS",
    "dof_descriptions",
    "dof_names",
    "dof_units",
    "joint_proprio_metadata",
    "joint_action_metadata",
    "position_metadata",
    "quaternion_metadata",
    "pose_frame",
    "attach_extras",
    "XYZ_NAMES",
    "QUAT_WXYZ_NAMES",
]

XYZ_NAMES = ["x", "y", "z"]
QUAT_WXYZ_NAMES = ["qw", "qx", "qy", "qz"]

DOF_QUANTITY_UNITS: Dict[str, Tuple[str, str]] = {
    "position": ("m", "rad"),
    "velocity": ("m/s", "rad/s"),
    "force": ("N", "N*m"),
}
"""``quantity -> (unit of a translational DOF, unit of a rotational DOF)``."""

# Per joint type: (name suffix, translational?) for each of its DOFs.
_JOINT_DOFS = {
    gs.JOINT_TYPE.REVOLUTE: [("", False)],
    gs.JOINT_TYPE.PRISMATIC: [("", True)],
    gs.JOINT_TYPE.SPHERICAL: [("rx", False), ("ry", False), ("rz", False)],
    gs.JOINT_TYPE.FREE: [
        ("x", True), ("y", True), ("z", True),
        ("rx", False), ("ry", False), ("rz", False),
    ],
}


def dof_descriptions(
    entity: Any,
    dofs_idx_local: Optional[Sequence[int]] = None,
) -> List[Tuple[str, bool]]:
    """``(name, translational)`` of each DOF of a rigid entity.

    DOFs are listed in the order of ``dofs_idx_local`` (all DOFs of the
    entity when ``None``), the order the entity's ``get_dofs_*`` methods
    return them in. A single-DOF joint names its DOF after the joint; the DOFs
    of a free / spherical joint get ``_x`` .. ``_rz`` suffixes.
    """
    by_dof: Dict[int, Tuple[str, bool]] = {}
    for joint in entity.joints:
        idxs = [int(i) for i in joint.dofs_idx_local]
        if not idxs:
            continue
        layout = _JOINT_DOFS.get(joint.type)
        if layout is None or len(layout) != len(idxs):
            layout = [(str(i), False) for i in range(len(idxs))]
        for idx, (suffix, translational) in zip(idxs, layout):
            name = joint.name if len(idxs) == 1 else f"{joint.name}_{suffix}"
            by_dof[idx] = (name, translational)
    order = range(entity.n_dofs) if dofs_idx_local is None else dofs_idx_local
    return [by_dof[int(i)] for i in order]


def dof_names(entity: Any, dofs_idx_local: Optional[Sequence[int]] = None) -> List[str]:
    """Names of the selected DOFs (see :func:`dof_descriptions`)."""
    return [name for name, _ in dof_descriptions(entity, dofs_idx_local)]


def dof_units(
    entity: Any,
    dofs_idx_local: Optional[Sequence[int]] = None,
    quantity: str = "position",
) -> List[str]:
    """Per-DOF units of ``quantity`` (``"position"``, ``"velocity"`` or ``"force"``).

    Translational (prismatic / free-joint linear) DOFs are in ``m``, ``m/s``,
    ``N``; rotational (revolute / spherical / free-joint angular) DOFs in
    ``rad``, ``rad/s``, ``N*m``.
    """
    translational_unit, rotational_unit = DOF_QUANTITY_UNITS[quantity]
    return [
        translational_unit if translational else rotational_unit
        for _, translational in dof_descriptions(entity, dofs_idx_local)
    ]


def joint_proprio_metadata(
    entity: Any,
    dofs_idx_local: Optional[Sequence[int]],
    quantity: str,
    **extra: Any,
) -> Dict[str, Any]:
    """``kind="proprio"`` metadata of a per-DOF joint reading."""
    return proprio_metadata(
        units=dof_units(entity, dofs_idx_local, quantity),
        names=dof_names(entity, dofs_idx_local),
        **extra,
    )


def joint_action_metadata(
    entity: Any,
    dofs_idx_local: Optional[Sequence[int]],
    control: str,
    **extra: Any,
) -> Dict[str, Any]:
    """``kind="action"`` metadata of absolute per-DOF joint targets.

    ``control`` is ``"joint_position"`` or ``"joint_velocity"``.
    """
    quantity = {"joint_position": "position", "joint_velocity": "velocity"}[control]
    return action_metadata(
        control=control,
        mode="absolute",
        units=dof_units(entity, dofs_idx_local, quantity),
        names=dof_names(entity, dofs_idx_local),
        **extra,
    )


def pose_frame(has_base_frame: bool) -> str:
    """The ``frame`` of a pose expressed in a configured base frame, else the world."""
    return "base" if has_base_frame else "world"


def position_metadata(kind: str = "proprio", *, frame: str, **extra: Any) -> Dict[str, Any]:
    """Metadata of a 3D position in metres (``names`` x, y, z)."""
    return metadata(kind, units="m", frame=frame, names=XYZ_NAMES, **extra)


def quaternion_metadata(kind: str = "proprio", *, frame: str, **extra: Any) -> Dict[str, Any]:
    """Metadata of a ``(w, x, y, z)`` orientation quaternion."""
    return metadata(kind, rotation="quat_wxyz", frame=frame, names=QUAT_WXYZ_NAMES, **extra)


def attach_extras(name: str, attach: Any) -> Dict[str, str]:
    """``source`` (the node name) plus the ``entity`` / ``link`` an attachment provider names.

    Providers that resolve their target otherwise (no ``entity_name`` /
    ``link_name`` string attributes) contribute nothing.
    """
    extra = {"source": name}
    for key, attr in (("entity", "entity_name"), ("link", "link_name")):
        value = getattr(attach, attr, None)
        if isinstance(value, str):
            extra[key] = value
    return extra
