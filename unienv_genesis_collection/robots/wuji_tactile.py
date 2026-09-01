"""Touch-sensor layout for the Wuji dexterous hand: a band of taxels per instrumented hand link."""

from __future__ import annotations

from typing import Any, Sequence

from unienv_genesis.nodes.sensors import (
    TactileSensorNode,
    TactileSensorNodeOptions,
)
from unienv_genesis.options.attach_option import AttachFromName

__all__ = ["make_wuji_hand_tactile"]


def make_wuji_hand_tactile(
    world: Any,
    *,
    robot_name: str = "tianji",
    sides: Sequence[str] = ("left", "right"),
    sensor_type: str = "contact_depth",
    taxels: tuple[int, int] | None = None,
) -> TactileSensorNode:
    """Build a TactileSensorNode covering the Wuji hand(s) of ``robot_name``; ``taxels`` is ``(n_around, n_along)``."""
    link_names = [f"{side}_wuji_palm" for side in sides] + [
        f"{side}_finger{finger}_link{seg}"  # link1 has no collision geoms; fingertips are geoms on link4
        for side in sides
        for finger in range(1, 6)
        for seg in (2, 3, 4)
    ]
    grid = dict(zip(("n_around", "n_along"), taxels)) if taxels else {}
    return TactileSensorNode(
        world,
        "hand_tactile",
        TactileSensorNodeOptions(
            attaches=[AttachFromName(robot_name, name) for name in link_names],
            sensor_type=sensor_type,
            **grid,
        ),
    )
