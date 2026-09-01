"""Touch-sensor layout for the Wuji dexterous hand: a band of taxels per instrumented hand link."""

from __future__ import annotations

from typing import Any, Sequence

from unienv_genesis.nodes.sensors import (
    TactileSensorNode,
    TactileSensorNodeOptions,
)
from unienv_genesis.options.attach_option import AttachFromName

__all__ = ["make_wuji_hand_tactile"]

# The thumb's links are rotated relative to the other four fingers, so the
# default "+x" face points off the palmar side for them. Measured against the
# palm's own +x: link2 is ~68 deg off on the right and mirrored on the left,
# and the left thumb's link3/link4 face the back of the hand outright.
_THUMB_FACE_AXES = {
    "right": {"finger1_link2": "-y"},
    "left": {"finger1_link2": "+y", "finger1_link3": "-x", "finger1_link4": "-x"},
}


def make_wuji_hand_tactile(
    world: Any,
    *,
    robot_name: str = "tianji",
    sides: Sequence[str] = ("left", "right"),
    sensor_type: str = "contact_depth",
    taxels: tuple[int, int] | None = None,
    face_axis_overrides: dict[str, str] | None = None,
) -> TactileSensorNode:
    """Build a TactileSensorNode covering the Wuji hand(s) of ``robot_name``; ``taxels`` is ``(n_around, n_along)``.

    The thumb links get a corrected ``face_axis`` by default;
    ``face_axis_overrides`` is merged on top of that, keyed by full link name.
    """
    link_names = [f"{side}_wuji_palm" for side in sides] + [
        f"{side}_finger{finger}_link{seg}"  # link1 has no collision geoms; fingertips are geoms on link4
        for side in sides
        for finger in range(1, 6)
        for seg in (2, 3, 4)
    ]
    axes = {
        f"{side}_{link}": axis
        for side in sides
        for link, axis in _THUMB_FACE_AXES.get(side, {}).items()
    }
    axes.update(face_axis_overrides or {})

    grid = dict(zip(("n_around", "n_along"), taxels)) if taxels else {}
    return TactileSensorNode(
        world,
        "hand_tactile",
        TactileSensorNodeOptions(
            attaches=[AttachFromName(robot_name, name) for name in link_names],
            sensor_type=sensor_type,
            face_axis_overrides=axes,
            **grid,
        ),
    )
