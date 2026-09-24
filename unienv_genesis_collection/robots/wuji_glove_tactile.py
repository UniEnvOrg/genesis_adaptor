"""Touch-sensor layout for the Wuji hand that copies the Tachin tactile glove's pad pattern.

The glove's vendor geometry (``tachin_glove/config_segments_glove_{side}hand_v5.json``,
copied unmodified from ``tachin_gloves``) gives every taxel of its 15 pads a position on a
flattened palm-side view of the hand, in mm. Each pad becomes one patch on the matching
Wuji link, keeping its shape and its place relative to the other pads rather than exact
distances:

* finger pads are spread out by ``spread`` (the Wuji is bigger than a human hand), wrapped
  around the link's palm side and laid along it; fingertip pads end just short of the tip,
  the others are centred on their segment. A pad too long or too wide for its segment is
  squeezed to fit.
* the palm pads -- thenar, hypothenar, and the ``upper_palm`` band -- are scaled together
  onto the palm plate, as large as fits, top edge just under the fingers. The Wuji palm is
  smaller than the glove's, so thenar and hypothenar are then spread downward toward the
  wrist, as far toward ``spread`` as they fit.

``upper_palm`` is the scalloped band across the top of the palm, one lobe under each finger.
It is on the physical glove but has no taxels in the vendor file (slots 6, 10, 14 and 17
were dropped), so it is drawn here -- a level strip just above the thenar and hypothenar
with a bump per finger, on a 2.5 mm grid -- and has no real-glove reading to compare against.

Each patch reads back as the pad's ``(rows, cols)`` grid, row 0 nearest the wrist and
columns in the glove's own order, so ``obs[f"{side}_{name}"]`` lines up cell for cell
with ``TachinGlove.read()["tactile"][slot]``. Cells the pad has no taxel for read 0. The
one known mismatch: the glove's wiring reports 13 hypothenar rows, its geometry (and so
this layout) only 12.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from genesis.utils.geom import trans_quat_to_T

from unienv_genesis.nodes.sensors import TactileSensorNode, TactileSensorNodeOptions, TaxelPatch
from unienv_genesis.nodes.sensors.tactile_node import _FACE_AXES, _build_link_mesh, _raycast_surface
from unienv_genesis.options.attach_option import AttachFromName

from .wuji_tactile import _THUMB_FACE_AXES

__all__ = ["GLOVE_REGIONS", "UPPER_PALM", "load_glove_pads", "load_hand_pads", "make_wuji_glove_tactile"]

_DATA_DIR = Path(__file__).with_name("tachin_glove")

GLOVE_REGIONS: Dict[int, Tuple[str, str]] = {
    # vendor slot id: (patch name, Wuji link it sits on). Slots 6, 10, 14 and 17 were
    # removed from the vendor file; the little finger has no proximal pad.
    0: ("thumb_distal", "finger1_link4"),
    1: ("thumb_proximal", "finger1_link3"),
    2: ("thenar", "wuji_palm"),
    3: ("index_distal", "finger2_link4"),
    4: ("index_middle", "finger2_link3"),
    5: ("index_proximal", "finger2_link2"),
    7: ("middle_distal", "finger3_link4"),
    8: ("middle_middle", "finger3_link3"),
    9: ("middle_proximal", "finger3_link2"),
    11: ("ring_distal", "finger4_link4"),
    12: ("ring_middle", "finger4_link3"),
    13: ("ring_proximal", "finger4_link2"),
    15: ("little_distal", "finger5_link4"),
    16: ("little_middle", "finger5_link3"),
    18: ("hypothenar", "wuji_palm"),
}
"""Glove slot id -> (patch name, Wuji link suffix); the observation key is ``f"{side}_{name}"``."""

UPPER_PALM = "upper_palm"
"""Patch name of the sim-drawn band under the fingers; it has no vendor slot."""

_TIP_SLOTS = (0, 3, 7, 11, 15)
_TIP_CLEARANCE = 0.0015  # metres left between a fingertip pad's last row and the end of the link
_END_CLEARANCE = 0.001  # metres left at both ends of any other finger segment
_MAX_WRAP = np.radians(80.0)  # widest a finger pad may reach around the link, each side of the palm side

# Upper-palm band, as on the physical glove: a level strip sitting just above the thenar and
# hypothenar pads, spanning from the hypothenar's outer edge to the thenar's, with one bump on
# its top edge per finger (index, middle, ring, little; the middle bump highest, the little lowest).
_UPPER_PALM_GAP = 0.003  # between the band's bottom and the pad below it
_UPPER_PALM_HALF_WIDTH = 0.005
_UPPER_PALM_LOBE_RADIUS = 0.0075  # > half width, so the lobes stand proud of the band's top edge
_UPPER_PALM_LOBE_RISE = (0.0, 0.002, 0.0, -0.002)  # index -> little
_UPPER_PALM_LOBE_HALF_RISE = _UPPER_PALM_LOBE_RADIUS - _UPPER_PALM_HALF_WIDTH  # lobe bottoms sit on the band's bottom
_UPPER_PALM_PITCH = 0.0025  # the band is drawn, not in the vendor file, so its grid is free to pick
_GLOVE_PITCH = 0.002

_PALM_SCALES = np.arange(1.5, 0.45, -0.05)  # tried scales for the palm layout as a whole
_PALM_SHIFTS = np.arange(0.0, 0.040, 0.002)  # tried drops of the palm pads below their highest start
_PALM_MIN_FACING = 0.6  # a palm taxel's surface must face at least this much toward the palm side
_PALM_MIN_GAP = 0.8  # taxels of two different palm pads stay this many pitches apart


def load_glove_pads(side: str) -> Dict[int, Dict[str, Any]]:
    """Per glove slot: ``shape`` (rows, cols), ``cells`` flat index of each taxel, ``xy`` (N, 2) flattened-hand position in metres."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right'; got {side!r}.")
    with open(_DATA_DIR / f"config_segments_glove_{side}hand_v5.json", encoding="utf-8") as handle:
        segments = json.load(handle)
    pads = {}
    for slot, segment in segments.items():
        cells = [i for i, point in enumerate(segment["coordinates"]) if point is not None]
        xy = np.array([segment["coordinates"][i][:2] for i in cells], dtype=np.float64) / 1000.0
        pads[int(slot)] = {"shape": tuple(segment["shape"]), "cells": cells, "xy": xy}
    return pads


def _distance_to_polyline(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    best = np.full(len(points), np.inf)
    for start, end in zip(vertices[:-1], vertices[1:]):
        edge = end - start
        t = np.clip((points - start) @ edge / (edge @ edge), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (start + t[:, None] * edge), axis=1))
    return best


def _upper_palm_pad(pads: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """The band under the fingers, drawn on its own grid in the same flattened-hand frame as the vendor pads."""
    thenar, hypothenar = pads[2]["xy"], pads[18]["xy"]
    toward_little = np.sign(hypothenar[:, 0].mean() - thenar[:, 0].mean())  # -1 on the right glove, +1 on the left
    ends = np.array([
        [(thenar[:, 0] * -toward_little).max() * -toward_little, thenar[:, 1].max()],  # index end, over the thenar
        [(hypothenar[:, 0] * toward_little).max() * toward_little, hypothenar[:, 1].max()],  # little end
    ])
    ends[:, 0] += toward_little * np.array([1.0, -1.0]) * _UPPER_PALM_LOBE_RADIUS  # keep the end lobes inside
    ends[:, 1] += _UPPER_PALM_GAP + _UPPER_PALM_HALF_WIDTH
    spine = ends[0] + np.linspace(0.0, 1.0, len(_UPPER_PALM_LOBE_RISE))[:, None] * (ends[1] - ends[0])
    lobes = spine + np.column_stack([np.zeros(len(spine)), _UPPER_PALM_LOBE_HALF_RISE + np.array(_UPPER_PALM_LOBE_RISE)])

    lo = np.minimum(spine.min(axis=0) - _UPPER_PALM_HALF_WIDTH, lobes.min(axis=0) - _UPPER_PALM_LOBE_RADIUS)
    hi = np.maximum(spine.max(axis=0) + _UPPER_PALM_HALF_WIDTH, lobes.max(axis=0) + _UPPER_PALM_LOBE_RADIUS)
    cols = int(np.floor((hi[0] - lo[0]) / _UPPER_PALM_PITCH)) + 1
    rows = int(np.floor((hi[1] - lo[1]) / _UPPER_PALM_PITCH)) + 1
    xs, ys = np.meshgrid(lo[0] + _UPPER_PALM_PITCH * np.arange(cols), lo[1] + _UPPER_PALM_PITCH * np.arange(rows))
    grid = np.column_stack([xs.ravel(), ys.ravel()])  # row-major, row 0 nearest the wrist

    inside = (_distance_to_polyline(grid, spine) <= _UPPER_PALM_HALF_WIDTH) | (
        (np.linalg.norm(grid[:, None] - lobes[None], axis=2).min(axis=1) <= _UPPER_PALM_LOBE_RADIUS)
        & (grid[:, 1] >= np.interp(grid[:, 0], np.sort(spine[:, 0]), spine[np.argsort(spine[:, 0]), 1]))
    )  # lobes only add to the top edge; the bottom edge stays straight
    cells = np.flatnonzero(inside).tolist()
    return {"shape": (rows, cols), "cells": cells, "xy": grid[cells]}


def load_hand_pads(side: str) -> Dict[str, Dict[str, Any]]:
    """Every pad this layout places, keyed by patch name: the vendor pads plus ``upper_palm``.

    Each has ``shape``, ``cells``, ``xy`` (as :func:`load_glove_pads`), ``link`` (Wuji link suffix)
    and ``slot`` (vendor slot id, None for ``upper_palm``).
    """
    pads = load_glove_pads(side)
    hand = {name: {**pads[slot], "link": link, "slot": slot} for slot, (name, link) in GLOVE_REGIONS.items()}
    hand[UPPER_PALM] = {**_upper_palm_pad(pads), "link": "wuji_palm", "slot": None}
    return hand


def _pad_uv(pad: Dict[str, Any]) -> np.ndarray:
    """Each taxel as (across, along) metres from the pad centre, along = the pad's row direction (toward the fingertip).

    'across' points the flattened view's +x, which is ``along x palm-normal`` on both gloves.
    """
    rows, cols = pad["shape"]
    grid = np.full((rows * cols, 2), np.nan)
    grid[pad["cells"]] = pad["xy"]
    steps = np.diff(grid.reshape(rows, cols, 2), axis=0).reshape(-1, 2)
    steps = steps[np.isfinite(steps).all(axis=1)]
    along = np.median(steps, axis=0)  # median: the rows that wrap the fingertip are irregular
    along /= np.linalg.norm(along)
    across = np.array([along[1], -along[0]])
    offset = pad["xy"] - pad["xy"].mean(axis=0)
    return np.stack([offset @ across, offset @ along], axis=1)


def _link_axes(mesh, face_axis: str):
    """Link-frame unit vectors (palm-normal, toward-tip, across) plus the long-axis index."""
    lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    long_axis = int(np.argmax(hi - lo))
    axis, sign = _FACE_AXES[face_axis]
    palmar = np.zeros(3)
    palmar[axis] = sign
    distal = np.zeros(3)
    distal[long_axis] = 1.0 if (lo + hi)[long_axis] >= 0.0 else -1.0  # link origin sits at its base joint
    return palmar, distal, np.cross(distal, palmar), long_axis


def _place_on_segment(uv: np.ndarray, face_axis: str, at_tip: bool):
    def place(link):
        mesh = _build_link_mesh(link)
        lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
        palmar, distal, across, long_axis = _link_axes(mesh, face_axis)
        centre = 0.5 * (lo + hi)
        base, tip = sorted(distal[long_axis] * np.array([lo[long_axis], hi[long_axis]]))
        room = (tip - base) - _TIP_CLEARANCE - _END_CLEARANCE
        along = uv[:, 1] - uv[:, 1].min()
        if along.max() > room:  # pad longer than the segment: squeeze it rather than run off the end
            along *= room / along.max()
        if at_tip:
            stations = tip - _TIP_CLEARANCE - along.max() + along
        else:
            stations = 0.5 * (base + tip) - 0.5 * along.max() + along
        cross = [i for i in range(3) if i != long_axis]
        radius = 0.25 * float(np.sum(hi[cross] - lo[cross]))
        angles = uv[:, 0] / radius  # arc length across the pad matches the (spread) glove's
        if np.abs(angles).max() > _MAX_WRAP:  # too wide to stay on the palm side: squeeze it
            angles *= _MAX_WRAP / np.abs(angles).max()
        origins = np.tile(centre, (len(uv), 1))
        origins[:, long_axis] = distal[long_axis] * stations
        dirs = np.cos(angles)[:, None] * palmar + np.sin(angles)[:, None] * across
        return _raycast_surface(mesh, origins, dirs)

    return place


def _pose_in(link, ancestor) -> np.ndarray:
    """``link``'s 4x4 pose in ``ancestor``'s frame with every joint in between at zero."""
    entity = link.entity
    pose = np.eye(4)
    while link.idx != ancestor.idx:
        pose = trans_quat_to_T(np.asarray(link.pos, float), np.asarray(link.quat, float)) @ pose
        if link.parent_idx < 0:
            raise ValueError(f"'{ancestor.name}' is not an ancestor of the starting link.")
        link = entity.links[link.parent_idx - entity.link_start]
    return pose


class _PalmLayout:
    """Places all palm pads together, keeping their shapes and where they sit relative to one another.

    The whole palm layout is scaled as one, as large as fits on the palm's front face. The glove's
    own pads (thenar, hypothenar) are then spread from the middle of their top edge toward
    ``spread`` as far as they fit without running into a neighbour; the drawn ``upper_palm`` band
    keeps the layout scale.
    """

    def __init__(self, pads: Dict[str, Dict[str, Any]], spread: float):
        self.pads = pads
        self.spread = spread
        self._placed: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

    def place(self, name: str):
        def place(palm):
            if self._placed is None:
                self._placed = self._solve(palm)
            return self._placed[name]

        return place

    def _solve(self, palm) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        mesh = _build_link_mesh(palm)
        placed = None
        for scale in _PALM_SCALES:
            placed = self._fit(mesh, scale, scale)
            if placed is not None:
                break
        if placed is None:
            raise ValueError(f"the glove's palm pads do not fit on '{palm.name}'.")
        for spread in np.arange(self.spread, scale, -0.05):
            wider = self._fit(mesh, scale, spread)
            if wider is not None:
                return wider
        return placed

    def _fit(self, mesh, scale: float, spread: float) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Palm-frame taxels at this layout scale and glove-pad spread, dropped as little as possible from the palm's
        top; None if no drop puts every taxel on the palm's front face with the pads kept apart."""
        palmar, distal, across, _ = _link_axes(mesh, "+x")
        plane = np.stack([across, distal])  # palm-frame point -> (across, along) on the palm face
        face = np.asarray(mesh.vertices) @ plane.T
        face_lo, face_hi = face.min(axis=0), face.max(axis=0)
        centre = 0.5 * (mesh.bounds[0] + mesh.bounds[1])

        names = list(self.pads)
        # Each pad grows from the middle of its top edge, so the thenar and hypothenar stay tucked
        # under the band and extend toward the wrist, as on the glove.
        tops = np.array([
            [self.pads[name]["xy"][:, 0].mean(), self.pads[name]["xy"][:, 1].max()] for name in names
        ])
        middle = 0.5 * (tops.min(axis=0) + tops.max(axis=0))
        shapes = [
            (self.pads[name]["xy"] - t) * (scale if self.pads[name]["slot"] is None else spread) + (t - middle) * scale
            for name, t in zip(names, tops)
        ]
        owner = np.repeat(np.arange(len(names)), [len(shape) for shape in shapes])
        splits = np.cumsum([len(shape) for shape in shapes])[:-1]
        layout = np.concatenate(shapes)
        apart = np.linalg.norm(layout[:, None] - layout[None], axis=2)
        if apart[owner[:, None] != owner[None]].min(initial=np.inf) < _PALM_MIN_GAP * _GLOVE_PITCH * spread:
            return None

        dirs = np.tile(palmar, (len(layout), 1))
        width = 0.5 * (layout[:, 0].min() + layout[:, 0].max())
        top = np.array([0.5 * (face_lo[0] + face_hi[0]) - width, face_hi[1] - 0.002 - layout[:, 1].max()])
        for shift in _PALM_SHIFTS:
            origins = centre - plane.T @ (plane @ centre) + (layout + top - [0.0, shift]) @ plane
            if not mesh.ray.intersects_any(origins, dirs).all():
                continue
            pos, nrm = _raycast_surface(mesh, origins, dirs)
            if (nrm @ palmar).min() >= _PALM_MIN_FACING:
                return dict(zip(names, zip(np.split(pos, splits), np.split(nrm, splits))))
        return None


def make_wuji_glove_tactile(world: Any, *, robot_name: str = "tianji",
                            sides: Sequence[str] = ("left", "right"), sensor_type: str = "contact_depth",
                            spread: float = 1.5, probe_radius: Optional[float] = None) -> TactileSensorNode:
    """Build a TactileSensorNode laying the Tachin glove's pad pattern onto ``robot_name``'s Wuji hand(s).

    ``spread`` scales the glove's 2 mm taxel pitch on the fingers (the palm gets as much as fits).
    ``probe_radius`` defaults to half the finger pitch, so neighbouring finger taxels just touch.
    """
    if spread <= 0.0:
        raise ValueError(f"spread must be positive; got {spread}.")
    patches = []
    for side in sides:
        pads = load_hand_pads(side)
        axes = _THUMB_FACE_AXES.get(side, {})
        palm = _PalmLayout({name: pad for name, pad in pads.items() if pad["link"] == "wuji_palm"}, spread)
        for name, pad in pads.items():
            if pad["link"] == "wuji_palm":
                place = palm.place(name)
            else:
                place = _place_on_segment(spread * _pad_uv(pad), axes.get(pad["link"], "+x"), pad["slot"] in _TIP_SLOTS)
            patches.append(
                TaxelPatch(
                    name=f"{side}_{name}",
                    attach=AttachFromName(robot_name, f"{side}_{pad['link']}"),
                    shape=pad["shape"],
                    cells=pad["cells"],
                    place=place,
                )
            )
    radius = probe_radius if probe_radius is not None else 0.5 * _GLOVE_PITCH * spread
    return TactileSensorNode(
        world,
        "hand_tactile",
        TactileSensorNodeOptions(patches=patches, sensor_type=sensor_type, probe_radius=radius),
    )
