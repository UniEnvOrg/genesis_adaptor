"""Tests for the Tachin-glove taxel layout on the Wuji hands; builds the Tianji robot on CPU."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import trimesh
import genesis as gs
from genesis.constants import backend as gs_backend
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_interface.world import WorldEnv

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.tactile_node import _build_link_mesh
from unienv_genesis_collection.robots.tianji_marvin_wuji import TianjiMarvinWuji
from unienv_genesis_collection.robots.wuji_glove_tactile import (
    UPPER_PALM,
    _DATA_DIR,
    _pose_in,
    load_hand_pads,
    make_wuji_glove_tactile,
)

SIDES = ("left", "right")
SPREAD = 1.5
PADS = {side: load_hand_pads(side) for side in SIDES}


@pytest.fixture(scope="module")
def hand():
    world = GenesisWorld()
    robot = TianjiMarvinWuji(world, name="tianji")
    node = make_wuji_glove_tactile(world, robot_name="tianji", sides=SIDES, spread=SPREAD)
    env = WorldEnv(world, [robot, node], render_mode=None)
    env.reset(seed=0, reload=True)
    yield node, world.scene.get_entity("tianji")
    env.close()


def _in_palm_frame(node, entity, side, name):
    pose = _pose_in(entity.get_link(f"{side}_{PADS[side][name]['link']}"), entity.get_link(f"{side}_wuji_palm"))
    return node.probe_positions[f"{side}_{name}"] @ pose[:3, :3].T + pose[:3, 3]


def test_every_pad_is_a_patch_with_its_grid_shape(hand):
    node, _ = hand
    obs = node.get_observation()
    assert set(obs) == {f"{side}_{name}" for side in SIDES for name in PADS[side]}
    for side in SIDES:
        with open(_DATA_DIR / f"config_segments_glove_{side}hand_v5.json", encoding="utf-8") as handle:
            segments = json.load(handle)
        for name, pad in PADS[side].items():
            key = f"{side}_{name}"
            assert obs[key].shape == pad["shape"] == node.observation_space.spaces[key].shape
            assert len(node.probe_positions[key]) == len(pad["cells"])
            assert torch.all(obs[key] >= 0.0)
            if pad["slot"] is not None:  # vendor pads match the glove file exactly
                segment = segments[str(pad["slot"])]
                assert pad["shape"] == tuple(segment["shape"])
                assert len(pad["cells"]) == segment["valid_point_count"]


def test_upper_palm_is_a_band_with_a_bump_per_finger():
    for side in SIDES:
        band = PADS[side][UPPER_PALM]
        rows, cols = band["shape"]
        grid = np.zeros(rows * cols, dtype=bool)
        grid[band["cells"]] = True
        columns = [col for col in grid.reshape(rows, cols).T if col.any()]
        top = np.array([np.flatnonzero(col).max() for col in columns])
        bottom = np.array([np.flatnonzero(col).min() for col in columns])
        x = np.arange(len(columns))
        # Bottom edge is a straight slope between the rounded ends; the top edge rises and falls
        # around it (one bump per finger).
        inner = slice(3, -3)
        straight = np.polyval(np.polyfit(x[inner], bottom[inner], 1), x[inner])
        assert np.abs(bottom[inner] - straight).max() <= 1.0, (side, bottom)
        assert np.ptp(top - np.polyval(np.polyfit(x, top, 1), x)) >= 2.0, (side, top)
        assert cols > 2 * rows  # wider than tall


def test_taxels_lie_on_their_link(hand):
    node, entity = hand
    for side in SIDES:
        for name, pad in PADS[side].items():
            mesh = _build_link_mesh(entity.get_link(f"{side}_{pad['link']}"))
            dist = trimesh.proximity.closest_point(mesh, node.probe_positions[f"{side}_{name}"])[1]
            assert float(dist.max()) < 1e-4, f"{side}_{name}: taxel {dist.max() * 1000:.2f} mm off the surface"


def test_taxels_are_spread_out_and_never_collapse(hand):
    node, entity = hand
    for side in SIDES:
        points = np.concatenate([_in_palm_frame(node, entity, side, name) for name in PADS[side]])
        gaps = cKDTree(points).query(points, k=2)[0][:, 1]
        assert gaps.min() > 0.0008, f"{side}: two taxels only {gaps.min() * 1000:.2f} mm apart"
        finger = np.concatenate([
            _in_palm_frame(node, entity, side, name) for name, pad in PADS[side].items()
            if pad["link"] != "wuji_palm" and pad["slot"] not in (0, 3, 7, 11, 15)
        ])
        spacing = np.median(cKDTree(finger).query(finger, k=2)[0][:, 1])
        assert abs(spacing - 0.002 * SPREAD) < 0.0005, spacing


@pytest.mark.parametrize("side", SIDES)
def test_pads_keep_their_shape(side, hand):
    """Placed taxel-to-taxel distances track the pad's own, up to scaling and squeezing."""
    node, _ = hand
    for name, pad in PADS[side].items():
        placed = pdist(node.probe_positions[f"{side}_{name}"])
        assert np.corrcoef(placed, pdist(pad["xy"]))[0, 1] > 0.95, name


@pytest.mark.parametrize("side", SIDES)
def test_rows_run_toward_the_fingertip_and_columns_follow_the_glove(side, hand):
    """Palm frame: +z toward the fingers, +y toward the right thumb. The glove numbers columns
    toward the thumb on the right hand and away from it on the left, so +y on both."""
    node, entity = hand
    names = ["index_middle", "index_proximal", "middle_middle", "middle_proximal", "ring_middle",
             "ring_proximal", "little_middle", "thenar", "hypothenar", UPPER_PALM]
    for name in names:
        points = _in_palm_frame(node, entity, side, name)
        rows, cols = np.divmod(node.probe_cells[f"{side}_{name}"], PADS[side][name]["shape"][1])
        assert np.corrcoef(rows, points[:, 2])[0, 1] > 0.8, name
        assert np.corrcoef(cols, points[:, 1])[0, 1] > 0.7, name


def test_palm_pads_sit_where_the_glove_puts_them(hand):
    """Thenar on the thumb side, hypothenar on the little-finger side, the band above both."""
    node, _ = hand
    for side, thumb in (("right", 1.0), ("left", -1.0)):
        thenar, hypothenar, band = (
            node.probe_positions[f"{side}_{name}"] for name in ("thenar", "hypothenar", UPPER_PALM)
        )
        assert thumb * thenar[:, 1].mean() > thumb * hypothenar[:, 1].mean()
        for pad in (thenar, hypothenar):  # the band tilts, so compare it with each pad right above it
            over = band[(band[:, 1] > pad[:, 1].min()) & (band[:, 1] < pad[:, 1].max())]
            assert len(over) and over[:, 2].min() > pad[:, 2].max()
        middles = sorted((thenar[:, 1].mean(), hypothenar[:, 1].mean()))
        assert band[:, 1].min() < middles[0] and middles[1] < band[:, 1].max()  # the band spans both
