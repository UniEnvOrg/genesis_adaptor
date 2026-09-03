"""Tests for the viz/ tactile view: layout, colours, and frames. No scene is built."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("PIL")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "viz"))

tactile_view = pytest.importorskip("tactile_view")
TactileView = tactile_view.TactileView

N_AROUND, N_ALONG = 3, 4
N_TAXELS = N_AROUND * N_ALONG

# A stand-in face table: a palm plate and two finger segments, in the mm palm-plane
# coordinates bake.face_row produces.
ROWS = [
    {"link": "right_wuji_palm", "area_cm2": 85.0, "w": 81.0, "h": 105.0,
     "ang": 90.0, "c": [-2.4, 43.4]},
    {"link": "right_finger2_link2", "area_cm2": 8.1, "w": 15.0, "h": 54.1,
     "ang": 82.9, "c": [31.8, 117.7]},
    {"link": "right_finger2_link3", "area_cm2": 5.8, "w": 15.0, "h": 38.6,
     "ang": 82.9, "c": [36.4, 154.8]},
]


class _FakeOption:
    sensor_type = "contact_depth"
    n_around = N_AROUND
    n_along = N_ALONG
    attaches = ()


class _FakeNode:
    """Enough of a TactileSensorNode for the view: its grid, its name, its type."""

    name = "hand_tactile"
    n_taxels = N_TAXELS

    def __init__(self, sensor_type="contact_depth"):
        self.option = _FakeOption()
        self.option.sensor_type = sensor_type


def _view(sensor_type="contact_depth", **kwargs):
    return TactileView(_FakeNode(sensor_type), layout=ROWS, **kwargs)


def _flat_obs(value):
    return {r["link"]: np.full(N_TAXELS, value) for r in ROWS}


def _pixel(view, img, link, k):
    """The colour at the centre of one taxel's pad."""
    poly = next(p for name, idx, p in view._taxels if name == link and idx == k)
    x, y = poly.mean(axis=0)
    return img[int(round(y)), int(round(x))]


def test_layout_covers_every_taxel_of_every_link():
    view = _view()
    assert len(view._taxels) == len(ROWS) * N_TAXELS
    assert len({(name, k) for name, k, _ in view._taxels}) == len(ROWS) * N_TAXELS
    assert {name for name, _, _ in view._taxels} == {r["link"] for r in ROWS}
    assert len(view._outlines) == len(ROWS)


def test_no_pad_is_painted_over_by_a_later_one():
    # the projected link boxes overlap by about a row, so the bands are drawn
    # short; without that, each segment hid the end of the one before it
    import cv2

    view = _view()
    for i, (_, _, poly) in enumerate(view._taxels):
        centre = tuple(float(x) for x in poly.mean(axis=0))
        covered = [
            name for name, _, later in view._taxels[i + 1:]
            if cv2.pointPolygonTest(later, centre, False) >= 0
        ]
        assert not covered


def test_pads_of_one_link_do_not_overlap():
    view = _view()
    palm = [p for name, _, p in view._taxels if name == "right_wuji_palm"]
    centres = np.array([p.mean(axis=0) for p in palm])
    gaps = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)
    assert gaps[~np.eye(len(palm), dtype=bool)].min() > 1.0


def test_frame_shape_and_dtype():
    view = _view(width=400)
    img = view.update(_flat_obs(0.0))
    assert img.shape == (view.height, view.width, 3)
    assert img.dtype == np.uint8
    assert view.width % 2 == 0 and view.height % 2 == 0  # h264 needs even sides


def test_no_contact_is_white_and_vmax_is_red():
    view = _view()
    cold = _pixel(view, view.update(_flat_obs(0.0)), "right_wuji_palm", 0)
    hot = _pixel(view, view.update(_flat_obs(view.vmax)), "right_wuji_palm", 0)
    assert tuple(cold) == (255, 255, 255)
    assert tuple(hot) == (214, 32, 28)


def test_colour_rises_with_contact_and_saturates_above_vmax():
    view = _view()
    mid = _pixel(view, view.update(_flat_obs(view.vmax / 2)), "right_wuji_palm", 0)
    hot = _pixel(view, view.update(_flat_obs(view.vmax)), "right_wuji_palm", 0)
    over = _pixel(view, view.update(_flat_obs(view.vmax * 10)), "right_wuji_palm", 0)
    assert 32 < mid[1] < 255                     # green falls off as red takes over
    assert mid[1] > hot[1]
    assert tuple(over) == tuple(hot)


def test_each_taxel_gets_its_own_reading():
    view = _view()
    obs = _flat_obs(0.0)
    obs["right_finger2_link2"] = np.linspace(0.0, view.vmax, N_TAXELS)
    img = view.update(obs)
    first = _pixel(view, img, "right_finger2_link2", 0)
    last = _pixel(view, img, "right_finger2_link2", N_TAXELS - 1)
    assert tuple(first) == (255, 255, 255)
    assert tuple(last) == (214, 32, 28)
    assert tuple(_pixel(view, img, "right_wuji_palm", 0)) == (255, 255, 255)


def test_batched_observation_picks_the_asked_for_env():
    view = _view()
    obs = {r["link"]: np.stack([np.zeros(N_TAXELS), np.full(N_TAXELS, view.vmax)])
           for r in ROWS}
    assert tuple(_pixel(view, view.update(obs, env=0), "right_wuji_palm", 0)) == (255, 255, 255)
    assert tuple(_pixel(view, view.update(obs, env=1), "right_wuji_palm", 0)) == (214, 32, 28)


def test_kinematic_readings_use_force_magnitude():
    view = _view("kinematic")
    force = np.zeros((N_TAXELS, 3))
    force[:, 1] = view.vmax               # magnitude vmax, off the first axis
    obs = {r["link"]: {"force": force, "torque": np.zeros((N_TAXELS, 3))} for r in ROWS}
    assert tuple(_pixel(view, view.update(obs), "right_wuji_palm", 0)) == (214, 32, 28)


def test_observation_may_be_nested_under_the_node_name():
    view = _view()
    img = view.update({"hand_tactile": _flat_obs(view.vmax)})
    assert tuple(_pixel(view, img, "right_wuji_palm", 0)) == (214, 32, 28)


def test_scene_frame_sits_beside_the_hand():
    view = _view()
    scene = np.zeros((240, 320, 3), np.uint8)
    frame = view.record(_flat_obs(0.0), scene_rgb=scene)
    assert frame.shape[0] == view.height
    assert frame.shape[1] > view.width
    assert frame.shape[1] % 2 == 0


def test_missing_link_raises():
    view = _view()
    obs = _flat_obs(0.0)
    del obs["right_finger2_link3"]
    with pytest.raises(ValueError, match="right_finger2_link3"):
        view.update(obs)


def test_wrong_taxel_count_raises():
    view = _view()
    obs = _flat_obs(0.0)
    obs["right_wuji_palm"] = np.zeros(N_TAXELS + 1)
    with pytest.raises(ValueError, match="expected"):
        view.update(obs)


def test_bad_vmax_and_width_raise():
    with pytest.raises(ValueError, match="vmax"):
        _view(vmax=0.0)
    with pytest.raises(ValueError, match="width"):
        _view(width=16)


def test_save_needs_recorded_frames():
    with pytest.raises(RuntimeError, match="nothing recorded"):
        _view().save("unused.mp4")


def test_save_writes_a_video(tmp_path):
    pytest.importorskip("imageio")
    view = _view(width=200)
    for step in range(5):
        view.record(_flat_obs(view.vmax * step / 4))
    path = view.save(str(tmp_path / "tactile.mp4"), fps=5)
    assert os.path.getsize(path) > 0


# -- against the real hand ---------------------------------------------------
# Builds a Genesis scene on CPU, so it is the slow one here (~15 s).


def test_pad_order_follows_the_probe_order():
    """Pad k must sit where probe k does: rows along the link, columns across it."""
    import math

    pytest.importorskip("genesis")
    common = pytest.importorskip("common")

    env, node = tactile_view.build_hand_with_tactile()
    try:
        view = TactileView(node)
        links = {}
        for attach in node.option.attaches:
            entity, link = attach.get_attachment(node.world)
            links[attach.link_name] = link if link is not None else entity.base_link
        palm = links["right_wuji_palm"]

        for row in view.rows:
            ang = math.radians(row["ang"])
            u = np.array([math.cos(ang), math.sin(ang)])
            v = np.array([-u[1], u[0]])
            probes = common.to_link(
                common.to_world(node.probe_positions[row["link"]], links[row["link"]]),
                palm,
            )
            xy = np.stack([probes[:, 1] * 1000, probes[:, 2] * 1000], 1)
            xy = xy - np.array(row["c"], float)
            index = np.arange(len(xy))
            along = index // view.n_around
            around = index % view.n_around
            # probes run with u along the link and against v across it; the pads
            # are laid out to match, so the correlations pin both directions
            assert np.corrcoef(xy @ u, along)[0, 1] > 0.9, row["link"]
            assert np.corrcoef(xy @ v, around)[0, 1] < -0.9, row["link"]
    finally:
        env.close()
