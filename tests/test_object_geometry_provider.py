"""ObjectGeometryGraspCandidateProvider selection hooks (CPU only, no scenes).

Covers:
  - ``lowest_finger_z``: the clearance is measured at the lowest point of the
    open fingers, not at the grasp point
  - ``filter``: table clearance + top-down preference, with fallbacks
  - ``cost``: approach tilt, contact depth below the top, centring
  - both hooks are off by default
"""

import math

import pytest
import torch

from unienv_genesis_collection.objects.base_object import GraspCandidates
from unienv_genesis_collection.objects.mesh_object import CONTACT_INSET
from unienv_genesis_refpolicy.grasp_providers.object_geometry import (
    FINGER_THICKNESS,
    FINGER_WIDTH,
    ObjectGeometryGraspCandidateProvider,
    lowest_finger_z,
)

WIDTH, DEPTH = 0.085, 0.035


def _candidates(position, approach, opening):
    return GraspCandidates(
        position=torch.tensor(position, dtype=torch.float32),
        approach_direction=torch.tensor(approach, dtype=torch.float32),
        opening_direction=torch.tensor(opening, dtype=torch.float32),
    )


class _BoxEntity:
    """Entity stub exposing an AABB (min corner, max corner)."""

    def __init__(self, low, high):
        self._aabb = torch.tensor([low, high], dtype=torch.float32)

    def get_AABB(self):
        return self._aabb


def _provider(entity=None, **kwargs):
    return ObjectGeometryGraspCandidateProvider(
        target_object=None,
        target_entity=entity,
        gripper_width=WIDTH,
        gripper_depth=DEPTH,
        **kwargs,
    )


def test_straight_down_grasp_drops_by_nothing():
    # Approach down, closing horizontally: the fingertips are level with the grasp point.
    candidates = _candidates([[0.0, 0.0, 0.10]], [[0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0]])
    lowest = lowest_finger_z(candidates, gripper_width=WIDTH, gripper_depth=DEPTH)
    assert float(lowest[0]) == pytest.approx(0.10)


def test_tilted_closing_axis_lowers_one_finger():
    # Closing axis tilted 30° out of the horizontal plane (approach still ⟂).
    tilt = math.radians(30.0)
    opening = [math.cos(tilt), 0.0, math.sin(tilt)]
    approach = [math.sin(tilt), 0.0, -math.cos(tilt)]
    candidates = _candidates([[0.0, 0.0, 0.10]], [approach], [opening])
    lowest = lowest_finger_z(candidates, gripper_width=WIDTH, gripper_depth=DEPTH)
    expected = 0.10 - (WIDTH / 2 + FINGER_THICKNESS) * math.sin(tilt)
    assert float(lowest[0]) == pytest.approx(expected, abs=1e-6)


def test_tilted_approach_lowers_the_finger_edge():
    # Approach tilted 60° from down about the (horizontal) closing axis.
    tilt = math.radians(60.0)
    approach = [0.0, math.sin(tilt), -math.cos(tilt)]
    candidates = _candidates([[0.0, 0.0, 0.10]], [approach], [[1.0, 0.0, 0.0]])
    lowest = lowest_finger_z(candidates, gripper_width=WIDTH, gripper_depth=DEPTH)
    expected = 0.10 - FINGER_WIDTH / 2 * math.sin(tilt)
    assert float(lowest[0]) == pytest.approx(expected, abs=1e-6)


def test_selection_is_off_by_default():
    candidates = _candidates([[0.0, 0.0, 0.10]], [[0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0]])
    provider = _provider()
    assert provider.filter(candidates) is None
    assert provider.cost(candidates) is None


def test_filter_prefers_fingers_above_the_clearance():
    tilt = math.radians(30.0)
    candidates = _candidates(
        [[0.0, 0.0, 0.010], [0.0, 0.0, 0.010]],
        [[0.0, 0.0, -1.0], [math.sin(tilt), 0.0, -math.cos(tilt)]],
        [[1.0, 0.0, 0.0], [math.cos(tilt), 0.0, math.sin(tilt)]],
    )
    keep = _provider(table_z=0.0, min_table_clearance=0.005, prefer_top_down=True).filter(candidates)
    # Both grasp points are 1 cm up, but the tilted one's lower finger is ~2.6 cm lower.
    assert keep.tolist() == [True, False]


def test_filter_prefers_top_down_and_falls_back_per_row():
    candidates = _candidates(
        [[[0.0, 0.0, 0.10], [0.0, 0.0, 0.10]], [[0.0, 0.0, 0.10], [0.0, 0.0, 0.10]]],
        [[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
    )
    keep = _provider(table_z=0.0, prefer_top_down=True).filter(candidates)
    # Row 0 keeps its top-down grasp; row 1 has none, so it keeps everything.
    assert keep.tolist() == [[True, False], [True, True]]


def test_filter_falls_back_to_every_top_down_candidate():
    candidates = _candidates([[0.0, 0.0, 0.001]], [[0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0]])
    keep = _provider(table_z=0.0, min_table_clearance=0.005, prefer_top_down=True).filter(candidates)
    assert keep.tolist() == [True]


def test_cost_prefers_vertical_deep_centred_grasps():
    # Box from z=0 to 0.10, centred at the origin.
    entity = _BoxEntity([-0.03, -0.03, 0.0], [0.03, 0.03, 0.10])
    provider = _provider(entity, quality_cost=True)
    down = [0.0, 0.0, -1.0]
    tilt = math.radians(60.0)
    tilted = [0.0, math.sin(tilt), -math.cos(tilt)]
    tip = lambda x, z: [x, 0.0, z - CONTACT_INSET]  # fingertip below the contacts
    candidates = _candidates(
        [tip(0.0, 0.05), tip(0.0, 0.095), tip(0.025, 0.05), tip(0.0, 0.05)],
        [down, down, down, tilted],
        [[1.0, 0.0, 0.0]] * 4,
    )
    candidates.position[3] = candidates.position[0] + CONTACT_INSET * (
        torch.tensor(tilted) - torch.tensor(down)
    )
    cost = provider.cost(candidates)
    assert cost.shape == (4,)
    assert float(cost[0]) == pytest.approx(0.0, abs=1e-6)          # vertical, deep, centred
    assert float(cost[1]) == pytest.approx(0.75, abs=1e-6)         # 5 mm below the top
    assert float(cost[2]) == pytest.approx(0.5, abs=1e-6)          # 2.5 cm off centre
    assert float(cost[3]) == pytest.approx(0.5, abs=1e-6)          # tilted 60°
