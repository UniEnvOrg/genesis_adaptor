"""Tests for MeshObject's antipodal grasp sampler (CPU only, no scenes).

Uses analytic boxes, whose valid grasps are known in closed form:

  - the jaws close square-on to a pair of faces, never across an axis wider
    than the gripper, with an approach perpendicular to it
  - the contact midpoint lies inside the object
  - approaches stay within 60° of the object's upright down
  - downward grasps on a tall box exist and keep the palm clear of the top
  - seeded sampling is reproducible
"""

import numpy as np
import pytest
import torch
import trimesh

from unienv_genesis_collection.objects.mesh_object import MeshObject

GRIPPER_WIDTH = 0.085
GRIPPER_DEPTH = 0.035
CONTACT_INSET = 0.01


def _box_object(tmp_path, extents, name="box", standing_quaternion=None):
    path = tmp_path / f"{name}.obj"
    trimesh.creation.box(extents=extents).export(path)
    return MeshObject(mesh_file=str(path), standing_quaternion=standing_quaternion)


def _candidates(obj, seed=0, **kwargs):
    rng = torch.Generator()
    rng.manual_seed(seed)
    return obj.list_box_gripper_grasping_candidates_local(
        None,
        GRIPPER_WIDTH,
        gripper_depth=GRIPPER_DEPTH,
        rng=rng,
        contact_inset=CONTACT_INSET,
        **kwargs,
    )


def _numpy(candidates):
    return (
        candidates.position.numpy(),
        candidates.approach_direction.numpy(),
        candidates.opening_direction.numpy(),
    )


def test_jaws_close_along_a_narrow_axis(tmp_path):
    # x (4 cm) and y (6 cm) fit in the gripper, z (20 cm) does not.
    obj = _box_object(tmp_path, (0.04, 0.06, 0.20))
    pos, approach, opening = _numpy(_candidates(obj))
    assert len(pos) > 0

    np.testing.assert_allclose(np.linalg.norm(approach, axis=-1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(opening, axis=-1), 1.0, atol=1e-9)
    np.testing.assert_allclose((approach * opening).sum(-1), 0.0, atol=1e-9)

    # Square-on: the jaws close along a face normal, and only across the face
    # pairs narrower than the gripper.
    along_x = np.abs(opening[:, 0]) >= 1.0 - 1e-6
    along_y = np.abs(opening[:, 1]) >= 1.0 - 1e-6
    assert np.all(along_x | along_y)


def test_contact_midpoint_lies_inside_the_object(tmp_path):
    extents = np.array([0.04, 0.06, 0.20])
    obj = _box_object(tmp_path, extents)
    pos, approach, _ = _numpy(_candidates(obj))
    midpoint = pos - CONTACT_INSET * approach
    assert np.all(np.abs(midpoint) <= extents / 2 + 1e-9)


def test_long_rod_is_never_closed_along_its_length(tmp_path):
    obj = _box_object(tmp_path, (0.25, 0.03, 0.03), name="rod")
    pos, _, opening = _numpy(_candidates(obj))
    assert len(opening) > 0
    assert np.all(np.abs(opening[:, 0]) <= 1e-6)
    # Spread along the rod rather than bunched in one spot.
    assert np.ptp(pos[:, 0]) > 0.15


def test_approaches_stay_near_the_upright_down(tmp_path):
    obj = _box_object(tmp_path, (0.04, 0.06, 0.20))
    _, approach, _ = _numpy(_candidates(obj))
    assert np.all(approach[:, 2] <= -np.cos(np.radians(60.0)) + 1e-9)


def test_approaches_follow_the_standing_quaternion(tmp_path):
    # Standing pose: 90° about x, so the object's local -y points down.
    half = np.sqrt(0.5)
    obj = _box_object(
        tmp_path, (0.04, 0.20, 0.06), name="lying", standing_quaternion=(half, half, 0.0, 0.0),
    )
    _, approach, _ = _numpy(_candidates(obj))
    assert len(approach) > 0
    assert np.all(-approach[:, 1] >= np.cos(np.radians(60.0)) - 1e-9)


def test_downward_grasps_keep_the_palm_clear_of_a_tall_box(tmp_path):
    half_height = 0.10
    obj = _box_object(tmp_path, (0.04, 0.06, 2 * half_height), name="tall")
    pos, approach, _ = _numpy(_candidates(obj))
    downward = approach[:, 2] <= -0.99
    # Grasps slide back until the palm clears, so a tall box keeps plenty.
    assert downward.sum() >= 20
    # The palm starts GRIPPER_DEPTH behind the fingertip; the top face must
    # not reach it.
    assert np.all(pos[downward, 2] + GRIPPER_DEPTH >= half_height - 1e-9)


def test_object_wider_than_the_gripper_has_no_antipodal_grasps(tmp_path):
    obj = _box_object(tmp_path, (0.2, 0.2, 0.2), name="crate")
    # Falls back to the bounding-box enumeration, which is empty as well.
    assert len(_candidates(obj)) == 0


def test_seeded_sampling_is_reproducible(tmp_path):
    first = _box_object(tmp_path, (0.04, 0.06, 0.20), name="a")
    second = _box_object(tmp_path, (0.04, 0.06, 0.20), name="b")
    for lhs, rhs in zip(_numpy(_candidates(first, seed=3)), _numpy(_candidates(second, seed=3))):
        np.testing.assert_array_equal(lhs, rhs)


@pytest.mark.parametrize("approach_tilts_deg", [(0.0,), (-30.0, 0.0, 30.0)])
def test_each_pair_yields_both_opening_signs(tmp_path, approach_tilts_deg):
    obj = _box_object(tmp_path, (0.04, 0.06, 0.20))
    pos, approach, opening = _numpy(_candidates(obj, approach_tilts_deg=approach_tilts_deg))
    np.testing.assert_array_equal(pos[0::2], pos[1::2])
    np.testing.assert_array_equal(approach[0::2], approach[1::2])
    np.testing.assert_array_equal(opening[0::2], -opening[1::2])


def test_thin_plate_is_pinched_across_its_thickness(tmp_path):
    # A 1 mm sheet standing upright, wider than the gripper (like a hang tab).
    obj = _box_object(tmp_path, (0.10, 0.001, 0.07), name="tab")
    _, approach, opening = _numpy(_candidates(obj))
    assert len(opening) > 0
    assert np.all(np.abs(opening[:, 1]) >= 1.0 - 1e-6)
    assert (approach[:, 2] <= -0.99).any()


def test_grasps_leave_width_slack_below_the_gripper_opening(tmp_path):
    # x (8 cm) fits the 8.5 cm opening but not with the default 1 cm of slack.
    obj = _box_object(tmp_path, (0.08, 0.04, 0.20), name="snug")
    _, _, opening = _numpy(_candidates(obj))
    assert len(opening) > 0
    assert np.all(np.abs(opening[:, 0]) <= 1e-6)
    _, _, opening = _numpy(_candidates(obj, width_margin=0.0))
    assert np.any(np.abs(opening[:, 0]) >= 1.0 - 1e-6)


def test_round_handle_gets_levelled_top_down_grasps(tmp_path):
    # A 2 cm-radius rod lying along x on the table (z from 0 to 0.04).
    rod = trimesh.creation.cylinder(radius=0.02, height=0.20, sections=48)
    rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    rod.apply_translation([0.0, 0.0, 0.02])
    path = tmp_path / "rod.obj"
    rod.export(path)
    obj = MeshObject(mesh_file=str(path))

    def clean_top_down(**kwargs):
        pos, approach, opening = _numpy(_candidates(obj, **kwargs))
        clean = (approach[:, 2] <= -0.999) & (np.abs(opening[:, 2]) < 1e-3)
        return pos[clean]

    # Without levelling the sampled diameters are tilted, so only the few
    # facets that happen to be vertical give clean top-down grasps.
    plain = clean_top_down(rest_clearance=None)
    levelled = clean_top_down(rest_clearance=0.007)
    assert len(levelled) >= 100 and len(levelled) >= 10 * len(plain)
    # Every fingertip clears the resting plane.
    assert np.all(levelled[:, 2] >= 0.007 - 1e-9)
