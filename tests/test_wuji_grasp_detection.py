"""Tests for Wuji hand grasp detection on the Tianji robot definition.

Covers the pad-link name arrays (contact-capable links only — link1 has no
collision geoms, and the MJCF ``*_tip_link`` fingertip meshes are geoms on
the link4 bodies, not standalone links) and the ``is_grasping_fn`` wiring of
the hand sub-controllers, using the same mock-entity pattern as the Franka
grasp-detection tests (no scene build).
"""

from __future__ import annotations

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis_collection.robots import tianji_marvin_wuji


class _FakeLink:
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx


class _FakeRobotEntity:
    """Minimal stand-in for a RigidEntity for grasp-detection tests."""

    def __init__(self, links, contacts):
        self.name = "fake_tianji"
        self.links = links
        self._contacts = contacts

    def get_contacts(self, with_entity=None):
        return self._contacts


_SIDES = ("left", "right")


def _pad_names(side: str):
    return (
        tianji_marvin_wuji.LEFT_FINGER_PAD_LINK_NAMES
        if side == "left"
        else tianji_marvin_wuji.RIGHT_FINGER_PAD_LINK_NAMES
    )


def _make_entity(side: str, batched: bool):
    from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

    names = _pad_names(side)
    fn = make_is_grasping(names)
    links = [_FakeLink("base", 0), _FakeLink("wrist", 4)]
    links += [_FakeLink(n, 5 + i) for i, n in enumerate(names)]
    if batched:
        # (B=2, n_contacts=2): env 0 touches pad link 5 (valid);
        # env 1 touches pad link 5 too, but the contact is invalid.
        contacts = {
            "link_a": torch.tensor([[5, 4], [5, 4]]),
            "link_b": torch.tensor([[4, 4], [4, 4]]),
            "force_a": torch.zeros(2, 2, 3),
            "valid_mask": torch.tensor([[True, True], [False, True]]),
        }
    else:
        # One contact on pad link 6, one on a non-pad link.
        contacts = {
            "link_a": torch.tensor([6, 4]),
            "link_b": torch.tensor([4, 4]),
            "force_a": torch.zeros(2, 3),
        }
    return fn, _FakeRobotEntity(links, contacts), names


@pytest.mark.parametrize("side", _SIDES)
def test_pad_link_names_cover_contact_capable_links_only(side: str):
    names = _pad_names(side)
    # 5 fingers x links 2-4.
    assert len(names) == 15
    expected = {
        f"{side}_finger{finger}_link{link}"
        for finger in range(1, 6)
        for link in (2, 3, 4)
    }
    assert set(names) == expected
    # link1 carries no collision geoms; *_tip_link meshes are geoms on the
    # link4 bodies and do not resolve as standalone Genesis links.
    assert not any(name.endswith("_link1") for name in names)
    assert not any(name.endswith("_tip_link") for name in names)


@pytest.mark.parametrize("side", _SIDES)
def test_is_grasping_unbatched(side: str):
    fn, robot, _ = _make_entity(side, batched=False)
    assert bool(fn(robot, target_entity=None)) is True

    # No pad contact -> False.
    robot._contacts["link_a"] = torch.tensor([4, 4])
    assert bool(fn(robot, target_entity=None)) is False


@pytest.mark.parametrize("side", _SIDES)
def test_is_grasping_batched_valid_mask(side: str):
    fn, robot, _ = _make_entity(side, batched=True)
    result = fn(robot, target_entity=None)
    assert result.shape == (2,)
    assert result.dtype == torch.bool
    # env 0 has a valid pad contact; env 1's pad contact is masked out.
    assert result.tolist() == [True, False]


@pytest.mark.parametrize("side", _SIDES)
def test_is_grasping_missing_pad_link_fails_loudly(side: str):
    fn, robot, names = _make_entity(side, batched=False)
    robot.links = [link for link in robot.links if link.name != names[0]]
    with pytest.raises(ValueError, match="pad link"):
        fn(robot, target_entity=None)


@pytest.mark.parametrize("side", _SIDES)
def test_hand_subcontroller_wires_is_grasping(side: str):
    cfg = tianji_marvin_wuji._DEFAULT_SUB_CONTROLLERS[f"{side}_hand"]
    assert cfg["controller_type"] == "joint_position"
    assert callable(cfg["is_grasping_fn"])
