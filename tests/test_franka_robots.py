"""Tests for the three Franka robot definition files (Phase 2 runtime lane).

Covers, for ``franka_emika_panda``, ``franka_fr3_hand`` and
``franka_fr3_robotiq_2f85``:

  - The active arm ``max_joint_pos_vel`` is the intended DROID limit of
    0.2 rad/s (a scalar), with the Franka hardware per-joint limits retained
    only as reference constants (including the legacy
    ``DROID_FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS`` alias).
  - Grasp detection resolves every configured pad link, raises a clear error
    when a configured pad link name does not exist on the robot (instead of
    silently reporting "not grasping"), and performs the pad comparison on
    the contact tensor's device (CPU, and CUDA where available), honouring
    the batched ``valid_mask``.
  - Integration: the default combined controller of ``FrankaEmikaPanda``
    wires the 0.2 rad/s limit into the arm sub-controller.
"""

from __future__ import annotations

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.robots import (
    franka_emika_panda,
    franka_fr3_hand,
    franka_fr3_robotiq_2f85,
)
from unienv_genesis_collection.robots.franka_emika_panda import FrankaEmikaPanda
from unienv_interface.world import WorldEnv

_VARIANT_MODULES = [franka_emika_panda, franka_fr3_hand, franka_fr3_robotiq_2f85]
_HARDWARE_LIMITS = [2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51]


# ---------------------------------------------------------------------------
# Velocity-limit configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", _VARIANT_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_arm_max_joint_pos_vel_is_droid_limit(mod):
    """Active arm rate limit must be the DROID 0.2 rad/s scalar."""
    assert mod.DROID_FRANKA_ARM_IK_VEL_LIMIT == 0.2
    limit = mod._DEFAULT_SUB_CONTROLLERS["arm"]["max_joint_pos_vel"]
    assert isinstance(limit, float) and limit == 0.2, (
        f"{mod.__name__}: arm max_joint_pos_vel should be the scalar 0.2 rad/s "
        f"(DROID limit), got {limit!r}"
    )


@pytest.mark.parametrize("mod", _VARIANT_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_hardware_limits_kept_as_reference_only(mod):
    """Hardware per-joint limits remain available as reference constants."""
    ref = mod.FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS
    assert torch.allclose(ref, torch.tensor(_HARDWARE_LIMITS))
    # The legacy DROID_-prefixed name remains as a backward-compatible
    # alias bound to the very same reference tensor.
    assert hasattr(mod, "DROID_FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS")
    alias = mod.DROID_FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS
    assert alias is ref
    assert torch.allclose(alias, torch.tensor(_HARDWARE_LIMITS))
    # The reference constant must not leak into the active controller config.
    arm_cfg = mod._DEFAULT_SUB_CONTROLLERS["arm"]
    assert arm_cfg["max_joint_pos_vel"] is not ref


# ---------------------------------------------------------------------------
# Grasp-detection helpers
# ---------------------------------------------------------------------------

class _FakeLink:
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx


class _FakeRobotEntity:
    """Minimal stand-in for a RigidEntity for grasp-detection tests."""

    def __init__(self, links, contacts):
        self.name = "fake_franka"
        self.links = links
        self._contacts = contacts

    def get_contacts(self, with_entity=None):
        return self._contacts


def _variant_pads(mod):
    fn = mod._DEFAULT_SUB_CONTROLLERS["gripper"]["is_grasping_fn"]
    if mod is franka_fr3_robotiq_2f85:
        names = ["g_left_pad", "g_right_pad"]
    else:
        names = ["left_finger", "right_finger"]
    return fn, names


def _make_entity(mod, device, batched):
    fn, names = _variant_pads(mod)
    links = [_FakeLink("base", 0), _FakeLink("wrist", 4)]
    links += [_FakeLink(n, 5 + i) for i, n in enumerate(names)]
    if batched:
        # (B=2, n_contacts=2): env 0 touches pad link 5 (valid);
        # env 1 touches pad link 5 too, but the contact is invalid.
        contacts = {
            "link_a": torch.tensor([[5, 4], [5, 4]], device=device),
            "link_b": torch.tensor([[4, 4], [4, 4]], device=device),
            "force_a": torch.zeros(2, 2, 3, device=device),
            "valid_mask": torch.tensor(
                [[True, True], [False, True]], device=device,
            ),
        }
    else:
        # One contact on pad link 6, one on a non-pad link.
        contacts = {
            "link_a": torch.tensor([6, 4], device=device),
            "link_b": torch.tensor([4, 4], device=device),
            "force_a": torch.zeros(2, 3, device=device),
        }
    return fn, _FakeRobotEntity(links, contacts), links


@pytest.mark.parametrize("mod", _VARIANT_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_is_grasping_unbatched(mod):
    fn, robot, _ = _make_entity(mod, torch.device("cpu"), batched=False)
    assert bool(fn(robot, target_entity=None)) is True

    # No pad contact -> False.
    robot._contacts["link_a"] = torch.tensor([4, 4])
    assert bool(fn(robot, target_entity=None)) is False


@pytest.mark.parametrize("mod", _VARIANT_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_is_grasping_batched_valid_mask(mod):
    fn, robot, _ = _make_entity(mod, torch.device("cpu"), batched=True)
    result = fn(robot, target_entity=None)
    assert result.shape == (2,)
    assert result.dtype == torch.bool
    # env 0 has a valid pad contact; env 1's pad contact is masked out.
    assert result.tolist() == [True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("mod", _VARIANT_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_is_grasping_cuda_contacts(mod):
    """Pad-index comparison must run on the contact tensor's device."""
    device = torch.device("cuda")
    fn, robot, _ = _make_entity(mod, device, batched=False)
    assert bool(fn(robot, target_entity=None)) is True

    fn, robot, _ = _make_entity(mod, device, batched=True)
    result = fn(robot, target_entity=None)
    assert result.device.type == "cuda"
    assert result.tolist() == [True, False]


@pytest.mark.parametrize("mod", _VARIANT_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_is_grasping_missing_pad_link_fails_loudly(mod):
    """Unresolvable configured pad links must raise, not silently report False."""
    fn, names = _variant_pads(mod)
    # Entity whose links lack the second pad link name.
    links = [_FakeLink("base", 0), _FakeLink(names[0], 5)]
    contacts = {
        "link_a": torch.tensor([5]),
        "link_b": torch.tensor([0]),
        "force_a": torch.zeros(1, 3),
    }
    robot = _FakeRobotEntity(links, contacts)
    with pytest.raises(ValueError, match=names[1]):
        fn(robot, target_entity=None)


# ---------------------------------------------------------------------------
# Integration: default combined controller wiring (Panda)
# ---------------------------------------------------------------------------

def test_panda_combined_controller_uses_droid_limit():
    world = GenesisWorld()
    robot = FrankaEmikaPanda(world, name="franka")  # default: combined controller
    env = WorldEnv(world, [robot], render_mode=None)
    env.reset(seed=0, reload=True)

    arm_limit = robot.controller["arm"].max_joint_pos_vel
    assert arm_limit is not None
    flat = arm_limit[0] if arm_limit.dim() == 2 else arm_limit
    assert flat.numel() == 7
    assert torch.allclose(flat, torch.full_like(flat, 0.2)), (
        f"Arm sub-controller limit should be 0.2 rad/s on all 7 joints, got {flat}"
    )

    env.close()


if __name__ == "__main__":
    for mod in _VARIANT_MODULES:
        test_arm_max_joint_pos_vel_is_droid_limit(mod)
        test_hardware_limits_kept_as_reference_only(mod)
        test_is_grasping_unbatched(mod)
        test_is_grasping_batched_valid_mask(mod)
        test_is_grasping_missing_pad_link_fails_loudly(mod)
    test_panda_combined_controller_uses_droid_limit()
    print("All tests passed!")
