"""Tests for RobotDescriptionNode reset lifecycle (Phase 1).

Covers:
  - Normal reset applies the rest/keyframe joint state during
    ``reset(priority=50)`` (before ``world.after_reset``), and the
    controller reset observes the final rest state.
  - ``after_reset`` consumes the final state (controller bookkeeping +
    observation refresh) and does NOT re-apply the rest mutation.
  - Reload still initialises the rest pose via ``after_reload`` (entities /
    controllers require a built scene).
  - Batched masked reset: only masked envs are returned to the rest state;
    unmasked envs keep their current joint state.
"""

import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.robots.franka_emika_panda import (
    FrankaEmikaPanda,
    HOME_JOINT_POSITIONS,
)
from unienv_interface.world import WorldEnv


def _make_robot_env(world):
    robot = FrankaEmikaPanda(world, name="franka", controller_type="joint_position")
    env = WorldEnv(world, [robot], render_mode=None)
    return robot, env


def test_robot_reset_ordering_and_ownership():
    """Rest pose applied in reset(); after_reset() only consumes state."""
    world = GenesisWorld()
    robot, env = _make_robot_env(world)
    env.reset(seed=0, reload=True)

    rest = HOME_JOINT_POSITIONS.to(dtype=world.dtype)

    # after_reload applied the rest pose on the freshly built scene.
    qpos = robot.entity.get_dofs_position()
    assert torch.allclose(qpos, rest, atol=1e-5), (
        f"after_reload did not leave the robot at its rest pose: {qpos}"
    )

    # --- Spy on the reset lifecycle -----------------------------------
    calls = []
    orig_apply_rest = robot._apply_rest
    orig_controller_reset = robot.controller.reset
    orig_controller_after_reset = robot.controller.after_reset

    def apply_rest_spy(mask=None):
        calls.append("apply_rest")
        return orig_apply_rest(mask=mask)

    def controller_reset_spy(mask=None):
        # The controller reset must observe the FINAL rest state.
        q = robot.entity.get_dofs_position()
        calls.append(("controller_reset", bool(torch.allclose(q, rest, atol=1e-5))))
        return orig_controller_reset(mask=mask)

    def controller_after_reset_spy(mask=None):
        calls.append("controller_after_reset")
        return orig_controller_after_reset(mask=mask)

    robot._apply_rest = apply_rest_spy
    robot.controller.reset = controller_reset_spy
    robot.controller.after_reset = controller_after_reset_spy

    # Perturb away from rest, then run a normal (non-reload) reset.
    robot.entity.set_dofs_position(torch.zeros_like(rest))
    env.reset(seed=1)

    robot._apply_rest = orig_apply_rest
    robot.controller.reset = orig_controller_reset
    robot.controller.after_reset = orig_controller_after_reset

    # The rest mutation ran exactly once, during reset(), BEFORE the
    # controller reset; after_reset only ran the controller hook.
    assert calls[0] == "apply_rest", f"Unexpected lifecycle order: {calls}"
    assert calls[1] == ("controller_reset", True), (
        f"controller.reset did not observe the final rest state: {calls}"
    )
    assert calls[2] == "controller_after_reset", f"Unexpected lifecycle order: {calls}"
    assert calls.count("apply_rest") == 1, (
        f"after_reset re-applied the rest mutation: {calls}"
    )

    # Final state is the rest pose, and the cached observation matches it.
    qpos = robot.entity.get_dofs_position()
    assert torch.allclose(qpos, rest, atol=1e-5)
    obs = robot.get_observation()
    assert obs is not None
    assert torch.allclose(obs["joint_position"], rest, atol=1e-5), (
        "Observation was not refreshed to the rest state after reset"
    )

    print("  PASS: robot reset ordering/ownership")
    env.close()


def test_robot_masked_reset():
    """Batched masked reset: masked envs go to rest, others keep state."""
    N_ENVS = 4
    world = GenesisWorld(batch_size=N_ENVS)
    robot, env = _make_robot_env(world)
    env.reset(seed=0, reload=True)

    rest = HOME_JOINT_POSITIONS.to(dtype=world.dtype)

    # Perturb every env away from the rest pose.
    perturb = rest.unsqueeze(0).repeat(N_ENVS, 1) + 0.1
    robot.entity.set_dofs_position(perturb)

    reset_mask = torch.zeros(N_ENVS, dtype=torch.bool)
    reset_mask[[1, 3]] = True
    env.reset(mask=reset_mask)

    qpos = robot.entity.get_dofs_position()
    for env_j in (1, 3):
        assert torch.allclose(qpos[env_j], rest, atol=1e-5), (
            f"Masked env {env_j} not returned to rest pose: {qpos[env_j]}"
        )
    for env_j in (0, 2):
        assert torch.allclose(qpos[env_j], perturb[env_j], atol=1e-5), (
            f"Unmasked env {env_j} state was overwritten: {qpos[env_j]}"
        )

    print("  PASS: robot masked reset")
    env.close()


if __name__ == "__main__":
    print("=== test_robot_reset_ordering_and_ownership ===")
    test_robot_reset_ordering_and_ownership()
    print()
    print("=== test_robot_masked_reset ===")
    test_robot_masked_reset()
    print()
    print("All tests passed!")
