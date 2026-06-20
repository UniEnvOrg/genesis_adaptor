"""Tests for the EEF controller with ``LinkAttachment`` (unbatched, CPU).

Verifies that ``DiscreteEntityController`` (via ``JointPositionController``)
correctly bundles the link name + offset through a :class:`LinkAttachment`
instance, instead of separate ``target_link`` / ``eef_offset_pos`` /
``eef_offset_quat`` / ``base_frame`` / ``base_frame_offset_*`` parameters.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import JointPositionController
from unienv_genesis.options.attach_option import LinkAttachment

BOX_LINK = "box_baselink"


class BoxEntityNode(WorldNode):
    """Adds a plane + free-floating box so the scene is non-empty.

    The box is placed with its centre at ``z = 0.05`` so it rests on the
    plane; its baselink has identity orientation at rest, which makes the
    EEF offset math easy to verify (local z == world z).
    """

    reload_priorities = {20}
    reset_priorities = {20}
    pre_environment_step_priorities = {20}
    post_environment_step_priorities = {20}

    def __init__(self, world: GenesisWorld, name: str = "box_env"):
        self.world = world
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reload(self, *, priority=0, **kw):
        if priority == 20:
            self.world.scene.add_entity(gs.morphs.Plane(), name="plane")
            self.world.scene.add_entity(
                gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05)),
                name="box",
            )

    def reset(self, *, priority=0, **kw):
        pass

    def pre_environment_step(self, dt, *, priority=0):
        pass

    def post_environment_step(self, dt, *, priority=0):
        pass

    def get_observation(self):
        return None


def _make_env(world: GenesisWorld) -> WorldEnv:
    """Build a WorldEnv from the box helper (no sensors, no controllers).

    The box entity is available via ``world.scene.get_entity("box")`` after
    ``env.reset(reload=True)`` returns.
    """
    box = BoxEntityNode(world)
    env = WorldEnv(world, [box], render_mode=None)
    env.reset(seed=0, reload=True)
    return env


def test_eef_controller_with_link_attachment():
    """A ``LinkAttachment`` target_link with offset enables EEF semantics and
    places the virtual EEF point at the offset from the physical link."""
    world = GenesisWorld()
    env = _make_env(world)

    entity = world.scene.get_entity("box")
    controller = JointPositionController(
        world=world,
        entity=entity,
        target_link=LinkAttachment(
            link_name=BOX_LINK, offset_pos=(0.0, 0.0, 0.1),
        ),
    )

    # 1. EEF semantics are enabled.
    assert controller.has_eef is True

    # 2. target_link resolves to the box's base link.
    assert controller.target_link is entity.get_link(BOX_LINK)

    # 3. The EEF offset position tensor matches the configured offset.
    assert controller.eef_offset_pos is not None
    assert torch.allclose(
        controller.eef_offset_pos,
        torch.tensor([0.0, 0.0, 0.1], dtype=controller.eef_offset_pos.dtype),
        atol=1e-6,
    ), f"Expected eef_offset_pos [0,0,0.1], got {controller.eef_offset_pos.tolist()}"

    # The target_link_attachment property reflects the original LinkAttachment.
    assert controller.target_link_attachment is not None
    assert controller.target_link_attachment.link_name == BOX_LINK
    assert list(controller.target_link_attachment.offset_pos) == [0.0, 0.0, 0.1]
    assert controller.base_frame_attachment is None

    # 4. The absolute EEF position is offset 0.1 along the link's local z-axis
    # (identity orientation at rest -> local z == world z).
    link_pos = controller.target_link.get_pos()
    eef_pos_abs, eef_quat_abs = controller.get_eef_pose_absolute()
    assert torch.allclose(
        eef_pos_abs, link_pos + torch.tensor([0.0, 0.0, 0.1], dtype=link_pos.dtype),
        atol=1e-4,
    ), f"Expected eef_pos_abs = link_pos + (0,0,0.1), got {eef_pos_abs.tolist()}"

    env.close()


def test_eef_controller_base_frame_offset():
    """A ``LinkAttachment`` base_frame with offset places the virtual base
    frame away from the physical base link origin, so the EEF position
    reported by ``get_eef_pose_base`` is relative to the offset base frame."""
    world = GenesisWorld()
    env = _make_env(world)

    entity = world.scene.get_entity("box")
    controller = JointPositionController(
        world=world,
        entity=entity,
        target_link=LinkAttachment(link_name=BOX_LINK),
        base_frame=LinkAttachment(
            link_name=BOX_LINK, offset_pos=(0.0, 0.0, 0.05),
        ),
    )

    assert controller.has_eef is True
    assert controller.base_frame is entity.get_link(BOX_LINK)
    assert controller.base_frame_offset_pos is not None
    assert torch.allclose(
        controller.base_frame_offset_pos,
        torch.tensor([0.0, 0.0, 0.05], dtype=controller.base_frame_offset_pos.dtype),
        atol=1e-6,
    )
    assert controller.base_frame_attachment is not None
    assert controller.base_frame_attachment.link_name == BOX_LINK

    # At rest the physical base link is at link_pos with identity orientation.
    # The virtual base frame is at link_pos + (0, 0, 0.05).  The EEF (no EEF
    # offset) is at link_pos.  In the virtual base frame that is (0, 0, -0.05).
    link_pos = controller.target_link.get_pos()
    eef_pos_base, eef_quat_base = controller.get_eef_pose_base()
    assert torch.allclose(
        eef_pos_base,
        torch.tensor([0.0, 0.0, -0.05], dtype=eef_pos_base.dtype),
        atol=1e-4,
    ), f"Expected eef_pos_base (0,0,-0.05), got {eef_pos_base.tolist()}"

    # Sanity: link_pos shouldn't be the zero vector (it's at z=0.05).
    assert link_pos[2].item() != 0.0

    env.close()


def test_eef_controller_str_target_link_backward_compat():
    """A bare ``str`` target_link (no ``LinkAttachment``) is still accepted
    and treated as a ``LinkAttachment`` with no offset."""
    world = GenesisWorld()
    env = _make_env(world)

    entity = world.scene.get_entity("box")
    controller = JointPositionController(
        world=world,
        entity=entity,
        target_link=BOX_LINK,
    )

    assert controller.has_eef is True
    assert controller.target_link is entity.get_link(BOX_LINK)
    # No offset configured -> offset tensors are None.
    assert controller.eef_offset_pos is None
    assert controller.eef_offset_quat is None
    # The attachment is still materialised as a LinkAttachment.
    assert controller.target_link_attachment is not None
    assert controller.target_link_attachment.link_name == BOX_LINK
    assert controller.target_link_attachment.offset_pos is None

    # Absolute EEF pose == physical link pose (no offset).
    link_pos = controller.target_link.get_pos()
    eef_pos_abs, _ = controller.get_eef_pose_absolute()
    assert torch.allclose(eef_pos_abs, link_pos, atol=1e-6)

    env.close()
