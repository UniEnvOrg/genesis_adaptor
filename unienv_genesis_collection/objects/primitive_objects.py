"""Primitive-geometry ManipulatableObject subclasses.

Each class wraps a Genesis primitive morph (Box, Sphere, Cylinder) and provides
an analytic bounding box.  The ``entity`` parameter is accepted for interface
consistency but not used by these implementations (dimensions are fixed at
construction time).
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch

from unienv_genesis_collection.objects.base_object import (
    ManipulatableObject,
    make_bbox,
)

if TYPE_CHECKING:
    from unienv_interface.space import BoxSpace
    from unienv_genesis.world.genesis_world import GenesisWorld

__all__ = ["BoxObject", "SphereObject", "CylinderObject"]


class BoxObject(ManipulatableObject):
    """A rectangular-prism manipulatable object (``gs.morphs.Box``).

    Args:
        size: Dimensions ``(width_x, depth_y, height_z)`` in metres.
        color: RGBA colour tuple.  Applied as a ``Plastic`` surface.
        **kwargs: Forwarded to :class:`ManipulatableObject` (``name``,
            ``mass``, ``fixed``).
    """

    def __init__(
        self,
        size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        color: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
        self.color = color

    def get_local_bounding_box(self, entity) -> "BoxSpace":
        half = torch.tensor(self.size, dtype=torch.float32) / 2.0
        return make_bbox(-half, half)

    def load_into_scene(
        self,
        world: "GenesisWorld",
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        name_override=None,
        batch_fixed_verts=False,
    ):
        import genesis as gs

        morph = gs.morphs.Box(
            size=self.size,
            pos=pos,
            quat=quat,
            fixed=self.fixed,
        )
        surface = gs.surfaces.Plastic(color=self.color) if self.color else None
        return world.scene.add_entity(
            morph,
            surface=surface,
            name=name_override or self.name,
        )


class SphereObject(ManipulatableObject):
    """A spherical manipulatable object (``gs.morphs.Sphere``).

    Args:
        radius: Sphere radius in metres.
        color: RGBA colour tuple.
        **kwargs: Forwarded to :class:`ManipulatableObject`.
    """

    def __init__(
        self,
        radius: float = 0.025,
        color: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.radius = radius
        self.color = color

    def get_local_bounding_box(self, entity) -> "BoxSpace":
        r = self.radius
        return make_bbox(
            torch.tensor([-r, -r, -r], dtype=torch.float32),
            torch.tensor([r, r, r], dtype=torch.float32),
        )

    def load_into_scene(
        self,
        world: "GenesisWorld",
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        name_override=None,
        batch_fixed_verts=False,
    ):
        import genesis as gs

        morph = gs.morphs.Sphere(
            radius=self.radius,
            pos=pos,
            quat=quat,
            fixed=self.fixed,
        )
        surface = gs.surfaces.Plastic(color=self.color) if self.color else None
        return world.scene.add_entity(
            morph,
            surface=surface,
            name=name_override or self.name,
        )


class CylinderObject(ManipulatableObject):
    """A cylindrical manipulatable object (``gs.morphs.Cylinder``).

    The cylinder's axis is aligned with Z in the local frame.

    Args:
        radius: Cylinder radius in metres.
        height: Cylinder height in metres.
        color: RGBA colour tuple.
        **kwargs: Forwarded to :class:`ManipulatableObject`.
    """

    def __init__(
        self,
        radius: float = 0.025,
        height: float = 0.1,
        color: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.radius = radius
        self.height = height
        self.color = color

    def get_local_bounding_box(self, entity) -> "BoxSpace":
        r = self.radius
        h = self.height / 2.0
        return make_bbox(
            torch.tensor([-r, -r, -h], dtype=torch.float32),
            torch.tensor([r, r, h], dtype=torch.float32),
        )

    def load_into_scene(
        self,
        world: "GenesisWorld",
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        name_override=None,
        batch_fixed_verts=False,
    ):
        import genesis as gs

        morph = gs.morphs.Cylinder(
            radius=self.radius,
            height=self.height,
            pos=pos,
            quat=quat,
            fixed=self.fixed,
        )
        surface = gs.surfaces.Plastic(color=self.color) if self.color else None
        return world.scene.add_entity(
            morph,
            surface=surface,
            name=name_override or self.name,
        )
