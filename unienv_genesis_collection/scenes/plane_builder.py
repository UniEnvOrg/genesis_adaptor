"""Plane builder for simulation scenes."""

from typing import Optional

import genesis as gs
from genesis.engine.entities.base_entity import Entity

from unienv_genesis import GenesisWorld
from unienv_interface.world.node import WorldNode

# Keyword arguments accepted by ``gs.morphs.Plane``. Any of these supplied to
# the builder are forwarded to the morph; everything else is forwarded to
# ``world.scene.add_entity`` as entity-level kwargs.
_PLANE_MORPH_KWARGS = frozenset(
    {
        "pos",
        "normal",
        "euler",
        "quat",
        "plane_size",
        "tile_size",
        "visualization",
        "collision",
        "contype",
        "conaffinity",
    }
)

class PlaneBuilder(WorldNode):
    """Builder for creating a flat plane (e.g. floor) at a specific height.

    Uses ``gs.morphs.Plane`` positioned at the given height. The plane is
    infinite in extent and acts as a static collision surface.

    Keyword arguments intended for ``gs.morphs.Plane`` (e.g. ``pos``,
    ``normal``, ``euler``, ``quat``, ``plane_size``, ``tile_size``,
    ``visualization``, ``collision``, ``contype``, ``conaffinity``) are
    forwarded to the morph, while any remaining kwargs are forwarded to
    ``world.scene.add_entity``. Unless the caller explicitly provides ``pos``,
    the plane is placed at ``(0, 0, height)``.

    Args:
        world: GenesisWorld instance
        name: Name for the plane entity
        height: Z-coordinate of the plane surface (default ``0.0`` = floor level)
        color: RGBA diffuse color tuple (default light grey)

    Example:
        >>> from unienv_genesis import GenesisWorld
        >>> from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
        >>>
        >>> world = GenesisWorld()
        >>> plane = PlaneBuilder(world, name="floor", height=0.0)
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    reload_priorities = {100}

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "plane",
        height: float = 0.0,
        **build_plane_kwargs,
    ):
        self.world : GenesisWorld = world
        self._name = name
        self.height = height
        self.build_plane_kwargs = build_plane_kwargs

        self.entity = None

    @property
    def name(self) -> str:
        return self._name

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        if priority == 100:
            self.entity = self.build_plane(self.world)
    
    def build_plane(
        self, 
        world : GenesisWorld
    ) -> Entity:
        """Helper method to build the plane morph.

        Splits ``self.build_plane_kwargs`` into morph-level kwargs (forwarded to
        ``gs.morphs.Plane``) and entity-level kwargs (forwarded to
        ``world.scene.add_entity``).
        """
        plane_kwargs = {}
        entity_kwargs = {}
        for key, value in self.build_plane_kwargs.items():
            if key in _PLANE_MORPH_KWARGS:
                plane_kwargs[key] = value
            else:
                entity_kwargs[key] = value

        # Default the plane position to (0, 0, height) unless the caller
        # explicitly supplies ``pos``.
        if "pos" not in plane_kwargs:
            plane_kwargs["pos"] = (0.0, 0.0, self.height)

        return world.scene.add_entity(
            gs.morphs.Plane(**plane_kwargs),
            name=self._name,
            **entity_kwargs
        )

    def get_observation(self):
        return None
