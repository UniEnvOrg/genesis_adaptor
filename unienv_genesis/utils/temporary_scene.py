"""Utility for building a minimal temporary Genesis scene to query entity properties.

During node ``__init__`` the main scene has not been compiled yet, so properties
like ``entity.n_dofs`` and ``entity.get_dofs_limit()`` are unavailable.
``TemporaryScene`` spins up a throw-away scene via the world's own scene-management
API, queries what is needed, then destroys it — enabling controllers to produce
accurate *preliminary* action / observation spaces before the real scene is built.

Example usage::

    from unienv_genesis.utils.temporary_scene import TemporaryScene

    with TemporaryScene(world) as tmp:
        entity = tmp.scene.add_entity(morph)
        tmp.build()
        n_dofs = entity.n_dofs
    # scene is destroyed on exit; do not use entity after this point
"""

import logging
import os
from typing import Optional

import genesis as gs
from unienv_genesis.world.genesis_world import GenesisWorld

LOGGER = logging.getLogger(__name__)

class TemporaryScene:
    def __init__(self, world : GenesisWorld):
        self.world = world
        self.scene = None
    
    def __enter__(self) -> "TemporaryScene":
        self.scene = self.world.create_scene()
        return self

    def build(self) -> None:
        if self.scene is not None and not self.scene.is_built:
            self.world.build_scene(self.scene, n_envs=0)
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.scene is not None:
            self.scene.destroy()
        self.scene = None