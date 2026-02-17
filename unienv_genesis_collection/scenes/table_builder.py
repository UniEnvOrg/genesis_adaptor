"""Table builder for simulation scenes."""

import random
from pathlib import Path
from typing import Optional, List, Tuple, Any

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity

from unienv_genesis import GenesisWorld
from unienv_interface.world.node import WorldNode
from unienv_genesis_collection.asset_path import get_asset_path


# Default table dimensions (in meters)
DEFAULT_TABLE_SIZE = (1.0, 0.6, 0.02)  # width, depth, thickness
DEFAULT_TABLE_HEIGHT = 0.8
DEFAULT_LEG_RADIUS = 0.025


def list_all_table_textures() -> List[Path]:
    """Get a list of all available table textures.
    
    Returns:
        List of Path objects for all available tabletop textures
    """
    texture_dir = get_asset_path("scenes", "table", "tabletop_textures")
    return list(texture_dir.glob("*.jpg"))


def get_random_table_texture() -> Path:
    """Get a random table texture from the available textures."""
    textures = list_all_table_textures()
    return random.choice(textures)


class TableBuilder(WorldNode):
    """Builder for creating tables in simulation scenes.
    
    Creates a table with a textured tabletop and four legs.
    Supports random texture selection during reload if no texture is specified.
    
    Args:
        world: GenesisWorld instance
        name: Name for the table entity
        size: Tabletop size as (width, depth, thickness) in meters
        height: Table height (distance from floor to tabletop surface) in meters
        leg_radius: Radius of table legs in meters
        position: Table position as (x, y, z) in world coordinates
        texture_path: Path to the tabletop texture image. If None, a random texture
            is selected during each reload.
        
    Example:
        >>> from unienv_genesis import GenesisWorld
        >>> from unienv_genesis_collection.scenes.table_builder import TableBuilder, list_all_table_textures
        >>>
        >>> world = GenesisWorld()
        >>> 
        >>> # Table with random texture on each reload
        >>> table = TableBuilder(world, name="table")
        >>>
        >>> # Table with specific texture
        >>> from unienv_genesis_collection.asset_path import get_asset_path
        >>> texture = get_asset_path("scenes", "table", "tabletop_textures", "wood_texture_1.jpg")
        >>> table = TableBuilder(world, name="table", texture_path=texture)
        >>>
        >>> # List all available textures
        >>> all_textures = list_all_table_textures()
    """
    
    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False
    
    # Priority for scene building - create table early (before robots)
    reload_priorities = {-100}
    reset_priorities = {0}
    after_reload_priorities = {0}
    after_reset_priorities = {0}
    
    def __init__(
        self,
        world: GenesisWorld,
        name: str = "table",
        size: tuple = DEFAULT_TABLE_SIZE,
        height: float = DEFAULT_TABLE_HEIGHT,
        leg_radius: float = DEFAULT_LEG_RADIUS,
        position: tuple = (0.0, 0.0, 0.0),
        texture_path: Optional[Path] = None,
    ):
        self.world = world
        self._name = name
        self.size = size
        self.height = height
        self.leg_radius = leg_radius
        self.position = position
        self.texture_path = texture_path
        
        # Entities will be created in reload
        self.tabletop_entity = None
        self.leg_entities = []
        
    @property
    def name(self) -> str:
        return self._name
    
    @staticmethod
    def create_table(
        world: GenesisWorld,
        name: str = "table",
        size: tuple = DEFAULT_TABLE_SIZE,
        height: float = DEFAULT_TABLE_HEIGHT,
        leg_radius: float = DEFAULT_LEG_RADIUS,
        position: tuple = (0.0, 0.0, 0.0),
        texture_path: Optional[Path] = None,
    ) -> Tuple[Any, List[Any]]:
        """Static method to create a table directly without instantiating TableBuilder.
        
        This is useful when you need to create a table immediately without going through
        the node lifecycle (reload/reset).
        
        Args:
            world: GenesisWorld instance
            name: Name for the table entity
            size: Tabletop size as (width, depth, thickness) in meters
            height: Table height (distance from floor to tabletop surface) in meters
            leg_radius: Radius of table legs in meters
            position: Table position as (x, y, z) in world coordinates
            texture_path: Path to the tabletop texture image. If None, a random texture is used.
        
        Returns:
            Tuple of (tabletop_entity, list_of_leg_entities)
        
        Example:
            >>> from unienv_genesis import GenesisWorld
            >>> from unienv_genesis_collection.scenes.table_builder import TableBuilder
            >>>
            >>> world = GenesisWorld()
            >>> tabletop, legs = TableBuilder.create_table(world, name="my_table")
        """
        # Select texture (random if not specified)
        if texture_path is None:
            texture_path = get_random_table_texture()
        
        # Create tabletop
        tabletop_pos = (
            position[0],
            position[1],
            position[2] + height - size[2] / 2
        )
        
        # Load texture
        texture = gs.textures.ImageTexture(
            image_path=str(texture_path),
        )
        
        tabletop_entity = world.scene.add_entity(
            gs.morphs.Box(
                size=size,
                pos=tabletop_pos,
            ),
            surface=gs.surfaces.Surface(
                color=(1.0, 1.0, 1.0, 1.0),
                metallic=0.5,
                roughness=0.5,
                texture=texture,
            ),
            name=f"{name}_top",
        )
        
        # Create four legs
        leg_height = height - size[2]
        leg_offset_x = size[0] / 2 - leg_radius * 2
        leg_offset_y = size[1] / 2 - leg_radius * 2
        
        leg_positions = [
            (position[0] + leg_offset_x, position[1] + leg_offset_y, position[2] + leg_height / 2),
            (position[0] - leg_offset_x, position[1] + leg_offset_y, position[2] + leg_height / 2),
            (position[0] + leg_offset_x, position[1] - leg_offset_y, position[2] + leg_height / 2),
            (position[0] - leg_offset_x, position[1] - leg_offset_y, position[2] + leg_height / 2),
        ]
        
        leg_entities = []
        for i, leg_pos in enumerate(leg_positions):
            leg = world.scene.add_entity(
                gs.morphs.Cylinder(
                    radius=leg_radius,
                    height=leg_height,
                    pos=leg_pos,
                ),
                surface=gs.surfaces.Metal(
                    color=(0.3, 0.3, 0.3, 1.0),
                    metallic=0.8,
                    roughness=0.2,
                ),
                name=f"{name}_leg_{i}",
            )
            leg_entities.append(leg)
        
        return tabletop_entity, leg_entities
    
    @staticmethod
    def set_table_transform(
        tabletop_entity: Any,
        leg_entities: List[Any],
        size: tuple,
        height: float,
        leg_radius: float,
        position: tuple,
    ) -> None:
        """Static method to set the position (transform) of an existing table.
        
        This is useful for moving a table to a new position without recreating it.
        
        Args:
            tabletop_entity: The tabletop entity returned by create_table
            leg_entities: List of leg entities returned by create_table
            size: Tabletop size as (width, depth, thickness) in meters
            height: Table height (distance from floor to tabletop surface) in meters
            leg_radius: Radius of table legs in meters
            position: New table position as (x, y, z) in world coordinates
        
        Example:
            >>> from unienv_genesis import GenesisWorld
            >>> from unienv_genesis_collection.scenes.table_builder import TableBuilder
            >>>
            >>> world = GenesisWorld()
            >>> tabletop, legs = TableBuilder.create_table(world, name="my_table")
            >>>
            >>> # Move table to new position
            >>> TableBuilder.set_table_transform(
            ...     tabletop, legs,
            ...     size=(1.0, 0.6, 0.02),
            ...     height=0.8,
            ...     leg_radius=0.025,
            ...     position=(0.5, 0.0, 0.0)
            ... )
        """
        # Update tabletop position
        tabletop_pos = (
            position[0],
            position[1],
            position[2] + height - size[2] / 2
        )
        tabletop_entity.set_pos(tabletop_pos)
        
        # Update leg positions
        leg_height = height - size[2]
        leg_offset_x = size[0] / 2 - leg_radius * 2
        leg_offset_y = size[1] / 2 - leg_radius * 2
        
        leg_positions = [
            (position[0] + leg_offset_x, position[1] + leg_offset_y, position[2] + leg_height / 2),
            (position[0] - leg_offset_x, position[1] + leg_offset_y, position[2] + leg_height / 2),
            (position[0] + leg_offset_x, position[1] - leg_offset_y, position[2] + leg_height / 2),
            (position[0] - leg_offset_x, position[1] - leg_offset_y, position[2] + leg_height / 2),
        ]
        
        for leg, leg_pos in zip(leg_entities, leg_positions):
            leg.set_pos(leg_pos)
    
    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        if priority == -100:
            # Set seed if provided for reproducible texture randomization
            if seed is not None:
                random.seed(seed)
            
            # Use create_table static method
            self.tabletop_entity, self.leg_entities = self.create_table(
                world=self.world,
                name=self._name,
                size=self.size,
                height=self.height,
                leg_radius=self.leg_radius,
                position=self.position,
                texture_path=self.texture_path,
            )
    
    def reset(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        pass
    
    def get_observation(self):
        return None
