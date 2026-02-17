"""Table builder for simulation scenes."""

import random
from pathlib import Path
from typing import Optional, List

import genesis as gs

from unienv_genesis import GenesisWorld
from unienv_genesis.utils.entity_group import EntityGroup
from unienv_interface.world.node import WorldNode
from unienv_genesis_collection.asset_path import get_asset_path

# Default table dimensions (in meters)
DEFAULT_TABLE_SIZE = (1.0, 0.6, 0.02)  # width, depth, thickness
DEFAULT_TABLE_HEIGHT = 0.6
DEFAULT_LEG_RADIUS = 0.025

# Unit-cube mesh with per-face UV coordinates — used for the tabletop so that
# ImageTexture renders correctly (gs.morphs.Box lacks UV coordinates).
_BOX_WITH_UV_OBJ = get_asset_path("scenes", "table", "box_with_uv.obj")


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


# Realistic leg colour presets (RGBA).
_LEG_COLORS = [
    (0.08, 0.08, 0.08, 1.0),  # black powder coat
    (0.30, 0.30, 0.30, 1.0),  # dark grey
    (0.75, 0.75, 0.75, 1.0),  # silver / stainless
    (0.92, 0.92, 0.92, 1.0),  # white
    (0.85, 0.82, 0.75, 1.0),  # warm beige
    (0.80, 0.65, 0.20, 1.0),  # brass / gold
    (0.35, 0.25, 0.15, 1.0),  # dark bronze
]

def get_random_leg_color() -> tuple:
    """Return a random realistic table-leg RGBA colour."""
    return random.choice(_LEG_COLORS)

class TableBuilder(WorldNode):
    """Builder for creating tables in simulation scenes.

    Creates a table with a textured tabletop and four legs.
    Supports random texture selection during reload if no texture is specified.

    The anchor point of the table is at the center of the top surface:
    - ``pos=(0, 0, 0)`` places the table surface center at the origin.
    - The tabletop and legs extend **downward** from ``pos[2]``.

    Args:
        world: GenesisWorld instance
        name: Name for the table entity
        size: Tabletop size as (width, depth, thickness) in meters
        height: Table height (distance from table surface down to the floor) in meters
        leg_radius: Radius of table legs in meters
        pos: Position of the table surface center as (x, y, z) in world coordinates
        quat: Orientation quaternion (w, x, y, z). Default is identity (no rotation).
        texture_path: Path to the tabletop texture image. If None, a random texture
            is selected during each reload.
        leg_color: RGBA colour tuple for the legs. If None, a random colour is chosen
            from a set of realistic presets during each reload.

    Example:
        >>> from unienv_genesis import GenesisWorld
        >>> from unienv_genesis_collection.scenes.table_builder import TableBuilder
        >>>
        >>> world = GenesisWorld()
        >>>
        >>> # Table surface at z=0.8, centered at origin
        >>> table = TableBuilder(world, name="table", pos=(0, 0, 0.8))
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    # Priority for scene building - create table early (before robots)
    reload_priorities = {100}
    reset_priorities = {0}
    after_reload_priorities = set()
    after_reset_priorities = {0}

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "table",
        size: tuple = DEFAULT_TABLE_SIZE,
        height: float = DEFAULT_TABLE_HEIGHT,
        leg_radius: float = DEFAULT_LEG_RADIUS,
        pos: tuple = (0.0, 0.0, 0.0),
        quat: tuple = (1.0, 0.0, 0.0, 0.0),
        texture_path: Optional[Path] = None,
        leg_color: Optional[tuple] = None,
    ):
        self.world = world
        self._name = name
        self.size = size
        self.height = height
        self.leg_radius = leg_radius
        self.pos = pos
        self.quat = quat
        self.texture_path = texture_path
        self.leg_color = leg_color

        # EntityGroup is created in reload() once entities exist
        self.group: Optional[EntityGroup] = None

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
        pos: tuple = (0.0, 0.0, 0.0),
        quat: tuple = (1.0, 0.0, 0.0, 0.0),
        texture_path: Optional[Path] = None,
        leg_color: Optional[tuple] = None,
    ) -> EntityGroup:
        """Static method to create a table directly without instantiating TableBuilder.

        See class docstring for the coordinate convention.

        Args:
            world: GenesisWorld instance
            name: Name for the table entity
            size: Tabletop size as (width, depth, thickness) in meters
            height: Table height (surface to floor) in meters
            leg_radius: Radius of table legs in meters
            pos: Position of table surface center as (x, y, z)
            quat: Orientation quaternion (w, x, y, z)
            texture_path: Path to the tabletop texture image. If None, a random texture is used.
            leg_color: RGBA colour tuple for the legs. If None, a random colour is chosen.

        Returns:
            ``EntityGroup`` whose anchor is the table surface center.
        """
        # The tabletop uses a UV-mapped mesh (box_with_uv.obj) so that
        # ImageTexture renders correctly.  gs.morphs.Box lacks UV coordinates
        # and would silently show white when a texture is applied.
        if texture_path is None:
            texture_path = get_random_table_texture()
        tabletop_surface = gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(texture_path))
        )

        # Build the group first so we can use world_pose() for morph placement.
        # Anchor = surface center; parts hang downward (negative z in anchor frame).
        group = EntityGroup(pos=pos, quat=quat)

        tabletop_local = (0.0, 0.0, -size[2] / 2)
        top_world_pos, top_world_quat = group.world_pose(tabletop_local)
        tabletop_entity = world.scene.add_entity(
            gs.morphs.Mesh(
                file=str(_BOX_WITH_UV_OBJ),
                scale=size,
                pos=top_world_pos.tolist(),
                quat=top_world_quat.tolist(),
                fixed=True,
                decimate=False,
            ),
            surface=tabletop_surface,
            name=f"{name}_top",
        )
        group.add_member(tabletop_entity, local_pos=tabletop_local)

        leg_height = height - size[2]
        leg_offset_x = size[0] / 2 - leg_radius * 2
        leg_offset_y = size[1] / 2 - leg_radius * 2

        leg_locals = [
            (+leg_offset_x, +leg_offset_y, -size[2] - leg_height / 2),
            (-leg_offset_x, +leg_offset_y, -size[2] - leg_height / 2),
            (+leg_offset_x, -leg_offset_y, -size[2] - leg_height / 2),
            (-leg_offset_x, -leg_offset_y, -size[2] - leg_height / 2),
        ]

        resolved_leg_color = leg_color if leg_color is not None else get_random_leg_color()
        for i, leg_local in enumerate(leg_locals):
            leg_world_pos, leg_world_quat = group.world_pose(leg_local)
            leg = world.scene.add_entity(
                gs.morphs.Cylinder(
                    radius=leg_radius,
                    height=leg_height,
                    pos=leg_world_pos.tolist(),
                    quat=leg_world_quat.tolist(),
                    fixed=True,
                ),
                surface=gs.surfaces.Metal(
                    color=resolved_leg_color,
                    metallic=0.8,
                    roughness=0.2,
                ),
                name=f"{name}_leg_{i}",
            )
            group.add_member(leg, local_pos=leg_local)

        return group

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        if priority == 100:
            self.group = self.create_table(
                world=self.world,
                name=self._name,
                size=self.size,
                height=self.height,
                leg_radius=self.leg_radius,
                pos=self.pos,
                quat=self.quat,
                texture_path=self.texture_path,
                leg_color=self.leg_color,
            )

    def reset(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        pass

    def get_observation(self):
        return None
