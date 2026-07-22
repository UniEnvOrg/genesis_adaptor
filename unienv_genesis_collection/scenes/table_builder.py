"""Table builder for simulation scenes."""

import math
import random
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import genesis as gs

from unienv_genesis import GenesisWorld
from unienv_genesis.utils.entity_group import EntityGroup
from unienv_interface.world.node import WorldNode
from unienv_genesis_collection.asset_path import get_asset_path
from xbarray.transformations.rotation_conversions.pytorch import quaternion_multiply

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
    return sorted(texture_dir.glob("*.jpg"))


def get_random_table_texture(rng: Optional[random.Random] = None) -> Path:
    """Get a random table texture from the available textures."""
    choice = rng.choice if rng is not None else random.choice
    return choice(list_all_table_textures())


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

def get_random_leg_color(rng: Optional[random.Random] = None) -> tuple:
    """Return a random realistic table-leg RGBA colour."""
    choice = rng.choice if rng is not None else random.choice
    return choice(_LEG_COLORS)


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Convert a Z-axis yaw angle (radians) to a quaternion ``(w, x, y, z)``."""
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


class TableBuilder(WorldNode):
    """Builder for creating tables in simulation scenes.

    Creates a table with a textured tabletop and four legs.
    Supports random texture selection during reload if no texture is specified.

    The anchor point of the table is at the center of the top surface:
    - ``pos=(0, 0, 0)`` places the table surface center at the origin.
    - The tabletop and legs extend **downward** from ``pos[2]``.

    Runtime parameter contract:
    - ``default_size``, ``default_height``, ``default_texture_path``, and
      ``default_leg_color`` store construction defaults.
    - ``size`` and ``height`` are fixed tensors set once during ``__init__``:
      ``(3,)`` / scalar in unbatched mode, ``(n_envs, 3)`` / ``(n_envs,)``
      in batched mode.

    Args:
        world: GenesisWorld instance
        name: Name for the table entity
        size: Tabletop size as (width, depth, thickness) in meters
        height: Table height (distance from table surface down to the floor) in meters
        leg_radius: Radius of table legs in meters
        pos: Position of the table surface center as (x, y, z) in world coordinates
        yaw: Rotation around the Z axis in radians. Default is 0 (no rotation).
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

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "table",
        size: tuple[float, float, float] = DEFAULT_TABLE_SIZE,
        height: float = DEFAULT_TABLE_HEIGHT,
        leg_radius: float = DEFAULT_LEG_RADIUS,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        texture_path: Optional[Path] = None,
        leg_color: Optional[tuple] = None,
        seed: Optional[int] = None,
    ):
        self.world = world
        self._name = name
        self.default_size = tuple(float(v) for v in size)
        self.default_height = float(height)
        self.default_texture_path = texture_path
        self.default_leg_color = (
            tuple(float(v) for v in leg_color) if leg_color is not None else None
        )

        size_t = torch.tensor(self.default_size, dtype=world.dtype)
        height_t = torch.tensor(self.default_height, dtype=world.dtype)
        if world.batch_size is None:
            self.size = size_t
            self.height = height_t
        else:
            self.size = size_t.unsqueeze(0).repeat(world.batch_size, 1)
            self.height = height_t.repeat(world.batch_size)

        self.leg_radius = leg_radius
        self.pos = pos
        self.yaw = yaw
        self.quat = yaw_to_quat(yaw)
        self._cos_yaw = math.cos(yaw)
        self._sin_yaw = math.sin(yaw)
        self._pos_t = torch.tensor(pos, dtype=world.dtype)
        self._quat_t = torch.tensor(self.quat, dtype=world.dtype)

        # EntityGroup is created in reload() once entities exist.
        self._group: Optional[EntityGroup] = None

        # Per-instance RNG for randomised texture/leg-colour selection during
        # reload().  Seeded from ``seed`` when provided so that reload(seed=...)
        # is reproducible; otherwise OS entropy.  Reseeded again in reload()
        # when an explicit seed is passed to that lifecycle pass.
        self._rng = random.Random(seed)

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
        yaw: float = 0.0,
        texture_path: Optional[Path] = None,
        leg_color: Optional[tuple] = None,
        batch_fixed_verts: bool = False,
        rng: Optional[random.Random] = None,
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
            yaw: Rotation around the Z axis in radians. Default is 0.
            texture_path: Path to the tabletop texture image. If None, a random texture is used.
            leg_color: RGBA colour tuple for the legs. If None, a random colour is chosen.
            batch_fixed_verts: If True, enables per-environment position overrides for
                fixed entities in a batched world (required by Genesis when calling
                ``set_pos`` with env-specific values on ``fixed=True`` morphs).

        Returns:
            ``EntityGroup`` whose anchor is the table surface center.
        """
        quat = yaw_to_quat(yaw)
        # The tabletop uses a UV-mapped mesh (box_with_uv.obj) so that
        # ImageTexture renders correctly.  gs.morphs.Box lacks UV coordinates
        # and would silently show white when a texture is applied.
        if texture_path is None:
            texture_path = get_random_table_texture(rng)
        tabletop_surface = gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(texture_path))
        )

        # Build the group first so we can use world_pose() for morph placement.
        # Anchor = surface center; parts hang downward (negative z in anchor frame).
        group = EntityGroup(pos=pos, quat=quat, dtype=world.dtype)

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
                batch_fixed_verts=batch_fixed_verts,
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

        if leg_color is not None:
            resolved_leg_color = leg_color
        else:
            resolved_leg_color = get_random_leg_color(rng)
        for i, leg_local in enumerate(leg_locals):
            leg_world_pos, leg_world_quat = group.world_pose(leg_local)
            leg = world.scene.add_entity(
                gs.morphs.Cylinder(
                    radius=leg_radius,
                    height=leg_height,
                    pos=leg_world_pos.tolist(),
                    quat=leg_world_quat.tolist(),
                    fixed=True,
                    batch_fixed_verts=batch_fixed_verts,
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

    def placement_world_pose(
        self,
        local_pos: torch.Tensor,
        local_quat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a table-local position/quaternion to world frame.

        The local frame origin is at the table surface center (``self.pos``),
        with X/Y axes rotated by ``self.yaw`` around Z.

        Args:
            local_pos: ``(..., 3)`` or ``(..., 2)`` position in table-local
                frame.  If 2-D, Z defaults to ``self.pos[2]``.
            local_quat: ``(..., 4)`` quaternion ``(w, x, y, z)`` in the
                table-local frame.  If ``None``, uses the table's own
                orientation.

        Returns:
            world_pos: ``(..., 3)`` world-frame position.
            world_quat: ``(..., 4)`` world-frame quaternion ``(w, x, y, z)``.
        """
        if local_pos.shape[-1] not in (2, 3):
            raise ValueError(
                f"local_pos must have last dimension 2 or 3, got {tuple(local_pos.shape)}"
            )

        world_xy = self.xy_to_world(local_pos[..., :2])
        table_pos_z = self._pos_t.to(device=local_pos.device, dtype=local_pos.dtype)[2]
        if local_pos.shape[-1] == 2:
            world_z = torch.full_like(world_xy[..., 0], table_pos_z)
        else:
            world_z = table_pos_z + local_pos[..., 2]
        world_pos = torch.cat([world_xy, world_z.unsqueeze(-1)], dim=-1)

        table_quat = self._quat_t.to(device=world_pos.device, dtype=world_pos.dtype)
        if local_quat is None:
            world_quat = table_quat.expand(world_pos.shape[:-1] + (4,))
        else:
            world_quat = quaternion_multiply(table_quat, local_quat.to(table_quat))

        return world_pos, world_quat
    
    def xy_to_world(self, local_xy: torch.Tensor) -> torch.Tensor:
        """Convert table-local XY coordinates to world-frame XY.

        Args:
            local_xy: ``(..., 2)`` local XY offset(s) from table centre.

        Returns:
            ``(..., 2)`` world-frame XY coordinates.
        """
        if local_xy.shape[-1] != 2:
            raise ValueError(
                f"local_xy must have last dimension 2, got {tuple(local_xy.shape)}"
            )

        table_xy = self._pos_t[:2].to(device=local_xy.device, dtype=local_xy.dtype)
        c = torch.as_tensor(self._cos_yaw, device=local_xy.device, dtype=local_xy.dtype)
        s = torch.as_tensor(self._sin_yaw, device=local_xy.device, dtype=local_xy.dtype)

        wx = table_xy[0] + local_xy[..., 0] * c - local_xy[..., 1] * s
        wy = table_xy[1] + local_xy[..., 0] * s + local_xy[..., 1] * c
        return torch.stack([wx, wy], dim=-1)

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        if priority == 100:
            # Reseed before appearance selection.
            if seed is not None:
                self._rng.seed(seed)
            self._group = self.create_table(
                world=self.world,
                name=self._name,
                size=self.default_size,
                height=self.default_height,
                leg_radius=self.leg_radius,
                pos=self.pos,
                yaw=self.yaw,
                texture_path=self.default_texture_path,
                leg_color=self.default_leg_color,
                batch_fixed_verts=False,
                rng=self._rng,
            )
    
