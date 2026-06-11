"""Goal region builder for simulation scenes."""

from typing import Optional

import genesis as gs

from unienv_genesis import GenesisWorld
from unienv_genesis.utils.entity_group import EntityGroup
from unienv_interface.world.node import WorldNode

# Default goal region radius (in meters)
DEFAULT_GOAL_RADIUS = 0.03

class GoalRegionBuilder(WorldNode):
    """Builder for creating goal regions in simulation scenes.

    Creates a goal region with the specified radius.

    Args:
        world: GenesisWorld instance
        name: Name for the goal region entity
        radius: Radius of the goal region in meters
        pos: Position of the goal region center as (x, y, z) in world coordinates
        quat: Orientation quaternion (w, x, y, z). Default is identity (no rotation).

    Example:
        >>> from unienv_genesis import GenesisWorld
        >>> from unienv_genesis_collection.scenes.goal_region_builder import GoalRegionBuilder
        >>>
        >>> world = GenesisWorld()
        >>>
        >>> # Goal region with radius 0.03m, centered at origin
        >>> goal_region = GoalRegionBuilder(world, name="goal", radius=0.02, pos=(0, 0, 0))
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    reload_priorities = {100}

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "goal_region",
        radius: float = DEFAULT_GOAL_RADIUS,
        pos: tuple = (0.0, 0.0, 0.0),
        quat: tuple = (1.0, 0.0, 0.0, 0.0),
        visible: bool = True,
    ):
        self.world = world
        self._name = name
        self.radius = radius
        self.pos = pos
        self.quat = quat
        self.visible = visible

        # EntityGroup is created in reload() once entities exist
        self.group: Optional[EntityGroup] = None

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def create_goal(
        world: GenesisWorld,
        name: str = "goal_region",
        radius: float = DEFAULT_GOAL_RADIUS,
        pos: tuple = (0.0, 0.0, 0.0),
        quat: tuple = (1.0, 0.0, 0.0, 0.0),
        visible: bool = True,
        batch_fixed_verts: bool = False,
    ) -> EntityGroup:
        """Static method to create a goal region directly without instantiating TableBuilder.

        See class docstring for the coordinate convention.

        Args:
            world: GenesisWorld instance
            name: Name for the goal region entity
            radius: Radius of the goal region in meters
            pos: Position of goal region center as (x, y, z)
            quat: Orientation quaternion (w, x, y, z)
            batch_fixed_verts: If True, enables per-environment position overrides for
                fixed entities in a batched world (required by Genesis when calling
                ``set_pos`` with env-specific values on ``fixed=True`` morphs).

        Returns:
            ``EntityGroup`` whose anchor is the goal region surface center.
        """
        group = EntityGroup(pos=pos, quat=quat, dtype=world.dtype)
        opacity = float(visible)

        for i in range(4):
            disc_local = (0.0, 0.0, i * 1e-5)
            disc_world_pos, disc_world_quat = group.world_pose(disc_local)
            disc_color = (1, 0, 0) if i % 2 == 0 else (1, 1, 1)
            disc = world.scene.add_entity(
                gs.morphs.Cylinder(
                    radius=(1-0.25*i) * radius,
                    height=1e-4,
                    pos=disc_world_pos.tolist(),
                    quat=disc_world_quat.tolist(),
                    fixed=True,
                    collision=False,
                    batch_fixed_verts=batch_fixed_verts,
                ),
                surface=gs.surfaces.Metal(
                    color=(*disc_color, opacity),
                    metallic=0.8,
                    roughness=0.2,
                ),
                name=f"{name}_disc_{i}",
            )
            group.add_member(disc, local_pos=disc_local)

        return group

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        if priority == 100:
            self.group = self.create_goal(
                world=self.world,
                name=self._name,
                radius=self.radius,
                pos=self.pos,
                quat=self.quat,
                visible=self.visible,
                batch_fixed_verts=False,
            )

    def get_observation(self):
        return None
