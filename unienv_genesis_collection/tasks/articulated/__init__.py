"""Articulated-furniture tasks for Genesis rooms.

Tasks are :class:`WorldNode` subclasses that define an observation, a reward
function and a termination condition for a goal expressed on **articulated
furniture** (joints addressed by name against the built entity, no per-asset
registry).

Available tasks:

* :class:`OpenArticulatedJointTask` — open one named joint of a
  :class:`~unienv_genesis_collection.scenes.room_builder.RoomBuilder` furniture
  instance to a randomly sampled target open fraction (dense fraction-tracking
  reward).
"""

from unienv_genesis_collection.tasks.articulated.open_fixture import (
    OpenArticulatedJointTask,
)

__all__ = [
    "OpenArticulatedJointTask",
]
