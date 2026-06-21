"""Tabletop manipulation tasks for Genesis scenes.

Tasks are :class:`WorldNode` subclasses that define a reward function,
termination condition, and observation for a specific manipulation goal.

Available tasks:

* :class:`ObjectRelocateTask` — pick-and-place: move an object from a random
  position to a goal position (3-D distance reward).
* :class:`ReachTargetTask` — reaching: move the end-effector to a random
  target position in the workspace (no object involved).
"""

from unienv_genesis_collection.tasks.tabletop.pickplace import ObjectRelocateTask
from unienv_genesis_collection.tasks.tabletop.reach import ReachTargetTask

__all__ = [
    "ObjectRelocateTask",
    "ReachTargetTask",
]
