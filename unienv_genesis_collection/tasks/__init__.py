"""Task collection for Genesis scenes.

Re-exports the tabletop task package so tasks can be imported as::

    from unienv_genesis_collection.tasks import ReachTargetTask
"""

from unienv_genesis_collection.tasks.tabletop import (
    ObjectRelocateTask,
    ReachTargetTask,
)

__all__ = [
    "ObjectRelocateTask",
    "ReachTargetTask",
]
