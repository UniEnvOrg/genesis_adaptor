"""Task collection for Genesis scenes.

Re-exports the tabletop and articulated task packages so tasks can be imported
as::

    from unienv_genesis_collection.tasks import ReachTargetTask
    from unienv_genesis_collection.tasks import OpenArticulatedJointTask
"""

from unienv_genesis_collection.tasks.tabletop import (
    ObjectRelocateTask,
    ReachTargetTask,
)
from unienv_genesis_collection.tasks.articulated import (
    OpenArticulatedJointTask,
)

__all__ = [
    "ObjectRelocateTask",
    "ReachTargetTask",
    "OpenArticulatedJointTask",
]
