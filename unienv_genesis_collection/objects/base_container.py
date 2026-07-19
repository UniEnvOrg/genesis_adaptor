"""ManipulatableContainer — extends ManipulatableObject for hollow objects.

A *container* is a manipulatable object that has an interior cavity
(bowls, cups, trays, drawers, boxes-with-lids, etc.).  It adds methods for
querying the interior bounding box and checking whether another object fits
inside.

Containers may be articulated (e.g. a drawer that must be opened) or have
lids, so runtime state matters. Local geometry queries (such as the local
bounding box) are entity-independent and may be used without a loaded
entity; runtime, world-frame, and interior queries depend on the loaded
entity.

Subclasses must implement:

* :meth:`get_interior_bounding_box` — AABB of the interior cavity.
* The inherited abstract methods from :class:`ManipulatableObject`
  (``load_into_scene``, ``get_local_bounding_box``).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from unienv_genesis_collection.objects.base_object import (
    ManipulatableObject,
    EntityT,
    bbox_size,
)

if TYPE_CHECKING:
    from unienv_interface.space import BoxSpace

__all__ = ["ManipulatableContainer"]


class ManipulatableContainer(ManipulatableObject[EntityT]):
    """Abstract base for container objects with an interior cavity.

    Subclasses must implement:

    * :meth:`get_interior_bounding_box`
    * :meth:`load_into_scene`  (inherited)
    * :meth:`get_local_bounding_box` (inherited)
    """

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def get_interior_bounding_box(self, entity: EntityT) -> "BoxSpace":
        """Return the AABB of the interior cavity in the local frame.

        This describes the usable internal volume — the space where objects
        can be placed *inside* the container.

        Args:
            entity: The loaded Genesis entity.

        Returns:
            ``BoxSpace`` (PyTorch, float32, shape ``(3,)``) for the interior.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def can_contain(
        self,
        entity: EntityT,
        other: ManipulatableObject,
        other_entity: EntityT,
    ) -> bool:
        """Check whether *other* fits inside this container's interior.

        Compares bounding-box dimensions axis-by-axis (no rotation).

        Args:
            entity: The loaded Genesis entity for this container.
            other: Another manipulatable object descriptor.
            other_entity: The loaded Genesis entity for *other*.

        Returns:
            ``True`` if every dimension of *other*'s bbox is ≤ the
            corresponding interior dimension.
        """
        interior = self.get_interior_bounding_box(entity)
        other_bbox = other.get_local_bounding_box()
        return bool(torch.all(bbox_size(other_bbox) <= bbox_size(interior)))
