"""ManipulatableTool — mixin interface for objects with tool functionality.

A *tool* is an object that has one or more **functional capabilities**
(e.g. sweeping, scooping, cutting, pushing).  Each capability is identified
by a string *function type* and is described as a sequence of
:class:`FunctionalPoint` instances sampling the relevant surface or edge.

``ManipulatableTool`` is a standalone ABC (not a subclass of
``ManipulatableObject``).  Concrete classes can inherit from both — e.g.
``class MySpatula(MeshObject, ManipulatableTool)`` — so that any
``ManipulatableObject`` subclass can also be a tool.

Subclasses must implement:

* :attr:`function_types` — declare supported function type strings.
* :meth:`list_functional_points` — sample points for a given function type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Sequence, Tuple

import torch

__all__ = [
    "ManipulatableTool",
    "FunctionalPoint",
]


@dataclass
class FunctionalPoint:
    """A point on a tool's functional edge or surface.

    All vectors are in the tool's **local frame**.

    Attributes:
        position: Point position, shape ``(3,)``.
        direction_vector: Optional direction at this point (e.g. surface
            normal, tangent, or push direction), shape ``(3,)``.
            ``None`` when only position matters.
    """

    position: torch.Tensor  # (3,)
    direction_vector: Optional[torch.Tensor] = None  # (3,) or None


class ManipulatableTool(ABC):
    """Abstract mixin for objects that can be used as tools.

    This is NOT a subclass of ``ManipulatableObject``.  Combine it with
    any ``ManipulatableObject`` subclass via multiple inheritance::

        class MySpatula(MeshObject, ManipulatableTool):
            ...

    Subclasses must implement:

    * :attr:`function_types`
    * :meth:`list_functional_points`
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def function_types(self) -> Sequence[str]:
        """Return the function types this tool supports.

        Each string identifies a capability (e.g. ``"sweep"``, ``"scoop"``,
        ``"cut"``, ``"push"``).  The set of supported types is fixed per
        tool class.
        """
        ...

    @abstractmethod
    def list_functional_points(
        self,
        entity,
        function_type: str,
    ) -> List[FunctionalPoint]:
        """Return points sampling a functional region for *function_type*.

        Args:
            entity: The loaded Genesis entity.
            function_type: One of :attr:`function_types`.

        Returns:
            List of :class:`FunctionalPoint` in the tool's local frame.

        Raises:
            ValueError: If *function_type* is not in :attr:`function_types`.
        """
        ...

