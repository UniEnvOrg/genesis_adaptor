from typing import Protocol, Tuple, Union, Optional, Sequence
from dataclasses import dataclass
import numpy as np

from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from unienv_genesis.world.genesis_world import GenesisWorld


class AttachmentProvider(Protocol):
    """Provider that resolves an attachment point (entity + optional link).

    Implementations *may* expose optional ``offset_pos`` / ``offset_quat``
    attributes describing a virtual frame offset from the resolved link
    origin, mirroring the EEF virtual-offset convention.  Consumers that
    care about offsets should read them defensively with
    ``getattr(provider, 'offset_pos', None)`` / ``getattr(provider,
    'offset_quat', None)``, since ``Protocol`` attributes cannot have
    defaults and providers that don't support offsets simply omit them.
    """

    offset_pos: Optional[Sequence[float]]
    """Virtual offset position in the resolved link's local frame, or
    ``None`` when the provider doesn't support offsets."""

    offset_quat: Optional[Sequence[float]]
    """Virtual offset quaternion ``(w, x, y, z)`` in the resolved link's
    local frame, or ``None`` when the provider doesn't support offsets.
    Defaults to identity ``[1, 0, 0, 0]`` in consumers when ``offset_pos``
    is set but this is ``None``."""

    def get_attachment(self, world: GenesisWorld) -> Tuple[RigidEntity, Optional[RigidLink]]:
        """
        Obtain an attachment point for a new node.
        Args:
            - world (GenesisWorld): The GenesisWorld instance to which the new node will be attached. This can be used to query existing entities and their states to determine a suitable attachment point.
        Returns:
            - entity (RigidEntity): The rigid entity to which the new node should be attached.
            - link (Optional[RigidLink]): The link on the rigid entity to which the new node should be attached. If None, the new node should be attached to the root of the rigid entity.
        """
        raise NotImplementedError("AttachmentProvider is a protocol and should not be instantiated directly. Implement get_attachment() in a subclass.")


@dataclass(frozen=True)
class AttachFromName(AttachmentProvider):
    entity_name: str
    link_name: Optional[str] = None
    offset_pos: Optional[Sequence[float]] = None
    offset_quat: Optional[Sequence[float]] = None

    def get_attachment(self, world: GenesisWorld) -> Tuple[RigidEntity, Optional[RigidLink]]:
        # Find the entity by name
        entity = world.scene.get_entity(self.entity_name)
        assert isinstance(entity, RigidEntity), f"Entity '{self.entity_name}' must be a RigidEntity."

        # If link_name is specified, find the link
        if self.link_name is not None:
            link : RigidLink = entity.get_link(self.link_name)
        else:
            link = None

        return entity, link


@dataclass(frozen=True)
class LinkAttachment:
    """A link reference on a known entity, with an optional virtual offset.

    Used by controllers that already hold a direct entity reference and only
    need to identify a link by name.  The offset, when set, places a virtual
    frame away from the physical link origin — mirroring the EEF virtual-offset
    convention.
    """
    link_name: str
    offset_pos: Optional[Sequence[float]] = None
    offset_quat: Optional[Sequence[float]] = None
