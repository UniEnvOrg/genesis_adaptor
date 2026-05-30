from typing import Protocol, Tuple, Union, Optional
from dataclasses import dataclass
import numpy as np

from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from unienv_genesis.world.genesis_world import GenesisWorld


class AttachmentProvider(Protocol):
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