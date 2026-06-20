"""Camera port for keypoint tracking.

This module quarantines all Genesis-specific camera access behind a narrow
interface (:class:`KeypointCameraPort`).  The concrete
:class:`GenesisCameraKeypointPort` wraps a :class:`CameraSensorNode` and
``GenesisWorld``, handling:

- Camera observation normalization (batch dimensions, depth→position fallback).
- Segmentation metadata parsing (``_visualizer.segmentation_idx_dict``).
- Configuration validation (render flags, segmentation level).
- Batch-size enforcement (all envs rendered in order).

The rest of the keypoint tracker is Genesis-agnostic and can be tested with
a fake port.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
from typing import Protocol

import torch
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink

from unienv_genesis.world.genesis_world import GenesisWorld

from ..camera_node import CameraSensorNode
from .types import CameraFrame, SegmentationIndex, SegmentationTarget

__all__ = [
    "KeypointCameraPort",
    "GenesisCameraKeypointPort",
]


class KeypointCameraPort(Protocol):
    """Interface for camera access required by keypoint tracking.

    Implementations provide normalized camera frames, build segmentation
    indices, and validate configuration.  This protocol allows the tracker
    to be tested with a fake port instead of a real Genesis scene.
    """

    @property
    def name(self) -> str:
        """Name of the delegated camera."""
        ...

    @property
    def render_seg(self) -> bool:
        """Whether the camera renders segmentation."""
        ...

    @property
    def render_depth(self) -> bool:
        """Whether the camera renders depth."""
        ...

    @property
    def render_position(self) -> bool:
        """Whether the camera renders per-pixel 3D positions."""
        ...

    def validate(self) -> None:
        """Validate camera configuration for keypoint tracking.

        Raises:
            ValueError: If the camera does not meet the tracker's
                requirements (segmentation, depth/position, same world).
        """
        ...

    def validate_segmentation_level(self) -> str:
        """Validate and return the scene's segmentation level.

        Returns:
            ``"link"`` or ``"geom"``.

        Raises:
            ValueError: If the level is ``"entity"`` or unsupported.
        """
        ...

    def current_frame(self) -> CameraFrame:
        """Fetch and normalize the current camera observation.

        All returned tensors have a guaranteed leading batch dimension.
        ``position`` is populated from ``depth`` via
        ``depth_to_position_map`` when only depth is available.

        Raises:
            RuntimeError: If the camera has no observation, is missing
                required keys, or the batch size does not match the world.
        """
        ...

    def build_segmentation_index(self) -> SegmentationIndex:
        """Build the immutable segmentation index from Genesis metadata.

        Raises:
            RuntimeError: If the camera's Genesis sensor does not expose
                a visualizer or segmentation index dictionary.
        """
        ...


class GenesisCameraKeypointPort:
    """Concrete port wrapping a :class:`CameraSensorNode` and ``GenesisWorld``.

    All Genesis-private attribute access (``segmentation_sensor``,
    ``_visualizer``, ``segmentation_idx_dict``, ``rigid_solver``) is
    confined to this class.
    """

    def __init__(self, camera_node: CameraSensorNode, world: GenesisWorld):
        self._camera_node = camera_node
        self._world = world
        self._n_envs = world.batch_size if world.batch_size is not None else 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._camera_node.name

    @property
    def render_seg(self) -> bool:
        return self._camera_node.render_seg

    @property
    def render_depth(self) -> bool:
        return self._camera_node.render_depth

    @property
    def render_position(self) -> bool:
        return self._camera_node.render_position

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate camera configuration for keypoint tracking."""
        if self._camera_node.world is not self._world:
            raise ValueError("camera_node must belong to the same world.")
        if not self._camera_node.render_seg:
            raise ValueError("camera_node must render segmentation (render_seg=True).")
        if not (self._camera_node.render_depth or self._camera_node.render_position):
            raise ValueError("camera_node must render depth or position.")

    def validate_segmentation_level(self) -> str:
        """Validate and return the scene's segmentation level."""
        seg_level = self._world.scene.vis_options.segmentation_level
        if seg_level == "entity":
            raise ValueError(
                f"KeypointTrackerNode does not support segmentation_level='entity'. "
                f"Use 'link' or 'geom' instead."
            )
        if seg_level not in ("link", "geom"):
            raise ValueError(
                f"Unsupported segmentation_level={seg_level!r}. "
                f"Must be 'link' or 'geom'."
            )
        return seg_level

    def _validate_camera_batch(self, n_camera_batch: int) -> None:
        """Assert that the camera batch size matches the world environment count.

        The tracker assumes **all environments are rendered in order** by the
        delegated camera, so the camera batch dimension must equal the number
        of environments (or 1 for an unbatched world).

        Raises:
            RuntimeError: If the sizes do not match.
        """
        if self._world.batch_size is None:
            if n_camera_batch != 1:
                raise RuntimeError(
                    f"Unbatched world expects camera batch size 1, got "
                    f"{n_camera_batch}. The keypoint tracker requires all "
                    f"environments to be rendered in order."
                )
            return
        if n_camera_batch != self._n_envs:
            raise RuntimeError(
                f"Camera batch size ({n_camera_batch}) does not match the "
                f"number of environments ({self._n_envs}). The keypoint "
                f"tracker requires all environments to be rendered in order "
                f"(no partial/subset rendering)."
            )

    # ------------------------------------------------------------------
    # Camera frame
    # ------------------------------------------------------------------

    def current_frame(self) -> CameraFrame:
        """Fetch and normalize the current camera observation.

        Returns a :class:`CameraFrame` with guaranteed batch dimensions on
        all tensors.  ``position`` is derived from ``depth`` via
        ``depth_to_position_map`` when only depth is available.

        Raises:
            RuntimeError: If the camera has no observation, is missing
                required keys, or the batch size does not match the world.
        """
        obs = self._camera_node.get_observation()
        if obs is None:
            raise RuntimeError(
                f"Camera node '{self._camera_node.name}' has no observation yet; "
                "ensure this tracking node runs after the camera."
            )

        required = ("seg", "intrinsics_cv", "extrinsics_cv")
        missing = [k for k in required if k not in obs]
        if missing:
            raise RuntimeError(
                f"Camera observation from '{self._camera_node.name}' is missing "
                f"required keys: {missing}."
            )

        seg = obs["seg"]
        intrinsics = obs["intrinsics_cv"]
        extrinsics = obs["extrinsics_cv"]

        # Normalize batch dimensions using ndim-based detection.
        if seg.ndim == 2:
            seg = seg.unsqueeze(0)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if extrinsics.ndim == 2:
            extrinsics = extrinsics.unsqueeze(0)

        B = seg.shape[0]
        self._validate_camera_batch(B)

        # Position: from "position" or derived from "depth".
        position = obs.get("position")
        depth = obs.get("depth")
        if position is None and depth is not None:
            position = self._camera_node.depth_to_position_map(depth)

        # Normalize position to batched.
        if position is not None:
            if position.ndim == 3 and position.shape[-1] == 3:
                position = position.unsqueeze(0)
            elif position.ndim != 4 or position.shape[-1] != 3:
                raise ValueError(f"Unsupported position map shape: {tuple(position.shape)}")
            if seg.shape[0] != position.shape[0]:
                raise ValueError(
                    f"Segmentation and position batch sizes mismatch: "
                    f"{seg.shape[0]} vs {position.shape[0]}"
                )

        # Normalize depth to batched.
        if depth is not None and depth.ndim == 2:
            depth = depth.unsqueeze(0)

        # Normalize auxiliary tensors (expand shared unbatched to B).
        rgb = obs.get("rgb")
        normal = obs.get("normal")
        if rgb is not None and rgb.ndim == 3:
            rgb = rgb.unsqueeze(0).expand(B, *rgb.shape)
        if normal is not None and normal.ndim == 3:
            normal = normal.unsqueeze(0).expand(B, *normal.shape)

        return CameraFrame(
            seg=seg,
            intrinsics_cv=intrinsics,
            extrinsics_cv=extrinsics,
            position=position,
            depth=depth,
            rgb=rgb,
            normal=normal,
        )

    # ------------------------------------------------------------------
    # Segmentation index
    # ------------------------------------------------------------------

    @staticmethod
    def _seg_key_to_target(
        seg_idxc: int, seg_key: Union[int, Tuple[int, ...]]
    ) -> Optional[SegmentationTarget]:
        """Parse a segmentation dict key into a :class:`SegmentationTarget`.

        Genesis segmentation keys have different structures depending on the
        segmentation level:
        - ``"entity"``: ``seg_key`` is an ``int`` (entity_idx)
        - ``"link"``: ``seg_key`` is ``(entity_idx, link_idx)``
        - ``"geom"``: ``seg_key`` is ``(entity_idx, link_idx, geom_idx)``

        Returns ``None`` if the key cannot be parsed (e.g. entity segmentation
        which is not supported).
        """
        if isinstance(seg_key, tuple):
            if len(seg_key) == 2:
                entity_idx, link_idx = int(seg_key[0]), int(seg_key[1])
                return SegmentationTarget(
                    seg_id=int(seg_idxc),
                    entity_idx=entity_idx,
                    link_idx=link_idx,
                    geom_idx=-1,
                )
            elif len(seg_key) == 3:
                entity_idx, link_idx, geom_idx = (
                    int(seg_key[0]), int(seg_key[1]), int(seg_key[2])
                )
                return SegmentationTarget(
                    seg_id=int(seg_idxc),
                    entity_idx=entity_idx,
                    link_idx=link_idx,
                    geom_idx=geom_idx,
                )
            else:
                return None
        # Entity mode (int) is not supported by this tracker.
        return None

    def build_segmentation_index(self) -> SegmentationIndex:
        """Build the segmentation index from the camera's Genesis metadata.

        This is the single place where Genesis segmentation metadata is
        interpreted.  It reads ``sensor._visualizer.segmentation_idx_dict``
        and resolves each entry to a concrete :class:`RigidEntity` and
        :class:`RigidLink`.

        Uses :attr:`CameraSensorNode.segmentation_sensor` to obtain the
        correct Genesis-backed sensor (important for NYX hybrid cameras
        where the primary sensor is a ``NyxCameraSensor``).

        Raises:
            RuntimeError: If the segmentation sensor does not expose a
                visualizer or segmentation index dictionary.
        """
        seg_to_entity_link: Dict[int, Tuple[int, int]] = {}
        entity_names: Dict[int, str] = {}
        seg_target_by_id: Dict[int, SegmentationTarget] = {}
        link_by_idx: Dict[int, RigidLink] = {}

        sensor = self._camera_node.segmentation_sensor
        if sensor is None:
            raise RuntimeError(
                f"Camera node '{self._camera_node.name}' has no Genesis sensor "
                f"for segmentation metadata.  Ensure the camera was initialised "
                f"(reload called) before the tracker."
            )

        visualizer = getattr(sensor, "_visualizer", None)
        if visualizer is None:
            raise RuntimeError(
                f"Camera node '{self._camera_node.name}': the Genesis sensor "
                f"({type(sensor).__name__}) does not expose '_visualizer'.  "
                f"The keypoint tracker requires Genesis segmentation metadata "
                f"(segmentation_idx_dict) to map pixels to entities."
            )

        seg_idx_dict = getattr(visualizer, "segmentation_idx_dict", None)
        if seg_idx_dict is None:
            raise RuntimeError(
                f"Camera node '{self._camera_node.name}': the Genesis sensor's "
                f"visualizer does not have 'segmentation_idx_dict'.  "
                f"Ensure the camera was created with segmentation rendering "
                f"enabled (render_seg=True) and that the scene contains rigid "
                f"entities."
            )

        rigid_solver = self._world.scene.rigid_solver
        rigid_entities = rigid_solver.entities
        rigid_links = rigid_solver.links
        n_entities = len(rigid_entities)
        n_links = len(rigid_links)

        for seg_idxc, seg_key in seg_idx_dict.items():
            target = self._seg_key_to_target(int(seg_idxc), seg_key)
            if target is None:
                continue

            entity_idx = target.entity_idx
            link_idx = target.link_idx

            if entity_idx < 0 or entity_idx >= n_entities:
                continue

            entity = rigid_entities[entity_idx]
            if not isinstance(entity, RigidEntity):
                continue

            if link_idx < 0 or link_idx >= n_links:
                continue

            link = rigid_links[link_idx]
            if not isinstance(link, RigidLink):
                continue

            seg_to_entity_link[target.seg_id] = (entity_idx, link_idx)
            entity_names[entity_idx] = entity.name
            link_by_idx[link_idx] = link
            seg_target_by_id[target.seg_id] = target

        seg_to_entity_idx = {
            seg_id: entity_idx
            for seg_id, (entity_idx, _link_idx) in seg_to_entity_link.items()
        }

        return SegmentationIndex(
            seg_to_entity_link=seg_to_entity_link,
            seg_to_entity_idx=seg_to_entity_idx,
            entity_names=entity_names,
            seg_target_by_id=seg_target_by_id,
            link_by_idx=link_by_idx,
        )
