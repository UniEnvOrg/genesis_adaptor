"""Keypoint tracking subpackage.

Public API:
    - :class:`KeypointTrackerNode` — lifecycle sensor node
    - :class:`PointProposer`, :class:`UniformSegmentationPointProposer`,
      :class:`FocusedSegmentationPointProposer` — point proposal
    - :class:`SegmentationTarget`, :class:`SegmentationIndex` — segmentation metadata
    - :class:`KeypointTrackState` — mutable track state buffers
    - :class:`CameraFrame` — normalized camera observation
    - :class:`KeypointCameraPort`, :class:`GenesisCameraKeypointPort` — camera port
"""

from .types import (
    SegmentationTarget,
    SegmentationIndex,
    KeypointTrackState,
    CameraFrame,
)
from .proposers import (
    FocusedSegmentationPointProposer,
    PointProposer,
    UniformSegmentationPointProposer,
)
from .camera_port import (
    KeypointCameraPort,
    GenesisCameraKeypointPort,
)
from .pose_port import (
    RigidPosePort,
    GenesisRigidPosePort,
)
from .keypoint_tracker_node import KeypointTrackerNode

__all__ = [
    "SegmentationTarget",
    "SegmentationIndex",
    "KeypointTrackState",
    "CameraFrame",
    "FocusedSegmentationPointProposer",
    "PointProposer",
    "UniformSegmentationPointProposer",
    "KeypointCameraPort",
    "GenesisCameraKeypointPort",
    "RigidPosePort",
    "GenesisRigidPosePort",
    "KeypointTrackerNode",
]
