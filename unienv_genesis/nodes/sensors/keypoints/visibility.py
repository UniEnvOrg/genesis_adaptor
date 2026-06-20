"""Visibility and projection logic for keypoint tracking.

This module contains pure functions for projecting tracked 3D points into
pixel space and determining visibility via a strict z-buffer depth test.
The functions operate on :class:`KeypointTrackState` and
:class:`CameraFrame` and have no Genesis-specific dependencies.
"""

from __future__ import annotations

import torch

from xbarray.transformations.pointcloud.pytorch import (
    world_to_pixel_coordinate_and_depth,
)

from .types import CameraFrame, KeypointTrackState

__all__ = [
    "update_visibility",
    "gather_rendered_depth",
]


def update_visibility(
    state: KeypointTrackState,
    frame: CameraFrame,
    depth_tolerance: float,
) -> None:
    """Project points to 2D and update visibility masks via a strict z-buffer test.

    A tracked point is visible when:

    1. Its rounded integer pixel falls inside the image.
    2. Both the projected depth and the rendered depth at that pixel are
       positive (in front of the camera, not background).
    3. The absolute difference between projected and rendered depth is within
       ``depth_tolerance`` (z-buffer consistency).

    This depth-only test is correct for all non-contact-patch cases: rigid
    surfaces do not interpenetrate, so a different object at the same pixel
    is almost always at a different depth.  The segmentation map is still
    required at initialisation (for point proposal and entity/link mapping)
    but is no longer consulted on the per-step hot path.

    Assumes all environments are rendered in order (batch index == env
    index), consistent with the port's batch validation.

    Args:
        state: Mutable track state — ``points_2d`` and ``visibility_mask``
            are updated in place.
        frame: Normalized camera observation with guaranteed batch dim.
        depth_tolerance: Maximum allowed absolute depth difference (m).
    """
    state.points_2d.zero_()
    state.visibility_mask.zero_()

    # No early-exit on torch.any(valid_mask) — that would force a GPU↔CPU
    # sync every step.  When no tracks are valid, the projection and gather
    # run on zero-points but the valid_mask in in_bounds suppresses all
    # results, so visibility_mask stays all-False (already zeroed above).

    if frame.depth is None and frame.position is None:
        raise RuntimeError(
            f"Camera frame must provide 'depth' or 'position' for "
            f"depth-based visibility checking."
        )

    H, W = frame.height, frame.width
    B = frame.batch_size

    # Project all points: (B, max_points, 3) -> (B, max_points, 2) + depth.
    pts = state.points_world
    pixel_xy, proj_depth = world_to_pixel_coordinate_and_depth(
        pts, frame.intrinsics_cv, frame.extrinsics_cv,
    )
    state.points_2d.copy_(pixel_xy)

    # Round first, then check bounds on integer pixels.
    u_px = pixel_xy[..., 0].round().long()
    v_px = pixel_xy[..., 1].round().long()

    in_bounds = (
        state.valid_mask
        & (proj_depth > 0)
        & (u_px >= 0) & (u_px < W)
        & (v_px >= 0) & (v_px < H)
    )

    # Clamp for gather (out-of-bounds slots are masked out by in_bounds).
    u_clamped = u_px.clamp(0, W - 1)
    v_clamped = v_px.clamp(0, H - 1)
    pixel_flat = v_clamped * W + u_clamped  # (B, max_points)

    # Gather rendered depth at projected pixel locations.
    rendered_depth = gather_rendered_depth(frame, pixel_flat, B, H, W)

    # Strict z-buffer test: both depths positive and close.
    visible = (
        in_bounds
        & (rendered_depth > 0)
        & ((proj_depth - rendered_depth).abs() < depth_tolerance)
    )

    state.visibility_mask.copy_(visible)


def gather_rendered_depth(
    frame: CameraFrame,
    pixel_flat: torch.Tensor,
    B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Gather rendered camera-space depth at the given flat pixel indices.

    If a depth map is available in the frame, gather directly.  Otherwise
    derive depth from the position map by gathering only the needed pixels
    (O(B*max_points) instead of O(B*H*W)) and transforming them to camera
    space.

    Args:
        frame: Normalized camera observation.
        pixel_flat: ``(B, max_points)`` flattened pixel indices (v*W + u).
        B: Batch size.
        H: Image height.
        W: Image width.

    Returns:
        ``(B, max_points)`` rendered camera-space depth tensor.
    """
    if frame.depth is not None:
        depth_flat = frame.depth.reshape(B, -1)
        return depth_flat.gather(1, pixel_flat)

    # Derive from position map -- gather first, then transform.
    position = frame.position
    pos_flat = position.reshape(B, H * W, 3)
    sampled_world = pos_flat.gather(
        1, pixel_flat[..., None].expand(-1, -1, 3),
    )  # (B, max_points, 3)

    extrinsics = frame.extrinsics_cv
    R = extrinsics[..., :3, :3]  # (B, 3, 3)
    t = extrinsics[..., :3, 3]   # (B, 3)
    sampled_cam = torch.matmul(
        R[:, None, :, :], sampled_world[..., None],
    ).squeeze(-1) + t[:, None, :]  # (B, max_points, 3)

    return sampled_cam[..., 2]  # (B, max_points)
