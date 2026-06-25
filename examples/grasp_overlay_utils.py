from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch


def rgb_tensor_to_bgr_uint8(rgb_tensor: torch.Tensor) -> np.ndarray:
    if rgb_tensor.dtype != torch.uint8:
        rgb_tensor = rgb_tensor.clamp(0, 255).to(torch.uint8)
    frame_rgb = rgb_tensor.detach().cpu().numpy()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def _project_world_points(points_world: np.ndarray, intrinsics_cv: np.ndarray, extrinsics_cv: np.ndarray):
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    R = extrinsics_cv[:, :3]
    t = extrinsics_cv[:, 3]
    cam = points_world @ R.T + t[None, :]
    z = cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((points_world.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        x = cam[valid, 0]
        y = cam[valid, 1]
        zv = z[valid]
        fx = float(intrinsics_cv[0, 0])
        fy = float(intrinsics_cv[1, 1])
        cx = float(intrinsics_cv[0, 2])
        cy = float(intrinsics_cv[1, 2])
        uv[valid, 0] = fx * x / zv + cx
        uv[valid, 1] = fy * y / zv + cy
    return uv, valid


def build_grasp_overlay_data(
    *,
    candidate_positions_world: np.ndarray,
    candidate_approach_world: np.ndarray,
    candidate_opening_world: np.ndarray,
    intrinsics_cv: np.ndarray,
    extrinsics_cv: np.ndarray,
    selected_idx: Optional[int],
    scores: Optional[np.ndarray] = None,
    openings: Optional[np.ndarray] = None,
    max_candidates: int = 64,
    approach_length: float = 0.05,
    default_opening_length: float = 0.03,
    finger_length: float = 0.045,
    jaw_half_extent: float = 0.006,
) -> Dict[str, Any]:
    total = int(candidate_positions_world.shape[0])
    selected_idx = None if selected_idx is None else int(selected_idx)
    selected_in_range = selected_idx is not None and 0 <= selected_idx < total

    # Build the list of *original* candidate indices to display. The caller
    # passes candidates already sorted by score (best first), so the first
    # `max_candidates` are the highest-scored ones. To guarantee the actually
    # selected grasp is always visible even when it is not among those top
    # candidates, reserve a display slot for it: show the top
    # `max_candidates - 1` plus the selected one, preserving original indices
    # for labeling.
    max_candidates = max(int(max_candidates), 1)
    if selected_in_range and selected_idx >= min(total, max_candidates):
        head = max(max_candidates - 1, 0)
        display_indices: List[int] = list(range(min(total, head)))
        display_indices.append(selected_idx)
    else:
        display_indices = list(range(min(total, max_candidates)))

    # De-duplicate while preserving order (harmless guard).
    seen: set = set()
    deduped: List[int] = []
    for idx in display_indices:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    display_indices = deduped

    n = len(display_indices)
    idx_arr = np.asarray(display_indices, dtype=np.int64)
    pos = candidate_positions_world[idx_arr]
    app = candidate_approach_world[idx_arr]
    opn = candidate_opening_world[idx_arr]
    scores = None if scores is None else scores[idx_arr]
    openings = None if openings is None else openings[idx_arr]

    # Map the original selected index back to its display slot, which the
    # drawing loop uses to highlight / label the chosen grasp.
    selected_display: Optional[int] = None
    if selected_in_range:
        try:
            selected_display = display_indices.index(selected_idx)
        except ValueError:
            selected_display = None

    if openings is None:
        opening_len = np.full((n,), default_opening_length, dtype=np.float32)
    else:
        opening_len = np.clip(openings.astype(np.float32), 0.015, 0.08)

    # Geometry of a parallel-jaw gripper projected to the image:
    #   - pos is the grasp center, located between the two jaw tips at the
    #     contact/object plane.
    #   - opn is the jaw separation axis (left/right of center).
    #   - app points from the gripper body toward the object (toward the
    #     fingertips), so the palm sits behind the tips along -app.
    tip_left = pos - 0.5 * opening_len[:, None] * opn
    tip_right = pos + 0.5 * opening_len[:, None] * opn
    palm_left = tip_left - finger_length * app
    palm_right = tip_right - finger_length * app
    # Jaw caps: short segments perpendicular to the fingers (along opn) at each
    # fingertip, giving the classic "hammer-head" gripper silhouette.
    jaw_left_a = tip_left - jaw_half_extent * opn
    jaw_left_b = tip_left + jaw_half_extent * opn
    jaw_right_a = tip_right - jaw_half_extent * opn
    jaw_right_b = tip_right + jaw_half_extent * opn
    approach_tip = pos + approach_length * app

    # Project all per-candidate points in one batch for efficiency.
    all_pts = np.concatenate(
        [
            pos,
            tip_left,
            tip_right,
            palm_left,
            palm_right,
            jaw_left_a,
            jaw_left_b,
            jaw_right_a,
            jaw_right_b,
            approach_tip,
        ],
        axis=0,
    )
    uv_all, valid_all = _project_world_points(all_pts, intrinsics_cv, extrinsics_cv)

    def _split(arr: np.ndarray, idx: int):
        return arr[idx * n : (idx + 1) * n]

    uv_center, valid_center = _split(uv_all, 0), _split(valid_all, 0)
    uv_left, valid_left = _split(uv_all, 1), _split(valid_all, 1)
    uv_right, valid_right = _split(uv_all, 2), _split(valid_all, 2)
    uv_palm_left, valid_palm_left = _split(uv_all, 3), _split(valid_all, 3)
    uv_palm_right, valid_palm_right = _split(uv_all, 4), _split(valid_all, 4)
    uv_jaw_la, valid_jaw_la = _split(uv_all, 5), _split(valid_all, 5)
    uv_jaw_lb, valid_jaw_lb = _split(uv_all, 6), _split(valid_all, 6)
    uv_jaw_ra, valid_jaw_ra = _split(uv_all, 7), _split(valid_all, 7)
    uv_jaw_rb, valid_jaw_rb = _split(uv_all, 8), _split(valid_all, 8)
    uv_tip, valid_tip = _split(uv_all, 9), _split(valid_all, 9)

    return {
        "count": n,
        "selected_idx": selected_display,
        "selected_original_idx": selected_idx if selected_in_range else None,
        "original_indices": idx_arr,
        "center_uv": uv_center,
        "left_uv": uv_left,
        "right_uv": uv_right,
        "palm_left_uv": uv_palm_left,
        "palm_right_uv": uv_palm_right,
        "jaw_left_uv": np.stack([uv_jaw_la, uv_jaw_lb], axis=1),
        "jaw_right_uv": np.stack([uv_jaw_ra, uv_jaw_rb], axis=1),
        "tip_uv": uv_tip,
        "valid_center": valid_center,
        "valid_left": valid_left,
        "valid_right": valid_right,
        "valid_palm_left": valid_palm_left,
        "valid_palm_right": valid_palm_right,
        "valid_jaw_left": np.stack([valid_jaw_la, valid_jaw_lb], axis=1),
        "valid_jaw_right": np.stack([valid_jaw_ra, valid_jaw_rb], axis=1),
        "valid_tip": valid_tip,
        "scores": scores,
        "openings": openings,
    }


def draw_grasp_overlay(frame_bgr: np.ndarray, overlay: Dict[str, Any], step: Optional[int] = None) -> np.ndarray:
    count = int(overlay.get("count", 0))
    selected_idx = overlay.get("selected_idx")
    center_uv = overlay["center_uv"]
    palm_left_uv = overlay["palm_left_uv"]
    palm_right_uv = overlay["palm_right_uv"]
    left_uv = overlay["left_uv"]
    right_uv = overlay["right_uv"]
    jaw_left_uv = overlay["jaw_left_uv"]
    jaw_right_uv = overlay["jaw_right_uv"]
    tip_uv = overlay["tip_uv"]
    valid_center = overlay["valid_center"]
    valid_palm_left = overlay["valid_palm_left"]
    valid_palm_right = overlay["valid_palm_right"]
    valid_left = overlay["valid_left"]
    valid_right = overlay["valid_right"]
    valid_jaw_left = overlay["valid_jaw_left"]
    valid_jaw_right = overlay["valid_jaw_right"]
    valid_tip = overlay["valid_tip"]
    scores = overlay.get("scores")
    original_indices = overlay.get("original_indices")
    selected_original_idx = overlay.get("selected_original_idx")

    def _pt(uv: np.ndarray) -> tuple:
        return tuple(np.round(uv).astype(int).tolist())

    for idx in range(count):
        if not valid_center[idx]:
            continue
        is_selected = selected_idx == idx
        line_thickness = 2 if is_selected else 1
        jaw_thickness = line_thickness + 1
        # Body (palm + fingers) in a greenish hue, jaw caps in orange, approach
        # cue as a faint thin line so it does not compete with the gripper body.
        body_color = (0, 255, 255) if is_selected else (0, 200, 120)
        jaw_color = (255, 180, 0) if is_selected else (255, 120, 0)
        approach_color = (0, 80, 255) if is_selected else (0, 0, 180)

        c = _pt(center_uv[idx])

        # Back bridge / palm connecting the two finger bases.
        if valid_palm_left[idx] and valid_palm_right[idx]:
            cv2.line(
                frame_bgr,
                _pt(palm_left_uv[idx]),
                _pt(palm_right_uv[idx]),
                body_color,
                line_thickness,
                cv2.LINE_AA,
            )

        # Two fingers from the palm to the jaw tips.
        if valid_palm_left[idx] and valid_left[idx]:
            cv2.line(
                frame_bgr,
                _pt(palm_left_uv[idx]),
                _pt(left_uv[idx]),
                body_color,
                line_thickness,
                cv2.LINE_AA,
            )
        if valid_palm_right[idx] and valid_right[idx]:
            cv2.line(
                frame_bgr,
                _pt(palm_right_uv[idx]),
                _pt(right_uv[idx]),
                body_color,
                line_thickness,
                cv2.LINE_AA,
            )

        # Jaw caps (perpendicular to fingers) to mark the gripping faces.
        jl_valid = valid_jaw_left[idx]
        if jl_valid[0] and jl_valid[1]:
            cv2.line(
                frame_bgr,
                _pt(jaw_left_uv[idx, 0]),
                _pt(jaw_left_uv[idx, 1]),
                jaw_color,
                jaw_thickness,
                cv2.LINE_AA,
            )
        jr_valid = valid_jaw_right[idx]
        if jr_valid[0] and jr_valid[1]:
            cv2.line(
                frame_bgr,
                _pt(jaw_right_uv[idx, 0]),
                _pt(jaw_right_uv[idx, 1]),
                jaw_color,
                jaw_thickness,
                cv2.LINE_AA,
            )

        # Subtle approach cue: short line forward from the grasp center.
        if valid_tip[idx]:
            cv2.line(frame_bgr, c, _pt(tip_uv[idx]), approach_color, 1, cv2.LINE_AA)

        # Highlight marker only for the selected candidate, to avoid clutter.
        if is_selected:
            cv2.circle(frame_bgr, c, 3, body_color, -1, cv2.LINE_AA)

    y = 20
    if step is not None:
        cv2.putText(frame_bgr, f"Step {step}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    cv2.putText(frame_bgr, f"Candidates: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += 20
    if selected_idx is not None:
        # Report the original candidate index (the rank in the full sorted
        # candidate list) rather than the reindexed display slot, so the label
        # stays meaningful when the selected grasp was appended past the top
        # `max_candidates` block.
        label_idx = selected_original_idx
        if label_idx is None and original_indices is not None and 0 <= selected_idx < len(original_indices):
            label_idx = int(original_indices[selected_idx])
        if label_idx is None:
            label_idx = selected_idx
        label = f"Selected: #{label_idx}"
        if scores is not None and 0 <= selected_idx < len(scores):
            label += f" score={float(scores[selected_idx]):.3f}"
        cv2.putText(frame_bgr, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


def build_robot_axis_overlay_data(
    *,
    eef_pos_world: np.ndarray,
    approach_world: np.ndarray,
    open_world: np.ndarray,
    intrinsics_cv: np.ndarray,
    extrinsics_cv: np.ndarray,
    approach_length: float = 0.06,
    open_half_length: float = 0.04,
) -> Dict[str, Any]:
    """Project the robot gripper's own EEF-frame axes into a camera image.

    This is a debug helper that draws, on top of the grasp-candidate overlay,
    the robot's current approach axis and opening/closing axis as inferred
    from the arm/gripper controller convention. It complements the grasp-
    candidate overlay so a viewer can sanity-check whether the gripper
    convention agrees with the annotated candidates.

    Args:
        eef_pos_world: (3,) robot EEF origin in world coordinates.
        approach_world: (3,) unit approach direction in world coordinates.
        open_world: (3,) unit opening/closing direction in world coordinates.
        intrinsics_cv: (3, 3) OpenCV-style camera intrinsics.
        extrinsics_cv: (3, 4) or (4, 4) world-to-camera extrinsics.
        approach_length: length (m) of the approach arrow.
        open_half_length: half-length (m) of the opening/closing span.

    Returns:
        Dict of projected pixel endpoints and per-point validity flags,
        consumed by :func:`draw_robot_axis_overlay`.
    """
    eef = np.asarray(eef_pos_world, dtype=np.float32).reshape(3)
    app = np.asarray(approach_world, dtype=np.float32).reshape(3)
    opn = np.asarray(open_world, dtype=np.float32).reshape(3)
    # Guard against non-unit / zero inputs so projection never blows up.
    app_n = np.linalg.norm(app)
    opn_n = np.linalg.norm(opn)
    app = app / app_n if app_n > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    opn = opn / opn_n if opn_n > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    approach_tip = eef + approach_length * app
    open_a = eef - open_half_length * opn
    open_b = eef + open_half_length * opn
    pts = np.stack([eef, approach_tip, open_a, open_b], axis=0)
    uv, valid = _project_world_points(pts, intrinsics_cv, extrinsics_cv)
    return {
        "eef_uv": uv[0],
        "eef_valid": bool(valid[0]),
        "approach_tip_uv": uv[1],
        "approach_valid": bool(valid[1]),
        "open_a_uv": uv[2],
        "open_a_valid": bool(valid[2]),
        "open_b_uv": uv[3],
        "open_b_valid": bool(valid[3]),
    }


def draw_robot_axis_overlay(frame_bgr: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
    """Draw a robot gripper's approach/opening axes onto an image.

    Colors (BGR) and labels, chosen to be distinct from the grasp-candidate
    overlay (which uses green/cyan/orange/red-ish hues):
      - approach axis: bright red, labeled "R:approach"
      - opening/closing span: bright blue, labeled "R:open"

    The approach is drawn as a line from the EEF origin to its tip; the
    opening/closing direction is drawn as a span through the EEF origin.
    """
    def _pt(uv: np.ndarray) -> tuple:
        return (int(round(float(uv[0]))), int(round(float(uv[1]))))

    approach_color = (0, 0, 255)  # red
    open_color = (255, 0, 0)      # blue
    label_color = (255, 255, 255)

    eef_uv = data["eef_uv"]
    eef_valid = data["eef_valid"]
    approach_tip_uv = data["approach_tip_uv"]
    approach_valid = data["approach_valid"]
    open_a_uv = data["open_a_uv"]
    open_a_valid = data["open_a_valid"]
    open_b_uv = data["open_b_uv"]
    open_b_valid = data["open_b_valid"]

    # Opening/closing span: a line through the EEF origin, perpendicular to
    # the approach in the gripper's own frame.
    if open_a_valid and open_b_valid:
        cv2.line(frame_bgr, _pt(open_a_uv), _pt(open_b_uv), open_color, 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, _pt(open_a_uv), 3, open_color, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, _pt(open_b_uv), 3, open_color, -1, cv2.LINE_AA)

    # Approach arrow: EEF origin -> fingertip direction.
    if eef_valid and approach_valid:
        eef_pt = _pt(eef_uv)
        tip_pt = _pt(approach_tip_uv)
        cv2.line(frame_bgr, eef_pt, tip_pt, approach_color, 2, cv2.LINE_AA)
        cv2.arrowedLine(frame_bgr, eef_pt, tip_pt, approach_color, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(frame_bgr, eef_pt, 3, approach_color, -1, cv2.LINE_AA)

    # Labels near the projected EEF origin for readability.
    if eef_valid:
        lx, ly = _pt(eef_uv)
        cv2.putText(frame_bgr, "R:approach", (lx + 6, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "R:open", (lx + 6, ly + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)

    return frame_bgr


def make_video_writer(path: str | Path, frame_bgr: np.ndarray, fps: float = 15.0):
    path = str(path)
    h, w = frame_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, float(fps), (w, h))
