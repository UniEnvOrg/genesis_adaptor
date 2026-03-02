"""Tianji clutter-table demo with GT 3D keypoint flow overlay video.

Builds the Tianji Marvin clutter-table task, adds ``GTKeypointFlowSensorNode``,
and writes an MP4 where tracked keypoints are overlaid as flow arrows on the
front camera image.

Usage:
    python examples/cluttered_table_tianji_gt_flow.py
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch

import genesis as gs

from cluttered_table_marvin import (
    TABLE_POS,
    TABLE_SIZE,
    TABLE_YAW,
    ROBOT_Z,
    CubeRelocateTask,
)

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors import (
    CameraSensorNode,
    CameraSensorNodeOptions,
    GTKeypointFlowSensorNode,
    UniformSegmentationPointProposer,
)
from unienv_genesis_collection.objects import BoxObject, SphereObject, CylinderObject
from unienv_genesis_collection.robots.tianji_marvin_robotiq_2f85 import (
    TianjiMarvinWithRobotiq2F85,
)
from unienv_genesis_collection.scenes.cluttered_table_manager import ClutteredTableManager
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_interface.world import WorldEnv


SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos_cluttered_tianji_gt_flow")


def match_points_to_pixels(
    points_world: torch.Tensor,
    seg_idxc: torch.Tensor,
    position_map: torch.Tensor | np.ndarray,
    segmentation: torch.Tensor | np.ndarray,
    *,
    prev_uv: torch.Tensor | None = None,
    prev_uv_valid: torch.Tensor | None = None,
    max_world_error: float = 0.03,
    temporal_px_weight: float = 2e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map tracked world points to image pixels using seg-constrained NN in position map.

    Returns:
        uv: ``(N, 2)`` float tensor in ``(u, v)`` order.
        uv_valid: ``(N,)`` bool mask for reliable matches.
        world_error: ``(N,)`` nearest-neighbor world distance in meters.
    """
    if points_world.numel() == 0:
        return (
            torch.empty((0, 2), dtype=torch.float32, device=points_world.device),
            torch.empty((0,), dtype=torch.bool, device=points_world.device),
            torch.empty((0,), dtype=torch.float32, device=points_world.device),
        )

    pts = torch.as_tensor(points_world, dtype=torch.float32, device=points_world.device)
    seg_ids = torch.as_tensor(seg_idxc, dtype=torch.long, device=pts.device)
    pos = torch.as_tensor(position_map, dtype=torch.float32, device=pts.device)
    seg = torch.as_tensor(segmentation, dtype=torch.long, device=pts.device)
    if pos.ndim == 4:
        pos = pos[0]
    if seg.ndim == 3:
        seg = seg[0]
    if pos.ndim != 3 or pos.shape[-1] != 3 or seg.ndim != 2:
        raise ValueError(
            f"Expected position (H, W, 3) and seg (H, W), got {tuple(pos.shape)} and {tuple(seg.shape)}"
        )

    N = pts.shape[0]
    uv = torch.full((N, 2), -1.0, dtype=torch.float32, device=pts.device)
    uv_valid = torch.zeros((N,), dtype=torch.bool, device=pts.device)
    world_error = torch.full((N,), float("inf"), dtype=torch.float32, device=pts.device)

    pos_valid = torch.isfinite(pos).all(dim=-1)
    pt_valid = torch.isfinite(pts).all(dim=-1) & (seg_ids >= 0)

    prev_uv_t = None if prev_uv is None else torch.as_tensor(prev_uv, dtype=torch.float32, device=pts.device)
    if prev_uv_valid is None:
        prev_uv_valid_t = (
            torch.isfinite(prev_uv_t).all(dim=-1)
            if prev_uv_t is not None
            else torch.zeros((N,), dtype=torch.bool, device=pts.device)
        )
    else:
        prev_uv_valid_t = torch.as_tensor(prev_uv_valid, dtype=torch.bool, device=pts.device)

    unique_seg = torch.unique(seg_ids[pt_valid])
    for seg_id in unique_seg.tolist():
        point_ids = torch.where(pt_valid & (seg_ids == seg_id))[0]
        if point_ids.numel() == 0:
            continue

        pix_mask = pos_valid & (seg == int(seg_id))
        if not torch.any(pix_mask):
            continue

        yx = torch.nonzero(pix_mask, as_tuple=False)  # (K, 2)
        cand_pos = pos[yx[:, 0], yx[:, 1]]  # (K, 3)
        cand_uv = torch.stack([yx[:, 1].to(torch.float32), yx[:, 0].to(torch.float32)], dim=-1)  # (K, 2)

        dist_world = torch.cdist(pts[point_ids], cand_pos)  # (n, K)
        cost = dist_world

        if prev_uv_t is not None:
            temporal_ids = torch.where(prev_uv_valid_t[point_ids])[0]
            if temporal_ids.numel() > 0:
                prev_uv_sub = prev_uv_t[point_ids[temporal_ids]]
                dist_px = torch.cdist(prev_uv_sub, cand_uv)
                cost = cost.clone()
                cost[temporal_ids] = cost[temporal_ids] + temporal_px_weight * dist_px

        nn = torch.argmin(cost, dim=-1)
        nn_world = dist_world[torch.arange(point_ids.shape[0], device=pts.device), nn]
        nn_uv = cand_uv[nn]

        uv[point_ids] = nn_uv
        world_error[point_ids] = nn_world
        uv_valid[point_ids] = torch.isfinite(nn_world) & (nn_world <= max_world_error)

    return uv, uv_valid, world_error


def project_points_to_pixels(
    points_world: torch.Tensor | np.ndarray,
    camera_node: CameraSensorNode,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world-space points to image pixels using camera intrinsics/extrinsics.

    Returns:
        uv: ``(N, 2)`` float tensor in ``(u, v)`` order.
        uv_valid: ``(N,)`` bool mask for points in front of camera and in frame.
    """
    pts = torch.as_tensor(points_world, dtype=torch.float32)
    if pts.ndim == 3:
        pts = pts[0]
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {tuple(pts.shape)}")

    N = pts.shape[0]
    uv = torch.full((N, 2), -1.0, dtype=torch.float32, device=pts.device)
    uv_valid = torch.zeros((N,), dtype=torch.bool, device=pts.device)
    if N == 0:
        return uv, uv_valid

    sensor = camera_node.sensor
    fx = float(sensor.f)
    cx = float(sensor.cx)
    cy = float(sensor.cy)
    W, H = camera_node.option.res

    camera_transform = torch.as_tensor(
        sensor.get_transform(),
        dtype=torch.float32,
        device=pts.device,
    )
    if camera_transform.ndim == 3:
        camera_transform = camera_transform[0]

    T_OPENGL_TO_OPENCV = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=pts.device,
    )
    cam_pose = camera_transform @ T_OPENGL_TO_OPENCV  # camera->world in OpenCV convention
    world_to_cam = torch.linalg.inv(cam_pose)

    pts_h = torch.cat([pts, torch.ones((N, 1), dtype=torch.float32, device=pts.device)], dim=-1)
    cam = pts_h @ world_to_cam.transpose(-1, -2)
    X = cam[:, 0]
    Y = cam[:, 1]
    Z = cam[:, 2]

    finite = torch.isfinite(cam).all(dim=-1)
    in_front = Z > 1e-6
    safe_Z = torch.where(in_front, Z, torch.ones_like(Z))

    u = fx * (X / safe_Z) + cx - 0.5
    v = fx * (Y / safe_Z) + cy - 0.5
    uv[:, 0] = u
    uv[:, 1] = v

    in_frame = (u >= 0.0) & (u < float(W)) & (v >= 0.0) & (v < float(H))
    uv_valid = finite & in_front & in_frame
    return uv, uv_valid


def draw_flow_overlay(
    rgb: torch.Tensor | np.ndarray,
    flow_obs: dict,
    uv_curr: torch.Tensor,
    uv_curr_valid: torch.Tensor,
    uv_prev: torch.Tensor,
    uv_prev_valid: torch.Tensor,
    world_error_curr: torch.Tensor | None = None,
    *,
    flow_world_thresh: float = 1e-4,
    max_draw: int = 96,
    arrow_scale: float = 4.0,
    draw_points: bool = True,
    max_points_draw: int | None = None,
    points_only: bool = True,
) -> np.ndarray:
    """Overlay GT tracked point visualization on a RGB frame."""
    frame_rgb = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    H, W = frame_bgr.shape[:2]

    points = torch.as_tensor(flow_obs["points_world"], dtype=torch.float32)
    flow = torch.as_tensor(flow_obs["flow_world"], dtype=torch.float32)
    valid = torch.as_tensor(flow_obs["valid_mask"], dtype=torch.bool)
    entity_idx = torch.as_tensor(flow_obs["entity_idx"], dtype=torch.int64)
    uv_curr = torch.as_tensor(uv_curr, dtype=torch.float32)
    uv_curr_valid = torch.as_tensor(uv_curr_valid, dtype=torch.bool)
    uv_prev = torch.as_tensor(uv_prev, dtype=torch.float32)
    uv_prev_valid = torch.as_tensor(uv_prev_valid, dtype=torch.bool)
    world_error_curr = (
        torch.as_tensor(world_error_curr, dtype=torch.float32)
        if world_error_curr is not None
        else torch.full((uv_curr.shape[0],), float("nan"), dtype=torch.float32)
    )

    if points.ndim == 3:  # batched fallback
        points = points[0]
        flow = flow[0]
        valid = valid[0]
        entity_idx = entity_idx[0]
    if uv_curr.ndim == 3:
        uv_curr = uv_curr[0]
    if uv_curr_valid.ndim == 2:
        uv_curr_valid = uv_curr_valid[0]
    if uv_prev.ndim == 3:
        uv_prev = uv_prev[0]
    if uv_prev_valid.ndim == 2:
        uv_prev_valid = uv_prev_valid[0]
    if world_error_curr.ndim == 2:
        world_error_curr = world_error_curr[0]

    if valid.numel() == 0 or not torch.any(valid):
        return frame_rgb

    if draw_points:
        visible = valid & uv_curr_valid
        if torch.any(visible):
            vis_idx = torch.where(visible)[0]
            if max_points_draw is not None and vis_idx.numel() > max_points_draw:
                vis_idx = vis_idx[:max_points_draw]

            palette = [
                (0, 255, 255),    # yellow
                (255, 80, 80),    # blue-ish in BGR
                (80, 255, 80),    # green
                (255, 160, 30),   # orange
                (220, 80, 255),   # purple
                (80, 200, 255),   # cyan
                (255, 255, 255),  # white
            ]
            for i in vis_idx.tolist():
                u, v = uv_curr[i].tolist()
                if not np.isfinite(u + v):
                    continue
                if not (0 <= u < W and 0 <= v < H):
                    continue
                e = int(entity_idx[i].item()) if i < entity_idx.shape[0] else -1
                color = palette[e % len(palette)] if e >= 0 else (180, 180, 180)
                cv2.circle(
                    frame_bgr,
                    (int(round(u)), int(round(v))),
                    2,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

            if points_only:
                cv2.putText(
                    frame_bgr,
                    f"tracked points visible: {int(visible.sum().item())}/{int(valid.sum().item())}",
                    (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (20, 220, 20),
                    2,
                    cv2.LINE_AA,
                )
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    speed = torch.linalg.norm(flow, dim=-1)
    moving = valid & torch.isfinite(speed) & (speed > flow_world_thresh) & uv_curr_valid & uv_prev_valid
    if not torch.any(moving):
        cv2.putText(
            frame_bgr, "GT flow: no visible moving tracked points", (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 2, cv2.LINE_AA,
        )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    idx = torch.where(moving)[0]
    speed_m = speed[idx]
    rank = torch.argsort(speed_m, descending=True)
    idx = idx[rank[:max_draw]]

    uv_curr_d = uv_curr[idx]
    uv_prev_d = uv_prev[idx]
    pix_speed = torch.linalg.norm(uv_curr_d - uv_prev_d, dim=-1)
    err_d = world_error_curr[idx]

    drawn = 0
    for i in range(idx.shape[0]):
        u0, v0 = uv_prev_d[i].tolist()
        u1, v1 = uv_curr_d[i].tolist()
        if not np.isfinite(u0 + v0 + u1 + v1):
            continue
        if not (0 <= u0 < W and 0 <= v0 < H and 0 <= u1 < W and 0 <= v1 < H):
            continue

        mag = float(pix_speed[i].item())
        color = (30, int(np.clip(80 + mag * 12, 80, 255)), int(np.clip(255 - mag * 6, 60, 255)))
        p0 = (int(round(u0)), int(round(v0)))
        du = (u1 - u0) * float(arrow_scale)
        dv = (v1 - v0) * float(arrow_scale)
        p1 = (
            int(round(np.clip(u0 + du, 0, W - 1))),
            int(round(np.clip(v0 + dv, 0, H - 1))),
        )
        cv2.arrowedLine(frame_bgr, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(frame_bgr, p1, 2, (255, 255, 255), -1, cv2.LINE_AA)
        drawn += 1

    err_text = ""
    finite_err = err_d[torch.isfinite(err_d)]
    if finite_err.numel() > 0:
        err_text = f"  mean match err={float(finite_err.mean().item())*1000:.1f}mm"

    cv2.putText(
        frame_bgr, f"GT tracked flow arrows: {drawn} (x{arrow_scale:.1f}){err_text}", (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 220, 20), 2, cv2.LINE_AA,
    )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_env() -> Tuple[WorldEnv, CameraSensorNode]:
    world = GenesisWorld(
        scene_init_kwargs={
            "sim_options": gs.options.SimOptions(substeps=5),
            "rigid_options": gs.options.RigidOptions(box_box_detection=True),
            "vis_options": gs.options.VisOptions(segmentation_level="link"),
        },
    )

    plane = PlaneBuilder(world, name="floor", height=0.0)
    table = TableBuilder(world, name="table", pos=TABLE_POS, yaw=TABLE_YAW)
    tianji = TianjiMarvinWithRobotiq2F85(
        world,
        name="tianji",
        pos=(-0.5, 0, ROBOT_Z),
        quat=(1, 0, 0, 0),
    )

    distractor_pool = [
        BoxObject(size=(0.05, 0.03, 0.06), name="green_box", color=(0, 0.8, 0.2, 1)),
        SphereObject(radius=0.025, name="blue_ball", color=(0, 0.3, 1, 1)),
        SphereObject(radius=0.02, name="yellow_ball", color=(1, 1, 0, 1)),
        CylinderObject(radius=0.02, height=0.08, name="orange_can", color=(1, 0.5, 0, 1)),
        CylinderObject(radius=0.015, height=0.06, name="purple_peg", color=(0.6, 0, 0.8, 1)),
    ]
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=distractor_pool,
        n_objects_range=(3, 5),
        table_size=TABLE_SIZE,
        table_pos=TABLE_POS,
        table_yaw=TABLE_YAW,
        random_quaternion=True,
    )

    target_cube = BoxObject(size=(0.04, 0.04, 0.04), name="target_cube", color=(1, 0, 0, 1))
    task = CubeRelocateTask(
        world,
        name="task",
        clutter_manager=clutter,
        target_object=target_cube,
        table_size=TABLE_SIZE,
        table_pos=TABLE_POS,
        table_yaw=TABLE_YAW,
        success_threshold=0.03,
    )

    cam_pos = (1.0, 0.0, 1.4)
    cam_lookat = (0.0, 0.0, 0.6)
    camera = CameraSensorNode(
        world,
        name="camera",
        option=CameraSensorNodeOptions(
            pos=cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
            res=(640, 480),
            fov=60,
        ),
        render_rgb=True,
        render_seg=True,
        render_position=True,
    )

    tracked_names = [
        "tianji",
        "task_target",
    ]

    gt_flow = GTKeypointFlowSensorNode(
        world,
        name="gt_flow",
        camera_node=camera,
        max_points=196,
        tracked_entity_names=tracked_names,
        tracked_entity_name_prefixes=["clutter_"],
        allow_articulated=True,
        point_proposer=UniformSegmentationPointProposer(
            max_points_per_entity=24,
            randomize=True,
        ),
    )

    env = WorldEnv(
        world,
        [plane, table, tianji, task, clutter, camera, gt_flow],
        render_mode="auto",
    )
    return env, camera


def make_policy_helpers(tianji, target_object, target_entity):
    from unienv_genesis_refpolicy.implementations.grasp_policy import BoxGripperGraspPolicy
    from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy
    from unienv_genesis_refpolicy.implementations.repeat_action_policy import RepeatActionRefPolicy
    from unienv_genesis_refpolicy.implementations.combined_policy import CombinedRefPolicy
    from unienv_genesis_refpolicy.implementations.graph_policy import GraphRefPolicy

    def top_down_filter(candidates):
        return candidates.approach_direction[..., 2] < 0

    OPEN = torch.tensor([-1.0])
    CLOSE = torch.tensor([1.0])

    left_eef = tianji.controller["left_arm"]
    left_grip = tianji.controller["left_gripper"]
    right_eef = tianji.controller["right_arm"]
    right_home_pos, right_home_quat = right_eef.get_eef_pose_absolute()
    right_hold = right_eef.get_action_world(right_home_pos.detach(), right_home_quat.detach())

    grasp_policy = BoxGripperGraspPolicy(
        world=tianji.world,
        eef_controller=left_eef,
        gripper_controller=left_grip,
        target_object=target_object,
        target_entity=target_entity,
        eef_name="left_arm",
        gripper_name="left_gripper",
        filter_candidate_fn=top_down_filter,
    )

    def make_lift_policy():
        eef_pos, eef_quat = left_eef.get_eef_pose_absolute()
        lift_pos = eef_pos.detach().clone()
        lift_pos[..., 2] += 0.15
        return CombinedRefPolicy(
            tianji.world,
            {
                "left_arm": GoToRefPolicy(tianji.world, left_eef, lift_pos, eef_quat.detach().clone()),
                "left_gripper": RepeatActionRefPolicy(tianji.world, CLOSE, steps=0),
            },
            success_keys=["left_arm"],
        )

    graph = GraphRefPolicy(
        world=tianji.world,
        nodes={"grasp": grasp_policy, "lift": make_lift_policy},
        success_edges=[("grasp", "lift")],
        fail_edges=[],
        initial="grasp",
        static_actions={"right_arm": right_hold, "right_gripper": OPEN},
    )
    graph.reset()
    return graph


def run_episode(
    env: WorldEnv,
    camera: CameraSensorNode,
    *,
    episode_seed: int,
    n_steps: int,
    flow_world_thresh: float,
    max_draw: int,
) -> List[np.ndarray]:
    _, obs, _ = env.reset(seed=episode_seed, reload=True)
    node_map = {n.name: n for n in env.node.nodes}
    tianji = node_map["tianji"]
    task = node_map["task"]

    graph = make_policy_helpers(tianji, task.target_object, task._target_entity)

    frames: List[np.ndarray] = []
    uv_prev, uv_prev_valid = project_points_to_pixels(
        torch.as_tensor(obs["gt_flow"]["points_world"], dtype=torch.float32),
        camera,
    )
    first = draw_flow_overlay(
        obs["camera"]["rgb"],
        obs["gt_flow"],
        uv_prev,
        uv_prev_valid,
        uv_prev,
        uv_prev_valid,
        flow_world_thresh=flow_world_thresh,
        max_draw=max_draw,
    )
    frames.append(first)

    for step in range(n_steps):
        action = graph.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        graph.step()

        uv_curr, uv_curr_valid = project_points_to_pixels(
            torch.as_tensor(obs["gt_flow"]["points_world"], dtype=torch.float32),
            camera,
        )

        overlay = draw_flow_overlay(
            obs["camera"]["rgb"],
            obs["gt_flow"],
            uv_curr,
            uv_curr_valid,
            uv_prev,
            uv_prev_valid,
            flow_world_thresh=flow_world_thresh,
            max_draw=max_draw,
        )
        uv_prev = uv_curr
        uv_prev_valid = uv_curr_valid
        cv2.putText(
            overlay,
            f"step={step:03d} reward={float(reward):+.3f}",
            (14, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        frames.append(overlay)

        if graph.get_success():
            print(f"Episode {episode_seed}: grasp+lift succeeded at step {step}.")
            break
        if graph.get_failure():
            print(f"Episode {episode_seed}: grasp failed at step {step}.")
            break
        if bool(terminated) or bool(truncated):
            print(f"Episode {episode_seed}: terminated={terminated}, truncated={truncated} at step {step}.")
            break

    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to render.")
    parser.add_argument("--steps", type=int, default=260, help="Max environment steps per episode.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_VIDEO_DIR, help="Directory for output mp4 files.")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--flow-world-thresh",
        type=float,
        default=1e-4,
        help="Only draw arrows for points with |flow_world| above this threshold (meters/step).",
    )
    parser.add_argument("--max-draw", type=int, default=96, help="Max arrows drawn per frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    env, camera = build_env()
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            frames = run_episode(
                env,
                camera,
                episode_seed=seed,
                n_steps=args.steps,
                flow_world_thresh=args.flow_world_thresh,
                max_draw=args.max_draw,
            )
            out_path = os.path.join(args.output_dir, f"episode_{ep:02d}_seed_{seed}_gt_flow.mp4")
            gs.tools.animate(frames, out_path, fps=args.fps)
            print(f"Saved GT flow video: {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
