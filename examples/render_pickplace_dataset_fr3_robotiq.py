"""Batch-render FR3 + Robotiq pick-place videos or stills for dataset objects.

This is a dataset-aware variant of ``pickplace_fr3_robotiq.py`` that renders
exactly one pick-place rollout per target object, or a still image of the
object posed on the table for scale/orientation inspection.

Typical usage::

    python examples/render_pickplace_dataset_fr3_robotiq.py \
        --dataset gso --limit 10

    python examples/render_pickplace_dataset_fr3_robotiq.py \
        --dataset gso --artifact still --selection nongraspable

By default, objects are filtered to the ``box_gripper_graspable`` subset so the
reference grasp policy has a reasonable chance of succeeding.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.gpu)

from xbarray.transformations.rotation_conversions.pytorch import quaternion_apply

from unienv_interface.world import WorldEnv
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper
from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import (
    CameraSensorNode,
    CameraSensorNodeOptions,
)
from unienv_genesis_collection.objects import (
    BoxObject,
    CylinderObject,
    GSOObject,
    SphereObject,
    YCBObject,
    list_gso_objects,
    list_ycb_objects,
)
from unienv_genesis_collection.robots.franka_fr3_robotiq_2f85 import (
    FrankaFR3WithRobotiq2F85,
    ZedSupportedResolution,
)
from unienv_genesis_collection.scenes.cluttered_table_manager import (
    ClutteredTableManager,
)
from unienv_genesis_collection.scenes.env_map_manager import EnvMapManager
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_genesis_collection.tasks.tabletop.pickplace import ObjectRelocateTask
from unienv_genesis_refpolicy.implementations.grasp_policy import BoxGripperGraspPolicy
from unienv_genesis_refpolicy.implementations.pickplace_policy import PickPlaceRefPolicy
from unienv_genesis_refpolicy.grasp_providers.contact_graspnet import (
    ContactGraspNetGraspCandidateProvider,
)
from grasp_overlay_utils import (
    build_grasp_overlay_data,
    build_robot_axis_overlay_data,
    draw_grasp_overlay,
    draw_robot_axis_overlay,
    rgb_tensor_to_bgr_uint8,
)

DEFAULT_CONTACT_GRASPNET_CHECKPOINT = str(
    ContactGraspNetGraspCandidateProvider.DEFAULT_CHECKPOINT_DIR
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "videos_pickplace_fr3_robotiq_datasets"
)

TABLE_SURFACE_Z = 0.6
TABLE_SIZE = (1.0, 0.6)
TABLE_POS = (0.0, 0.0, TABLE_SURFACE_Z)
TABLE_YAW = math.pi / 2
ROBOT_POS = (-0.45, 0.0, TABLE_SURFACE_Z)
TOTAL_STEPS = 500


def _build_target_object(dataset: str, object_id: str):
    if dataset == "gso":
        return GSOObject(object_id, name=f"gso_{object_id}")
    if dataset == "ycb":
        return YCBObject(object_id, name=f"ycb_{object_id}")
    raise ValueError(f"Unsupported dataset: {dataset}")


def _build_distractor_pool():
    return [
        BoxObject(size=(0.05, 0.03, 0.06), name="green_box", color=(0, 0.8, 0.2, 1)),
        SphereObject(radius=0.025, name="blue_ball", color=(0, 0.3, 1, 1)),
        SphereObject(radius=0.02, name="yellow_ball", color=(1, 1, 0, 1)),
        CylinderObject(
            radius=0.02, height=0.08, name="orange_can", color=(1, 0.5, 0, 1),
        ),
        CylinderObject(
            radius=0.015, height=0.06, name="purple_peg", color=(0.6, 0, 0.8, 1),
        ),
    ]


def _save_rgb_png(rgb: object, dest: Path) -> None:
    import numpy as np
    import torch

    dest.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(rgb, "detach"):
        arr_t = rgb.detach().to("cpu")
        if arr_t.dtype != torch.uint8:
            arr_t = arr_t.clamp(0, 255).to(torch.uint8)
        arr = arr_t.contiguous().numpy()
    else:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

    try:
        from PIL import Image

        Image.fromarray(arr).save(dest, format="PNG")
        return
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover
        import imageio  # type: ignore[no-redef]

    imageio.imwrite(dest, arr, format=".png")


def _save_bgr_png(frame_bgr, dest: Path) -> None:
    """Save a BGR uint8 frame (as produced by ``rgb_tensor_to_bgr_uint8``) as a PNG."""
    import numpy as np

    arr = np.asarray(frame_bgr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr[..., ::-1])  # BGR -> RGB
    _save_rgb_png(arr, dest)


def _obs_matrix_to_np(x) -> np.ndarray:
    """Coerce a camera-observation matrix (torch or numpy) to a numpy array."""
    if hasattr(x, "detach"):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


def _get_camera_node(env: WorldEnv, camera_name: str) -> CameraSensorNode:
    for node in env.get_nodes_by_type(CameraSensorNode):
        if node.name == camera_name:
            return node
    raise KeyError(f"Camera node not found: {camera_name}")


def init_env(
    target_object,
    *,
    use_clutter: bool,
    render_grasp_depth_seg: bool = False,
) -> WorldEnv:
    world = GenesisWorld(
        scene_init_kwargs=dict(
            sim_options=gs.options.SimOptions(dt=1 / 15, substeps=10),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
        ),
        lights={},
    )

    env_map = EnvMapManager(
        world,
        env_maps=[
            EnvMapManager.from_asset(asset_id)
            for asset_id in EnvMapManager.list_assets()
        ],
        name="env_map",
    )

    table = TableBuilder(
        world,
        name="table",
        pos=TABLE_POS,
        yaw=TABLE_YAW,
    )

    franka = FrankaFR3WithRobotiq2F85(
        world,
        name="franka",
        pos=ROBOT_POS,
        quat=(1, 0, 0, 0),
    )

    franka_gripper_cam = CameraSensorNode(
        world,
        name="gripper_cam",
        option=franka.get_droid_gripper_cam_options(
            backend="nyx",
            resolution=ZedSupportedResolution.HD720,
        ),
        render_rgb=True,
    )

    clutter = None
    if use_clutter:
        clutter = ClutteredTableManager(
            world,
            name="clutter",
            object_pool=_build_distractor_pool(),
            n_objects_range=(3, 5),
            table_size=TABLE_SIZE,
            table_pos=TABLE_POS,
            table_yaw=TABLE_YAW,
            random_quaternion=True,
        )

    task = ObjectRelocateTask(
        world,
        table_builder=table,
        clutter_manager=clutter,
        target_object=target_object,
        success_threshold=0.03,
    )
    task.has_termination_signal = False

    cam_pos = (1.0, 0.0, 1.4)
    cam_lookat = (0.0, 0.0, 0.6)
    camera = CameraSensorNode(
        world,
        name="camera",
        option=CameraSensorNodeOptions(
            backend="nyx",
            pos=cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
            res=(1280, 720),
            fov=60,
        ),
        render_rgb=True,
        render_depth=render_grasp_depth_seg,
        render_seg=render_grasp_depth_seg,
    )

    side_cam_pos = (0.0, 1.2, 1.0)
    side_cam_lookat = (0.0, 0.0, 0.6)
    side_camera = CameraSensorNode(
        world,
        name="side_camera",
        option=CameraSensorNodeOptions(
            backend="nyx",
            pos=side_cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(side_cam_pos, side_cam_lookat),
            res=(1280, 720),
            fov=60,
        ),
        render_rgb=True,
        render_depth=render_grasp_depth_seg,
        render_seg=render_grasp_depth_seg,
    )

    nodes = [env_map, table, franka, franka_gripper_cam, task]
    if clutter is not None:
        nodes.append(clutter)
    nodes.extend([camera, side_camera])

    if render_grasp_depth_seg:
        # Dedicated overhead camera for Contact-GraspNet grasp generation. A
        # top-down view tends to produce denser, more reliable grasp candidates
        # on tabletop scenes than the angled "camera" view. It is only created
        # when grasp depth/seg rendering is enabled (i.e. the Contact-GraspNet
        # path) to avoid affecting other artifacts.
        top_down_cam_pos = (0.0, 0.0, 1.4)
        top_down_cam_lookat = (0.0, 0.0, TABLE_SURFACE_Z)
        top_down_camera = CameraSensorNode(
            world,
            name="top_down_camera",
            option=CameraSensorNodeOptions(
                backend="nyx",
                pos=top_down_cam_pos,
                quat=CameraSensorNodeOptions.quat_from_lookat(
                    top_down_cam_pos,
                    top_down_cam_lookat,
                    # The view direction is straight down (-Z), so the default
                    # up vector (0, 0, 1) is collinear and degenerate. Use the
                    # world +X axis as up instead.
                    up=(1.0, 0.0, 0.0),
                ),
                res=(1280, 720),
                fov=60,
            ),
            render_rgb=True,
            render_depth=True,
            render_seg=True,
        )
        nodes.append(top_down_camera)

    return WorldEnv(world, nodes, render_mode="auto")


def _disable_move_object_slip(policy: PickPlaceRefPolicy, eef_name: str = "arm") -> None:
    """Zero out slip thresholds on the move-object phases of a pick-place policy.

    ``PickPlaceRefPolicy`` forwards ``goto_kwargs`` to *every* motion phase,
    including the EEF-centric ``retreat`` (a :class:`GoToLambdaRefPolicy`),
    which does not accept slip thresholds. To keep ``retreat`` untouched while
    still disabling slip detection during transport, reach into the already
    constructed graph and set ``slip_position_threshold`` /
    ``slip_orientation_threshold`` to ``0.0`` on the underlying
    :class:`MoveObjectLambdaPolicy` instances for the relevant phases only.

    This also disables per-step recomputation of the object→EEF relative pose
    (``track_relative_pose = False``) on the same move-object phases, so the
    transport relies solely on the transform cached at reset instead of
    querying the object/EEF pose every frame.
    """
    move_object_phases = ("lift", "move_to_goal", "lower", "release")
    node_policies = policy._graph._node_policies
    for phase in move_object_phases:
        combined = node_policies.get(phase)
        if combined is None:
            # Lazily-created factory node; not expected for these phases.
            continue
        move_object_policy = combined._policies.get(eef_name)
        if move_object_policy is None:
            continue
        move_object_policy.slip_position_threshold = 0.0
        move_object_policy.slip_orientation_threshold = 0.0
        move_object_policy.track_relative_pose = False


def init_ref_policy(env: WorldEnv, *, top_down_only: bool) -> PickPlaceRefPolicy:
    world: GenesisWorld = env.get_wrapper_attr("world")
    franka: FrankaFR3WithRobotiq2F85 = env.get_nodes_by_type(FrankaFR3WithRobotiq2F85)[0]
    task: ObjectRelocateTask = env.get_nodes_by_type(ObjectRelocateTask)[0]

    grasp_kwargs = {}
    if top_down_only:
        grasp_kwargs["filter_candidate_fn"] = BoxGripperGraspPolicy.filter_fn_top_down

    policy = PickPlaceRefPolicy(
        world=world,
        eef_controller=franka.controller["arm"],
        gripper_controller=franka.controller["gripper"],
        task=task,
        target_object=task.target_object,
        eef_name="arm",
        gripper_name="gripper",
        lift_height=0.15,
        retreat_height=0.15,
        release_seconds=1.0,
        grasp_kwargs=grasp_kwargs,
    )
    # Disable slip detection during the move-object phases so the policy does
    # not abort transport on small object drift (dataset rendering should not
    # terminate early due to slip). Applied post-construction so the
    # ``retreat`` GoToLambda phase (which does not accept slip kwargs) is not
    # affected.
    _disable_move_object_slip(policy, eef_name="arm")
    return policy


def init_ref_policy_with_contact_graspnet(
    env: WorldEnv,
    *,
    camera_name: str,
    checkpoint_dir: str,
    forward_passes: int,
) -> PickPlaceRefPolicy:
    world: GenesisWorld = env.get_wrapper_attr("world")
    franka: FrankaFR3WithRobotiq2F85 = env.get_nodes_by_type(FrankaFR3WithRobotiq2F85)[0]
    task: ObjectRelocateTask = env.get_nodes_by_type(ObjectRelocateTask)[0]
    camera = _get_camera_node(env, camera_name)
    provider = ContactGraspNetGraspCandidateProvider(
        world=world,
        camera_node=camera,
        target_entity=task.target_object_entity,
        checkpoint_dir=checkpoint_dir,
        forward_passes=forward_passes,
        gripper_width_limit=franka.controller["gripper"].gripper_width,
    )
    policy = PickPlaceRefPolicy(
        world=world,
        eef_controller=franka.controller["arm"],
        gripper_controller=franka.controller["gripper"],
        task=task,
        target_object=task.target_object,
        eef_name="arm",
        gripper_name="gripper",
        lift_height=0.15,
        retreat_height=0.15,
        release_seconds=1.0,
        grasp_kwargs={"candidate_provider": provider},
    )
    # Disable slip detection during the move-object phases so the policy does
    # not abort transport on small object drift (dataset rendering should not
    # terminate early due to slip). Applied post-construction so the
    # ``retreat`` GoToLambda phase (which does not accept slip kwargs) is not
    # affected.
    _disable_move_object_slip(policy, eef_name="arm")
    return policy


def _resolve_tags(args: argparse.Namespace) -> Optional[List[str]]:
    if args.object_id:
        return None

    tags = list(args.tag or [])
    if args.selection == "graspable":
        if "box_gripper_graspable" not in tags:
            tags.append("box_gripper_graspable")
    return tags or None


def _list_dataset_objects(
    dataset: str,
    *,
    tags: Optional[List[str]],
    contains: Optional[str],
) -> List[str]:
    if dataset == "gso":
        return list_gso_objects(tags=tags, contains=contains)
    if dataset == "ycb":
        if contains is not None:
            raise ValueError("--contains is only supported for --dataset gso")
        return list_ycb_objects(tags=tags)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _select_object_ids(dataset: str, args: argparse.Namespace) -> List[str]:
    if args.object_id:
        return list(args.object_id)

    tags = _resolve_tags(args)
    ids = _list_dataset_objects(dataset, tags=tags, contains=args.contains)

    if args.selection == "nongraspable":
        graspable_tags = list(args.tag or [])
        if "box_gripper_graspable" not in graspable_tags:
            graspable_tags.append("box_gripper_graspable")
        graspable_ids = set(
            _list_dataset_objects(dataset, tags=graspable_tags, contains=args.contains)
        )
        ids = [object_id for object_id in ids if object_id not in graspable_ids]

    if args.offset:
        ids = ids[args.offset:]
    if args.limit is not None:
        ids = ids[: args.limit]
    return ids


def _write_manifest(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected manifest list in {path}")
    return data


def _upsert_manifest_row(rows: List[dict], row: dict) -> None:
    key = (row.get("artifact"), row.get("dataset"), row.get("object_id"))
    for index, existing in enumerate(rows):
        existing_key = (
            existing.get("artifact"),
            existing.get("dataset"),
            existing.get("object_id"),
        )
        if existing_key == key:
            rows[index] = row
            return
    rows.append(row)


def _discover_videos(store_dir: Path) -> List[str]:
    return sorted(
        str(path.resolve())
        for path in store_dir.glob("*.mp4")
        if not path.name.startswith("debug_grasp_overlay_")
    )


def _discover_images(store_dir: Path) -> List[str]:
    return sorted(str(path.resolve()) for path in store_dir.glob("*.png"))


def render_video_object(
    *,
    dataset: str,
    object_id: str,
    seed: int,
    output_dir: Path,
    total_steps: int,
    use_clutter: bool,
    top_down_only: bool,
) -> dict:
    object_output_dir = output_dir / dataset / object_id
    object_output_dir.mkdir(parents=True, exist_ok=True)

    env = init_env(_build_target_object(dataset, object_id), use_clutter=use_clutter)
    policy = init_ref_policy(env, top_down_only=top_down_only)
    env = EpisodeVideoWrapper(env, store_dir=str(object_output_dir), format="mp4")

    status = "timeout"
    failure_step = None
    success_step = None
    error_message = None

    try:
        env.reset(seed=seed, reload=True)
        policy.reset(seed=seed)

        for step in range(total_steps):
            action = policy.get_action()
            env.step(action)
            policy.step()

            if policy.get_success():
                status = "success"
                success_step = step
                print(f"[{dataset}] {object_id}: success at step {step}")
                break
            if policy.get_failure():
                status = "failure"
                failure_step = step
                print(f"[{dataset}] {object_id}: failure at step {step}")
                break
        else:
            print(f"[{dataset}] {object_id}: timeout after {total_steps} steps")
    except Exception as exc:  # pragma: no cover - runtime safety path
        status = "error"
        error_message = repr(exc)
        print(f"[{dataset}] {object_id}: error: {exc!r}")
    finally:
        env.close()

    return {
        "artifact": "video",
        "dataset": dataset,
        "object_id": object_id,
        "seed": seed,
        "status": status,
        "success_step": success_step,
        "failure_step": failure_step,
        "error": error_message,
        "videos": _discover_videos(object_output_dir),
        "output_dir": str(object_output_dir.resolve()),
    }


def render_still_object(
    *,
    dataset: str,
    object_id: str,
    seed: int,
    output_dir: Path,
) -> dict:
    object_output_dir = output_dir / dataset / object_id
    object_output_dir.mkdir(parents=True, exist_ok=True)

    env = init_env(_build_target_object(dataset, object_id), use_clutter=False)
    status = "success"
    error_message = None

    try:
        env.reset(seed=seed, reload=True)
        for camera_name in ("camera", "side_camera"):
            camera = _get_camera_node(env, camera_name)
            rgb = camera.get_observation()["rgb"]
            _save_rgb_png(rgb, object_output_dir / f"{camera_name}.png")
        print(f"[{dataset}] {object_id}: saved stills")
    except Exception as exc:  # pragma: no cover - runtime safety path
        status = "error"
        error_message = repr(exc)
        print(f"[{dataset}] {object_id}: error: {exc!r}")
    finally:
        env.close()

    return {
        "artifact": "still",
        "dataset": dataset,
        "object_id": object_id,
        "seed": seed,
        "status": status,
        "error": error_message,
        "images": _discover_images(object_output_dir),
        "output_dir": str(object_output_dir.resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("gso", "ycb", "both"),
        required=True,
        help=(
            "Dataset to render. Note: in this codebase, GSO is the repackaged "
            "MuJoCo scanned objects dataset."
        ),
    )
    parser.add_argument(
        "--object-id",
        action="append",
        help="Specific object id to render. Repeat to render multiple ids.",
    )
    parser.add_argument(
        "--artifact",
        choices=("video", "still"),
        default="video",
        help="Output type: pick-place video or still image(s).",
    )
    parser.add_argument(
        "--selection",
        choices=("graspable", "nongraspable", "all"),
        default="graspable",
        help=(
            "Which subset of objects to render. 'graspable' uses the "
            "box_gripper_graspable tag, 'nongraspable' uses the complement, "
            "and 'all' renders every filtered object."
        ),
    )
    parser.add_argument(
        "--tag",
        action="append",
        help="Catalog tag filter. Repeat to require multiple tags.",
    )
    parser.add_argument(
        "--grasp-source",
        choices=("geometry", "contact_graspnet"),
        default="geometry",
        help="Source of grasp proposals for video rollouts.",
    )
    parser.add_argument(
        "--contact-graspnet-checkpoint",
        default=DEFAULT_CONTACT_GRASPNET_CHECKPOINT,
        help="Checkpoint directory for the vendored Contact-GraspNet PyTorch model.",
    )
    parser.add_argument(
        "--contact-graspnet-camera",
        default="camera",
        help=(
            "External camera name to use for Contact-GraspNet inference. "
            "'top_down_camera' (a dedicated overhead camera) is recommended for "
            "tabletop scenes and is available when grasp depth/seg rendering is "
            "enabled (i.e. the Contact-GraspNet path)."
        ),
    )
    parser.add_argument(
        "--contact-graspnet-forward-passes",
        type=int,
        default=5,
        help="Number of Contact-GraspNet forward passes per scene.",
    )
    parser.add_argument(
        "--debug-grasp-overlay",
        action="store_true",
        help=(
            "Save debug PNGs of the first frame with grasp proposal overlays "
            "projected into each available camera view (front 'camera', "
            "'side_camera', and the overhead 'top_down_camera' when present). "
            "Each view is projected using that camera's own intrinsics/"
            "extrinsics, so overlays stay registered to the right image."
        ),
    )
    parser.add_argument(
        "--debug-grasp-overlay-camera",
        default=None,
        help=(
            "Deprecated/no-op. Grasp-overlay PNGs are now rendered for every "
            "available view (camera, side_camera, top_down_camera). This flag "
            "is kept for backward compatibility and no longer has any effect."
        ),
    )
    parser.add_argument(
        "--debug-grasp-overlay-max-candidates",
        type=int,
        default=5,
        help="Maximum number of grasp candidates to draw in the overlay image.",
    )
    parser.add_argument(
        "--contains",
        help="Case-insensitive substring filter for GSO object ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of objects to render after filtering.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many filtered objects before rendering.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TOTAL_STEPS,
        help=f"Max environment steps per rollout (default: {TOTAL_STEPS}).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed; each object uses seed_base + object_index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-clutter",
        action="store_true",
        help="Disable distractor objects and render only the target object.",
    )
    parser.add_argument(
        "--allow-side-grasps",
        action="store_true",
        help="Do not restrict grasp candidates to top-down approaches.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.dataset == "both" and args.object_id:
        raise ValueError("--object-id cannot be combined with --dataset both")
    if args.contains is not None and args.dataset in ("ycb", "both"):
        raise ValueError("--contains is only supported with --dataset gso")
    if args.selection == "nongraspable" and args.dataset != "gso":
        raise ValueError("--selection nongraspable is currently only supported for --dataset gso")
    if args.grasp_source == "contact_graspnet" and args.artifact != "video":
        raise ValueError("--grasp-source contact_graspnet is only meaningful for --artifact video")
    if args.grasp_source == "contact_graspnet":
        pass

    datasets: Iterable[str] = ("gso", "ycb") if args.dataset == "both" else (args.dataset,)
    manifest_path = args.output_dir / "render_manifest.json"
    all_rows = _load_manifest(manifest_path)
    _write_manifest(manifest_path, all_rows)

    for dataset in datasets:
        object_ids = _select_object_ids(dataset, args)
        print(f"[{dataset}] rendering {len(object_ids)} objects")

        for object_index, object_id in enumerate(object_ids):
            if args.artifact == "video":
                if args.grasp_source == "geometry":
                    row = render_video_object(
                        dataset=dataset,
                        object_id=object_id,
                        seed=args.seed_base + object_index,
                        output_dir=args.output_dir,
                        total_steps=args.steps,
                        use_clutter=not args.no_clutter,
                        top_down_only=not args.allow_side_grasps,
                    )
                else:
                    object_output_dir = args.output_dir / dataset / object_id
                    object_output_dir.mkdir(parents=True, exist_ok=True)
                    base_env = init_env(
                        _build_target_object(dataset, object_id),
                        use_clutter=not args.no_clutter,
                        render_grasp_depth_seg=True,
                    )
                    env = EpisodeVideoWrapper(base_env, store_dir=str(object_output_dir), format="mp4")
                    status = "timeout"
                    failure_step = None
                    success_step = None
                    error_message = None
                    debug_overlay_images: Dict[str, str] = {}
                    grasp_debug_summary: dict = {}
                    try:
                        env.reset(seed=args.seed_base + object_index, reload=True)
                        policy = init_ref_policy_with_contact_graspnet(
                            base_env,
                            camera_name=args.contact_graspnet_camera,
                            checkpoint_dir=args.contact_graspnet_checkpoint,
                            forward_passes=args.contact_graspnet_forward_passes,
                        )
                        policy.reset(seed=args.seed_base + object_index)
                        grasp_policy = policy._graph._node_policies.get("grasp")
                        if grasp_policy is not None:
                            candidate_provider = getattr(grasp_policy, "_candidate_provider", None)
                            if candidate_provider is not None and hasattr(candidate_provider, "save_debug"):
                                candidate_provider.save_debug(object_output_dir)
                            if hasattr(grasp_policy, "save_debug"):
                                grasp_policy.save_debug(object_output_dir)
                            # Capture core grasp-debug fields for the manifest so
                            # failures can be inspected later without re-running
                            # the policy. These come from the policy's most
                            # recent candidate-selection pass.
                            grasp_debug = getattr(grasp_policy, "_debug_last", {}) or {}
                            overlay_debug = getattr(candidate_provider, "last_debug", None) if candidate_provider is not None else None
                            overlay_meta = (overlay_debug or {}).get("meta", {}) or {}
                            best_idx_raw = grasp_debug.get("best_idx")
                            if best_idx_raw is not None and hasattr(best_idx_raw, "item"):
                                best_idx_raw = int(best_idx_raw.item())
                            elif best_idx_raw is not None:
                                best_idx_raw = int(best_idx_raw)
                            best_error_raw = grasp_debug.get("best_error")
                            if isinstance(best_error_raw, (int, float)):
                                best_error_raw = float(best_error_raw)
                            elif best_error_raw is not None and hasattr(best_error_raw, "item"):
                                best_error_raw = float(best_error_raw.item())
                            filtered_scores = None
                            if overlay_debug is not None:
                                filtered_scores = overlay_debug.get("arrays", {}).get("filtered_scores")
                            selected_score = None
                            if (
                                best_idx_raw is not None
                                and filtered_scores is not None
                                and 0 <= best_idx_raw < len(filtered_scores)
                            ):
                                try:
                                    selected_score = float(np.asarray(filtered_scores)[best_idx_raw])
                                except Exception:
                                    selected_score = None
                            grasp_debug_summary = {
                                "grasp_status": grasp_debug.get("status"),
                                "selected_candidate_idx": best_idx_raw,
                                "selected_candidate_score": selected_score,
                                "best_error": best_error_raw,
                                "raw_candidate_count": overlay_meta.get("raw_candidate_count"),
                                "post_filter_candidate_count": overlay_meta.get(
                                    "post_filter_candidate_count",
                                    grasp_debug.get("post_filter_candidate_count"),
                                ),
                            }
                            print(
                                f"[{dataset}] {object_id}: grasp selected_idx={best_idx_raw} "
                                f"score={selected_score if selected_score is not None else 'n/a'} "
                                f"best_error={best_error_raw if best_error_raw is not None else 'n/a'} "
                                f"status={grasp_debug.get('status')}"
                            )
                            if args.debug_grasp_overlay and candidate_provider is not None:
                                # Robot gripper own-axes overlay (debug): project
                                # the arm/gripper controller's current approach
                                # and open/closing axes from the EEF frame into
                                # world coordinates using the current EEF pose
                                # (captured after reset, before any rollout
                                # stepping, since we save the first-frame PNG).
                                # These are world-frame directions, identical for
                                # every camera view; each view re-projects them
                                # with its own intrinsics/extrinsics below.
                                robot_axis_world = None
                                try:
                                    franka_node = base_env.get_nodes_by_type(FrankaFR3WithRobotiq2F85)[0]
                                    arm_ctrl = franka_node.controller["arm"]
                                    grip_ctrl = franka_node.controller["gripper"]
                                    eef_pos_t, eef_quat_t = arm_ctrl.get_eef_pose_absolute()
                                    eef_pos = eef_pos_t.detach().to("cpu").float().numpy().reshape(3)
                                    q_wxyz = eef_quat_t.detach().to("cpu").float().reshape(4)
                                    app_axis = torch.tensor(grip_ctrl.approach_direction, dtype=torch.float32)
                                    opn_axis = torch.tensor(grip_ctrl.open_direction, dtype=torch.float32)
                                    app_world_t = quaternion_apply(q_wxyz, app_axis)
                                    opn_world_t = quaternion_apply(q_wxyz, opn_axis)
                                    robot_axis_world = (
                                        eef_pos,
                                        app_world_t.detach().to("cpu").numpy().reshape(3),
                                        opn_world_t.detach().to("cpu").numpy().reshape(3),
                                    )
                                except Exception as axis_exc:
                                    print(
                                        f"[{dataset}] {object_id}: robot axis overlay "
                                        f"skipped: {axis_exc!r}"
                                    )
                                try:
                                    overlay_debug = getattr(candidate_provider, "last_debug", None)
                                    if (
                                        overlay_debug is not None
                                        and overlay_debug.get("meta", {}).get("status") == "ok"
                                    ):
                                        arrays = overlay_debug["arrays"]
                                        candidate_positions_world = arrays["candidate_positions_world"]
                                        candidate_approach_world = arrays["candidate_approach_world"]
                                        candidate_opening_world = arrays["candidate_opening_world"]
                                        overlay_scores = arrays.get("filtered_scores")
                                        overlay_openings = arrays.get("filtered_openings")
                                        # Render overlays for every available view.
                                        # Each overlay projects grasp candidates
                                        # using THAT camera's own intrinsics/
                                        # extrinsics (from its observation),
                                        # rather than the Contact-GraspNet
                                        # provider's stored matrices, so the
                                        # overlay stays registered to the right
                                        # image regardless of which view it is.
                                        for view_name in (
                                            "camera",
                                            "side_camera",
                                            "top_down_camera",
                                        ):
                                            try:
                                                view_camera = _get_camera_node(base_env, view_name)
                                            except KeyError:
                                                continue
                                            try:
                                                view_obs = view_camera.get_observation()
                                                view_intrinsics = _obs_matrix_to_np(view_obs["intrinsics_cv"])
                                                view_extrinsics = _obs_matrix_to_np(view_obs["extrinsics_cv"])
                                                overlay_data = build_grasp_overlay_data(
                                                    candidate_positions_world=candidate_positions_world,
                                                    candidate_approach_world=candidate_approach_world,
                                                    candidate_opening_world=candidate_opening_world,
                                                    intrinsics_cv=view_intrinsics,
                                                    extrinsics_cv=view_extrinsics,
                                                    selected_idx=best_idx_raw,
                                                    scores=overlay_scores,
                                                    openings=overlay_openings,
                                                    max_candidates=args.debug_grasp_overlay_max_candidates,
                                                )
                                                frame_bgr = rgb_tensor_to_bgr_uint8(view_obs["rgb"])
                                                draw_grasp_overlay(frame_bgr, overlay_data, step=0)
                                                if robot_axis_world is not None:
                                                    _r_eef, _r_app, _r_opn = robot_axis_world
                                                    robot_axis_data = build_robot_axis_overlay_data(
                                                        eef_pos_world=_r_eef,
                                                        approach_world=_r_app,
                                                        open_world=_r_opn,
                                                        intrinsics_cv=view_intrinsics,
                                                        extrinsics_cv=view_extrinsics,
                                                    )
                                                    draw_robot_axis_overlay(frame_bgr, robot_axis_data)
                                                overlay_dest = object_output_dir / f"debug_grasp_overlay_{view_name}.png"
                                                legacy_overlay_mp4 = object_output_dir / f"debug_grasp_overlay_{view_name}.mp4"
                                                if legacy_overlay_mp4.is_file():
                                                    try:
                                                        legacy_overlay_mp4.unlink()
                                                    except OSError:
                                                        pass
                                                _save_bgr_png(frame_bgr, overlay_dest)
                                                debug_overlay_images[view_name] = str(overlay_dest.resolve())
                                            except Exception as view_exc:
                                                print(
                                                    f"[{dataset}] {object_id}: debug grasp overlay "
                                                    f"skipped for view '{view_name}': {view_exc!r}"
                                                )
                                except Exception as overlay_exc:
                                    print(
                                        f"[{dataset}] {object_id}: debug grasp overlay skipped: {overlay_exc!r}"
                                    )
                        for step in range(args.steps):
                            action = policy.get_action()
                            env.step(action)
                            policy.step()
                            if policy.get_success():
                                status = "success"
                                success_step = step
                                print(f"[{dataset}] {object_id}: success at step {step}")
                                break
                            if policy.get_failure():
                                status = "failure"
                                failure_step = step
                                print(f"[{dataset}] {object_id}: failure at step {step}")
                                break
                    except Exception as exc:
                        status = "error"
                        error_message = repr(exc)
                        print(f"[{dataset}] {object_id}: error: {exc!r}")
                    finally:
                        env.close()
                    row = {
                        "artifact": "video",
                        "dataset": dataset,
                        "object_id": object_id,
                        "seed": args.seed_base + object_index,
                        "status": status,
                        "success_step": success_step,
                        "failure_step": failure_step,
                        "error": error_message,
                        "videos": _discover_videos(object_output_dir),
                        "output_dir": str(object_output_dir.resolve()),
                        "grasp_source": args.grasp_source,
                    }
                    if grasp_debug_summary:
                        row["grasp_debug"] = grasp_debug_summary
                    if debug_overlay_images:
                        row["debug_grasp_overlay_images"] = dict(debug_overlay_images)
            else:
                row = render_still_object(
                    dataset=dataset,
                    object_id=object_id,
                    seed=args.seed_base + object_index,
                    output_dir=args.output_dir,
                )
            _upsert_manifest_row(all_rows, row)
            _write_manifest(manifest_path, all_rows)

    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
