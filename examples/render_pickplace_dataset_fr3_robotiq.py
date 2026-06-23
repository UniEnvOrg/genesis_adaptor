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
from typing import Iterable, List, Optional

import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.gpu)

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


def _get_camera_node(env: WorldEnv, camera_name: str) -> CameraSensorNode:
    for node in env.get_nodes_by_type(CameraSensorNode):
        if node.name == camera_name:
            return node
    raise KeyError(f"Camera node not found: {camera_name}")


def init_env(target_object, *, use_clutter: bool) -> WorldEnv:
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
    )

    nodes = [env_map, table, franka, franka_gripper_cam, task]
    if clutter is not None:
        nodes.append(clutter)
    nodes.extend([camera, side_camera])

    return WorldEnv(world, nodes, render_mode="auto")


def init_ref_policy(env: WorldEnv, *, top_down_only: bool) -> PickPlaceRefPolicy:
    world: GenesisWorld = env.get_wrapper_attr("world")
    franka: FrankaFR3WithRobotiq2F85 = env.get_nodes_by_type(FrankaFR3WithRobotiq2F85)[0]
    task: ObjectRelocateTask = env.get_nodes_by_type(ObjectRelocateTask)[0]

    grasp_kwargs = {}
    if top_down_only:
        grasp_kwargs["filter_candidate_fn"] = BoxGripperGraspPolicy.filter_fn_top_down

    return PickPlaceRefPolicy(
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
    return sorted(str(path.resolve()) for path in store_dir.glob("*.mp4"))


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

    datasets: Iterable[str] = ("gso", "ycb") if args.dataset == "both" else (args.dataset,)
    manifest_path = args.output_dir / "render_manifest.json"
    all_rows = _load_manifest(manifest_path)
    _write_manifest(manifest_path, all_rows)

    for dataset in datasets:
        object_ids = _select_object_ids(dataset, args)
        print(f"[{dataset}] rendering {len(object_ids)} objects")

        for object_index, object_id in enumerate(object_ids):
            if args.artifact == "video":
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
