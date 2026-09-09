import json
from pathlib import Path

import numpy as np
from genesis.utils.geom import inv_transform_by_trans_quat, transform_by_trans_quat

from common import (ARC, MARGIN, N_ALONG, N_AROUND, PLAIN, SENSED, SIDE, build_hand,
                    get_face_axis_for, get_link_mesh, get_link_pose, get_output_path,
                    get_world_transform, transform_to_world)
from renderer import HandMesh, compute_band_parameters, encode_jpeg, render_hand

SUPERSAMPLING = 2
IMAGE_WIDTH = 1150
IMAGE_HEIGHT = 1300
FIELD_OF_VIEW = 30
JPEG_QUALITY = 88


def compute_face_row(name, mesh, link_pose, palm_pose):
    lower, upper = mesh.bounds
    area = float((upper[1] - lower[1]) * (upper[2] - lower[2]) * 1e4)
    centre = np.array([upper[0], 0.5 * (lower[1] + upper[1]), 0.5 * (lower[2] + upper[2])])
    face_axis = np.asarray([centre, centre + [0.0, 0.0, 0.01]])
    palm_centre, palm_direction = inv_transform_by_trans_quat(
        transform_by_trans_quat(face_axis, *link_pose), *palm_pose
    )
    angle = float(np.degrees(np.arctan2(palm_direction[2] - palm_centre[2], palm_direction[1] - palm_centre[1])))
    return {
        "link": name,
        "area_cm2": area,
        "w": float((upper[1] - lower[1]) * 1000),
        "h": float((upper[2] - lower[2]) * 1000),
        "ang": angle,
        "c": [float(palm_centre[1] * 1000), float(palm_centre[2] * 1000)],
    }


def collect_hand_geometry(entity, palm):
    world_vertices, local_vertices, triangles, triangle_link_ids = [], [], [], []
    bands, rows = [], []
    vertex_offset = 0
    palm_pose = get_link_pose(palm)
    for link_id, name in enumerate(SENSED + PLAIN):
        link = entity.get_link(name)
        if not link.geoms:
            continue
        mesh = get_link_mesh(link)
        vertices = np.asarray(mesh.vertices, float)
        local_vertices.append(vertices)
        world_vertices.append(transform_to_world(vertices, link))
        triangles.append(np.asarray(mesh.faces) + vertex_offset)
        triangle_link_ids.append(np.full(len(mesh.faces), link_id if name in SENSED else -1))
        bands.append(
            compute_band_parameters(mesh, get_face_axis_for(name), ARC, MARGIN, N_AROUND, N_ALONG)
            if name in SENSED else None
        )
        vertex_offset += len(vertices)
        if name in SENSED:
            rows.append(compute_face_row(name, mesh, get_link_pose(link), palm_pose))
            print(f"{name:26s} area {rows[-1]['area_cm2']:5.1f} cm^2")

    hand = HandMesh(np.concatenate(world_vertices), np.concatenate(local_vertices),
                    np.concatenate(triangles), np.concatenate(triangle_link_ids), bands)
    print(f"{len(hand.triangles)} triangles, {len(hand.world_vertices)} vertices")
    return hand, rows


def build_static_artifacts():
    environment, entity = build_hand()
    palm = entity.get_link(f"{SIDE}_wuji_palm")
    hand, rows = collect_hand_geometry(entity, palm)
    palm_transform = get_world_transform(palm)
    palm_x, palm_y, palm_z = palm_transform[:3, 0], palm_transform[:3, 1], palm_transform[:3, 2]
    environment.close()

    Path(get_output_path("taxels.json")).write_text(json.dumps(rows, indent=1))
    print("wrote taxels.json")
    target = 0.5 * (hand.world_vertices.min(axis=0) + hand.world_vertices.max(axis=0))
    views = {
        "hand_palm": palm_x,
        "hand_three_quarter": palm_x * 0.62 + palm_y * 0.78,
        "hand_edge": palm_x * 0.12 + palm_y * 0.99 + palm_z * 0.10,
    }
    images = {}
    for name, direction in views.items():
        rendered = render_hand(hand, direction, target, FIELD_OF_VIEW,
                               IMAGE_WIDTH * SUPERSAMPLING, IMAGE_HEIGHT * SUPERSAMPLING, up=palm_z)
        images[name] = encode_jpeg(rendered, IMAGE_WIDTH, IMAGE_HEIGHT, JPEG_QUALITY)
        Path(get_output_path(f"{name}.jpg")).write_bytes(images[name])
        print(f"wrote {name}.jpg  ({len(images[name]) // 1024} KB)")
    return rows, images
