from dataclasses import dataclass
import io

import numpy as np
from PIL import Image

from common import compute_link_frame, compute_view_matrix

SKIN_COLOR = np.array([0.63, 0.67, 0.70])
TAXEL_COLOR = np.array([0.93, 0.60, 0.20])
SEAM_COLOR = np.array([0.31, 0.17, 0.03])


@dataclass
class BandParameters:
    long_axis: int
    face_axis: int
    cross_axis: int
    face_sign: float
    centre: np.ndarray
    stations: np.ndarray
    angles: np.ndarray
    station_spacing: float
    angle_spacing: float


@dataclass
class HandMesh:
    world_vertices: np.ndarray
    local_vertices: np.ndarray
    triangles: np.ndarray
    triangle_link_ids: np.ndarray
    bands: list[BandParameters | None]


@dataclass
class RasterBuffers:
    link_ids: np.ndarray
    local_positions: np.ndarray
    normals: np.ndarray


def compute_band_parameters(mesh, face_axis, arc, margin, n_around, n_along):
    frame = compute_link_frame(mesh, face_axis)
    span = frame["hi"][frame["la"]] - frame["lo"][frame["la"]]
    stations = np.linspace(frame["lo"][frame["la"]] + margin * span,
                           frame["hi"][frame["la"]] - margin * span, n_along)
    angles = np.linspace(-np.radians(arc) / 2, np.radians(arc) / 2, n_around)
    return BandParameters(frame["la"], frame["u"], frame["v"], frame["sign"], frame["centre"],
                          stations, angles, float(stations[1] - stations[0]), float(angles[1] - angles[0]))


def _compute_eye_position(vertices, direction, target, fov, width, height, up, padding=1.06):
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    right = np.cross(-direction, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, -direction)
    vertical_tangent = np.tan(np.radians(fov) / 2) / padding
    horizontal_tangent = vertical_tangent * width / height
    relative = vertices - target
    depth, horizontal, vertical = relative @ direction, relative @ right, relative @ camera_up
    distance = np.max(np.maximum(depth + np.abs(horizontal) / horizontal_tangent,
                                 depth + np.abs(vertical) / vertical_tangent))
    return target + direction * float(distance)


def _project_vertices(hand, direction, target, fov, width, height, up):
    eye = _compute_eye_position(hand.world_vertices, direction, target, fov, width, height, up)
    view = compute_view_matrix(eye, target, up)
    camera_vertices = (view[:3, :3] @ hand.world_vertices.T).T + view[:3, 3]
    depth = -camera_vertices[:, 2]
    focal_length = (height / 2) / np.tan(np.radians(fov) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        screen_x = width / 2 + camera_vertices[:, 0] / depth * focal_length
        screen_y = height / 2 - camera_vertices[:, 1] / depth * focal_length
    return screen_x, screen_y, depth, 1.0 / np.maximum(depth, 1e-6)


def _rasterize(hand, screen_x, screen_y, depth, inverse_depth, width, height):
    depth_buffer = np.full((height, width), np.inf, np.float32)
    link_ids = np.full((height, width), -2, np.int16)
    local_positions = np.zeros((height, width, 3), np.float32)
    normals = np.zeros((height, width, 3), np.float32)
    world_triangles = hand.world_vertices[hand.triangles]
    face_normals = np.cross(world_triangles[:, 1] - world_triangles[:, 0],
                            world_triangles[:, 2] - world_triangles[:, 0])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)

    for triangle_index in np.flatnonzero(np.all(depth[hand.triangles] > 1e-4, axis=1)):
        first, second, third = hand.triangles[triangle_index]
        x0, x1, x2 = screen_x[first], screen_x[second], screen_x[third]
        y0, y1, y2 = screen_y[first], screen_y[second], screen_y[third]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        x_min = max(int(np.floor(min(x0, x1, x2))), 0)
        x_max = min(int(np.ceil(max(x0, x1, x2))), width - 1)
        y_min = max(int(np.floor(min(y0, y1, y2))), 0)
        y_max = min(int(np.ceil(max(y0, y1, y2))), height - 1)
        if x_max < x_min or y_max < y_min:
            continue

        grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max + 1) + 0.5,
                                     np.arange(y_min, y_max + 1) + 0.5)
        weight0 = ((x1 - grid_x) * (y2 - grid_y) - (x2 - grid_x) * (y1 - grid_y)) / area
        weight1 = ((x2 - grid_x) * (y0 - grid_y) - (x0 - grid_x) * (y2 - grid_y)) / area
        weight2 = 1.0 - weight0 - weight1
        inside = (weight0 >= 0) & (weight1 >= 0) & (weight2 >= 0)
        if not inside.any():
            continue
        inverse_z = weight0 * inverse_depth[first] + weight1 * inverse_depth[second] + weight2 * inverse_depth[third]
        pixel_depth = 1.0 / np.maximum(inverse_z, 1e-12)
        depth_slice = depth_buffer[y_min:y_max + 1, x_min:x_max + 1]
        visible = inside & (pixel_depth < depth_slice)
        if not visible.any():
            continue

        perspective0 = (weight0 * inverse_depth[first] / inverse_z)[visible]
        perspective1 = (weight1 * inverse_depth[second] / inverse_z)[visible]
        perspective2 = (weight2 * inverse_depth[third] / inverse_z)[visible]
        position = (hand.local_vertices[first] * perspective0[:, None]
                    + hand.local_vertices[second] * perspective1[:, None]
                    + hand.local_vertices[third] * perspective2[:, None])
        depth_slice[visible] = pixel_depth[visible]
        link_ids[y_min:y_max + 1, x_min:x_max + 1][visible] = hand.triangle_link_ids[triangle_index]
        local_positions[y_min:y_max + 1, x_min:x_max + 1][visible] = position
        normals[y_min:y_max + 1, x_min:x_max + 1][visible] = face_normals[triangle_index]
    return RasterBuffers(link_ids, local_positions, normals)


def _shade(hand, raster, direction, up):
    colors = np.zeros((*raster.link_ids.shape, 3), np.float32)
    body = raster.link_ids > -2
    colors[body] = SKIN_COLOR
    for link_id, band in enumerate(hand.bands):
        selected = raster.link_ids == link_id
        if band is None or not selected.any():
            continue
        positions = raster.local_positions[selected]
        face = (positions[:, band.face_axis] - band.centre[band.face_axis]) * band.face_sign
        cross = positions[:, band.cross_axis] - band.centre[band.cross_axis]
        angles = np.arctan2(cross, face)
        stations = positions[:, band.long_axis]
        nearest_angle = np.abs(angles[:, None] - band.angles[None, :]).argmin(axis=1)
        nearest_station = np.abs(stations[:, None] - band.stations[None, :]).argmin(axis=1)
        angle_offset = (angles - band.angles[nearest_angle]) / band.angle_spacing
        station_offset = (stations - band.stations[nearest_station]) / band.station_spacing
        sensed = (np.abs(angle_offset) <= 0.5) & (np.abs(station_offset) <= 0.5)
        seam = np.maximum(np.abs(angle_offset), np.abs(station_offset)) > 0.5 - 0.032
        link_colors = np.repeat(SKIN_COLOR[None, :], len(positions), axis=0).astype(np.float32)
        link_colors[sensed] = TAXEL_COLOR
        link_colors[sensed & seam] = SEAM_COLOR
        colors[selected] = link_colors

    back = np.asarray(direction, float)
    back /= np.linalg.norm(back)
    right = np.cross(-back, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, -back)
    key = back * 0.80 + camera_up * 0.42 + right * 0.32
    key /= np.linalg.norm(key)
    fill = back * 0.45 - right * 0.72 - camera_up * 0.20
    fill /= np.linalg.norm(fill)
    light = np.clip(raster.normals @ key, 0, 1) * 0.62 + np.clip(raster.normals @ fill, 0, 1) * 0.22 + 0.36
    image = np.clip(colors * light[..., None], 0, 1)
    image[~body] = 0.965
    return (image ** (1 / 1.05) * 255).astype(np.uint8)


def render_hand(hand, direction, target, fov, width, height, up=None):
    up = np.array([0.0, 0.0, 1.0]) if up is None else up
    projection = _project_vertices(hand, direction, np.asarray(target, float), fov, width, height, up)
    return _shade(hand, _rasterize(hand, *projection, width, height), direction, up)


def encode_jpeg(image, width, height, quality) -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray(image).resize((width, height), Image.LANCZOS).convert("RGB")
    image.save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
