import math

import numpy as np

LINK_PADDING = 2.2
CONNECTOR_REACH = 52.0
CANVAS_PADDING = 6.0


def compute_face_frame(row, padding=LINK_PADDING):
    angle = math.radians(row["ang"])
    along = np.array([math.cos(angle), math.sin(angle)])
    across = np.array([-along[1], along[0]])
    centre = np.asarray(row["c"], float)
    return centre, along, across, row["h"] + 2 * padding, row["w"] + 2 * padding


def compute_quad(frame, along_start, along_end, across_start, across_end):
    centre, along, across, height, width = frame
    corners = (
        (along_start, across_start),
        (along_start, across_end),
        (along_end, across_end),
        (along_end, across_start),
    )
    return np.array([centre + along * height * y + across * width * x for y, x in corners])


def compute_rectangle(row, padding=LINK_PADDING):
    return compute_quad(compute_face_frame(row, padding), -0.5, 0.5, -0.5, 0.5)


def generate_tiles(row, n_around, n_along, *, padding=LINK_PADDING, inset=0.0,
                   length_scale=1.0, reverse_across=False):
    frame = list(compute_face_frame(row, padding))
    frame[3] *= length_scale
    for along_index in range(n_along):
        for around_index in range(n_around):
            column = n_around - 1 - around_index if reverse_across else around_index
            tile = compute_quad(
                frame,
                along_index / n_along - 0.5,
                (along_index + 1) / n_along - 0.5,
                column / n_around - 0.5,
                (column + 1) / n_around - 0.5,
            )
            if inset:
                centre = tile.mean(axis=0)
                tile = centre + (tile - centre) * (1 - inset)
            yield along_index * n_around + around_index, tile


def compute_connector(rectangle, palm_centre, reach=CONNECTOR_REACH):
    nearest = sorted(range(4), key=lambda index: np.linalg.norm(rectangle[index] - palm_centre))[:2]
    start, end = rectangle[nearest]
    direction = palm_centre - (start + end) / 2
    direction /= max(np.linalg.norm(direction), 1e-9)
    return np.array([start, end, end + direction * reach, start + direction * reach])


def compute_bounds(polygons, padding=CANVAS_PADDING):
    points = np.concatenate(polygons)
    return points.min(axis=0) - padding, points.max(axis=0) + padding


def transform_to_canvas(points, upper_bound, scale=1.0, margin=0.0):
    points = np.asarray(points)
    mirrored = np.stack([upper_bound[0] - points[:, 0], upper_bound[1] - points[:, 1]], axis=1)
    return mirrored * scale + margin
