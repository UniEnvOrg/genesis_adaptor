from __future__ import annotations

import numpy as np


def distance_by_translation_point(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def farthest_points(data, nclusters, dist_func, return_center_indexes=False, return_distances=False, verbose=False):
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)
        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for _iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1
        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = _iter
        if verbose:
            print(f"farthest points max distance : {np.max(distances)}")

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)
    return clusters


def reject_median_outliers(data, m=0.4, z_only=False):
    if z_only:
        d = np.abs(data[:, 2:3] - np.median(data[:, 2:3]))
    else:
        d = np.abs(data - np.median(data, axis=0, keepdims=True))
    return data[np.sum(d, axis=1) < m]


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(
                pc, npoints, distance_by_translation_point, return_center_indexes=True,
            )
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def depth2pc(depth, K, rgb=None):
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]
    normalized_x = (x.astype(np.float32) - K[0, 2])
    normalized_y = (y.astype(np.float32) - K[1, 2])
    world_x = normalized_x * depth[y, x] / K[0, 0]
    world_y = normalized_y * depth[y, x] / K[1, 1]
    world_z = depth[y, x]
    if rgb is not None:
        rgb = rgb[y, x, :]
    pc = np.vstack((world_x, world_y, world_z)).T
    return pc, rgb


def preprocess_pc_for_inference(input_pc, num_point, pc_mean=None, return_mean=False, use_farthest_point=False, convert_to_internal_coords=False):
    normalize_pc_count = input_pc.shape[0] != num_point
    if normalize_pc_count:
        pc = regularize_pc_point_count(input_pc, num_point, use_farthest_point=use_farthest_point).copy()
    else:
        pc = input_pc.copy()
    if convert_to_internal_coords:
        pc[:, :2] *= -1
    if pc_mean is None:
        pc_mean = np.mean(pc, 0)
    pc -= np.expand_dims(pc_mean, 0)
    if return_mean:
        return pc, pc_mean
    return pc
