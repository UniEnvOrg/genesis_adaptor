from __future__ import annotations

import torch

from xbarray.transformations.pointcloud.pytorch import depth_image_to_world


def distance_by_translation_point(p1, p2):
    return torch.sqrt(((p1 - p2) ** 2).sum(dim=-1))


def farthest_points(
    data,
    nclusters,
    dist_func,
    return_center_indexes=False,
    return_distances=False,
    verbose=False,
):
    """Greedy farthest-point sampling on a ``(N, D)`` torch point set.

    Mirrors the original NumPy implementation's behaviour, including the
    deterministic start at index 0 (distances init to a large constant).
    """
    data = data.to(torch.float32)
    n = data.shape[0]
    if nclusters >= n:
        clusters = torch.arange(n, dtype=torch.int64, device=data.device)
        if return_center_indexes:
            centers = torch.arange(n, dtype=torch.int64, device=data.device)
            if return_distances:
                return clusters, centers, torch.full((n,), 0.0, dtype=torch.float32, device=data.device)
            return clusters, centers
        return clusters

    clusters = torch.full((n,), -1, dtype=torch.int64, device=data.device)
    distances = torch.full((n,), 1e7, dtype=torch.float32, device=data.device)
    centers = []
    for _iter in range(nclusters):
        index = int(torch.argmax(distances).item())
        centers.append(index)
        new_distances = dist_func(data[index], data).to(torch.float32)
        distances = torch.minimum(distances, new_distances)
        clusters = torch.where(distances == new_distances, torch.full_like(clusters, _iter), clusters)
        if verbose:
            print(f"farthest points max distance : {torch.max(distances).item()}")

    centers_t = torch.as_tensor(centers, dtype=torch.int64, device=data.device)
    if return_center_indexes:
        if return_distances:
            return clusters, centers_t, distances
        return clusters, centers_t
    return clusters


def reject_median_outliers(data, m=0.4, z_only=False):
    if data.shape[0] == 0:
        return data
    if z_only:
        col = data[:, 2:3]
        d = torch.abs(col - torch.median(col, dim=0, keepdim=True).values)
    else:
        d = torch.abs(data - torch.median(data, dim=0, keepdim=True).values)
    keep = d.sum(dim=1) < m
    return data[keep]


def regularize_pc_point_count(pc, npoints, use_farthest_point=False, seed=None):
    """Resample a point cloud to exactly ``npoints`` rows.

    Subsets use FPS (deterministic) or random permutation; padding samples
    with replacement. ``seed`` makes the random subset deterministic.
    """
    n = pc.shape[0]

    if n > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(
                pc, npoints, distance_by_translation_point, return_center_indexes=True,
            )
        else:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=pc.device).manual_seed(int(seed))
            center_indexes = torch.randperm(n, generator=generator, device=pc.device)[:npoints]
        pc = pc[center_indexes, :]
    else:
        required = npoints - n
        if required > 0:
            if n == 0:
                return pc
            generator = None
            if seed is not None:
                generator = torch.Generator(device=pc.device).manual_seed(int(seed))
            index = torch.randint(0, n, (required,), generator=generator, device=pc.device)
            pc = torch.cat((pc, pc[index, :]), dim=0)
    return pc


def depth2pc(depth, K, rgb=None):
    """Depth image -> camera-frame point cloud (torch).

    Uses :func:`xbarray.transformations.pointcloud.pytorch.depth_image_to_world`,
    which returns per-pixel ``(x, y, z, valid_mask)`` in camera coordinates when
    no extrinsic matrix is supplied. Only pixels with valid (positive) depth are
    kept, matching the original NumPy implementation.
    """
    depth = depth.to(torch.float32)
    K = K.to(dtype=torch.float32, device=depth.device)
    world = depth_image_to_world(depth, K)  # (H, W, 4)
    valid = world[..., 3] > 0
    pc = world[valid][..., :3]
    if rgb is not None:
        rgb = rgb.to(device=depth.device)
        rgb = rgb[valid]
    return pc, rgb


def preprocess_pc_for_inference(
    input_pc,
    num_point,
    pc_mean=None,
    return_mean=False,
    use_farthest_point=False,
    convert_to_internal_coords=False,
    seed=None,
):
    input_pc = input_pc.to(torch.float32)
    if input_pc.shape[0] != num_point:
        pc = regularize_pc_point_count(
            input_pc, num_point, use_farthest_point=use_farthest_point, seed=seed,
        ).clone()
    else:
        pc = input_pc.clone()
    if convert_to_internal_coords:
        pc[:, :2] *= -1
    if pc_mean is None:
        pc_mean = pc.mean(dim=0)
    pc = pc - pc_mean.unsqueeze(0)
    if return_mean:
        return pc, pc_mean
    return pc
