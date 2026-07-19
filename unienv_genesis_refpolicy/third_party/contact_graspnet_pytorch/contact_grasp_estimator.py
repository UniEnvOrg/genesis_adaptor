from __future__ import annotations

import importlib

import torch

from unienv_genesis_refpolicy.third_party.contact_graspnet_pytorch import config_utils
from unienv_genesis_refpolicy.third_party.contact_graspnet_pytorch.data import (
    depth2pc,
    distance_by_translation_point,
    farthest_points,
    preprocess_pc_for_inference,
    regularize_pc_point_count,
    reject_median_outliers,
)


class GraspEstimator:
    """Minimal inference-only estimator adapted from the community port."""

    def __init__(self, cfg):
        if "surface_grasp_logdir_folder" in cfg:
            self._contact_grasp_cfg = config_utils.load_config(
                cfg["surface_grasp_logdir_folder"], batch_size=1, arg_configs=cfg.arg_configs,
            )
            self._cfg = cfg
            self._num_samples = self._cfg.num_samples
        else:
            self._contact_grasp_cfg = cfg

        self._model_func = importlib.import_module(
            "unienv_genesis_refpolicy.third_party.contact_graspnet_pytorch.contact_graspnet"
        )
        self._num_input_points = self._contact_grasp_cfg["DATA"].get(
            "raw_num_points",
            self._contact_grasp_cfg["DATA"]["num_point"],
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._model_func.ContactGraspnet(self._contact_grasp_cfg, self.device)
        self.model.to(self.device)

    def filter_segment(self, contact_pts, segment_pc, thres=0.00001):
        if contact_pts.shape[0] > 0 and segment_pc.shape[0] > 0:
            try:
                dists = contact_pts[:, :3].reshape(-1, 1, 3) - segment_pc.reshape(1, -1, 3)
                min_dists = torch.linalg.norm(dists, dim=2).min(dim=1).values
                return torch.nonzero(min_dists < thres, as_tuple=True)[0]
            except Exception:
                pass
        return torch.zeros((0,), dtype=torch.int64, device=self.device)

    def extract_3d_cam_boxes(self, full_pc, pc_segments, min_size=0.3, max_size=0.6):
        pc_regions, obj_centers = {}, {}
        for i in pc_segments:
            pc_segments[i] = reject_median_outliers(pc_segments[i], m=0.4, z_only=False)
            seg = pc_segments[i]
            if seg.shape[0] > 0:
                max_bounds = seg[:, :3].max(dim=0).values
                min_bounds = seg[:, :3].min(dim=0).values
                obj_extent = max_bounds - min_bounds
                obj_center = min_bounds + obj_extent / 2
                size = torch.clamp(torch.max(obj_extent) * 2, min=min_size, max=max_size)
                mask = (full_pc > (obj_center - size / 2)).all(dim=1) & (full_pc < (obj_center + size / 2)).all(dim=1)
                partial_pc = full_pc[mask]
                if partial_pc.shape[0] > 0:
                    partial_pc = regularize_pc_point_count(
                        partial_pc,
                        self._contact_grasp_cfg["DATA"]["raw_num_points"],
                        use_farthest_point=self._contact_grasp_cfg["DATA"]["use_farthest_point"],
                    )
                    pc_regions[i] = partial_pc
                    obj_centers[i] = obj_center
        return pc_regions, obj_centers

    def filter_pc_segments(self, pc_segments):
        for i in pc_segments:
            pc_segments[i] = reject_median_outliers(pc_segments[i], m=0.4, z_only=False)
        return pc_segments

    def predict_grasps(self, pc, constant_offset=False, convert_cam_coords=True, forward_passes=1, seed=None):
        pc = pc.to(dtype=torch.float32, device=self.device)
        if pc.shape[0] == 0:
            empty_grasps = torch.zeros((0, 4, 4), dtype=torch.float32, device=self.device)
            empty_scores = torch.zeros((0,), dtype=torch.float32, device=self.device)
            empty_points = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            empty_openings = torch.zeros((0,), dtype=torch.float32, device=self.device)
            return empty_grasps, empty_scores, empty_points, empty_openings
        pc, pc_mean = preprocess_pc_for_inference(
            pc.squeeze(),
            self._num_input_points,
            return_mean=True,
            convert_to_internal_coords=convert_cam_coords,
            seed=seed,
        )
        pc_batch = pc.unsqueeze(0) if pc.dim() == 2 else pc
        if forward_passes > 1:
            pc_batch = pc_batch.repeat(forward_passes, 1, 1)
        pc_batch = pc_batch.to(self.device)
        pred = self.model(pc_batch)
        pred_grasps_cam = pred["pred_grasps_cam"]
        pred_scores = pred["pred_scores"]
        pred_points = pred["pred_points"]
        offset_pred = pred["offset_pred"]

        pred_grasps_cam = pred_grasps_cam.reshape(-1, *pred_grasps_cam.shape[-2:])
        pred_points = pred_points.reshape(-1, pred_points.shape[-1])
        pred_scores = pred_scores.reshape(-1)
        offset_pred = offset_pred.reshape(-1)

        pred_grasps_cam[:, :3, 3] = pred_grasps_cam[:, :3, 3] + pc_mean.reshape(-1, 3)
        pred_points[:, :3] = pred_points[:, :3] + pc_mean.reshape(-1, 3)

        if constant_offset:
            value = self._contact_grasp_cfg["DATA"]["gripper_width"] - self._contact_grasp_cfg["TEST"]["extra_opening"]
            offset_pred = torch.full((offset_pred.shape[0],), value, dtype=torch.float32, device=self.device)
        gripper_openings = torch.minimum(
            offset_pred + self._contact_grasp_cfg["TEST"]["extra_opening"],
            torch.tensor(float(self._contact_grasp_cfg["DATA"]["gripper_width"]), dtype=torch.float32, device=self.device),
        )
        selection_idcs = self.select_grasps(
            pred_points[:, :3],
            pred_scores,
            self._contact_grasp_cfg["TEST"]["max_farthest_points"],
            self._contact_grasp_cfg["TEST"]["num_samples"],
            self._contact_grasp_cfg["TEST"]["first_thres"],
            self._contact_grasp_cfg["TEST"].get("second_thres", self._contact_grasp_cfg["TEST"]["first_thres"]),
            with_replacement=self._contact_grasp_cfg["TEST"]["with_replacement"],
        )
        if selection_idcs.numel() == 0:
            selection_idcs = torch.zeros((0,), dtype=torch.int64, device=self.device)
        if self._contact_grasp_cfg["TEST"].get("center_to_tip"):
            pred_grasps_cam[:, :3, 3] = pred_grasps_cam[:, :3, 3] - pred_grasps_cam[:, :3, 2] * (self._contact_grasp_cfg["TEST"]["center_to_tip"] / 2)
        if convert_cam_coords:
            pred_grasps_cam[:, :2, :] = pred_grasps_cam[:, :2, :] * -1
            pred_points[:, :2] = pred_points[:, :2] * -1
        return (
            pred_grasps_cam[selection_idcs],
            pred_scores[selection_idcs],
            pred_points[selection_idcs].squeeze(),
            gripper_openings[selection_idcs].squeeze(),
        )

    def predict_scene_grasps(self, pc_full, pc_segments=None, local_regions=False, filter_grasps=False, forward_passes=1, use_cam_boxes=True, seed=None):
        pc_full = pc_full.to(dtype=torch.float32, device=self.device)
        pc_segments = {} if pc_segments is None else pc_segments
        pc_segments = {
            k: v.to(dtype=torch.float32, device=self.device)
            for k, v in pc_segments.items()
        }
        pred_grasps_cam, scores, contact_pts, gripper_openings = {}, {}, {}, {}
        if local_regions:
            if use_cam_boxes:
                pc_regions, _ = self.extract_3d_cam_boxes(pc_full, pc_segments, min_size=0.2)
            else:
                pc_regions = self.filter_pc_segments(pc_segments)
            for k, pc_region in pc_regions.items():
                pred_grasps_cam[k], scores[k], contact_pts[k], gripper_openings[k] = self.predict_grasps(
                    pc_region, convert_cam_coords=True, forward_passes=forward_passes, seed=seed,
                )
        else:
            pc_full = regularize_pc_point_count(pc_full, self._contact_grasp_cfg["DATA"]["raw_num_points"], seed=seed)
            pred_grasps_cam[-1], scores[-1], contact_pts[-1], gripper_openings[-1] = self.predict_grasps(
                pc_full, convert_cam_coords=True, forward_passes=forward_passes, seed=seed,
            )

        if filter_grasps:
            segment_keys = list(contact_pts.keys()) if local_regions else list(pc_segments.keys())
            for k in segment_keys:
                j = k if local_regions else -1
                seg_k = pc_segments[k]
                if seg_k.shape[0] > 0 and contact_pts[j].shape[0] > 0:
                    seg_t = contact_pts[j].to(dtype=torch.float32, device=self.device)
                    segment_idcs = self.filter_segment(seg_t, seg_k, thres=self._contact_grasp_cfg["TEST"]["filter_thres"])
                    pred_grasps_cam[k] = pred_grasps_cam[j][segment_idcs]
                    scores[k] = scores[j][segment_idcs]
                    contact_pts[k] = contact_pts[j][segment_idcs]
                    gripper_openings[k] = gripper_openings[j][segment_idcs]
        return pred_grasps_cam, scores, contact_pts, gripper_openings

    def select_grasps(self, contact_pts, contact_conf, max_farthest_points=150, num_grasps=200, first_thres=0.25, second_thres=0.2, with_replacement=False):
        grasp_conf = contact_conf.to(dtype=torch.float32, device=self.device).squeeze()
        contact_pts = contact_pts.to(dtype=torch.float32, device=self.device).squeeze()
        if grasp_conf.dim() == 0:
            grasp_conf = grasp_conf.unsqueeze(0)
        if contact_pts.dim() == 1:
            contact_pts = contact_pts.unsqueeze(0)
        conf_idcs_greater_than = torch.nonzero(grasp_conf > first_thres, as_tuple=True)[0]
        _, center_indexes = farthest_points(
            contact_pts[conf_idcs_greater_than, :3],
            min(max_farthest_points, int(conf_idcs_greater_than.shape[0])),
            distance_by_translation_point,
            return_center_indexes=True,
        )
        all_idcs = torch.arange(grasp_conf.shape[0], dtype=torch.int64, device=self.device)
        centers_taken = conf_idcs_greater_than[center_indexes]
        remaining_confidences = all_idcs[~torch.isin(all_idcs, centers_taken)]
        sorted_confidences = torch.argsort(grasp_conf, descending=True)
        mask = torch.isin(sorted_confidences, remaining_confidences)
        sorted_remaining_confidence_idcs = sorted_confidences[mask]
        if with_replacement:
            selection_idcs = list(centers_taken.tolist())
            j = len(selection_idcs)
            while j < num_grasps and conf_idcs_greater_than.shape[0] > 0:
                selection_idcs.append(conf_idcs_greater_than[j % conf_idcs_greater_than.shape[0]].item())
                j += 1
            return torch.as_tensor(selection_idcs, dtype=torch.int64, device=self.device)
        remaining_idcs = sorted_remaining_confidence_idcs[: num_grasps - int(center_indexes.shape[0])]
        remaining_conf_idcs_greater_than = torch.nonzero(grasp_conf[remaining_idcs] > second_thres, as_tuple=True)[0]
        return torch.cat((centers_taken, remaining_idcs[remaining_conf_idcs_greater_than])).unique()

    def extract_point_clouds(self, depth, K, segmap=None, rgb=None, z_range=(0.2, 1.8), segmap_id=0, skip_border_objects=False, margin_px=5):
        if K is None:
            raise ValueError("K is required")
        depth = depth.to(torch.float32)
        K = K.to(dtype=torch.float32, device=depth.device)
        pc_full, pc_colors = depth2pc(depth, K, rgb)
        z_mask = (pc_full[:, 2] < z_range[1]) & (pc_full[:, 2] > z_range[0])
        if pc_colors is not None:
            pc_colors = pc_colors[z_mask]
        pc_full = pc_full[z_mask]
        pc_segments = {}
        if segmap is not None:
            if segmap_id:
                obj_instances = [segmap_id]
            else:
                obj_instances = torch.unique(segmap[segmap > 0]).tolist()
            for i in obj_instances:
                i_int = int(i)
                if skip_border_objects and not i_int == segmap_id:
                    coords = torch.nonzero(segmap == i_int, as_tuple=True)
                    obj_i_y, obj_i_x = coords[0], coords[1]
                    if (torch.any(obj_i_x < margin_px)
                            or torch.any(obj_i_x > segmap.shape[1] - margin_px)
                            or torch.any(obj_i_y < margin_px)
                            or torch.any(obj_i_y > segmap.shape[0] - margin_px)):
                        continue
                inst_mask = segmap == i_int
                pc_segment, _ = depth2pc(torch.where(inst_mask, depth, torch.zeros_like(depth)), K)
                seg_z_mask = (pc_segment[:, 2] < z_range[1]) & (pc_segment[:, 2] > z_range[0])
                pc_segments[i_int] = pc_segment[seg_z_mask]
        return pc_full, pc_segments, pc_colors

    def predict_scene_grasps_from_depth_K_and_2d_seg(self, _sess_unused, depth, segmap, K, z_range=(0.2, 1.8), local_regions=False, filter_grasps=False, segmap_id=0, skip_border_objects=False, margin_px=5, rgb=None, forward_passes=1, seed=None):
        pc_full, pc_segments, _ = self.extract_point_clouds(
            depth,
            K,
            segmap=segmap,
            segmap_id=segmap_id,
            skip_border_objects=skip_border_objects,
            margin_px=margin_px,
            z_range=z_range,
            rgb=rgb,
        )
        return self.predict_scene_grasps(
            pc_full,
            pc_segments,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes,
            seed=seed,
        )
