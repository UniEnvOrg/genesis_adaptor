from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from unienv_genesis_refpolicy.third_party.contact_graspnet_pytorch.pointnet2_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
)


class ContactGraspnet(nn.Module):
    """Minimal inference-only Contact-GraspNet PyTorch model."""

    def __init__(self, global_config, device):
        super().__init__()
        self.device = device
        self.global_config = global_config
        self.model_config = global_config["MODEL"]
        self.data_config = global_config["DATA"]

        radius_list_0 = self.model_config["pointnet_sa_modules_msg"][0]["radius_list"]
        radius_list_1 = self.model_config["pointnet_sa_modules_msg"][1]["radius_list"]
        radius_list_2 = self.model_config["pointnet_sa_modules_msg"][2]["radius_list"]
        nsample_list_0 = self.model_config["pointnet_sa_modules_msg"][0]["nsample_list"]
        nsample_list_1 = self.model_config["pointnet_sa_modules_msg"][1]["nsample_list"]
        nsample_list_2 = self.model_config["pointnet_sa_modules_msg"][2]["nsample_list"]
        mlp_list_0 = self.model_config["pointnet_sa_modules_msg"][0]["mlp_list"]
        mlp_list_1 = self.model_config["pointnet_sa_modules_msg"][1]["mlp_list"]
        mlp_list_2 = self.model_config["pointnet_sa_modules_msg"][2]["mlp_list"]
        npoint_0 = self.model_config["pointnet_sa_modules_msg"][0]["npoint"]
        npoint_1 = self.model_config["pointnet_sa_modules_msg"][1]["npoint"]
        npoint_2 = self.model_config["pointnet_sa_modules_msg"][2]["npoint"]
        fp_mlp_0 = self.model_config["pointnet_fp_modules"][0]["mlp"]
        fp_mlp_1 = self.model_config["pointnet_fp_modules"][1]["mlp"]
        fp_mlp_2 = self.model_config["pointnet_fp_modules"][2]["mlp"]
        sa_mlp = self.model_config["pointnet_sa_module"]["mlp"]
        sa_group_all = self.model_config["pointnet_sa_module"]["group_all"]

        self.input_normals = self.data_config["input_normals"]
        self.offset_bins = self.data_config["labels"]["offset_bins"]
        additional_channel = 3 if self.input_normals else 0

        if not self.model_config.get("asymmetric_model", False):
            raise NotImplementedError("Only asymmetric_model=True is supported")

        self.sa1 = PointNetSetAbstractionMsg(
            npoint=npoint_0,
            radius_list=radius_list_0,
            nsample_list=nsample_list_0,
            in_channel=3 + additional_channel,
            mlp_list=mlp_list_0,
        )
        sa1_out_channels = sum([mlp_list_0[i][-1] for i in range(len(mlp_list_0))])
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=npoint_1,
            radius_list=radius_list_1,
            nsample_list=nsample_list_1,
            in_channel=sa1_out_channels,
            mlp_list=mlp_list_1,
        )
        sa2_out_channels = sum([mlp_list_1[i][-1] for i in range(len(mlp_list_1))])
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=npoint_2,
            radius_list=radius_list_2,
            nsample_list=nsample_list_2,
            in_channel=sa2_out_channels,
            mlp_list=mlp_list_2,
        )
        sa3_out_channels = sum([mlp_list_2[i][-1] for i in range(len(mlp_list_2))])
        self.sa4 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=3 + sa3_out_channels,
            mlp=sa_mlp,
            group_all=sa_group_all,
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=sa_mlp[-1] + sa3_out_channels, mlp=fp_mlp_2)
        self.fp2 = PointNetFeaturePropagation(in_channel=fp_mlp_2[-1] + sa2_out_channels, mlp=fp_mlp_1)
        self.fp1 = PointNetFeaturePropagation(in_channel=fp_mlp_1[-1] + sa1_out_channels, mlp=fp_mlp_0)

        if self.model_config.get("joint_heads", False):
            raise NotImplementedError("joint_heads=True is not supported")

        self.grasp_dir_head = nn.Sequential(
            nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 3, 1, padding=0),
        )
        self.grasp_approach_head = nn.Sequential(
            nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 3, 1, padding=0),
        )
        if self.model_config.get("dir_vec_length_offset", False):
            raise NotImplementedError("dir_vec_length_offset=True is not supported")
        if not self.model_config.get("bin_offsets", False):
            raise NotImplementedError("Only bin_offsets=True is supported")
        self.grasp_offset_head = nn.Sequential(
            nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, len(self.offset_bins) - 1, 1, padding=0),
        )
        self.binary_seg_head = nn.Sequential(
            nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, 1, padding=0),
        )

    def forward(self, point_cloud):
        point_cloud = torch.transpose(point_cloud, 1, 2)
        l0_xyz = point_cloud[:, :3, :]
        l0_points = point_cloud[:, 3:6, :] if self.input_normals else l0_xyz.clone()

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = l1_points
        pred_points = l1_xyz

        grasp_dir_head = self.grasp_dir_head(l0_points)
        grasp_dir_head_normed = F.normalize(grasp_dir_head, p=2, dim=1)
        approach_dir_head = self.grasp_approach_head(l0_points)
        dot_product = torch.sum(grasp_dir_head_normed * approach_dir_head, dim=1, keepdim=True)
        projection = dot_product * grasp_dir_head_normed
        approach_dir_head_orthog = F.normalize(approach_dir_head - projection, p=2, dim=1)
        grasp_offset_head = self.grasp_offset_head(l0_points)
        binary_seg_head = self.binary_seg_head(l0_points)

        torch_bin_vals = self.get_bin_vals()
        argmax_indices = torch.argmax(grasp_offset_head, dim=1, keepdim=True)
        offset_bin_pred_vals = torch_bin_vals[argmax_indices]
        pred_grasps_cam = self.build_6d_grasp(
            approach_dir_head_orthog.permute(0, 2, 1),
            grasp_dir_head_normed.permute(0, 2, 1),
            pred_points.permute(0, 2, 1),
            offset_bin_pred_vals.permute(0, 2, 1),
            gripper_depth=0.1034,
        )
        pred_scores = torch.sigmoid(binary_seg_head).permute(0, 2, 1)
        pred_points = pred_points.permute(0, 2, 1)
        offset_pred = offset_bin_pred_vals
        return {
            "pred_grasps_cam": pred_grasps_cam,
            "pred_scores": pred_scores,
            "pred_points": pred_points,
            "offset_pred": offset_pred,
            "grasp_offset_head": grasp_offset_head,
        }

    def get_bin_vals(self):
        bins_bounds = self.data_config["labels"]["offset_bins"]
        bins_bounds = torch.tensor(bins_bounds, dtype=torch.float32, device=self.device)
        if self.global_config["TEST"]["bin_vals"] == "max":
            bin_vals = (bins_bounds[1:] + bins_bounds[:-1]) / 2
            bin_vals[-1] = bins_bounds[-1]
        elif self.global_config["TEST"]["bin_vals"] == "mean":
            bin_vals = bins_bounds[1:]
        else:
            raise NotImplementedError
        if not self.global_config["TEST"]["allow_zero_margin"]:
            bin_vals = torch.minimum(
                bin_vals,
                torch.tensor(
                    self.global_config["DATA"]["gripper_width"] - self.global_config["TEST"]["extra_opening"],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        return bin_vals

    def build_6d_grasp(self, approach_dirs, base_dirs, contact_pts, thickness, gripper_depth=0.1034):
        grasp_R = torch.stack([base_dirs, torch.cross(approach_dirs, base_dirs), approach_dirs], dim=3)
        grasp_t = contact_pts + (thickness / 2) * base_dirs - gripper_depth * approach_dirs
        grasp_t = grasp_t.unsqueeze(3)
        ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32, device=self.device)
        zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32, device=self.device)
        homog_vec = torch.cat([zeros, ones], dim=3)
        return torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)
