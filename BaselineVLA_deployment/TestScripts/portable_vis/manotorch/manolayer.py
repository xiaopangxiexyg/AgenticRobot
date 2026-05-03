import os
import warnings
from collections import namedtuple
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from .utils.geometry import axis_angle_to_matrix, quaternion_to_axis_angle, quaternion_to_matrix

# Removed external project dependencies (PathConfig, teleop, etc.)

MANO_KEYPOINTS_CONNECTIONS = [
    (0, 5), (5, 6), (6, 7), (7, 8),    # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]
MANO_KEYPOINTS_CONNECTIONS_THUMB = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # 大拇指
]
MANOOutput = namedtuple(
    "MANOOutput",
    [
        "verts",
        "joints",
        "center_idx",
        "center_joint",
        "full_poses",
        "betas",
        "transforms_abs",
    ],
)
MANOOutput.__new__.__defaults__ = (None,) * len(MANOOutput._fields)


def _as_numpy_array(value):
    if hasattr(value, "r"):
        return np.asarray(value.r)
    return np.asarray(value)


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False
    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res

def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation matrix to an axis-angle representation.
    (Corrected and robust version)
    
    Args:
        matrix (torch.Tensor): Rotation matrix of shape (..., 3, 3).
    Returns:
        torch.Tensor: Axis-angle vector of shape (..., 3).
    """
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("Input tensor must have shape (..., 3, 3)")
    
    batch_shape = matrix.shape[:-2]
    
    # --- Basic conversion from matrix to angle and unnormalized axis ---
    diags = torch.diagonal(matrix, offset=0, dim1=-2, dim2=-1)
    trace = torch.sum(diags, dim=-1)
    cos_angle = (trace - 1) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)

    # --- Handle edge cases: angle is close to 0 or pi ---
    mask_close_to_0 = angle < 1e-6
    mask_close_to_pi = torch.abs(angle - np.pi) < 1e-6

    # Case 1: Angle is not near 0 or pi (standard case)
    axis_unnormalized = torch.stack([
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1],
    ], dim=-1)

    # Case 2: Angle is close to pi
    # Find eigenvector for eigenvalue 1 of R. This is a column of R+I.
    # We find the column of R+I with the largest norm.
    R_plus_I = matrix + torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    max_diag_idx = torch.argmax(torch.diagonal(R_plus_I, offset=0, dim1=-2, dim2=-1), dim=-1)
    
    # Flatten batch dimensions for easier, robust indexing
    num_matrices = R_plus_I.reshape(-1, 3, 3).shape[0]
    flat_R_plus_I = R_plus_I.reshape(num_matrices, 3, 3)
    flat_max_diag_idx = max_diag_idx.reshape(num_matrices)
    
    # Select the correct column for each matrix in the batch
    arange_idx = torch.arange(num_matrices, device=matrix.device)
    axis_at_pi_flat = flat_R_plus_I[arange_idx, :, flat_max_diag_idx]
    
    # Reshape back to the original batch shape
    axis_at_pi = axis_at_pi_flat.view(*batch_shape, 3)

    # --- Combine and normalize ---
    axis = torch.where(
        mask_close_to_pi[..., None], 
        axis_at_pi, 
        axis_unnormalized
    )

    # Normalize the axis
    axis = torch.nn.functional.normalize(axis, p=2, dim=-1, eps=1e-8)
    
    # If angle was close to 0, axis is undefined; use a default like [1, 0, 0]
    # The final result will be zero anyway due to multiplication by angle.
    axis = torch.where(
        mask_close_to_0[..., None], 
        torch.tensor([1.0, 0.0, 0.0], device=matrix.device, dtype=matrix.dtype), 
        axis
    )

    return axis * angle[..., None]
def robust_normalize(v, eps=1e-8):
    """带epsilon的向量归一化，防止除以零。"""
    v_mag = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / (v_mag + eps)

class ManoLayer(torch.nn.Module):

    def __init__(
        self,
        rot_mode: str = "axisang",
        side: str = "right",
        center_idx: Optional[int] = None,
        mano_assets_root: str = None, # Updated default to None
        use_pca: bool = False,
        flat_hand_mean: bool = True,  # Only used in pca mode
        ncomps: int = 15,  # Only used in pca mode
        device = "cuda",
        **kargs,
    ):
        super().__init__()
        self.center_idx = center_idx
        self.rot_mode = rot_mode
        self.side = side
        self.use_pca = use_pca
        if mano_assets_root is None:
             current_dir = os.path.dirname(os.path.abspath(__file__))
             portable_root = os.path.dirname(current_dir)
             mano_assets_root = os.path.join(portable_root, "assets", "mano")
             
        self.mano_assets_root = mano_assets_root
        self.flat_hand_mean = flat_hand_mean
        self.ncomps = ncomps if use_pca else -1

        if rot_mode == "axisang":
            self.rot_dim = 3
        elif rot_mode == "quat":
            self.rot_dim = 4
            if use_pca == True or flat_hand_mean == False:
                warnings.warn("Quat mode doesn't support PCA pose or non flat_hand_mean !")
        else:
            raise NotImplementedError(f"Unrecognized rotation mode, expect [pca|axisang|quat], got {rot_mode}")

        # load model according to side flag
        mano_assets_path = os.path.join(mano_assets_root, "models", f"MANO_{side.upper()}.pkl")  # eg.  MANO_RIGHT.pkl
        assert os.path.isfile(
            mano_assets_path), f"Can not find MANO assets {mano_assets_path}, please follow steps in README.md"

        # parse and register stuff
        self.device = device
        smpl_data = ready_arguments(mano_assets_path)
        self.register_buffer("th_betas", torch.Tensor(_as_numpy_array(smpl_data["betas"])).unsqueeze(0).to(device))
        self.register_buffer("th_shapedirs", torch.Tensor(_as_numpy_array(smpl_data["shapedirs"])).to(device))
        self.register_buffer("th_posedirs", torch.Tensor(_as_numpy_array(smpl_data["posedirs"])).to(device))
        self.register_buffer("th_v_template", torch.Tensor(_as_numpy_array(smpl_data["v_template"])).unsqueeze(0).to(device))
        self.register_buffer("th_J_regressor", torch.Tensor(np.asarray(smpl_data["J_regressor"].toarray())).to(device))
        self.register_buffer("th_weights", torch.Tensor(_as_numpy_array(smpl_data["weights"])).to(device))
        self.register_buffer("th_faces", torch.Tensor(np.array(smpl_data["f"]).astype(np.int32)).long().to(device))

        kintree_table = smpl_data["kintree_table"]
        self.kintree_parents = list(kintree_table[0].tolist())
        hands_components = smpl_data["hands_components"]

        if rot_mode == "axisang":
            hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data["hands_mean"]
            hands_mean = hands_mean.copy()
            hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
            self.register_buffer("th_hands_mean", hands_mean)

        if rot_mode == "axisang" and use_pca == True:
            selected_components = hands_components[:ncomps]
            selected_components = torch.Tensor(selected_components)
            self.register_buffer("th_selected_comps", selected_components)

        self.register_buffer("parents_tensor", torch.tensor(self.kintree_parents, dtype=torch.long), persistent=False)

        # 为构建局部坐标系，定义每个关节的子关节
        # 注意: 指尖 (16-20) 分别是 (15, 3, 6, 12, 9) 的子关节
        child_map = {
            0: 1, 1: 2, 2: 3, 3: 17, 4: 5, 5: 6, 6: 18,
            7: 8, 8: 9, 9: 20, 10: 11, 11: 12, 12: 19, 13: 14, 14: 15, 15: 16
        }
        # 我们只需要16个骨骼关节的子关节
        child_indices = [child_map[i] for i in range(16)]
        self.register_buffer("child_map_tensor", torch.tensor(child_indices, dtype=torch.long), persistent=False)

        # 预计算非根关节 (1-15) 的索引、它们的父关节索引和子关节索引
        non_root_indices = torch.arange(1, 16, device=self.device)
        self.register_buffer("non_root_indices", non_root_indices, persistent=False)
        self.register_buffer("non_root_parents", torch.tensor([self.kintree_parents[i] for i in range(1, 16)], dtype=torch.long), persistent=False)
        self.register_buffer("non_root_children", torch.tensor([child_indices[i] for i in range(1, 16)], dtype=torch.long), persistent=False)

        # 预计算从 "SNAP" 顺序到 "MANO" 顺序的映射
        snap_to_mano_map = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        mano_to_snap_map = [0] * 21
        for i, j in enumerate(snap_to_mano_map):
            mano_to_snap_map[j] = i
        self.register_buffer("mano_to_snap_map", torch.tensor(mano_to_snap_map, dtype=torch.long, device=self.device), persistent=False)

        
    def joints_to_mano_parameters(self, joints: torch.Tensor, betas: Optional[torch.Tensor] = None):
        """
        [已优化 V2 - 稳定版] 通过直接的逆运动学计算，从21个关节点坐标还原MANO姿态参数（轴角）。
        此版本使用手腕法线作为参考，消除了所有共线性导致的180度翻转问题。
        """
        batch_size = joints.shape[0]
        device = joints.device
        self.mano_to_snap_map.to(device)

        # --- 步骤 1: 获取模板骨架 (A-Pose) 的关节点坐标 J (已优化) ---
        if betas is None:
            betas = self.th_betas.repeat(batch_size, 1).to(device)
        
        shape_blend = torch.matmul(self.th_shapedirs, betas.t()).permute(2, 0, 1)
        v_shaped = self.th_v_template + shape_blend
        
        j_template_16 = torch.matmul(self.th_J_regressor, v_shaped)
        
        tip_indices = [745, 317, 444, 556, 673] if self.side == "right" else [745, 317, 445, 556, 673]
        j_template_tips = v_shaped[:, tip_indices]
        j_template = torch.cat([j_template_16, j_template_tips], dim=1)

        # print(joints)
        # print(self.mano_to_snap_map)
        # breakpoint()

        # --- 步骤 2: 使用预计算的映射来重排输入的关节点 ---
        mano_joints = joints[:, self.mano_to_snap_map, :]

        # --- 步骤 3: 全局旋转计算 (使用稳定逻辑) ---
        R_global = torch.empty(batch_size, 16, 3, 3, device=device, dtype=joints.dtype)

        # -- (不变) 特殊处理手腕 (关节 0) --
        # 我们需要这个坐标系来获取手掌法线
        y0_target = robust_normalize(mano_joints[:, 1, :] - mano_joints[:, 0, :])
        temp_vec_target = robust_normalize(mano_joints[:, 7, :] - mano_joints[:, 0, :])
        x0_target = robust_normalize(torch.cross(y0_target, temp_vec_target, dim=-1))
        z0_target = robust_normalize(torch.cross(x0_target, y0_target, dim=-1)) # 这是手掌法线
        TargetFrame_0 = torch.stack([x0_target, y0_target, z0_target], dim=-1)

        y0_template = robust_normalize(j_template[:, 1, :] - j_template[:, 0, :])
        temp_vec_template = robust_normalize(j_template[:, 7, :] - j_template[:, 0, :])
        x0_template = robust_normalize(torch.cross(y0_template, temp_vec_template, dim=-1))
        z0_template = robust_normalize(torch.cross(x0_template, y0_template, dim=-1)) # 模板手掌法线
        TemplateFrame_0 = torch.stack([x0_template, y0_template, z0_template], dim=-1)
            
        R_global[:, 0] = torch.bmm(TargetFrame_0, TemplateFrame_0.transpose(1, 2))

        # -- 一次性计算其余15个关节 --
        
        # 提取所有相关关节点的坐标 (B, 15, 3)
        k_joints_target = mano_joints[:, self.non_root_indices, :]
        c_joints_target = mano_joints[:, self.non_root_children, :]
        
        k_joints_template = j_template[:, self.non_root_indices, :]
        c_joints_template = j_template[:, self.non_root_children, :]
        
        # ==============================================================================
        # ==== FIX: 使用稳定的、基于手腕法线的坐标系构建逻辑 ====
        # ==============================================================================

        # 1. 向量化地构建目标坐标系
        # Y 轴 (骨骼方向) (B, 15, 3)
        y_target = robust_normalize(c_joints_target - k_joints_target)
        
        # 获取手腕(Palm)的 Z 轴 (法线) (B, 3)
        palm_z_target = TargetFrame_0[:, :, 2]
        # 将其扩展以匹配 15 个关节 (B, 15, 3)
        palm_z_target_expanded = palm_z_target.unsqueeze(1).expand(-1, 15, -1)

        # X 轴 (弯曲轴) = Y 轴 x 手掌法线 (B, 15, 3)
        x_target = robust_normalize(torch.cross(y_target, palm_z_target_expanded, dim=-1))
        
        # Z 轴 (实际的骨骼 Z 轴) = X 轴 x Y 轴 (B, 15, 3)
        z_target = robust_normalize(torch.cross(x_target, y_target, dim=-1))
        
        TargetFrame_k = torch.stack([x_target, y_target, z_target], dim=-1) # (B, 15, 3, 3)

        # 2. 向量化地构建模板坐标系 (使用完全相同的逻辑)
        y_template = robust_normalize(c_joints_template - k_joints_template)
        palm_z_template = TemplateFrame_0[:, :, 2]
        palm_z_template_expanded = palm_z_template.unsqueeze(1).expand(-1, 15, -1)
        
        x_template = robust_normalize(torch.cross(y_template, palm_z_template_expanded, dim=-1))
        z_template = robust_normalize(torch.cross(x_template, y_template, dim=-1))
        
        TemplateFrame_k = torch.stack([x_template, y_template, z_template], dim=-1) # (B, 15, 3, 3)

        # ==============================================================================

        # (不变) 一次性计算所有15个全局旋转
        R_global[:, 1:] = torch.matmul(TargetFrame_k, TemplateFrame_k.transpose(-1, -2))

        # --- 步骤 4: 局部旋转计算 (向量化) ---
        R_local = torch.empty_like(R_global)
        R_local[:, 0] = R_global[:, 0]

        # 提取所有父关节的全局旋转矩阵 (B, 15, 3, 3)
        R_global_parents = R_global[:, self.non_root_parents]
        
        # 一次性计算所有15个局部旋转: R_local_k = inv(R_global_parent_k) @ R_global_k
        R_local[:, 1:] = torch.matmul(R_global_parents.transpose(-1, -2), R_global[:, 1:])
        
        # --- 步骤 5: 将局部旋转矩阵转换为轴角表示 (本身已是向量化) ---
        axis_angle = matrix_to_axis_angle(R_local.reshape(-1, 3, 3))
        # print(axis_angle.shape)
        # axis_angle[...,3,:] = axis_angle[...,4,:]
        # axis_angle[...,4,:] = axis_angle[...,5,:]
        # breakpoint()
        full_poses = axis_angle.view(batch_size, 16 * 3)
            
        return {
            "full_poses": full_poses,
            "betas": betas
        }
    def rotation_by_axisang(self, pose_coeffs):
        batch_size = pose_coeffs.shape[0]
        hand_pose_coeffs = pose_coeffs[:, self.rot_dim:]
        root_pose_coeffs = pose_coeffs[:, :self.rot_dim]
        if self.use_pca:
            full_hand_pose = hand_pose_coeffs.mm(self.th_selected_comps)
        else:
            full_hand_pose = hand_pose_coeffs

        # Concatenate back global rot
        self.th_hands_mean = self.th_hands_mean.to(pose_coeffs.device)  # Ensure hands mean is on the same device
        # print(self.th_hands_mean.device, full_hand_pose.device)
        full_poses = torch.cat([root_pose_coeffs, self.th_hands_mean + full_hand_pose], 1)

        pose_vec_reshaped = full_poses.contiguous().view(-1, 3)  # (B x N, 3)
        rot_mats = axis_angle_to_matrix(pose_vec_reshaped)  # (B x N, 3, 3)
        # rot_mats = lietorch.SO3.exp(pose_vec_reshaped).matrix()[..., :3, :3]  # (B x N, 3, 3)
        full_rots = rot_mats.view(batch_size, 16, 3, 3)
        rotation_blob = {"full_rots": full_rots, "full_poses": full_poses}
        return rotation_blob

    def rotation_by_quaternion(self, pose_coeffs):
        batch_size = pose_coeffs.shape[0]
        full_quat_poses = pose_coeffs.view((batch_size, 16, 4))  # [B. 16, 4]
        full_rots = quaternion_to_matrix(full_quat_poses)  # [B, 16, 3, 3]
        full_poses = quaternion_to_axis_angle(full_quat_poses).reshape(batch_size, -1)  # [B, 16 x 3]

        rotation_blob = {"full_rots": full_rots, "full_poses": full_poses}
        return rotation_blob

    def skinning_layer(self, full_rots: torch.Tensor, betas: Optional[torch.Tensor]):
        full_rots = full_rots.to(torch.float32)
        betas = betas.to(torch.float32) if betas is not None else None
        batch_size = full_rots.shape[0]
        n_rot = int(full_rots.shape[1])  # 16

        root_rot = full_rots[:, 0, :, :]  # (B, 3, 3)
        hand_rot = full_rots[:, 1:, :, :]  # (B, 15, 3, 3)
        # Full axis angle representation with root joint

        # ============== Shape Blend Shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # $ B_S = \sum_{n=1}^{|\arrow{\beta}|} \beta_n \mathbf{S}_n $  #Eq.4 in MANO
        _betas = self.th_betas if betas is None else betas
        # print(f"betas device: {_betas.device}, shapedirs device: {self.th_shapedirs.device}")
        B_S = torch.matmul(self.th_shapedirs, _betas.transpose(1, 0)).permute(2, 0, 1)  # (?, 778, 3), ? = 1, or B

        # $ \mathcal{J}(\bar{\mathbf{T}} + B_S)$ # Eq.10 in SMPL
        J = torch.matmul(self.th_J_regressor, (self.th_v_template + B_S))  # (?, 16, 3)
        if betas is None:
            J = J.repeat(batch_size, 1, 1)  # (B, 16, 3)

        # ============== Pose Blender Shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        flat_rot = torch.eye(3, dtype=full_rots.dtype, device=full_rots.device)  # (3, 3)
        flat_rot = flat_rot.view(1, 1, 3, 3).repeat(batch_size, hand_rot.shape[1], 1, 1)  # (B, 15, 3, 3)

        # $ R_n (\arrow{\theta}) -  R_n (\arrow{\theta}^{*}) $
        rot_minus_mean_flat = (hand_rot - flat_rot).reshape(batch_size, hand_rot.shape[1] * 9)  # (B, 15 x 9)

        # $ B_P = \sum_{n=1}^{9K} (R_n (\arrow{\theta}) -  R_n (\arrow{\theta}^{*})) * \mathbf{P}_n $  #Eq.3 in MANO
        B_P = torch.matmul(self.th_posedirs, rot_minus_mean_flat.transpose(0, 1)).permute(2, 0, 1)  # (B, 778, 3)

        # $ T_P =\bar{\mathbf{T}} + B_S + B_P $ # Eq.2 in MANO
        T_P = self.th_v_template + B_S + B_P

        # ============== Constructing $ G_{k} $ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Global rigid transformation
        root_j = J[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_transf = th_with_zeros(torch.cat([root_rot, root_j], 2))

        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = hand_rot[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = hand_rot[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = hand_rot[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = J[:, lev1_idxs]
        lev2_j = J[:, lev2_idxs]
        lev3_j = J[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_transf.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_transf.unsqueeze(1).repeat(1, 5, 1, 1).view(root_transf.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(hand_rot.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(hand_rot.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(hand_rot.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]

        # Eq. 4 in SMPL
        G_k = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_transf_global = G_k

        # ============== Constructing $ G^{\prime}_{k} $ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        joint_js = torch.cat([J, J.new_zeros(batch_size, 16, 1)], 2)
        tmp2 = torch.matmul(G_k, joint_js.unsqueeze(3))
        G_prime_k = (G_k - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        # ============== Finally, blender skinning >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # we define $ T = w_{k, i} * G^{\prime}_k $
        T = torch.matmul(G_prime_k, self.th_weights.transpose(0, 1))  # (B, 4, 4, 778)

        T_P_homo = torch.cat(
            [T_P.transpose(2, 1),
             torch.ones((batch_size, 1, B_P.shape[1]), dtype=T.dtype, device=T.device)], dim=1)
        T_P_homo = T_P_homo.unsqueeze(1)  # (B, 1, 4, 778)

        # Eq. 7 in SMPL
        # Theorem: A \cdot B = (A * B^{T}).sum(1) # A is a matrix, B is a vector
        verts = (T * T_P_homo).sum(2).transpose(2, 1)  # (B, 778, 4)
        joints = th_transf_global[:, :, :3, 3]  # (B, 16, 3)
        verts = verts[:, :, :3]  # (B, 778, 3)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == "right":
            tips = verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = verts[:, [745, 317, 445, 556, 673]]

        joints = torch.cat([joints, tips], 1)

        # ** original MANO joint order (right hand)
        #                16-15-14-13-\
        #                             \
        #          17 --3 --2 --1------0
        #        18 --6 --5 --4-------/
        #        19 -12 -11 --10-----/
        #          20 --9 --8 --7---/

        # Reorder joints to match SNAP definition
        joints = joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        if self.center_idx is not None:
            center_joint = joints[:, self.center_idx].unsqueeze(1)
        else:  # dummy center joint (B, 1, 3)
            center_joint = torch.zeros_like(joints[:, 0].unsqueeze(1))

        # apply center shift on verts and joints
        joints = joints - center_joint
        verts = verts - center_joint

        # apply center shift on global
        global_rot = th_transf_global[:, :, :3, :3]  # (B, 16, 3, 3)
        global_tsl = th_transf_global[:, :, :3, 3:]  # (B, 16, 3, 1)
        global_tsl = global_tsl - center_joint.unsqueeze(-1)  # (B, [16], 3, 1)
        global_transf = torch.cat([global_rot, global_tsl], dim=3)  # (B, 16, 3, 4)
        global_transf = th_with_zeros(global_transf.view(-1, 3, 4))
        global_transf = global_transf.view(batch_size, 16, 4, 4)

        skinning_blob = {
            "verts": verts,
            "joints": joints,
            "center_joint": center_joint,
            "transforms_abs": global_transf,
            "betas": _betas,
        }
        return skinning_blob


    def forward(self, pose_coeffs: torch.Tensor, betas: Optional[torch.Tensor] = None,  **kwargs):
        if self.rot_mode == "axisang":
            rot_blob = self.rotation_by_axisang(pose_coeffs)
        elif self.rot_mode == "quat":
            rot_blob = self.rotation_by_quaternion(pose_coeffs)

        full_rots = rot_blob["full_rots"].to(self.device)  # TENSOR
        if betas is not None:
            betas = betas.to(self.device)
        else:
            betas = self.th_betas.to(self.device)
        # print(f"betas device: {betas.device}, full_rots device: {full_rots.device}")
        skinning_blob = self.skinning_layer(full_rots, betas)
        output = MANOOutput(
            verts=skinning_blob["verts"],
            joints=skinning_blob["joints"],
            center_idx=self.center_idx,
            center_joint=skinning_blob["center_joint"],
            full_poses=rot_blob["full_poses"],
            betas=skinning_blob["betas"],
            transforms_abs=skinning_blob["transforms_abs"],
        )
        return output

    def get_rotation_center(self, betas: Optional[torch.Tensor] = None):
        """

        V = MANO(theta, beta)

        Then we apply a rotation R on the vertices V
        V_1 = R @ V

        or, we can apply a rotation R on the global components of theta: first 3 elements of the theta
        theta' = CONCAT( SO3.log(R @ SO3.exp(theta[:3])), theta[3:] )

        V_2 = MANO(theta', beta)

        No doubt that, V_1 != V_2
        we found V_1 = V_2 + t, the t is an unknown translation offset

        Directly apply R on V would rotate V w.r.t the rotation center at V's [0,0,0] coordinate.
        However, apply R on the theta[:3] would cause the vertices rotate w.r.t to a rotation center at
        a non-zero, soley beta-sepcified center, C

        In other word, apply any disturb on the theta[:3] would not change the C's coordinates.
        the following code describe how we acquire the rotation center C

        This function will be called at artiboost/utils/refineunit.py in our upcoming work ArtiBoost
        """

        if betas is None:
            betas = self.th_betas

        batch_size = betas.shape[0]
        if self.center_idx is not None:
            return torch.zeros((batch_size, 3), device=betas.device)

        # $ B_S = \sum_{n=1}^{|\arrow{\beta}|} \beta_n \mathbf{S}_n $  #Eq.4 in MANO
        B_S = torch.matmul(self.th_shapedirs, betas.transpose(1, 0)).permute(2, 0, 1)

        # $ \mathcal{J}(\bar{\mathbf{T}} + B_S)$ # Eq.10 in SMPL
        J = torch.matmul(self.th_J_regressor, (self.th_v_template + B_S))  # (B, 16, 3)

        root_rotation_center = J[:, 0, :].contiguous().view(-1, 3)
        return root_rotation_center

    def get_mano_closed_faces(self):
        """
        The default MANO mesh is "open" at the wrist. By adding additional faces, the hand mesh is closed,
        which looks much better.
        https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py
        """
        close_faces = torch.Tensor([
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ])
        th_closed_faces = torch.cat([self.th_faces.clone().detach().cpu(), close_faces.long()])
        # Indices of faces added during closing --> should be ignored as they match the wrist
        # part of the hand, which is not an external surface of the human

        # Valid because added closed faces are at the end
        hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]
        return th_closed_faces  # , hand_ignore_faces
