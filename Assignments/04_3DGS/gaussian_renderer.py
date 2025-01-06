import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # 预先计算像素坐标网格
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # 形状: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. 将点转换到相机空间
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. 在投影前获取深度以便正确排序和裁剪
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. 使用相机内参将点投影到屏幕空间
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # 4. 将协方差从世界空间转换到相机空间，然后转换到2D
        # 计算透视投影的雅可比矩阵
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        ### 修改: 计算雅可比矩阵时加入相机内参 K
        J_proj[:, 0, 0] = K[0, 0] / cam_points[:, 2]
        J_proj[:, 0, 2] = -K[0, 0] * cam_points[:, 0] / (cam_points[:, 2] ** 2)
        J_proj[:, 1, 1] = K[1, 1] / cam_points[:, 2]
        J_proj[:, 1, 2] = -K[1, 1] * cam_points[:, 1] / (cam_points[:, 2] ** 2)
        
        # 计算世界到相机的旋转矩阵的逆
        ### 修改: 使用旋转矩阵的转置作为逆矩阵
        R_inv = R.transpose(0, 1)  # (3, 3)
        
        # 将协方差从世界空间转换到相机空间
        # covs_cam = R @ covs3d @ R^T
        covs_cam = torch.matmul(R, covs3d)       # (N, 3, 3)
        covs_cam = torch.matmul(covs_cam, R_inv) # (N, 3, 3)

        # 投影到2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        return means2D, covs2D, depths


    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # 计算与均值的偏移 (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # 为数值稳定性在对角线上添加小的epsilon
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # 计算协方差矩阵的逆和行列式
        ### 修改: 添加高斯分布的归一化因子
        covs_inv = torch.inverse(covs2D)  # (N, 2, 2)
        det = torch.det(covs2D)  # (N,)
        norm_factor = 1. / (2 * np.pi * torch.sqrt(det)).unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        
        # 计算指数部分
        P = -0.5 * torch.einsum('nhwc,ncd,nhwd->nhw', dx, covs_inv, dx)  # (N, H, W)

        # 计算高斯值
        gaussian = norm_factor * torch.exp(P)  # (N, H, W)
        
        return gaussian


    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. 投影到2D，means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. 深度掩码
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. 按深度排序
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[indices]        # (N, 3)
        opacities = opacities[indices]  # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. 计算高斯值
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. 应用有效掩码
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha 组合设置
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. 计算权重
        ### 修改: 使用累积乘积计算权重，改进alpha组合
        alphas_shifted = torch.cat([
            torch.ones((1, self.H, self.W), device=alphas.device),
            1 - alphas + 1e-10  # 为数值稳定性添加epsilon
        ], dim=0)  # (N+1, H, W)
        transmittance = torch.cumprod(alphas_shifted, dim=0)[:-1]  # (N, H, W)
        weights = alphas * transmittance  # (N, H, W)
        
        # 8. 最终渲染
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
