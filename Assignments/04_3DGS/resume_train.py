import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import cv2
import os
import sys

from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from data_utils import ColmapDataset

@dataclass
class TrainConfig:
    num_epochs: int = 200  # 总训练轮数
    batch_size: int = 1
    learning_rate: float = 0.01
    grad_clip: float = 1.0
    save_every: int = 20
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    debug_every: int = 1  # 每N轮保存一次调试图像
    debug_samples: int = 1  # 调试时保存的图像数量
    render_every: int = 50  # 每N轮导出一次debug_rendering视频

class GaussianTrainer:
    def __init__(
        self, 
        model: GaussianModel,
        renderer: GaussianRenderer,
        config: TrainConfig,
        device: torch.device,
        start_epoch: int = 0  # 记录起始轮数
    ):
        self.model = model.to(device)
        self.renderer = renderer.to(device)
        self.config = config
        self.device = device
        self.start_epoch = start_epoch  # 存储起始轮数
        
        # 初始化优化器，使用不同的学习率
        optable_params = [
            {'params': [self.model.positions], 'lr': 0.000016, "name": "xyz"},
            {'params': [self.model.colors], 'lr': 0.025, "name": "color"},
            {'params': [self.model.opacities], 'lr': 0.05, "name": "opacity"},
            {'params': [self.model.scales], 'lr': 0.005, "name": "scaling"},
            {'params': [self.model.rotations], 'lr': 0.001, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(optable_params, lr=0.001, eps=1e-15)
        
        # 创建检查点和日志目录
        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(config.log_dir).mkdir(exist_ok=True, parents=True)
        
        # 记录调试样本的索引
        self.debug_indices = None

    def save_debug_images(self, epoch: int, rendered_images: torch.Tensor, 
                         gt_images: torch.Tensor, image_paths: list):
        """
        保存对比图像，比较ground truth和渲染结果
        """
        # 将张量转换为numpy数组
        rendered = rendered_images.detach().cpu().numpy()
        gt = gt_images.detach().cpu().numpy()
        epoch_dir = Path(self.config.log_dir) / f"epoch_{epoch:06d}"  # 使用绝对轮数
        epoch_dir.mkdir(exist_ok=True)
        
        for b in range(rendered.shape[0]):
            base_name = Path(image_paths[b]).stem
            rendered_img = (rendered[b] * 255).clip(0, 255).astype(np.uint8)
            gt_img = (gt[b] * 255).clip(0, 255).astype(np.uint8)
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            comparison = np.concatenate([gt_img, rendered_img], axis=1)
            output_path = epoch_dir / f"{base_name}.png"
            cv2.imwrite(str(output_path), comparison)

    def save_checkpoint(self, epoch: int):
        """保存模型检查点，包含绝对轮数"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{epoch:06d}.pt"  # 更新文件名
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch} to {path}")

    def load_checkpoint(self, path: str) -> int:
        """加载模型检查点并返回加载的轮数"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {path} at epoch {loaded_epoch}")
        return loaded_epoch

    def visualize_rendering(self, dataset, save_vid_path: str, epoch: int, num_frames: int = 300):
        """
        从原始相机路径生成场景渲染的视频
            
        Args:
            dataset: ColmapDataset实例，包含相机参数
            save_vid_path: 保存视频的路径
            epoch: 当前轮数，用于在视频文件名中
            num_frames: 圆形路径中的帧数
        """
        print(f"Generating rendering visualization for epoch {epoch}...")
        
        # 获取示例K矩阵和图像尺寸
        sample = dataset[0]
        K = sample['K'].to(self.device)
        H, W = sample['image'].shape[:2]
        
        # 更新视频文件名以包含轮数
        vid_path_with_epoch = save_vid_path.replace(".mp4", f"_{epoch:06d}.mp4")
        
        # 初始化视频写入器
        out = cv2.VideoWriter(vid_path_with_epoch, cv2.VideoWriter_fourcc(*'mp4v'), 3, (W*2, H))
        
        # 获取高斯参数（只需计算一次）
        with torch.no_grad():
            gaussian_params = self.model()
        
        # 渲染帧
        for data_item in tqdm(dataset, desc=f"Rendering frames for epoch {epoch}"):
            # 将相机姿态转换为torch张量
            R_torch = data_item['R'].to(self.device)
            t_torch = data_item['t'].to(self.device).reshape(-1, 3)
            
            # 渲染图像
            with torch.no_grad():
                rendered_image = self.renderer(
                    means3D=gaussian_params['positions'],
                    covs3d=gaussian_params['covariance'],
                    colors=gaussian_params['colors'],
                    opacities=gaussian_params['opacities'],
                    K=K.squeeze(0),
                    R=R_torch.squeeze(0),
                    t=t_torch.squeeze(0),
                )
            
            # 转换为numpy并转换为BGR格式用于OpenCV
            frame = rendered_image.cpu().numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            ori_img = (data_item['image']*255).cpu().numpy().astype(np.uint8)
            vis = cv2.cvtColor(np.concatenate((ori_img, frame), axis=1), cv2.COLOR_RGB2BGR)
            # 写入帧
            out.write(vis)
        
        # 释放视频写入器
        out.release()
        print(f"Video saved to: {vid_path_with_epoch}")

    def train_step(self, batch: dict, in_train=True):
        """单次训练步骤"""
        # 获取批次数据并准备相机矩阵
        images = batch['image'].to(self.device)            # (B, H, W, 3)
        K = batch['K'].to(self.device)                     # (B, 3, 3)
        R = batch['R'].to(self.device)                     # (B, 3, 3)
        t = batch['t'].to(self.device).reshape(-1, 3)      # (B, 3)
        
        # 前向传播
        gaussian_params = self.model()
        rendered_images = self.renderer(
            means3D=gaussian_params['positions'],
            covs3d=gaussian_params['covariance'],
            colors=gaussian_params['colors'],
            opacities=gaussian_params['opacities'],
            K = K.squeeze(0),
            R = R.squeeze(0),
            t = t.squeeze(0),
        )
        rendered_images = rendered_images.unsqueeze(0)
        
        if not in_train:
            return rendered_images

        # 计算RGB损失
        loss = torch.abs(rendered_images - images).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.grad_clip
        )
        
        # 优化步骤
        self.optimizer.step()
        
        return loss.item(), rendered_images

    def train(self, train_loader: DataLoader):
        """主训练循环"""
        # 选择固定的调试索引
        if self.debug_indices is None:
            dataset_size = len(train_loader.dataset)
            self.debug_indices = np.random.choice(
                dataset_size, 
                min(self.config.debug_samples, dataset_size), 
                replace=False
            )
        
        for epoch in range(self.config.num_epochs):
            current_epoch = self.start_epoch + epoch + 1  # 当前轮数，从1开始
            
            # 新增：打印当前实际 epoch
            print(f"\n=== Starting Epoch {current_epoch} ===")  # 新增
            
            # 训练循环
            pbar = tqdm(train_loader, desc=f"Epoch {current_epoch}")
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(pbar):
                # 训练步骤
                loss, rendered_images = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # 更新进度条
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
            
            # 保存检查点
            if current_epoch % self.config.save_every == 0:
                self.save_checkpoint(current_epoch)
                
            # 每N轮保存调试图像
            if current_epoch % self.config.debug_every == 0:
                debug_batches = []
                for idx in self.debug_indices:
                    debug_batches.append(train_loader.dataset[idx])
                
                # 堆叠调试批次
                debug_batch = {
                    k: torch.stack([b[k] for b in debug_batches], 0) 
                    if torch.is_tensor(debug_batches[0][k])
                    else [b[k] for b in debug_batches]
                    for k in debug_batches[0].keys()
                }
                
                # 获取调试批次的渲染图像
                with torch.no_grad():
                    debug_rendered = self.train_step(debug_batch, in_train=False)
                
                # 保存调试图像
                self.save_debug_images(
                    epoch=current_epoch,
                    rendered_images=debug_rendered,
                    gt_images=debug_batch['image'],
                    image_paths=debug_batch['image_path']
                )
            
            # 每N轮导出debug_rendering视频
            if current_epoch % self.config.render_every == 0:
                vid_path = os.path.join(self.config.checkpoint_dir, f"debug_rendering.mp4")
                self.visualize_rendering(
                    dataset=train_loader.dataset, 
                    save_vid_path=vid_path, 
                    epoch=current_epoch
                )

    def visualize_final_rendering(self, dataset, save_vid_path: str, epoch: int, num_frames: int = 300):
        """
        创建最终的视频，包含当前轮数
        """
        self.visualize_rendering(dataset, save_vid_path, epoch, num_frames)

def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D Gaussian Splatting')
    
    # 数据路径
    parser.add_argument('--colmap_dir', type=str, required=True,
                      help='Directory containing COLMAP data (with sparse/0/ and images/)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    
    # 创建互斥组，确保 --total_epochs 和 --additional_epochs 只能选择一个
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--total_epochs', type=int, default=None,
                      help='总训练轮数。如果指定了此参数，则训练将持续到该轮数。')
    group.add_argument('--additional_epochs', type=int, default=None,
                      help='继续训练的轮数。如果指定了此参数，必须同时提供 --resume 参数。')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping value')
    
    # 调试参数
    parser.add_argument('--debug_every', type=int, default=1,
                      help='Save debug images every N epochs')
    parser.add_argument('--debug_samples', type=int, default=1,
                      help='Number of images to save for debugging')
    parser.add_argument('--render_every', type=int, default=50,
                      help='Export debug_rendering video every N epochs')  # 新增参数
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.additional_epochs is not None and args.resume is None:
        parser.error("--additional_epochs requires --resume.")
    
    return args

def main():
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化数据集
    dataset = ColmapDataset(args.colmap_dir)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 获取数据集中的图像尺寸
    sample = dataset[0]['image']
    H, W = sample.shape[:2]
    
    # 使用COLMAP点初始化模型
    model = GaussianModel(
        points3D_xyz=dataset.points3D_xyz,
        points3D_rgb=dataset.points3D_rgb
    )
    
    # 初始化渲染器
    renderer = GaussianRenderer(
        image_height=H,
        image_width=W
    )
    
    # 初始化训练配置
    config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=os.path.join(args.checkpoint_dir, "debug_images"),
        debug_every=args.debug_every,
        debug_samples=args.debug_samples,
        render_every=args.render_every  # 从参数中设置render_every
    )
    
    # 初始化训练器
    trainer = GaussianTrainer(model, renderer, config, device, start_epoch=0)
    
    # 设置训练轮数
    if args.total_epochs is not None:
        total_epochs = args.total_epochs
        print(f"将训练总轮数设置为 {total_epochs} 轮。")
    elif args.additional_epochs is not None:
        # 从检查点恢复训练
        print(f"Resuming from checkpoint: {args.resume}")
        loaded_epoch = trainer.load_checkpoint(args.resume)
        total_epochs = loaded_epoch + args.additional_epochs
        trainer.start_epoch = loaded_epoch
        print(f"将继续训练 {args.additional_epochs} 轮，从轮数 {loaded_epoch + 1} 到 {total_epochs} 轮。")
    else:
        print("必须指定 --total_epochs 或 --additional_epochs 参数。")
        sys.exit(1)
    
    # 更新config.num_epochs为需要训练的轮数
    if args.total_epochs is not None:
        config.num_epochs = total_epochs
    elif args.additional_epochs is not None:
        config.num_epochs = args.additional_epochs
    
    # 验证如果使用 --additional_epochs，确保 total_epochs > loaded_epoch
    if args.additional_epochs is not None:
        if total_epochs <= trainer.start_epoch:
            print(f"错误: 总训练轮数 {total_epochs} 必须大于已训练轮数 {trainer.start_epoch}。")
            sys.exit(1)
    
    # 计算本次会话要训练的轮数
    if args.total_epochs is not None:
        epochs_to_train = config.num_epochs
        print(f"开始从轮数 1 到 {epochs_to_train} 进行训练。")
    elif args.additional_epochs is not None:
        epochs_to_train = config.num_epochs
        print(f"继续训练，从轮数 {trainer.start_epoch +1} 到 {total_epochs}（共训练 {epochs_to_train} 轮）。")
    
    # 开始训练
    print("开始训练...")
    print(f"Training on {len(dataset)} images for {epochs_to_train} epochs.")
    print(f"Debug images will be saved every {config.debug_every} epochs.")
    print(f"Debug_rendering videos will be exported every {config.render_every} epochs.")
    print(f"Using {config.debug_samples} debug samples.")
    trainer.train(train_loader)
    print("Training completed!")
    
    # 导出最终的debug_rendering视频
    final_vid_path = os.path.join(config.checkpoint_dir, f"debug_rendering.mp4")
    final_epoch = trainer.start_epoch + config.num_epochs
    trainer.visualize_final_rendering(dataset, final_vid_path, final_epoch)
    print(f"Final debug_rendering video saved to: {final_vid_path}")

if __name__ == "__main__":
    main()
