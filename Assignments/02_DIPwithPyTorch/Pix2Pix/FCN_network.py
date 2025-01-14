import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super(FullyConvNetwork, self).__init__()
        
        # 编码器（卷积层）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),  # 输出通道数：64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  # 输出通道数：128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 输出通道数：256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),  # 输出通道数：512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),  # 输出通道数：1024
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # 解码器（反卷积层）
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),  # 输出通道数：512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1),  # 输出通道数：256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1),   # 输出通道数：128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),    # 输出通道数：64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),      # 输出通道数：3（RGB）
            nn.Tanh()  # 使用 Tanh 激活函数将输出值限制在 [-1, 1] 之间
        )

    def forward(self, x):
        # 编码器前向传播
        e1 = self.encoder1(x)  # 输出尺寸减半，通道数：64
        e2 = self.encoder2(e1) # 进一步减半，通道数：128
        e3 = self.encoder3(e2) # 进一步减半，通道数：256
        e4 = self.encoder4(e3) # 进一步减半，通道数：512
        e5 = self.encoder5(e4) # 进一步减半，通道数：1024

        # 解码器前向传播
        d1 = self.decoder1(e5)                # 上采样，通道数：512
        d1 = torch.cat((d1, e4), dim=1)        # 跳跃连接，与编码器第五层连接，拼接后通道数：512 + 512 = 1024

        d2 = self.decoder2(d1)                 # 上采样，通道数：256
        d2 = torch.cat((d2, e3), dim=1)        # 跳跃连接，与编码器第四层连接，拼接后通道数：256 + 256 = 512

        d3 = self.decoder3(d2)                 # 上采样，通道数：128
        d3 = torch.cat((d3, e2), dim=1)        # 跳跃连接，与编码器第三层连接，拼接后通道数：128 + 128 = 256

        d4 = self.decoder4(d3)                 # 上采样，通道数：64
        d4 = torch.cat((d4, e1), dim=1)        # 跳跃连接，与编码器第二层连接，拼接后通道数：64 + 64 = 128

        output = self.decoder5(d4)             # 最终上采样，恢复到原始尺寸，通道数：3

        return output