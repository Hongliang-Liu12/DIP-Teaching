import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        # 第二层卷积，输入8通道，输出16通道
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 第三层卷积，输入16通道，输出32通道
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        # 第一个反卷积层，输入32通道，输出16通道
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # 第二个反卷积层，输入16通道，输出8通道
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        # 第三个反卷积层，输入8通道，输出3通道（恢复到RGB）
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 输出RGB图像，确保值在[0, 1]之间
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)  # 输出大小: [batch_size, 8, H/2, W/2]
        x2 = self.conv2(x1)  # 输出大小: [batch_size, 16, H/4, W/4]
        x3 = self.conv3(x2)  # 输出大小: [batch_size, 32, H/8, W/8]        
        # Decoder forward pass
        x4 = self.deconv1(x3)  # 输出大小: [batch_size, 16, H/4, W/4]
        x5 = self.deconv2(x4)  # 输出大小: [batch_size, 8, H/2, W/2]
        ### FILL: encoder-decoder forward pass

        output = self.deconv3(x5)  # 输出大小: [batch_size, 3, H, W]      
        
        return output
    