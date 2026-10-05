import torch
import torch.nn as nn

# Helper block for downsampling
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

# Helper block for upsampling
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

# 1. The Generator (U-Net)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownsampleBlock(3, 64, norm=False) # 128x128
        self.down2 = DownsampleBlock(64, 128)          # 64x64
        self.down3 = DownsampleBlock(128, 256)         # 32x32
        
        self.up1 = UpsampleBlock(256, 128)             # 64x64
        self.up2 = UpsampleBlock(128 + 128, 64)        # 128x128 (with Skip Connection)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()                                  # Output scale: [-1, 1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        u1 = self.up1(d3)
        u2 = self.up2(torch.cat([u1, d2], dim=1)) # Skip Connection
        out = self.final(torch.cat([u2, d1], dim=1))
        return out

# 2. The Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # In Pix2Pix, Discriminator takes both Input Layout + Generated Image combined (6 channels)
        self.model = nn.Sequential(
            DownsampleBlock(6, 64, norm=False),
            DownsampleBlock(64, 128),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, layout, city_plan):
        x = torch.cat([layout, city_plan], dim=1)
        return self.model(x)
