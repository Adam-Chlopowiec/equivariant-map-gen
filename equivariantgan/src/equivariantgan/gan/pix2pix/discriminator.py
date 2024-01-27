from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from equivariantgan.gan.pix2pix.generator import DownConvNormAct


class UNetDiscriminator(nn.Module):

    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 1, 
        image_size: int = 256,
    ) -> None:
        super().__init__()

        assert image_size in [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        # initial projection
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, padding=1, stride=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # downblock
        self.down1 = DownConvNormAct(
            in_channels=64,
            out_channels=128,
            residual=False,
        )
        
        self.down2 = DownConvNormAct(
            in_channels=128,
            out_channels=256,
            residual=False,
        )
        
        self.down3 = DownConvNormAct(
            in_channels=256,
            out_channels=512,
            residual=False,
        )

        self.outc = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=out_channels, 
                kernel_size=1
            ),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x) # 64
        x = self.down1(x) # 128
        x = self.down2(x) # 256
        x = self.down3(x) # 512
        x = self.outc(x)

        return x
