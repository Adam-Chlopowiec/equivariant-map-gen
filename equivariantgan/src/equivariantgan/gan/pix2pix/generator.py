from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownConvNormAct(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        residual: bool = False,
        slope: float = 0.2
    ) -> None:
        super().__init__()
        self.residual = residual

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DownConvNormDropAct(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        residual: bool = False,
        p: float = 0.5,
        slope: float = 0.2
    ) -> None:
        super().__init__()
        self.residual = residual

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p),
            nn.LeakyReLU(slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UpConvNormAct(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return torch.cat([res, self.conv(x)], dim=1)
        else:
            return self.conv(x)


class UpConvNormDropAct(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        residual: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.residual = residual

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return torch.cat([res, self.conv(x)], dim=1)
        else:
            return self.conv(x)


class UNetGenerator(nn.Module):

    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3, 
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
        
        self.down4 = DownConvNormAct(
            in_channels=512,
            out_channels=512,
            residual=False,
        )

        # upsample
        self.up1 = UpConvNormDropAct(
            in_channels=512,
            out_channels=512,
            residual=True,
        )
        
        self.up2 = UpConvNormAct(
            in_channels=1024,
            out_channels=256,
            residual=True,
        )

        self.up3 = UpConvNormAct(
            in_channels=512,
            out_channels=128,
            residual=True,
        )

        self.up4 = UpConvNormAct(
            in_channels=256,
            out_channels=64,
            residual=True,
        )

        self.outc = nn.Sequential(
            nn.Conv2d(
                in_channels=128, 
                out_channels=out_channels, 
                kernel_size=1
            ),
            nn.Tanh()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.inc(x) # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) # 512
        x5 = self.down4(x4) # 512

        x = self.up1(x5, x4) # 512 + 512 = 1024
        x = self.up2(x, x3) # 1024->256 + 256 = 512
        x = self.up3(x, x2) # 512->128 + 128 = 256
        x = self.up4(x, x1) # 256->64 + 64 = 128
        output = self.outc(x)
        return output
