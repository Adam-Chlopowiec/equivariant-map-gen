from typing import Optional, Union, List, Callable, Tuple

from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import escnn
import escnn.nn as enn
from escnn.nn.modules.utils import indexes_from_labels
from escnn import gspaces

from guidedcontrolcom.gan.e2_pix2pix.generator import E2DownConvNormAct, trivial_feature_type, regular_feature_type, create_conv_instance, named_apply, init_weights

class UNetDiscriminator(nn.Module):

    def __init__(
        self, 
        gspace: escnn.gspaces.GSpace,
        N: int,
        in_channels: int = 3, 
        out_channels: int = 1, 
        image_size: int = 256,
        restriction: int = 3,
        deltaorthonormal: bool = False,
        channel_div: int = 8,
        to_trivial: bool = True,
        r1_subgroup: int = 2,
        r2_subgroup: int = 4,
    ) -> None:
        super().__init__()

        assert image_size in [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        self.r2_act = gspace
        self.G = self.r2_act.fibergroup
        self.restriction = restriction
        self.deltaorthonormal = deltaorthonormal
        self.channel_div = channel_div
        self.to_trivial = to_trivial
        self.r1_subgroup = r1_subgroup
        self.r2_subgroup = r2_subgroup
        self.r3_subgroup = N
        self.N = N

        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * in_channels)

        # initial projection
        self.inc = enn.SequentialModule(
            create_conv_instance(
                trivial_feature_type(self.r2_act, in_channels),
                regular_feature_type(self.r2_act, 64 // self.channel_div),
                kernel_size=4,
                padding=1,
                stride=2,
                bias=True,
                sigma=None,
                frequencies_cutoff=lambda r: 3*r,
                conv_type="r2conv"
            ),
            enn.LeakyReLU(regular_feature_type(self.r2_act, 64 // self.channel_div), 0.2)
        )

        # downblock
        self.down1 = E2DownConvNormAct(
            in_channels=regular_feature_type(self.r2_act, 64 // self.channel_div),
            in_channels=regular_feature_type(self.r2_act, 128 // self.channel_div),
            residual=False,
        )
        in_type = self.down1.out_type

        if self.restriction >= 1:
            self.r1_gspace, self.r1_G, restriction_layer, in_type = self.restrict_layer(
                (0, self.N // self.r1_subgroup if self.N >= 8 else self.N // 2), in_type
            )
            self.add_module("r1", restriction_layer)
            self.r1 = restriction_layer
        else:
            self.r1 = enn.IdentityModule()
            self.r1_gspace = self.r2_act
            self.r1_G = self.G
        
        self.down2 = E2DownConvNormAct(
            in_channels=regular_feature_type(self.r1_gspace, 128 // self.channel_div),
            in_channels=regular_feature_type(self.r1_gspace, 256 // self.channel_div),
            residual=False,
        )
        in_type = self.down2.out_type

        if self.restriction >= 2:
            self.r1_gspace, self.r1_G, restriction_layer, in_type = self.restrict_layer(
                (0, self.N // self.r1_subgroup if self.N >= 8 else self.N // 2), in_type
            )
            self.add_module("r2", restriction_layer)
            self.r2 = restriction_layer
        else:
            self.r2 = enn.IdentityModule()
            self.r2_gspace = self.r2_act
            self.r2_G = self.G

        self.down3 = E2DownConvNormAct(
            in_channels=regular_feature_type(self.r2_gspace, 256 // self.channel_div),
            in_channels=regular_feature_type(self.r2_gspace, 512 // self.channel_div),
            residual=False,
        )
        in_type = self.down3.out_type

        self.outc = create_conv_instance(
            self.down3.out_type,
            trivial_feature_type(self.r2_gspace, out_channels),
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
            sigma=None,
            frequencies_cutoff=lambda r: 3*r,
            conv_type="r2conv"
        )
        self.out_act = nn.Tanh()
        # self.outc = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.down3.out_type.size, 
        #         out_channels=out_channels, 
        #         kernel_size=1
        #     ),
        #     nn.Sigmoid()
        # )

        named_apply(partial(init_weights, deltaorthonormal=self.deltaorthonormal), self)

    def restrict_layer(
        self,
        id: Tuple[int, int],
        in_type: enn.FieldType
    ) -> Tuple[enn.SequentialModule, enn.FieldType]:
        layers = list()
        layers.append(enn.RestrictionModule(in_type, id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        r2_act = layers[-1].out_type.gspace
        G = self.r2_act.fibergroup
        
        restrict_layer = enn.SequentialModule(*layers)
        return r2_act, G, restrict_layer, layers[-1].out_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_type(x)
        x = self.inc(x) # 64
        x = self.down1(x) # 128
        x = self.r1(x)
        x = self.down2(x) # 256
        x = self.r2(x)
        x = self.down3(x) # 512
        x = self.outc(x).tensor
        x = self.out_act(x)

        return x
