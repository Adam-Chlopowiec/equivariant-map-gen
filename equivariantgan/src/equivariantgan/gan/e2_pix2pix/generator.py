from typing import Optional, Union, List, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import escnn
import escnn.nn as enn
from escnn.nn.modules.utils import indexes_from_labels
from escnn import gspaces
from functools import partial
import math


def regular_feature_type(gspace: gspaces.GSpace, channels: int) -> enn.FieldType:
    """Build a regular feature map with the specified number of channels"""
    return enn.FieldType(gspace, [gspace.regular_repr] * channels)


def trivial_feature_type(gspace: gspaces.GSpace, channels: int) -> enn.FieldType:
    """Build a trivial feature map with the specified number of channels"""
    return enn.FieldType(gspace, [gspace.trivial_repr] * channels)



def create_conv_instance(
    in_type: enn.FieldType,
    out_type: enn.FieldType,
    kernel_size: int,
    stride: int, 
    padding: int,
    sigma: Union[List[float], float], 
    frequencies_cutoff: Callable, 
    bias: bool = False, 
    dilation: int = 1,
    groups: int = 1,
    initialize: bool = False,
    conv_type: str = "r2conv",
) -> enn.R2Conv:
    if conv_type == "r2conv":
        return enn.R2Conv(
            in_type,
            out_type,
            kernel_size=kernel_size,
            padding=padding,
            initialize=initialize,
            sigma=sigma,
            frequencies_cutoff=frequencies_cutoff,
            stride=stride,
            bias=bias,
            dilation=dilation,
            groups=groups,
        )
    elif conv_type == "r2convtrans":
        return enn.R2ConvTransposed(
            in_type,
            out_type,
            kernel_size=kernel_size,
            padding=padding,
            initialize=initialize,
            sigma=sigma,
            frequencies_cutoff=frequencies_cutoff,
            stride=stride,
            bias=bias,
            dilation=dilation,
            groups=groups,
        )


class E2DownConvNormAct(nn.Module):

    def __init__(
        self, 
        in_type: int, 
        out_type: int,
        slope: float = 0.2
    ) -> None:
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type

        self.conv = enn.SequentialModule(
            create_conv_instance(
                in_type,
                out_type,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                sigma=None,
                frequencies_cutoff=lambda r: 3*r,
            ),
            enn.InnerBatchNorm(out_type),
            enn.LeakyReLU(out_type, slope)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class E2DownConvNormDropAct(nn.Module):

    def __init__(
        self, 
        in_type: int, 
        out_type: int,
        p: float = 0.5,
        slope: float = 0.2
    ) -> None:
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type

        self.conv = enn.SequentialModule(
            create_conv_instance(
                in_type,
                out_type,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                sigma=None,
                frequencies_cutoff=lambda r: 3*r,
                conv_type="r2conv"
            ),
            enn.InnerBatchNorm(out_type),
            enn.FieldDropout(out_type, p),
            enn.LeakyReLU(out_type, slope)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class E2UpConvNormAct(nn.Module):

    def __init__(
        self, 
        in_type: enn.FieldType, 
        out_type: enn.FieldType,
        conv_out_type: enn.FieldType,
        res_type: enn.FieldType,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.in_type = in_type
        self.out_type = out_type
        self.res_type = res_type
        self.conv_out_type = conv_out_type
        rotations = in_type.gspace.rotations_order

        self.conv = enn.SequentialModule(
            create_conv_instance(
                in_type,
                conv_out_type,
                kernel_size=3 if rotations in [0, 2, 4] else 5,
                stride=2,
                padding=1 if rotations in [0, 2, 4] else 2,
                bias=False,
                sigma=None,
                frequencies_cutoff=lambda r: 3*r,
                conv_type="r2convtrans"
            ),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )
        if res_type.gspace != out_type.gspace:
            restrict_layer, res_type = self.restrict_layer((0, out_type.gspace._sg_id[1]), res_type)
            self.add_module("restrict", restrict_layer)
        else:
            self.restrict = enn.IdentityModule(res_type)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        if self.residual:
            res = self.restrict(res)
            return self.out_type(torch.cat([res.tensor, self.conv(x).tensor], dim=1))
        else:
            return self.conv(x)
    
    def restrict_layer(
        self,
        id: Tuple[int, int],
        in_type: enn.FieldType
    ) -> Tuple[enn.SequentialModule, enn.FieldType]:
        layers = list()
        layers.append(enn.RestrictionModule(in_type, id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        
        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer, layers[-1].out_type


class E2UpConvNormDropAct(nn.Module):

    def __init__(
        self, 
        in_type: enn.FieldType, 
        out_type: enn.FieldType,
        conv_out_type: enn.FieldType,
        res_type: enn.FieldType,
        residual: bool = False,
        p: float = 0.5
    ) -> None:
        super().__init__()
        self.residual = residual
        self.in_type = in_type
        self.out_type = out_type
        self.res_type = res_type
        self.conv_out_type = conv_out_type
        rotations = in_type.gspace.rotations_order

        self.conv = enn.SequentialModule(
            create_conv_instance(
                in_type,
                conv_out_type,
                kernel_size=3 if rotations in [0, 2, 4] else 5,
                stride=2,
                padding=1 if rotations in [0, 2, 4] else 2,
                bias=False,
                sigma=None,
                frequencies_cutoff=lambda r: 3*r,
                conv_type="r2convtrans"
            ),
            enn.InnerBatchNorm(conv_out_type),
            enn.FieldDropout(conv_out_type, p),
            enn.ReLU(conv_out_type)
        )
        if res_type.gspace != out_type.gspace:
            restrict_layer, res_type = self.restrict_layer((0, out_type.gspace._sg_id[1]), res_type)
            self.add_module("restrict", restrict_layer)
        else:
            self.restrict = enn.IdentityModule(res_type)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        if self.residual:
            res = self.restrict(res)
            return self.out_type(torch.cat([res.tensor, self.conv(x).tensor], dim=1))
        else:
            return self.conv(x)
    
    def restrict_layer(
        self,
        id: Tuple[int, int],
        in_type: enn.FieldType
    ) -> Tuple[enn.SequentialModule, enn.FieldType]:
        layers = list()
        layers.append(enn.RestrictionModule(in_type, id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        
        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer, layers[-1].out_type


class E2UNetGenerator(nn.Module):

    def __init__(
        self, 
        N: int,
        in_channels: int = 3, 
        out_channels: int = 3, 
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
        
        self.r2_act = escnn.gspaces.flipRot2dOnR2(N=N)
        self.G = self.r2_act.fibergroup
        self.restriction = restriction
        self.deltaorthonormal = deltaorthonormal
        self.channel_div = channel_div
        self.to_trivial = to_trivial
        self.r1_subgroup = r1_subgroup
        self.r2_subgroup = r2_subgroup
        self.r3_subgroup = N
        self.N = N
        rotations = trivial_feature_type(self.r2_act, in_channels).gspace.rotations_order

        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * in_channels)
        in_type = self.in_type

        # initial projection

        self.inc = enn.SequentialModule(
            create_conv_instance(
                trivial_feature_type(self.r2_act, in_channels),
                regular_feature_type(self.r2_act, int(64 // self.channel_div)),
                kernel_size=3 if rotations in [0, 2, 4] else 5,
                padding=1 if rotations in [0, 2, 4] else 2,
                stride=1,
                bias=True,
                sigma=None,
                frequencies_cutoff=lambda r: 3*r,
                conv_type="r2conv"
            ),
            enn.LeakyReLU(regular_feature_type(self.r2_act, int(64 // self.channel_div)), 0.2)
        )

        # downblock
        self.down1 = E2DownConvNormAct(
            in_type=regular_feature_type(self.r2_act, int(64 // self.channel_div)),
            out_type=regular_feature_type(self.r2_act, int(128 // self.channel_div)),
        )
        in_type = self.down1.out_type
        
        self.down2 = E2DownConvNormAct(
            in_type=regular_feature_type(self.r2_act, int(128 // self.channel_div)),
            out_type=regular_feature_type(self.r2_act, int(256 // self.channel_div)),
        )
        in_type = self.down2.out_type

        if self.restriction >= 1:
            self.r1_gspace, self.r1_G, restriction_layer, in_type = self.restrict_layer(
                (0, self.N // self.r1_subgroup if self.N >= 8 else self.N // 2), in_type
            )
            self.add_module("r1", restriction_layer)
            self.r1 = restriction_layer
        else:
            self.r1 = enn.IdentityModule(in_type)
            self.r1_gspace = self.r2_act
            self.r1_G = self.G
        
        self.down3 = E2DownConvNormAct(
            in_type=self.r1.out_type,
            out_type=regular_feature_type(self.r1_gspace, int(512 // self.channel_div)),
        )
        in_type = self.down3.out_type

        if self.restriction >= 2:
            self.r2_gspace, self.r2_G, restriction_layer, in_type = self.restrict_layer(
                (0, self.N // self.r2_subgroup if self.N >= 8 else self.N // 2), in_type
            )
            self.add_module("r2", restriction_layer)
            self.r2 = restriction_layer
        else:
            self.r2 = enn.IdentityModule(in_type)
            self.r2_gspace = self.r1_gspace
            self.r2_G = self.r1_G
        
        self.down4 = E2DownConvNormAct(
            in_type=self.r2.out_type,
            out_type=regular_feature_type(self.r2_gspace, int(512 // self.channel_div)),
        )
        in_type = self.down4.out_type

        if self.restriction >= 3:
            self.r3_gspace, self.r3_G, restriction_layer, in_type = self.restrict_layer(
                (None, self.N // self.r3_subgroup if self.N >= 8 else 1), in_type
            )
            self.add_module("r3", restriction_layer)
            self.r3 = restriction_layer
        else:
            self.r3 = enn.IdentityModule(in_type)
            self.r3_gspace = self.r2_gspace
            self.r3_G = self.r2_G

        # upsample
        self.up1 = E2UpConvNormDropAct(
            in_type=self.r3.out_type,
            res_type=self.r2.out_type,
            conv_out_type=regular_feature_type(self.r3_gspace, int(512 // self.channel_div)),
            out_type=regular_feature_type(self.r3_gspace, int(1024 // self.channel_div)),
            residual=True,
        )
        self.up1.out_type = regular_feature_type(self.r3_gspace, self.up1.restrict.out_type.size + self.up1.conv.out_type.size)
        
        self.up2 = E2UpConvNormDropAct(
            in_type=self.up1.out_type,
            res_type=self.r1.out_type,
            conv_out_type=regular_feature_type(self.r3_gspace, int(256 // self.channel_div)),
            out_type=regular_feature_type(self.r3_gspace, int(512 // self.channel_div)),
            residual=True,
        )
        self.up2.out_type = regular_feature_type(self.r3_gspace, self.up2.restrict.out_type.size + self.up2.conv.out_type.size)

        self.up3 = E2UpConvNormDropAct(
            in_type=self.up2.out_type,
            res_type=self.down1.out_type,
            conv_out_type=regular_feature_type(self.r3_gspace, int(128 // self.channel_div)),
            out_type=regular_feature_type(self.r3_gspace, int(256 // self.channel_div)),
            residual=True,
        )
        self.up3.out_type = regular_feature_type(self.r3_gspace, self.up3.restrict.out_type.size + self.up3.conv.out_type.size)

        self.up4 = E2UpConvNormDropAct(
            in_type=self.up3.out_type,
            res_type=regular_feature_type(self.r2_act, int(64 // self.channel_div)),
            conv_out_type=regular_feature_type(self.r3_gspace, int(64 // self.channel_div)),
            out_type=regular_feature_type(self.r3_gspace, int(128 // self.channel_div)),
            residual=True,
        )
        self.up4.out_type = regular_feature_type(self.r3_gspace, self.up4.restrict.out_type.size + self.up4.conv.out_type.size)

        self.outc = create_conv_instance(
            self.up4.out_type,
            trivial_feature_type(self.r3_gspace, out_channels),
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
        #         in_channels=self.up4.out_type.size, 
        #         out_channels=out_channels, 
        #         kernel_size=1
        #     ),
        #     nn.Tanh()
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
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x3 = self.r1(x3)
        x4 = self.down3(x3) # 512
        x4 = self.r2(x4)
        x5 = self.down4(x4) # 512
        x5 = self.r3(x5)

        x = self.up1(x5, x4) # 512 + 512 = 1024
        x = self.up2(x, x3) # 1024->256 + 256 = 512
        x = self.up3(x, x2) # 512->128 + 128 = 256
        x = self.up4(x, x1) # 256->64 + 64 = 128
        x = self.outc(x).tensor
        x = self.out_act(x)
        return x


def named_apply(
        fn: Callable,
        module: nn.Module,
        name: str = '',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights(
    module: nn.Module,
    name: str = '',
    deltaorthonormal: bool = False
) -> None:
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, enn.R2Conv):
        if deltaorthonormal:
            enn.init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
        else:
            enn.init.generalized_he_init(module.weights.data, module.basisexpansion, cache=True)
        if module.bias is not None:
            module.bias.data.zero_()