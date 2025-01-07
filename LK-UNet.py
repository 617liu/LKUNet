# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            groups=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)



class ChannelSELayer(BaseModule):
    def __init__(self, 
                channel, 
                reduction=2,
                act_cfg=None,
                init_cfg=None):
        super().__init__(init_cfg)

        # self.act_cfg = act_cfg
        # self.activate = build_activation_layer(act_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialSELayer(BaseModule):
    def __init__(self, 
                channel,
                init_cfg=None):
        super().__init__(init_cfg)
        self.conv = Conv2dReLU(channel, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ConcurrentSELayer(BaseModule):
    def __init__(self, 
                channel, 
                reduction=2,
                init_cfg=None):
        super().__init__(init_cfg)
        self.cSE = ChannelSELayer(channel, reduction)
        self.sSE = SpatialSELayer(channel)
         # cat后的1x1卷积
        self.pool_conv = Conv2dReLU(
            channel * 2,
            channel,
            kernel_size=1,
            stride=1,
        )
    def forward(self, x):
        out = torch.cat([self.cSE(x), self.sSE(x)], dim=1)
        output = self.pool_conv(out)
        return output


class EASPP(nn.Module):
    def __init__(self, in_channels, out_channels, num):
        super(EASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        self.branch2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2 * num, padding=2 * num, bias=False)
        
        self.branch3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=4 * num, padding=4 * num, bias=False)
        
        self.branch4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=8 * num, padding=8 * num, bias=False)
        
        self.branch5_pool = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        self.final_conv = nn.Conv2d(in_channels * 5, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        branch1_out = self.branch1(x) * x
        
        branch2_out = self.branch2(x)
        
        branch3_out = self.branch3(x)
        
        branch4_out = self.branch4(x)
        
        branch5_out = self.branch5_pool(x)
        branch5_out = self.branch5_conv(branch5_out)
        branch5_out = F.interpolate(branch5_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        concatenated = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out, branch5_out], dim=1)
        
        output = self.final_conv(concatenated)
        return output

@MODELS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.bilinear_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.scSE = nn.ModuleList()
        self.feature_channels = [1024, 512, 256, 128]
        for feature_channel in self.feature_channels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(feature_channel//2, feature_channel//2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feature_channel//2),
                    nn.ReLU(inplace=True),
                )
            )
            self.merge_convs.append(
                nn.Sequential(
                    nn.Conv2d(feature_channel, feature_channel//2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(feature_channel//2),
                    nn.ReLU(inplace=True),
                )
            )
            self.conv_up.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature_channel, feature_channel//2, kernel_size=2, stride=2),
                    nn.BatchNorm2d(feature_channel//2),
                    nn.ReLU(inplace=True),
                )
            )
            scse = ConcurrentSELayer(channel=feature_channel)
            self.scSE.append(scse)

        self.EASPP1 = EASPP(512, 512, 1)
        self.EASPP2 = EASPP(1024, 1024, 2)


        self.conv_finup = nn.Sequential(
                    nn.ConvTranspose2d(64, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    )
        self.conv_320_256 = nn.Sequential(
                        nn.Conv2d(320, 256, kernel_size=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True))
    def forward(self, inputs):

        x = self._transform_inputs(inputs)  


        for i in range(len(inputs)):
            if i == 0:
                x[i] = self.EASPP2(x[i])
            if i == 1:
                x[i] = self.EASPP1(x[i])
            x[i] = self.scSE[i](x[i])
        for i in range(len(inputs) - 1):

            upsampled_feature = self.conv_up[i](x[i])

            temp = self.merge_convs[i](torch.cat([x[i+1], upsampled_feature], dim=1))
            x[i+1] = self.convs[i](temp)

        x[3] = self.merge_convs[3](x[3])
        x[3] = self.convs[3](x[3])
        x[3] = self.conv_finup(x[3])

        output = self.cls_seg(x[3])



        return output
