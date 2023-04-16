# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .van import VAN, Block, VAN_Block
from .hr_layers import ConvBlock, fSEModule, Conv3x3, Conv1x1, upsample, ConvBlockSELU
from layers import *
from .hr_layers_diffnet import Attention_Module

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


class HRDepthDecoder(nn.Module):
    """
    Adopted from paper HR-Depth: https://github.com/shawLyu/HR-Depth/blob/main/networks/HR_Depth_Decoder.py
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1):
        super(HRDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        self.convs = nn.ModuleDict()
        self.van_blocks = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])

            fse_high_ch = num_ch_enc[row + 1] // 2
            fse_low_ch = self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1)
            fse_out_ch = fse_high_ch if index == "04" else self.num_ch_enc[row]
            van_depths = 2 if index == "22" else 1

            self.convs["X_" + index + "_attention"] = fSEModule(fse_high_ch, fse_low_ch, fse_out_ch)
            self.van_blocks["X_" + index] = VAN_Block(num_ch=fse_out_ch, depth=van_depths, mlp_ratio=4)

        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])

            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                                 self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                               + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
                features["X_" + index] = self.van_blocks["X_" + index](features["X_" + index])
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[('disp', 0)] = self.sigmoid(self.convs["dispConvScale0"](x))  # (16, 1)
        outputs[('disp', 1)] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))  # (32, 1)
        outputs[('disp', 2)] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))  # (64, 1)
        outputs[('disp', 3)] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))  # (128, 1)
        return outputs


class VAN_decoder(VAN):
    def __init__(self,
                 num_ch_enc=(64, 64, 128, 320, 512),
                 mlp_ratios=(4, 4, 4, 4),
                 depths=(1, 1, 2, 1),
                 linear=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(VAN, self).__init__()
        self.depths = depths
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (32, 64, 64, 128, 320)
        self.linear = linear

        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_enc[i] * 2
            num_ch_out = self.num_ch_dec[i]
            halving_conv = ConvBlockSELU(num_ch_in, num_ch_out)
            setattr(self, f"halving_conv_{i + 1}", halving_conv)

            if i < 4:  # there is no VAN block on last level
                block = nn.ModuleList([Block(dim=self.num_ch_enc[i] * 2,
                                             mlp_ratio=mlp_ratios[i],
                                             drop=0.,
                                             drop_path=0.,
                                             linear=linear,
                                             norm_cfg=norm_cfg)
                                       for j in range(depths[i])])

                norm = nn.LayerNorm(self.num_ch_enc[i] * 2)
                setattr(self, f"block{i + 1}", block)
                setattr(self, f"norm{i + 1}", norm)

        for d in range(3, -1, -1):
            num_ch_in = self.num_ch_dec[d] * 2 if d > 0 else self.num_ch_dec[d]
            disparity_head = Conv3x3(num_ch_in, 1)
            setattr(self, f"disp_head_{d + 1}", disparity_head)

    def forward(self, input_features):
        outs = {}
        x = input_features[-1]
        for s in range(5, 0, -1):
            halving_conv = getattr(self, f"halving_conv_{s}")
            x = halving_conv(x)
            x = upsample(x)
            if s >= 2:
                x = torch.cat([x, input_features[s - 2]], dim=1)
                block = getattr(self, f"block{s-1}")
                norm = getattr(self, f"norm{s-1}")
                B, C, H, W = x.shape
                x = x.flatten(2).permute(0, 2, 1)  # [B, N, C]
                for blk in block:
                    x = blk(x, H, W)
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if s <= 4:
                disp_head = getattr(self, f"disp_head_{s}")
                outs[("disp", s - 1)] = disp_head(x).sigmoid()
        return outs


class DiffnetDecoder(nn.Module):
    """From DIFFNet paper"""
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()

        # decoder
        self.convs = nn.ModuleDict()

        # adaptive block
        if self.num_ch_dec[0] < 16:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])

            # adaptive block
            self.convs["72"] = Attention_Module(2 * self.num_ch_dec[4],  2 * self.num_ch_dec[4]  , self.num_ch_dec[4])
            self.convs["36"] = Attention_Module(self.num_ch_dec[4], 3 * self.num_ch_dec[3], self.num_ch_dec[3])
            self.convs["18"] = Attention_Module(self.num_ch_dec[3], self.num_ch_dec[2] * 3 + 64 , self.num_ch_dec[2])
            self.convs["9"] = Attention_Module(self.num_ch_dec[2], 64, self.num_ch_dec[1])
        else:
            self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
            self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

            self.convs["72"] = Attention_Module(self.num_ch_enc[4], self.num_ch_enc[3] * 1, 256)  # (512, 320 * 2, 256)
            self.convs["36"] = Attention_Module(256, self.num_ch_enc[2] * 1, 128)  # (256, 128 * 3, 128)
            self.convs["18"] = Attention_Module(128, self.num_ch_enc[1] * 1, 64)  # (128, 64 * 4, 64)
            self.convs["9"] = Attention_Module(64, 64, 32)
        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        feature144 = input_features[4]
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]
        x72 = self.convs["72"](feature144, feature72)  # attention (512, 320, 320) -> (256)
        x36 = self.convs["36"](x72 , feature36)
        x18 = self.convs["18"](x36 , feature18)
        x9 = self.convs["9"](x18, feature64)
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))

        outputs[("disp", 0)] = self.sigmoid(self.convs["dispConvScale0"](x6))
        outputs[("disp", 1)] = self.sigmoid(self.convs["dispConvScale1"](x9))
        outputs[("disp", 2)] = self.sigmoid(self.convs["dispConvScale2"](x18))
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispConvScale3"](x36))
        return outputs
