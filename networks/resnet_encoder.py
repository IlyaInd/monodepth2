# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import gc

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from timm.models.vision_transformer import _cfg
from mmcv.cnn.utils import revert_sync_batchnorm

from .van import VAN, ZeroVANlayer

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


def load_weights(model, weights_path):
    model.default_cfg = _cfg()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    checkpoint = torch.load(weights_path, map_location=torch.device(device))
    strict = True
    if not hasattr(model, 'num_classes') or model.num_classes != 1000:
        strict = False
        del checkpoint["state_dict"]["head.weight"]
        del checkpoint["state_dict"]["head.bias"]
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    print(f'CUDA memory allocated --- {torch.cuda.memory_allocated() / 1024 // 1024} MB')
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    del checkpoint
    gc.collect()
    print(f'CUDA memory allocated --- {torch.cuda.memory_allocated() / 1024 // 1024} MB')
    return model


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, width, height, nhead):
        super().__init__()
        self.dim = dim
        self.width = width
        self.height = height
        self.attention = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dropout=0,
                                                          activation='gelu', batch_first=True)

    def forward(self, x):
        # [bs, c, w, h]
        x = x.flatten(2).permute(0, 2, 1) # [bs, n, c]
        x = self.attention(x)
        return x.permute(0, 2, 1).view(-1, self.dim, self.width, self.height)


class VAN_encoder(nn.Module):
    def __init__(self, zero_layer_mlp_ratio=4, zero_layer_depths=3):
        super().__init__()
        self.register_buffer('imagenet_mean', torch.Tensor([0.485, 0.456, 0.406]))
        self.register_buffer('imagenet_std', torch.Tensor([0.229, 0.224, 0.225]))
        self.num_ch_enc = np.array([64, 64, 128, 320, 512])
        # self.conv_stem = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=(3, 3), stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        # )
        van = VAN(embed_dims=[64, 128, 320, 512],  mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 3])
        self.zero_layer = ZeroVANlayer(mlp_ratio=zero_layer_mlp_ratio, depths=zero_layer_depths)
        self.van = load_weights(van, 'networks/van_base_828.pth.tar')
        self.mha_block = MultiHeadAttentionBlock(self.num_ch_enc[-1], 6, 20, nhead=8)

    def forward(self, x):
        x = (x - self.imagenet_mean[None, :, None, None]) / self.imagenet_std[None, :, None, None]
        # out = [self.conv_stem(x)]
        out = [self.zero_layer(x)]
        van_out = self.van(x)
        out.extend(van_out)
        out[-1] += self.mha_block(out[-1])
        return out
