#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: cbam.py
@time: 2019/1/14 15:33
@desc: Convolutional Block Attention Module in ECCV 2018, including channel attention module and spatial attention module.
'''

import torch
from torch import nn
import time
from .attention import *


class identity(nn.Module):
    def forward(self, out):
        return out, []
    
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class BottleNeck(nn.Module):
    '''Improved Residual Bottleneck with Channel Attention Module and Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match, attention_type='ir'):
        super(BottleNeck, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))

        self.attention_type = attention_type
        if attention_type == 'ir':
            self.attention = identity()
        elif attention_type == 'cbam':
            self.attention = CBAM(out_channel, 16)
        elif attention_type == 'se':
            self.attention = SELayer(out_channel, 16)
        else:
            raise('Select Proper Attention Type')
        

        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity = x
        out = self.res_layer(x)
        out, _ = self.attention(out)

        if self.shortcut_layer is not None:
            identity = self.shortcut_layer(x)

        out += identity
        return out

filter_list = [64, 64, 128, 256, 512]
def get_layers(num_layers):
    if num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 100:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]

class ResNet(nn.Module):
    def __init__(self, num_layers, feature_dim=512, drop_ratio=0.4, mode='ir',filter_list=filter_list):
        super(ResNet, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'se', 'cam', 'sam', 'cbam'], 'mode should be ir, ir_se, ir_cam, ir_sam or ir_cbam'
        self.mode = mode
        
        layers = get_layers(num_layers)
        block = BottleNeck
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))

        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)


        self.conv_bridge = nn.Conv2d(512, 3072, kernel_size=3, stride=1, padding=1)
        
        
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False, attention_type=self.mode))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True, attention_type=self.mode))
        return nn.Sequential(*layers)

    def forward(self, x, train=True):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.output_layer(x)
        
        if train:
            x_recon = self.conv_bridge(x)
            return out, x_recon
        else:
            return out

if __name__ == '__main__':
    input = torch.Tensor(4, 3, 112, 112)
    net = CBAMResNet(50, mode='cbam')
    out = net(input)
    print(out.size())