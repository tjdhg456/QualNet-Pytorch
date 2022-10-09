import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from .attention import *

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class target(nn.Module): 
    def __init__(self, feat_type='feature'):
        super(target, self).__init__()
        self.feat_type = feat_type
        
    def forward(self, x):
        if self.feat_type == 'feature':
            return x
        elif self.feat_type == 'attention':
            return x
        else:
            raise('Select Proper Feat Type')
        
class identity(nn.Module):
    def forward(self, out):
        return out, []
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention_type='ir'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.attention_type = attention_type
        
        
        # Attention Type
        if attention_type == 'ir':
            self.attention = identity()
        elif attention_type == 'cbam':
            self.attention = CBAM(planes, 16)
        elif attention_type == 'se':
            self.attention = SELayer(planes, 16)
        else:
            raise('Select Proper Attention Type')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Calculation Attention
        out, _ = self.attention(out)
        out += residual
        out = self.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention_type='ir'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention_type = attention_type
        
        
        # Attention Type
        if attention_type == 'ir':
            self.attention = identity()
        elif attention_type == 'cbam':
            self.attention = CBAM(planes * 4, 16)
        elif attention_type == 'se':
            self.attention = SELayer(planes * 4, 16)
        else:
            raise('Select Proper Attention Type')


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Calculation Attention
        out, _ = self.attention(out)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, attention_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # different model config between ImageNet and CIFAR 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)


        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2, attention_type=attention_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, attention_type=attention_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention_type=attention_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, attention_type=attention_type)

        self.conv_bridge = nn.Conv2d(512 * block.expansion, 3072, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        
        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0


    def _make_layer(self, block, planes, blocks, stride=1, attention_type='ir'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention_type=attention_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention_type=attention_type))

        return nn.Sequential(*layers)

    def forward(self, x, train=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Bridge and Reconstruction
        if train:
            x_recon = self.conv_bridge(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if train:
            return x, x_recon
        else:
            return x


def load_resnet(depth, num_classes, attention_type):
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, attention_type=attention_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, attention_type=attention_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, attention_type=attention_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, attention_type=attention_type)

    return model


if __name__=='__main__':
    model = load_resnet(50, 1000, 'cbam')
    x = torch.ones([1, 3, 256, 256])
    out, out_recon = model(x, train=True)
    print(out)