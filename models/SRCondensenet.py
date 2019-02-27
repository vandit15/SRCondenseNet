from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from layers import Conv, LearnedGroupConv
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from PIL import Image
__all__ = ['CondenseNet']

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = LearnedGroupConv(in_channels, args.bottleneck * growth_rate,
                                       kernel_size=1, groups=self.group_1x1,
                                       condense_factor=args.condense_factor)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3, flag = False)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)

                                  
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)

class SRCondenseNet(nn.Module):
    def __init__(self, args):

        super(SRCondenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        self.init_stride = 1
        self.pool_size = 1


        self.num_features = 10 * self.growth[0]
        self.feature_out = 0
        
        self.init_conv = nn.Conv2d(args.c_dim, self.num_features, kernel_size=3,
                                    stride=self.init_stride,padding=1,
                                    bias=False)
        self.init_relu = nn.LeakyReLU(0.1,inplace=True)
        self.end_relu  = nn.LeakyReLU(0.1,inplace=True)
        self.dense_block_0 = _DenseBlock(num_layers=self.stages[0],
            in_channels=self.num_features,
            growth_rate=self.growth[0],
            args = self.args)
        self.num_features += self.stages[0] * self.growth[0]
        self.feature_out += self.num_features

        self.dense_block_1 = _DenseBlock(num_layers=self.stages[1],
            in_channels=self.num_features,
            growth_rate=self.growth[1],
            args = self.args)
        self.num_features += self.stages[1] * self.growth[1]
        self.feature_out += self.num_features
        

        self.dense_block_2 = _DenseBlock(num_layers=self.stages[2],
            in_channels=self.num_features,
            growth_rate=self.growth[2],
            args = self.args)
        self.num_features += self.stages[2] * self.growth[2]
        self.feature_out += self.num_features
        

        self.dense_block_3 = _DenseBlock(num_layers=self.stages[3],
            in_channels=self.num_features,
            growth_rate=self.growth[3],
            args = self.args)
        self.num_features += self.stages[3] * self.growth[3]
        self.feature_out += self.num_features
        
        
        
        self.bottleneck_conv = nn.Conv2d(self.feature_out, 128,
                                            kernel_size=1,
                                            stride=self.init_stride,
                                            padding=0,
                                            bias=False)

        self.init_deconv = nn.ConvTranspose2d(128, 128,
                                            kernel_size = 3,
                                            stride = 3,
                                            padding = 0,
                                            bias = False)
        self.relu_deconv = nn.LeakyReLU(0.1,inplace = True)
        self.recons = nn.Conv2d(128, args.c_dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        return



    def forward(self, x, x_,progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        x = self.init_conv(x)
        x = self.init_relu(x)
        x_0 = self.dense_block_0(x)
        x_1 = self.dense_block_1(x_0)
        x_2 = self.dense_block_2(x_1)
        x_3 = self.dense_block_3(x_2)
        x_3 = self.end_relu(x_3)
        x = torch.cat([x_0,x_1,x_2,x_3],1)
        x = self.bottleneck_conv(x)
        x = self.init_deconv(x)
        x = self.relu_deconv(x)
        x = self.recons(x)
        return torch.add(x, x_)
