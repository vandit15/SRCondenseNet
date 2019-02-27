from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from layers import ShuffleLayer, Conv, CondenseConv, CondenseLinear

__all__ = ['CondenseNet']

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = CondenseConv(in_channels, args.bottleneck * growth_rate,
                                   kernel_size=1, groups=self.group_1x1)
        ### 3x3 conv b*k-->k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

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


class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):
    def __init__(self, args):

        super(CondenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        self.bottleneck_input = self.num_features
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.init_conv = nn.Conv2d(args.c_dim, self.num_features, kernel_size=3,
                                    stride=self.init_stride,padding=1,
                                    bias=False)
        self.init_relu = nn.LeakyReLU(0.1,inplace=True)
        self.dense_layers = [None] * len(self.stages)
        for i in range(len(self.stages)):
            self.dense_layers[i] = [None] * self.stages[i]
        for i in range(len(self.stages)):
            for j in range(self.stages[i]):
                self.dense_layers[i][j] = _DenseLayer(self.num_features + j * self.growth[i], self.growth[i], args).cuda()
            self.num_features += self.stages[i] * self.growth[i]
            self.bottleneck_input += self.num_features #+ self.num_features
        self.end_relu  = nn.LeakyReLU(0.1,inplace=True)
        self.bottleneck_conv = CondenseConv(self.num_features, 128, kernel_size = 1, groups=self.group_1x1)
        
        self.deconv = nn.ConvTranspose2d(128, 128,
                                            kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            bias=False)
        self.relu_deconv = nn.LeakyReLU(0.1,inplace = True)
        self.reconv = nn.Conv2d(128, args.c_dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, x_, progress=None):
        x = self.init_conv(x)
        x = self.init_relu(x)
        h = []
        h.append(x)
        for i in range(len(self.stages)):
            for j in range(self.stages[i]):
                x = self.dense_layers[i][j](x)
                h.append(x)
        x = self.end_relu(x)
        x = self.bottleneck_conv(x)
        for i in range(int(math.log(self.args.scaling_factor,2))):
            x = self.deconv(x)
            x = self.relu_deconv(x)
        x = self.reconv(x)
        return torch.add(x, x_)
