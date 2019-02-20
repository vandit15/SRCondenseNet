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
        ########### EDITED by Vandit 30-3 ###############
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
            # self.dense_block[i] =  _DenseBlock(
            #                     num_layers=self.stages[i],
            #                     in_channels=self.num_features,
            #                     growth_rate=self.growth[i],
            #                     args=self.args)
            self.num_features += self.stages[i] * self.growth[i]
            self.bottleneck_input += self.num_features #+ self.num_features
        self.end_relu  = nn.LeakyReLU(0.1,inplace=True)
        # self.bottleneck_conv = nn.Conv2d(self.num_features, 256,
        #                                     kernel_size=1,
        #                                     stride=self.init_stride,
        #                                     padding=0,
        #                                     bias=False)
        # deconv = [None]*int(math.log(args.scaling_factor,2))
        # Prelu_deconv = [None]*int(math.log(args.scaling_factor,2))
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

        ######################### edited by Vandit 30-3 #################
        ### Dense-block 1 (224x224)
    #     self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
    #                                                     kernel_size=3,
    #                                                     stride=self.init_stride,
    #                                                     padding=1,
    #                                                     bias=False))
    #     for i in range(len(self.stages)):
    #         ### Dense-block i
    #         self.add_block(i)
    #     ### Linear layer
    #     # self.classifier = nn.Linear(self.num_features, args.num_classes)
    #     self.features.add_module('bottleneck_conv', nn.Conv2d(self.num_features, 256,
    #                                                     kernel_size=1,
    #                                                     stride=self.init_stride,
    #                                                     padding=0,
    #                                                     bias=False))
    #     for i in range(int(math.log(args.scaling_factor,2))):
    #         self.features.add_module('deconv', nn.ConvTranspose2d(256, 256,
    #                                                     kernel_size=2,
    #                                                     stride=2,
    #                                                     padding=0,
    #                                                     bias=False))
    #     self.features.add_module('reconv', nn.Conv2d(256, args.c_dim,
    #                                                     kernel_size=1,
    #                                                     stride=1,
    #                                                     padding=0,
    #                                                     bias=False))
    #     ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             m.bias.data.zero_()

    # def add_block(self, i):
    #     ### Check if ith is the last one
    #     last = (i == len(self.stages) - 1)
    #     block = _DenseBlock(
    #         num_layers=self.stages[i],
    #         in_channels=self.num_features,
    #         growth_rate=self.growth[i],
    #         args=self.args,
    #     )
    #     self.features.add_module('denseblock_%d' % (i + 1), block)
    #     self.num_features += self.stages[i] * self.growth[i]
    #     if not last:
    #         trans = _Transition(in_channels=self.num_features,
    #                             args=self.args)
    #         self.features.add_module('transition_%d' % (i + 1), trans)
    #     else:
    #         self.features.add_module('norm_last',
    #                                  nn.BatchNorm2d(self.num_features))
    #         self.features.add_module('relu_last',
    #                                  nn.ReLU(inplace=True))
    #         self.features.add_module('pool_last',
    #                                  nn.AvgPool2d(self.pool_size))

    def forward(self, x, x_, progress=None):
        # features = self.features(x)
        # out = features.view(features.size(0), -1)
        # out = self.classifier(out)
        x = self.init_conv(x)
        x = self.init_relu(x)
        # p = x
        h = []
        h.append(x)
        for i in range(len(self.stages)):
            for j in range(self.stages[i]):
                x = self.dense_layers[i][j](x)
                h.append(x)
                # x = torch.cat([x,p],1)
                # for k in range(j+1):
                #   x = torch.add(x,list[k])
                # list.append(x)
            # p = x
        # for i in h:
        #     x = torch.cat([x, i], 1)
        x = self.end_relu(x)
        x = self.bottleneck_conv(x)
        for i in range(int(math.log(self.args.scaling_factor,2))):
            x = self.deconv(x)
            x = self.relu_deconv(x)
        x = self.reconv(x)
        # x_ = bicubic(x_,self.args)
        return torch.add(x, x_)
