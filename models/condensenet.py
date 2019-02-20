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
                                       condense_factor=args.condense_factor,
                                       dropout_rate=args.dropout_rate)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, in_channels,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.add(x, x_)

                                  
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels , growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)
# class _DenseBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate, args):
#         super(_DenseBlock, self).__init__()

#         self.layer_1 = _DenseLayer(in_channels , growth_rate, args)
#         self.layer_2 = _DenseLayer(in_channels + growth_rate, growth_rate, args)
#         self.layer_3 = _DenseLayer(in_channels + 2*growth_rate, growth_rate, args)
#         self.layer_4 = _DenseLayer(in_channels + 3*growth_rate , growth_rate, args)
#         self.layer_5 = _DenseLayer(in_channels + 4*growth_rate , growth_rate, args)
#         self.layer_6 = _DenseLayer(in_channels + 5*growth_rate , growth_rate, args)
#         self.layer_7 = _DenseLayer(in_channels + 6*growth_rate , growth_rate, args)

    # def forward(self, x):
    #     x_1 = self.layer_1(x)
    #     #x = torch.cat([x,x_1],1)
    #     x_2 = self.layer_2(x_1)
    #     #x = torch.cat([x,x_1],1)
    #     x_3 = self.layer_3(x_2)
    #     #x = torch.cat([x,x_1],1)
    #     x_4 = self.layer_4(x_3)
    #     #x = torch.cat([x,x_1],1)
    #     x_5 = self.layer_5(x_4)
    #     #x = torch.cat([x,x_1],1)
    #     x_6 = self.layer_6(x_5)
    #     # x = torch.cat([x,x_1],1)
    #     x_7 = self.layer_7(x_6)
    #     x = torch.cat([x,x_1,x_2,x_3,x_4,x_5, x_6, x_7],1)
    #     return x

class CondenseNet(nn.Module):
    def __init__(self, args):

        super(CondenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        self.init_stride = 1

        self.features = nn.Sequential()

####################        Edited by me     ###################################

        self.num_features = 10 * self.growth[0]
        self.init_conv = nn.Conv2d(args.c_dim, self.num_features, kernel_size=3,
                                    stride=self.init_stride,padding=1,
                                    bias=False)
        # self.init_relu = nn.LeakyReLU(0.1,inplace=True)
        # self.block_0 = _DenseBlock(in_channels=self.num_features,
        #                             growth_rate=self.growth[0],
        #                             args=self.args)
        # self.num_features = 8 * self.num_features + 28 * self.growth[0] 
        # self.block_1 = _DenseBlock(in_channels=self.num_features,
        #                             growth_rate=self.growth[1],
        #                             args=self.args)
        # self.num_features = 8 * self.num_features + 28 * self.growth[1]
        # for i in range(len(self.stages)):
        #     self.add_block(i)
        # self.dense_layers = [None] * len(self.stages)
        # for i in range(len(self.stages)):
        #     self.dense_layers[i] = [None] * self.stages[i]
        # for i in range(len(self.stages)):
        #     for j in range(self.stages[i]):
        #         self.dense_layers[i][j] = _DenseLayer(self.num_features + j * self.growth[i], self.growth[i], args).cuda()
        #         self.num_features += self.num_features + j * self.growth[i]
        #     self.num_features += self.stages[i] * self.growth[i]
                # self.num_features2 = self.num_features2
        #     # self.dense_block[i] =  _DenseBlock(
        #     #                     num_layers=self.stages[i],
        #     #                     in_channels=self.num_features,
        #     #                     growth_rate=self.growth[i],
        #     #                     args=self.args)
        #     self.num_features += self.stages[i] * self.growth[i]
        #     self.feature_out += self.num_features   #+ self.num_features
        # self.block_0 = _DenseBlock(num_layers=self.stages[0],
        #   in_channels=self.num_features,
        #   growth_rate=self.growth[0],
        #   args=self.args)
        # self.num_features += self.stages[0] * self.growth[0] + self.num_features

        # self.block_1 = _DenseBlock(num_layers=self.stages[1],
        #   in_channels=self.num_features,
        #   growth_rate=self.growth[1],
        #   args=self.args)
        # self.num_features += self.stages[1] * self.growth[1] + self.num_features

        #self.block_2 = _DenseBlock(num_layers=self.stages[2],
            #in_channels=self.num_features,
            #growth_rate=self.growth[2],
            #args=self.args)
        #self.num_features += self.stages[2] * self.growth[2] + self.num_features

        #self.block_3 = _DenseBlock(num_layers=self.stages[3],
            #in_channels=self.num_features,
            #growth_rate=self.growth[3],
            #args=self.args)
        #self.num_features += self.stages[3] * self.growth[3] + self.num_features
        self.block1 = _DenseBlock(
            num_layers=self.stages[0],
            in_channels=self.num_features,
            growth_rate=self.growth[0],
            args=self.args,
        )
        self.mid_relu = nn.LeakyReLU(0.1, inplace=True)
        self.mid_conv = nn.Conv2d(self.num_features, 2*self.num_features, kernel_size=3,
                                    stride=self.init_stride,padding=1,
                                    bias=False)

        self.block2 = _DenseBlock(
            num_layers=self.stages[1],
            in_channels=2*self.num_features,
            growth_rate=self.growth[1],
            args=self.args,
        )
        self.num_features += self.num_features
        # self.num_features += self.stages[1] * self.growth[1]
        self.end_relu  = nn.LeakyReLU(0.1, inplace=True)
        # self.bottleneck_conv = Conv(self.feature_out, 128, kernel_size=1, padding=0, flag = True)
        # print (self.num_features)
        self.bottleneck_conv = nn.Conv2d(self.num_features, 256,
                                            kernel_size=1,
                                            stride=self.init_stride,
                                            padding=0,
                                            bias=False)
        self.bottl_relu  = nn.LeakyReLU(0.1, inplace=True)
        self.deconv = nn.ConvTranspose2d(256, 256,
                                            kernel_size = 2,
                                            stride = 2,
                                            padding = 0,
                                            bias = False)
        self.deconv_relu  = nn.LeakyReLU(0.1, inplace=True)
        self.recons = nn.Conv2d(256, args.c_dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False)
#######################################################################################

        # self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
        #                                                 kernel_size=3,
        #                                                 stride=self.init_stride,
        #                                                 padding=1,
        #                                                 bias=False))
        # self.features.add_module('init_relu', nn.ReLU(inplace=True))
        # for i in range(len(self.stages)):
        #     ### Dense-block i
        #     self.add_block(i)
        ### Linear layer
        # self.classifier = nn.Linear(self.num_features, args.num_classes)
        # self.features.add_module('bottleneck_conv', nn.Conv2d(self.num_features, 256,
        #                                                 kernel_size=1,
        #                                                 stride=self.init_stride,
        #                                                 padding=0,
        #                                                 bias=False))
        # for i in range(int(math.log(args.scaling_factor,2))):
        #     self.features.add_module('deconv_%d' %(i+1), nn.ConvTranspose2d(256, 256,
        #                                                 kernel_size=2,
        #                                                 stride=2,
        #                                                 padding=0,
        #                                                 bias=False))
        #     self.features.add_module('Prelu_deconv_%d' %(i+1),
        #                              nn.PReLU())
        # self.features.add_module('reconv', nn.Conv2d(256, args.c_dim,
        #                                                 kernel_size=1,
        #                                                 stride=1,
        #                                                 padding=0,
        #                                                 bias=False))
        ### initialize
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
            # if isinstance(m, nn.ConvTranspose2d):
                #c1, c2, h, w = m.weight.data.size()
                #weight = get_upsample_filter(h)
                #m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                #if m.bias is not None:
                    #m.bias.data.zero_()   
        return

    def add_block(self, i):
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        # self.num_features += self.stages[i] * self.growth[i]


    def forward(self, x, x_,progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        x = self.init_conv(x)
        # x = self.init_relu(x)
        # x = self.features(x)
        x = self.block1(x)
        x = self.mid_relu(x)
        x = self.mid_conv(x)
        x = self.block2(x)
        x = self.end_relu(x)
        x = self.bottleneck_conv(x)
        x = self.bottl_relu(x)
        for i in range(int(math.log(self.args.scaling_factor,2))):
            x = self.deconv(x)
            x = self.deconv_relu(x)
        x = self.recons(x)
        return torch.add(x, x_)