#!/usr/bin/python
'''
Here implemented: densenet with single layer with a layer from densenet architecture
'''
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import time
from model import *
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SingleLayer(nn.Module):
    def __init__(self, J, num_channels, num_features, track, normalizer_name, activation_name):
        super(SingleLayer, self).__init__()
        self.bn = get_normalizer(normalizer_name, num_channels*J, track)
        self.conv = nn.Conv2d(num_channels*J, num_features, 1)
        self.act = get_activation_function(activation_name)

    def forward(self, input):
        W, x = input
        out = gmul(W,x)
        out = self.conv(self.act(self.bn(out)))
        out = torch.cat((x,out),1)
        return W, out.contiguous()

class Transition(nn.Module):
    def __init__(self, J, num_channels, num_out_channels, track, normalizer_name, activation_name):
        super(Transition, self).__init__()
        self.bn = get_normalizer(normalizer_name, J*num_channels, track).to(device)
        self.conv = nn.Conv2d(J*num_channels, num_out_channels, 1)
        self.act = get_activation_function(activation_name)

    def forward(self, input):
        W, x = input
        x = gmul(W,x)
        x = self.conv(self.act(self.bn(x)))
        return W, x.contiguous()

class ConvLayer(nn.Module):
    def __init__(self, J, num_channels, nclasses):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(num_channels*J, nclasses, 1).to(device)

    def forward(self, input):
        W, x = input
        x = gmul(W,x)
        x = self.conv(x)
        return W, x

class DenseGNN_2(nn.Module):
    def __init__(self, num_layers, num_features, num_classes, J, reduction, track, normalizer_name, activation_name):
        super(DenseGNN_2, self).__init__()
        num_blocks = (num_layers-4)//3
        num_channels = num_features
        self.first_layer = ConvLayer(J, 1, num_channels)

        self.dense1 =  self.make_dense(J, num_channels, num_features, num_blocks, track, normalizer_name, activation_name)
        num_channels += num_blocks * num_features
        num_out_channels = int(math.floor(num_channels*reduction))
        self.trans1 = Transition(J, num_channels, num_out_channels, track, normalizer_name, activation_name)

        num_channels = num_out_channels
        self.dense2 = self.make_dense(J, num_channels, num_features, num_blocks, track, normalizer_name, activation_name)
        num_channels += num_blocks * num_features
        num_out_channels = int(math.floor(num_channels*reduction))
        self.trans2 = Transition(J, num_channels, num_out_channels, track, normalizer_name, activation_name)

        num_channels = num_out_channels
        self.dense3 = self.make_dense(J, num_channels, num_features, num_blocks, track, normalizer_name, activation_name)
        num_channels += num_blocks * num_features
        num_out_channels = int(math.floor(num_channels*reduction))
        self.trans3 = Transition(J, num_channels, num_out_channels, track, normalizer_name, activation_name)

        self.last_layer = ConvLayer(J, num_out_channels, num_classes)


    def make_dense(self, J, num_channels, num_features, num_blocks, track, normalizer_name, activation_name):
        layers = []
        for i in range(int(num_blocks)):
            layers.append(SingleLayer(J, num_channels, num_features, track, normalizer_name, activation_name))
            num_channels += num_features
        return nn.Sequential(*layers)

    def forward(self, input):
        input = self.first_layer(input)
        input = self.trans1(self.dense1(input))
        input = self.trans2(self.dense2(input))
        input = self.trans3(self.dense3(input))
        input = self.last_layer(input)
        input = input[1].squeeze(3).transpose(1,2).squeeze(2).contiguous().to(device)
        return input




# test
from model import get_normalizer
if __name__ == "__main__":
    num_layers = 100
    num_features = 16
    num_classes = 1
    J = 5
    reduction = 1/2
    track = 0
    normalizer_name = 'bn'
    model = DenseGNN_2(num_layers, num_features, num_classes, J, reduction, track, normalizer_name)
    W = torch.rand(1, 100, 100, 5)
    x = torch.rand(1, 1, 100, 1)
    input = (W,x)
    print(model(input).shape)


    print('done')


#
#
#
#
#
#
#
# class DenseNet(nn.Module):
#     def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, args=None, **kwargs):
#         super(DenseNet, self).__init__()
#         self.args = args
#         self.growth_rate = growth_rate
#
#         num_planes = 2*growth_rate
#         self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
#
#         self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], 0, args, **kwargs)
#         num_planes += nblocks[0]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans1 = Transition(num_planes, out_planes, 0, args, **kwargs)
#         num_planes = out_planes
#
#         self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], 1, args, **kwargs)
#         num_planes += nblocks[1]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans2 = Transition(num_planes, out_planes, 1, args, **kwargs)
#         num_planes = out_planes
#
#         self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], 2, args, **kwargs)
#         num_planes += nblocks[2]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans3 = Transition(num_planes, out_planes, 2, args, **kwargs)
#         num_planes = out_planes
#
#         self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], 3, args, **kwargs)
#         num_planes += nblocks[3]*growth_rate
#
#         self.bn = CustomNorm2d(num_planes, args, **kwargs)
#         self.linear = nn.Linear(num_planes, self.args.num_classes)
#
#         self.activation = get_activation(args.activation, 'out', args, **kwargs)
#
#     def _make_dense_layers(self, block, in_planes, nblock, ndense, args, **kwargs):
#         layers = []
#         for i in range(nblock):
#             layers.append(block(in_planes, self.growth_rate, ndense, i, args, **kwargs))
#             in_planes += self.growth_rate
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.trans1(self.dense1(out))
#         out = self.trans2(self.dense2(out))
#         out = self.trans3(self.dense3(out))
#         out = self.dense4(out)
#         out = F.avg_pool2d(self.activation(self.bn(out)), 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
# def DenseNet121(args, **kwargs):
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, args=args, **kwargs)
#
# def DenseNet169(args, **kwargs):
#     return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, args=args, **kwargs)
#
# def DenseNet201(args, **kwargs):
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, args=args, **kwargs)
#
# def DenseNet161(args, **kwargs):
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, args=args, **kwargs)
#
# def densenet_cifar(args, **kwargs):
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, args=args, **kwargs)
#
# def test_densenet():
#     net = densenet_cifar()
#     x = torch.randn(1,3,32,32)
#     y = net(Variable(x))
#     print(y)
#
# # test_densenet()
