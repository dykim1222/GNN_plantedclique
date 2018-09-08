#!/usr/bin/python
'''
Here implemented: densenet with bottleneck with layers like densenet architecture
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

class Bottleneck(nn.Module): #original
    def __init__(self, J, num_channels, num_features, track, normalizer_name, activation_name, inter_factor):
        super(Bottleneck, self).__init__()
        inter_channels = inter_factor*num_features

        self.bn1 = get_normalizer(normalizer_name, J*num_channels, track)
        self.conv1 = nn.Conv2d(J*num_channels, inter_channels, 1)
        self.act1 = get_activation_function(activation_name)

        self.bn2 = get_normalizer(normalizer_name, inter_channels, track)
        self.conv2 = nn.Conv2d(inter_channels, num_features, 1)
        self.act2 = get_activation_function(activation_name)

    def forward(self, input):
        W, x = input
        out = gmul(W,x)
        out = self.conv1(self.act1(self.bn1(out)))
        out = self.conv2(self.act2(self.bn2(out)))
        out = torch.cat((x,out),1)
        return W, out.contiguous()

class Transition(nn.Module):
    def __init__(self, J, num_channels, num_out_channels, track, normalizer_name, activation_name):
        super(Transition, self).__init__()
        self.bn = get_normalizer(normalizer_name, num_channels*J, track).to(device)
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

class DenseGNN_5(nn.Module):
    def __init__(self, num_layers, num_features, num_classes, J, reduction, track, normalizer_name, activation_name, inter_factor):
        super(DenseGNN_5, self).__init__()
        num_blocks = ((num_layers-4)//3)//2
        num_channels = num_features
        self.first_layer = ConvLayer(J, 1, num_channels)

        self.dense1 =  self.make_dense(J, num_channels, num_features, num_blocks, track, normalizer_name, activation_name, inter_factor)
        num_channels += num_blocks * num_features
        num_out_channels = int(math.floor(num_channels*reduction))
        self.trans1 = Transition(J, num_channels, num_out_channels, track, normalizer_name, activation_name)

        num_channels = num_out_channels
        self.dense2 = self.make_dense(J, num_channels, num_features, num_blocks, track, normalizer_name, activation_name, inter_factor)
        num_channels += num_blocks * num_features
        num_out_channels = int(math.floor(num_channels*reduction))
        self.trans2 = Transition(J, num_channels, num_out_channels, track, normalizer_name, activation_name)

        num_channels = num_out_channels
        self.dense3 = self.make_dense(J, num_channels, num_features, num_blocks, track, normalizer_name, activation_name, inter_factor)
        num_channels += num_blocks * num_features
        num_out_channels = int(math.floor(num_channels*reduction))
        self.trans3 = Transition(J, num_channels, num_out_channels, track, normalizer_name, activation_name)

        self.last_layer = ConvLayer(J, num_out_channels, num_classes)


    def make_dense(self, J, num_channels, num_features, num_blocks, track, normalizer_name, activation_name, inter_factor):
        layers = []
        for i in range(int(num_blocks)):
            layers.append(Bottleneck(J, num_channels, num_features, track, normalizer_name, activation_name, inter_factor))
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
    num_layers = 30
    num_features = 16
    num_classes = 1
    J = 5
    reduction = 1/2
    track = 0
    normalizer_name = 'bn'
    model = DenseGNN_4(num_layers, num_features, num_classes, J, reduction, track, normalizer_name)
    W = torch.rand(1, 100, 100, 5)
    x = torch.rand(1, 1, 100, 1)
    input = (W,x)
    print(model)
    print(model(input).shape)


    print('done')
