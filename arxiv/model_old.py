import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataGenerator import dataGenerator
import pdb
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def gmul(input):
# 	# GMul.  Debugging complete.
# 	W, x= input
# 	N = W.shape[2]
# 	W = W.split(1,3) # a list of J tensors of size [bs, N, N, 1]
# 	W = torch.cat(W,2).squeeze(3) # [bs, N, J*N]
# 	W = torch.bmm(x.squeeze(3), W) # [bs, nf, J*N]
# 	W = W.split(N, 2) # a list of J tensors of size [bs, nf, N]
# 	W = torch.cat(W,1).unsqueeze(3) # [bs, J*nf, N, 1]
# 	return W

class MixtureSoftmax(nn.Module):
    # implementing == Breaking the Softmax Bottleneck: A High-Rank RNN Language Model
    # https://arxiv.org/abs/1711.03953
    def __init__(self, N, num_classes, k = 5):
        super(MixtureSoftmax, self).__init__()
        self.N = N
        self.k = k
        self.fc = nn.Linear(N, (1+num_classes)*k)

    def forward(self, input):
        # input dim [bs, N]

        # input = input.to(device).transpose(1,2)   # input dim [bs, 1, N]
        input = self.fc(input)  # now [bs, 1, 3*k]
        # input = input.squeeze() # now [bs, 3*k]
        # pdb.set_trace()
        input = input.view(input.shape[0], self.k, 3)  # now [bs, k, 3]
        Y, w = input[:,:,1:], input[:,:,0]  # Y=[bs, k, num_clas], w=[bs, k]
        w = F.softmax(w, 1).unsqueeze(1)
        Y = F.softmax(Y, 2)
        # output
        Y = torch.bmm(w, Y).squeeze().to(device)  # dim  = [bs, num_classes]
        return Y

def gmul_old(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    N = W.size()[-2]
    W = W.to(device).split(1, 3) # W is a list of J tensors of size (bs, N, N, 1)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    # output = torch.bmm(W, x) # matrix multiplication (J*N,N) x (N,num_features): output has size (bs, J*N, num_features)
    # output = output.split(N, 1) # output is a list of J tensors of size (bs, N, num_features)
    # output = torch.cat(output, 2)
    W = torch.bmm(W, x) # matrix multiplication (J*N,N) x (N,num_features): output has size (bs, J*N, num_features)
    W = W.split(N, 1) # output is a list of J tensors of size (bs, N, num_features)
    W = torch.cat(W, 2)
    # if first layer: size : (bs, N, 1)
    # else: output has size (bs, N, J*num_features)
    return W

class Gconv(nn.Module):
    def __init__(self, feature_maps, J, Jd, initializer, track_running_stats):
        super(Gconv, self).__init__()
        self.num_inputs = J*feature_maps[0] # size of the input
        self.num_outputs = feature_maps[1] # size of the output
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2).to(device)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs // 2).to(device)
        self.bn = nn.BatchNorm1d(self.num_outputs, track_running_stats=track_running_stats).to(device)
        if initializer == 'base':
            pass
        elif initializer == 'xu':
            init.xavier_uniform_(self.fc1.weight)
            init.xavier_uniform_(self.fc2.weight)
        elif initializer == 'xn':
            init.xavier_normal_(self.fc1.weight)
            init.xavier_normal_(self.fc2.weight)
        elif initializer == 'ku_in':
            init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        elif initializer == 'ku_out':
            init.kaiming_uniform_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_uniform_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        elif initializer == 'kn_in':
            init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        elif initializer == 'kn_out':
            init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        W = input[0]
        x = gmul(input).to(device) # x has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous() # makes sure that x is stored in a contiguous chunk of memory
        x = x.view(-1, self.num_inputs)
        x1 = F.relu(self.fc1(x)) # x_1 has size (bs*N, num_outputs // 2)
        x2 = self.fc2(x) # x_2 has size (bs*N, num_outputs // 2)
        x = torch.cat((x1, x2), 1) # x has size (bs*N, num_outputs)
        x = self.bn(x)
        x = x.view(*x_size[:-1], self.num_outputs) # x has size (bs, N, num_outputs)
        return W.to(device), x.to(device)

class Gconv_last(nn.Module):
    def __init__(self, feature_maps, J, Jd, initializer):
        super(Gconv_last, self).__init__()
        self.num_inputs = J*feature_maps[0] # size of the input
        self.num_outputs = feature_maps[1] # size of the output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs).to(device) # the only difference is that there is no activations layer
        if initializer == 'base':
            pass
        elif initializer == 'xu':
            init.xavier_uniform_(self.fc.weight)
        elif initializer == 'xn':
            init.xavier_normal_(self.fc.weight)
        elif initializer == 'ku_in':
            init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
        elif initializer == 'ku_out':
            init.kaiming_uniform_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        elif initializer == 'kn_in':
            init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        elif initializer == 'kn_out':
            init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        W = input[0]
        x = gmul(input).to(device) # out has size (bs, N, num_inputs) ###  num_inputs = num_features * num_filters
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0]*x_size[1], -1) # x has size (bs*N, num_inputs)
        x = self.fc(x) # x has size (bs*N, num_outputs)
        x = x.view(*x_size[:-1], self.num_outputs) # x has size (bs, N, num_outputs)
        x = x.squeeze(2)
        return x.to(device)

class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J, Jd, initializer, track_running_stats):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_first = [1, num_features]
        self.featuremap = [num_features, num_features]
        self.featuremap_last = [num_features, 1]
        self.layer0 = Gconv(self.featuremap_first, J, Jd, initializer, track_running_stats).to(device)
        for i in range(num_layers):
            module = Gconv(self.featuremap, J, Jd, initializer, track_running_stats).to(device)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv_last(self.featuremap_last, J, Jd, initializer)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        cur = self.layerlast(cur)
        return cur



def predict_clique(y):
    return torch.ge(y, 0.).to(device, dtype = torch.float)

def compute_overlap(pred, labels):
    pred = predict_clique(pred)
    intersection = torch.eq(pred + labels, 2).to(device, dtype = torch.float)
    union = torch.clamp(pred+labels,0,1).to(device, dtype = torch.float)
    intersection = torch.sum(intersection, 1)
    union = torch.sum(union, 1)
    overlap = (intersection / union).mean(0).squeeze()
    return overlap.item()


# Loss function
# criterion = nn.CrossEntropyLoss()
# def compute_loss(pred, labels):
#     # pred: [bs, 2], labels: [bs]
#     labels = (labels.contiguous().view(-1)).to(torch.float)
#     return base_loss(pred, labels).to(device)

base_loss = nn.BCEWithLogitsLoss()
def compute_loss(pred, labels):
    pred = pred.view(-1)
    labels = (labels.contiguous().view(-1)).to(torch.float)
    return base_loss(pred, labels).to(device)


# Optimizer
def get_optimizer(model,learning_rate, optimizer_name):
    if optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

# def adjust_lr(optimizer, epoch, init_lr, freq, lr_decay):
#     lr = init_lr * (lr_decay ** (epoch // freq))
#     for param_group in optimizer.param_groups:
#         # print(len(param_group))#,optimizer.param_groups)
#         # print(str(param_group['lr'])+' changed to '+str(lr))
#         print('Learning rate decayed from {} to {}.'.format(param_group['lr'],lr))
#         param_group['lr'] = lr
