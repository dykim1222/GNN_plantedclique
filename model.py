import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataGenerator import dataGenerator
import pdb
import sys
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# save for later...!
# class MixtureSoftmax(nn.Module):
#     # implementing == Breaking the Softmax Bottleneck: A High-Rank RNN Language Model
#     # https://arxiv.org/abs/1711.03953
#     def __init__(self, N, num_classes, k = 5):
#         super(MixtureSoftmax, self).__init__()
#         self.N = N
#         self.k = k
#         self.fc = nn.Linear(N, (1+num_classes)*k)
#
#     def forward(self, input):
#         # input dim [bs, N]
#
#         # input = input.to(device).transpose(1,2)   # input dim [bs, 1, N]
#         input = self.fc(input)  # now [bs, 1, 3*k]
#         # input = input.squeeze() # now [bs, 3*k]
#         input = input.view(input.shape[0], self.k, 3)  # now [bs, k, 3]
#         Y, w = input[:,:,1:], input[:,:,0]  # Y=[bs, k, num_clas], w=[bs, k]
#         w = F.softmax(w, 1).unsqueeze(1)
#         Y = F.softmax(Y, 2)
#         # output
#         Y = torch.bmm(w, Y).squeeze().to(device)  # dim  = [bs, num_classes]
#         return Y

def gmul(W, x):
    # GMul.  Debugging complete.
    N = W.shape[2]
    W = W.to(device).split(1,3) # a list of J tensors of size [bs, N, N, 1]
    W = torch.cat(W,2).squeeze(3) # [bs, N, J*N]
    W = torch.bmm(x.squeeze(3), W) # [bs, nf, J*N]
    W = W.split(N, 2) # a list of J tensors of size [bs, nf, N]
    W = torch.cat(W,1).unsqueeze(3).contiguous().to(device) # [bs, J*nf, N, 1]
    return W

class gnn_atomic(nn.Module):
    def __init__(self, feature_maps, J, track, check_activation, normalizer_name, activation_name):  #[1,nf], [nf,nf]
        super(gnn_atomic, self).__init__()
        # if check_activation:
        #     self.check_activation = check_activation
        #     self.activation_ratio = [0]
        num_output = int(feature_maps[1]/2)
        self.conv1 = nn.Conv2d(feature_maps[0]*J, num_output, 1).to(device)
        self.conv2 = nn.Conv2d(feature_maps[0]*J, num_output, 1).to(device)
        self.bn = get_normalizer(normalizer_name, feature_maps[1], track).to(device)
        self.act = get_activation_function(activation_name)

    def forward(self, input):
        W, x = input # [bs, N, N, J+extra_ops], [bs, 1, N, 1]
        x = gmul(W,x) # [bs, J+extra_ops, N, 1]
        x1 = self.act(self.conv1(x))
        x2 = self.conv2(x)
        x = torch.cat((x1,x2),1)
        x = self.bn(x).contiguous().to(device)
        # if self.check_activation:
        #     self.activation_ratio.append(((~(torch.abs(x1)<1e-6)).sum().item() / (x1.view(-1).shape[0])))
        return W, x

class gnn_last(nn.Module):
    def __init__(self, num_features, J, nclasses):
        super(gnn_last, self).__init__()
        self.conv = nn.Conv2d(num_features*J, nclasses, 1).to(device)

    def forward(self, input):
        W, x = input
        x = gmul(W,x)
        x = self.conv(x)
        x = x.squeeze(3).transpose(1,2).squeeze(2).contiguous().to(device)
        return x

class gnn_line_atomic(nn.Module):
    def __init__(self, feature_maps, J, Jd, track, last, check_activation, normalizer_name, activation_name):
        super(gnn_line_atomic, self).__init__()
        # if check_activation:
        #     self.check_activation = check_activation
        #     self.activation_ratio = [0]
        self.last = last
        num_output = int(feature_maps[1]/2)
        self.conv1 = nn.Conv2d(feature_maps[0]*(J+2), num_output, 1).to(device)
        self.conv2 = nn.Conv2d(feature_maps[0]*(J+2), num_output, 1).to(device)
        self.conv3 = nn.Conv2d(Jd*feature_maps[0]+2*feature_maps[1], num_output, 1).to(device)
        self.conv4 = nn.Conv2d(Jd*feature_maps[0]+2*feature_maps[1], num_output, 1).to(device)
        self.bn1 = get_normalizer(normalizer_name, feature_maps[1], track).to(device)
        self.bn2 = get_normalizer(normalizer_name, feature_maps[1], track).to(device)
        self.act1 = get_activation_function(activation_name)
        self.act2 = get_activation_function(activation_name)
        # self.bn1 = nn.BatchNorm2d(feature_maps[1], track_running_stats=track).to(device)
        # self.bn2 = nn.BatchNorm2d(feature_maps[1], track_running_stats=track).to(device)

    def forward(self, input):
        W, x, Wd, P, y = input

        x1 = gmul(W,x)  # ch: x.ch * (J+ex)
        x2 = gmul(P.transpose(1,2),y) # ch: y.ch * 2
        x  = torch.cat((x1,x2),1) # ch: sum of above two
        x1  = self.act1(self.conv1(x)) # ch: out
        # if self.check_activation:
        #     self.activation_ratio.append(((~(torch.abs(x1)<1e-6)).sum().item() / (x1.view(-1).shape[0])))
        x2 = self.conv2(x) # ch: out
        x = torch.cat((x1,x2),1) # ch: 2*out
        x = self.bn1(x).contiguous().to(device)

        x1 = gmul(Wd,y) # ch: y.ch*(Jd+ex)
        x2 = gmul(P,x) #or P.t()???  # ch: 2*out*2
        y  = torch.cat((x1,x2),1) # sum above
        x1 = self.act2(self.conv3(y)) # out
        x2 = self.conv4(y) # out
        y =torch.cat((x1,x2),1) # 2*out
        y =self.bn2(y).contiguous().to(device)
        # if self.check_activation:
        #     self.activation_ratio.append(((~(torch.abs(x1)<1e-6)).sum().item() / (x1.view(-1).shape[0])))
        if self.last:
            return W, x, P, y
        else:
            return W, x, Wd, P, y

class gnn_line_last(nn.Module):
    def __init__(self, num_features, J, nclasses):
        super(gnn_line_last, self).__init__()
        self.conv = nn.Conv2d(num_features*(J+2), nclasses, 1).to(device)

    def forward(self, input):
        W, x, P, y = input
        x1 = gmul(W,x)
        x2 = gmul(P.transpose(1,2),y)
        x = torch.cat((x1,x2),1)
        x = self.conv(x)
        x = x.squeeze(3).transpose(1,2).squeeze(2).contiguous().to(device)
        return x

class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J, Jd, nclasses, track, line_on, check_activation, normalizer_name, activation_name):
        # HERE (J, Jd) IS ACTUALLY (J+extra_ops, Jd+extra_ops) in main.py
        super(GNN, self).__init__()
        self.num_layers = num_layers

        if line_on: # line GNN
            featuremap_beg, featuremap_mid = [1,num_features], [num_features, num_features]
            self.add_module('layer{}'.format(0), gnn_line_atomic(featuremap_beg, J, Jd, track, 0, check_activation, normalizer_name, activation_name).to(device))
            for i in range(1, num_layers):
                self.add_module('layer{}'.format(i), gnn_line_atomic(featuremap_mid, J, Jd, track, 0, check_activation, normalizer_name, activation_name).to(device))
            self.add_module('layer{}'.format(num_layers), gnn_line_atomic(featuremap_mid, J, Jd, track, 1, check_activation, normalizer_name, activation_name).to(device))
            self.last_layer = gnn_line_last(num_features, J, nclasses).to(device)

        else: # regular GNN
            featuremap_beg, featuremap_mid = [1,num_features], [num_features, num_features]
            self.add_module('layer{}'.format(0), gnn_atomic(featuremap_beg, J, track, check_activation, normalizer_name, activation_name).to(device))
            for i in range(1, num_layers):
                self.add_module('layer{}'.format(i), gnn_atomic(featuremap_mid, J, track, check_activation, normalizer_name, activation_name).to(device))
            self.add_module('layer{}'.format(num_layers), gnn_atomic(featuremap_mid, J, track, check_activation, normalizer_name, activation_name).to(device))
            self.last_layer = gnn_last(num_features, J, nclasses).to(device)

    def forward(self, input):
        for i in range(self.num_layers+1):
            input = self._modules['layer{}'.format(i)](input)
        input = self.last_layer(input)
        return input















# def predict_clique(y):
#     return torch.ge(y, 0.).to(device, dtype = torch.float)
#
# def compute_overlap(pred, labels):
#     pred = predict_clique(pred)
#     intersection = torch.eq(pred + labels, 2).to(device, dtype = torch.float)
#     union = torch.clamp(pred+labels,0,1).to(device, dtype = torch.float)
#     intersection = torch.sum(intersection, 1)
#     union = torch.sum(union, 1)
#     overlap = (intersection / union).mean(0).squeeze()
#     return overlap.item()


# Loss function
# criterion = nn.CrossEntropyLoss()
# def compute_loss(pred, labels):
#     # pred: [bs, 2], labels: [bs]
#     labels = (labels.contiguous().view(-1)).to(torch.float)
#     return base_loss(pred, labels).to(device)

def predict_clique(y, clique_sizes):
    # return torch.ge(y, 0.).to(device, dtype = torch.float)
    # shapes: y=[bs, N], clique_sizes=[bs]
    _, idx = torch.sort(-y, dim=1)
    prediction = torch.zeros_like(y, device=device, dtype=torch.float).to(device, dtype=torch.float)
    for i in range(idx.shape[0]): # running over batch_size
        id = (idx[i][:clique_sizes[i].item()]).clone()
        prediction[i][id] = 1
    return prediction

def compute_overlap(pred, labels):
    # pred and labels shape [bs, N]
    clique_sizes = labels.sum(1).int() # justification on using this info: I can use this info on prediction as bp and sp do.
    pred = predict_clique(pred, clique_sizes)
    intersection = torch.eq(pred + labels, 2).to(device, dtype = torch.float)
    union = torch.clamp(pred+labels,0,1).to(device, dtype = torch.float)
    intersection = torch.sum(intersection, 1)
    union = torch.sum(union, 1)
    overlap = (intersection / union).mean(0).squeeze()
    return overlap.item()


class TPLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(TPLoss, self).__init__()
        self.pos_weight = pos_weight if pos_weight is not None else 1.

    def forward(self, pred, labels):
        #  torch.Size([2, 1000]) torch.Size([2, 1000])
        out = torch.zeros(pred.shape[0], device=device)
        pred = F.sigmoid(pred.contiguous())
        labels = (labels.contiguous()).to(torch.float)
        for i in range(pred.shape[0]):
            TP = torch.dot(pred[i], labels[i])
            TN = torch.dot(1-pred[i], 1-labels[i])
            # print(TP, TN)
            # print(pred[i].sum())
            # out[i] =  TP/(1-TN)
            out[i] =  TP/(1-TP-TN)
        # return -torch.log(out).mean()
        return -out.mean()
        # TP = pred*labels   # true positives
        # out = (-torch.log(self.pos_weight*(TP)/(pred+labels-TP) + (1+TP-pred-labels)/(1-TP))).mean()
        # print(out)
        # return out

class TPTNLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(TPTNLoss, self).__init__()
        self.pos_weight = pos_weight if pos_weight is not None else 1.

    def forward(self, pred, labels):
        #  torch.Size([2, 1000]) torch.Size([2, 1000])
        out = torch.zeros(pred.shape[0], device=device)
        pred = F.sigmoid(pred.contiguous())
        labels = (labels.contiguous()).to(torch.float)
        for i in range(pred.shape[0]):
            TP = torch.dot(pred[i], labels[i])
            TN = torch.dot(1-pred[i], 1-labels[i])
            out[i] =  (self.pos_weight*TP + TN)/(1-TN)
        return -torch.log(out).mean()

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight if pos_weight is not None else 1.

    def forward(self, input, target):
        # pred = F.sigmoid(pred.contiguous().view(-1))
        # labels = (labels.contiguous().view(-1)).to(torch.float)
        # loss = (-(self.pos_weight*(labels*torch.log(pred)) + ((1-labels)*torch.log(1-pred)))).mean()
        # return loss
        input = input.contiguous().view(-1)
        target = (target.contiguous().view(-1)).to(torch.float)
        max_val = (-input).clamp(min=0)
        # loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        loss = input - input*target + (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())*(1+(self.pos_weight-1)*target)
        return loss.mean()




def get_loss_function(loss_name, pos_weight = None):
    if loss_name == 'bce':
        return WeightedBCEWithLogitsLoss(pos_weight).to(device)
        # return nn.BCEWithLogitsLoss().to(device)
    elif loss_name == 'tp':
        return TPLoss(pos_weight).to(device)
    elif loss_name == 'tptn':
        return TPTNLoss(pos_weight).to(device)
    else:
        print('Loss name is not recognized.')
        sys.exit()

# def compute_loss(crit, pred, labels):
#     pred = pred.contiguous().view(-1)
#     labels = (labels.contiguous().view(-1)).to(torch.float)
#     return base_loss(pred, labels).to(device)


# base_loss = nn.BCEWithLogitsLoss().to(device)
# def compute_loss(pred, labels):
#     pred = pred.contiguous().view(-1)
#     labels = (labels.contiguous().view(-1)).to(torch.float)
#     return base_loss(pred, labels).to(device)

# activation function
def get_activation_function(activation_name):
    if activation_name == 'elu':
        return nn.ELU(alpha=1.0, inplace=False)
    elif activation_name == 'hardshrink':
        return nn.Hardshrink(lambd=0.5)
    elif activation_name == 'hardtanh':
        return nn.Hardtanh(min_val=-1, max_val=1, inplace=False, min_value=None, max_value=None)
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.03, inplace=False)
    elif activation_name == 'logsigmoid':
        return nn.LogSigmoid()
    elif activation_name == 'prelu':
        return nn.PReLU(num_parameters=1, init=0.25)
    elif activation_name == 'relu':
        return nn.ReLU(inplace=False)
    elif activation_name == 'relu6':
        return nn.ReLU6(inplace=False)
    elif activation_name == 'rrelu':
        return nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)
    elif activation_name == 'selu':
        return nn.SELU(inplace=False)
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'softplus':
        return nn.Softplus(beta=1, threshold=20)
    elif activation_name == 'softshrink':
        return nn.Softshrink(lambd=0.5)
    elif activation_name == 'softsign':
        return nn.Softsign()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'tanhshrink':
        return nn.Tanhshrink()
    elif activation_name == 'swish':
        def swish(x):
            return x * F.sigmoid(x)
        return swish
    else:
        print('Activation function name is not recognized.')
        sys.exit()

# models
def get_model(model_name, num_layers, num_features, num_classes, J, Jd, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor):
    if model_name == 'gnn':
        return GNN(num_features, num_layers, J, Jd, num_classes, track, line_on, check_activation, normalizer_name, activation_name).to(device)
    elif model_name == 'dense1':
        from dense_1 import DenseGNN_1
        return DenseGNN_1(num_layers, num_features, num_classes, J, reduction, track, normalizer_name, activation_name).to(device)
    elif model_name == 'dense2':
        from dense_2 import DenseGNN_2
        return DenseGNN_2(num_layers, num_features, num_classes, J, reduction, track, normalizer_name, activation_name).to(device)
    elif model_name == 'dense3':
        from dense_3 import DenseGNN_3
        return DenseGNN_3(num_layers, num_features, num_classes, J, reduction, track, normalizer_name, activation_name, inter_factor).to(device)
    elif model_name == 'dense4':
        from dense_4 import DenseGNN_4
        return DenseGNN_4(num_layers, num_features, num_classes, J, reduction, track, normalizer_name, activation_name, inter_factor).to(device)
    elif model_name == 'dense42':
        from dense_4 import DenseGNN_42
        return DenseGNN_42(num_layers, num_features, num_classes, Jd, reduction, track, normalizer_name, activation_name, inter_factor).to(device)
    elif model_name == 'dense5':
        from dense_5 import DenseGNN_5
        return DenseGNN_5(num_layers, num_features, num_classes, J, reduction, track, normalizer_name, activation_name, inter_factor).to(device)
    else:
        print('Model name is not recongnized.')
        sys.exit()

# normlaization
def get_normalizer(normalizer_name, nftrs, track):
    if normalizer_name == 'bn':
        normalizer = nn.BatchNorm2d(nftrs, track_running_stats=track)
    elif normalizer_name == 'ln':
        nm_size = torch.Size((1,nftrs,1000,1))
        normalizer = nn.LayerNorm(nm_size, elementwise_affine=track)
    elif normalizer_name == 'in':
        normalizer = nn.InstanceNorm2d(nftrs, track_running_stats=track)
    else:
        print('Normalizer name is not recognized.')
        sys.exit()
    return normalizer

# Optimizer
def get_optimizer(model,learning_rate, optimizer_name):
    if optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'lbfgs':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'rprop':
        optimizer = optim.Rprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5, nesterov=True)
    else:
        print('Optimizer name is not recognized')
        sys.exit()
    return optimizer

def get_initializer(initializer, activation_name):
    if initializer == 'base':
        def init_weights(m):
            pass
        return init_weights
    elif initializer == 'xu':
        def init_weights(m):
            if type(m) == nn.Conv2d:
                init.xavier_uniform_(m.weight)
        return init_weights
    elif initializer == 'xn':
        def init_weights(m):
            if type(m) == nn.Conv2d:
                init.xavier_normal_(m.weight)
        return init_weights
    elif initializer == 'ku_in':
        def init_weights(m):
            if type(m) == nn.Conv2d:
                init.kaiming_uniform_(m.weight, nonlinearity=activation_name)
        return init_weights
    elif initializer == 'ku_out':
        def init_weights(m):
            if type(m) == nn.Conv2d:
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity=activation_name)
        return init_weights
    elif initializer == 'kn_in':
        def init_weights(m):
            if type(m) == nn.Conv2d:
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation_name)
        return init_weights
    elif initializer == 'kn_out':
        def init_weights(m):
            if type(m) == nn.Conv2d:
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_name)
    else:
        print('Initializer method is not recognized.')
        sys.exit()









# def adjust_lr(optimizer, epoch, init_lr, freq, lr_decay):
#     lr = init_lr * (lr_decay ** (epoch // freq))
#     for param_group in optimizer.param_groups:
#         # print(len(param_group))#,optimizer.param_groups)
#         # print(str(param_group['lr'])+' changed to '+str(lr))
#         print('Learning rate decayed from {} to {}.'.format(param_group['lr'],lr))
#         param_group['lr'] = lr

# if __name__ == '__main__':
    # featuremap_in= [1, 1, 4]
    # featuremap_mi= [8, 8, 4]
    # featuremap_end=[8, 8, 1]
    # J = 5
    # Jd = 5
    # initializer = 'base'
    # track=True
    #
    #
    # conv = gnn_sq_atomic(featuremap_in, J, Jd, initializer, track)
