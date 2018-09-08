#!/usr/bin/python
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdb
from dataGenerator import dataGenerator
from model import *
from logger import *
from trainer import train, test
import time
nsml_avail = True
try:
    import nsml
except:
    nsml_avail = False

# set seed and device
npr.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
####################################################################################
####################################################################################
####################################################################################
parser.add_argument('--num_features', nargs='?', const=1, type=int, default=16)       # num_features must be divisible by (1/reduction)^4=16
parser.add_argument('--num_layers', nargs='?', const=1, type=int, default=40)
parser.add_argument('--dd', nargs='?', const=1, type=str, default='delta')
parser.add_argument('--tt', nargs='?', const=1, type=float, default=0.766)
parser.add_argument('--ss', nargs='?', const=1, type=float, default=0.1)
parser.add_argument('--initializer', nargs='?', const=1, type=str, default='base')
parser.add_argument('--loss_name', nargs='?', const=1, type=str, default='bce')
parser.add_argument('--pos_weight', nargs='?', const=1, type=float,default=10.)
parser.add_argument('--model_name', nargs='?', const=1, type=str, default='dense4')
parser.add_argument('--optimizer_name', nargs='?', const=1, type=str, default='rmsprop')
parser.add_argument('--inter_factor', nargs='?', const=1, type=int, default=10)
parser.add_argument('--lr', nargs='?', const=1, type=float,default=0.001)
parser.add_argument('--normalizer_name', nargs='?', const=1, type=str, default='bn')
parser.add_argument('--activation_name', nargs='?', const=1, type=str, default='relu')
parser.add_argument('--max_iter', nargs='?', const=1, type=int, default=500000)
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--num_test', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--N', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--mos', nargs='?', const=1, type=int, default=5)
parser.add_argument('--track', nargs='?', const=1, type=int, default=0)
parser.add_argument('--M', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--Jd', nargs='?', const=1, type=int, default=3)
parser.add_argument('--reduction', nargs='?', const=1, type=float, default=1/2)
parser.add_argument('--line_on', nargs='?', const=1, type=int, default=0)
parser.add_argument('--check_activation', nargs='?', const=1, type=int, default=0)
####################################################################################
####################################################################################
####################################################################################
parser.add_argument('--num_classes', nargs='?', const=1, type=int, default=1)
parser.add_argument('--J', nargs='?', const=1, type=int, default=2)
parser.add_argument('--edge_density', nargs='?', const=1, type=float, default=1/2)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='regular')
parser.add_argument('--clique_size', nargs='?', const=1, type=int, default=100)
parser.add_argument('--save_path', nargs='?', const=1, type=str, default='./')
parser.add_argument('--load_path', nargs='?', const=1, type=str, default='./')
parser.add_argument('--log_dir', nargs='?', const=1, type=str, default='logs/')
parser.add_argument('--lr_decay', nargs='?', const=1, type=float, default=1)
####################################################################################
####################################################################################
####################################################################################
parser_args = parser.parse_args()

extra_ops = 3  # how many extra graph operators : currently:: U (average), D (degree), A (from T. Kipf)
clique_size = parser_args.clique_size
NUM_SAMPLES_test = parser_args.num_test
N = parser_args.N
M = parser_args.M
mos = parser_args.mos
J = parser_args.J
Jd= parser_args.Jd
reduction = parser_args.reduction
num_features = parser_args.num_features
num_layers = parser_args.num_layers
num_classes= parser_args.num_classes
edge_density = parser_args.edge_density
max_iter = parser_args.max_iter
batch_size = parser_args.batch_size
mode = parser_args.mode
save_path = parser_args.save_path
os.makedirs(save_path, exist_ok=True)
load_path = parser_args.load_path
log_dir = parser_args.log_dir
log_dir = save_path + log_dir
lr = parser_args.lr
lr_decay = parser_args.lr_decay
tt = parser_args.tt
ss = parser_args.ss
dd = parser_args.dd
initializer = parser_args.initializer
loss_name = parser_args.loss_name
pos_weight = parser_args.pos_weight
model_name = parser_args.model_name
optimizer_name = parser_args.optimizer_name
inter_factor = parser_args.inter_factor
normalizer_name = parser_args.normalizer_name
activation_name = parser_args.activation_name
track = parser_args.track
line_on = parser_args.line_on
if line_on:
    model_name = 'gnn'
check_activation = parser_args.check_activation
# if nsml_avail:
#     nsml.report(
#         nf = num_features,
#         nl = num_layers,
#         dd = dd,
#         tt = tt,
#         ss = ss,
#         init = initializer,
#         opt = optimizer_name,
#         bs = batch_size,
#         nm = normalizer_name,
#         run_stat = track,
#         lr = lr,
#         line_on = line_on,
#         scope=locals(),
#         summary=True)

logger = Logger(log_dir)
logger.write_settings(parser_args)

generator = dataGenerator()
generator.N = N
generator.M = M
generator.J = J
generator.Jd = Jd
generator.line_on = line_on
generator.dd = dd
generator.edge_density = edge_density
generator.clique_size = clique_size
generator.NUM_SAMPLES_test = NUM_SAMPLES_test
generator.tt = tt
generator.ss = ss
generator.extra_ops = extra_ops


################################################################################
# # # DEBUGGING PURPOSES
# write stuff here.



################################################################################
if mode == 'regular':
    # train
    gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
    weight_initializer = get_initializer(initializer, activation_name)
    gnn.apply(weight_initializer)
    optimizer = get_optimizer(gnn,lr,optimizer_name)
    train(gnn, optimizer, generator, logger, loss_name=loss_name, pos_weight=pos_weight, iterations=max_iter, batch_size=batch_size, lr_decay = lr_decay, mos=mos)#10000

    # # resume
    # path = load_path+'models/gnn.pt'
    # data = torch.load(path)
    # gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
    # gnn.load_state_dict(data['model'])
    # optimizer = get_optimizer(gnn,lr,optimizer_name)
    # optimizer.load_state_dict(data['optimizer'])
    # start_epoch = data['epoch']
    # logger.loss_train = data['loss']
    # logger.overlap_train = data['overlap']
    # train(gnn, optimizer, generator, logger, loss_name=loss_name, pos_weight=pos_weight, iterations=max_iter, start_epoch=start_epoch, batch_size=batch_size, lr_decay = lr_decay, mos=mos)#, mode=mode)#10000

elif mode == 'curriculum':
    generator.tt = 1.0
    gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
    weight_initializer = get_initializer(initializer, activation_name)
    gnn.apply(weight_initializer)
    optimizer = get_optimizer(gnn,lr,optimizer_name)
    train(gnn, optimizer, generator, logger, loss_name=loss_name, pos_weight=pos_weight, iterations=20003, batch_size=batch_size, lr_decay = lr_decay, mos=mos)

    # resume
    generator.tt = 0.9
    path = load_path+'models/gnn.pt'
    data = torch.load(path)
    gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
    gnn.load_state_dict(data['model'])
    optimizer = get_optimizer(gnn,lr,optimizer_name)
    optimizer.load_state_dict(data['optimizer'])
    start_epoch = data['epoch']
    logger.loss_train = data['loss']
    logger.overlap_train = data['overlap']
    train(gnn, optimizer, generator, logger, loss_name=loss_name, pos_weight=pos_weight, iterations=50003, start_epoch=start_epoch, batch_size=batch_size, lr_decay = lr_decay, mos=mos)

    # resume
    generator.tt = 0.8
    path = load_path+'models/gnn.pt'
    data = torch.load(path)
    gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
    gnn.load_state_dict(data['model'])
    optimizer = get_optimizer(gnn,lr,optimizer_name)
    optimizer.load_state_dict(data['optimizer'])
    start_epoch = data['epoch']
    logger.loss_train = data['loss']
    logger.overlap_train = data['overlap']
    train(gnn, optimizer, generator, logger, loss_name=loss_name, pos_weight=pos_weight, iterations=100003, start_epoch=start_epoch, batch_size=batch_size, lr_decay = lr_decay, mos=mos)

    # resume
    generator.tt = 0.7
    path = load_path+'models/gnn.pt'
    data = torch.load(path)
    gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
    gnn.load_state_dict(data['model'])
    optimizer = get_optimizer(gnn,lr,optimizer_name)
    optimizer.load_state_dict(data['optimizer'])
    start_epoch = data['epoch']
    logger.loss_train = data['loss']
    logger.overlap_train = data['overlap']
    train(gnn, optimizer, generator, logger, loss_name=loss_name, pos_weight=pos_weight, iterations=180003, start_epoch=start_epoch, batch_size=batch_size, lr_decay = lr_decay, mos=mos)

    # resume
    generator.tt = 0.6
    path = load_path+'models/gnn.pt'
    data = torch.load(path)
    gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
    gnn.load_state_dict(data['model'])
    optimizer = get_optimizer(gnn,lr,optimizer_name)
    optimizer.load_state_dict(data['optimizer'])
    start_epoch = data['epoch']
    logger.loss_train = data['loss']
    logger.overlap_train = data['overlap']
    train(gnn, optimizer, generator, logger, loss_name=loss_name, pos_weight=pos_weight, iterations=280003, start_epoch=start_epoch, batch_size=batch_size, lr_decay = lr_decay, mos=mos)


    # resume
    generator.tt = 0.5
    path = load_path+'models/gnn.pt'
    data = torch.load(path)
    gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
    gnn.load_state_dict(data['model'])
    optimizer = get_optimizer(gnn,lr,optimizer_name)
    optimizer.load_state_dict(data['optimizer'])
    start_epoch = data['epoch']
    logger.loss_train = data['loss']
    logger.overlap_train = data['overlap']
    train(gnn, optimizer, generator, logger, loss_name=loss_name, pos_weight=pos_weight, iterations=380003, start_epoch=start_epoch, batch_size=batch_size, lr_decay = lr_decay, mos=mos)

# test
t1 = time.time()
model_path = 'gnn.pt'
load_path = load_path+'models/' + model_path
load_data = torch.load(load_path)
gnn = gnn = get_model(model_name, num_layers, num_features, num_classes, J+extra_ops, Jd+extra_ops, track, normalizer_name, line_on, check_activation, reduction, activation_name, inter_factor).to(device)
gnn.load_state_dict(load_data['model'])
test(gnn,generator,logger)
print('Test finished.')
print('Total Testing Time Spent: ', time.time()-t1)
