import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pdb
from dataGenerator import dataGenerator
from model import *
from logger import *
import time
nsml_avail = True
try:
    import nsml
except:
    nsml_avail = False
# import visdom as viz

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, optim, generator, logger, loss_name, pos_weight, mos=5, iterations=60000, start_epoch = None,
        batch_size=32, clip_grad_norm=40.0, learning_decay_freq = 2000, lr_decay = 1.0,
        print_freq=10, save_freq = 1000, test_freq = 1000):
    # model should be a gnn
    # generator is the data_generator
    # mixture_softmax = MixtureSoftmax(generator.N, 2, mos).to(device)
    optimizer = optim
    criterion = get_loss_function(loss_name, pos_weight)
    if batch_size == 1:
        save_freq = 5000
        test_freq = 5000
    plot_freq = 1000
    tb_freq   = 10000
    if start_epoch is not None:
        iteration_range = range(start_epoch, iterations+1)
    else:
        iteration_range = range(iterations+1)

    for iter_count in iteration_range:
        t1 = time.time()
        G, labels = generator.sample_batch(batch_size)
        pred = model(G)
        # pred = mixture_softmax(pred)   # MOS!  Comment out if not using.
        # loss = compute_loss(pred, labels)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        logger.add_train_loss(loss, iter_count)
        logger.add_train_overlap(pred,labels, iter_count)


        if iter_count % print_freq == 0: #and iter_count > 0:
            t2 = time.time()
            if batch_size > 1:
                print('Mode: Train || BS: {:<4} It: {:<7} Ls: {:<10.5f} OvLp: {:<10.5f} T: {:<7.2f}'.format(
                labels.shape[0],
                iter_count,
                loss.item(),
                compute_overlap(pred,labels),
                t2-t1))
            elif batch_size == 1:
                print('Mode: Train || K: {:<4} It: {:<7} Ls: {:<10.5f} OvLp: {:<10.5f} T: {:<7.2f}'.format(
                int(labels.sum().item()),
                iter_count,
                loss.item(),
                compute_overlap(pred,labels),
                t2-t1))

        if (iter_count % plot_freq == 0):
            logger.plot_train_loss()
            logger.plot_train_overlap()

        if (iter_count % test_freq == 0):
            # logger.plot_layer_activation(model)
            logger.save_model(model,optim,iter_count,batch_size)
            test(model, generator, logger, epoch = iter_count)

        # # 1. TB Log scalar values (scalar summary)
        # info = { 'loss': loss.item(), 'overlap': compute_overlap(pred,labels) }
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, iter_count+1)

        # if iter_count % tb_freq == 0:
        #     # 2. Log values and gradients of the parameters (histogram summary)
        #     for tag, value in model.named_parameters():
        #         tag = tag.replace('.', '/')
        #         logger.histo_summary(tag, value.data.cpu().numpy(), iter_count+1)
        #         logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), iter_count+1)

        # if (iter_count % save_freq == 0):
        #     logger.save_model(model,optim,iter_count,batch_size)
        #     # logger.save_results()

        # if iter_count % learning_decay_freq == 0: #and iter_count > 0:
        #     adjust_lr(optimizer, iter_count, logger.args['learning_rate'],learning_decay_freq, lr_decay)
    logger.save_results()
    print('Optimization finished.')

def test(model, generator, logger, epoch=None):
    # model should be a gnn

    model.eval()  # don't use moving mean... worst performance.
    # with torch.no_grad():
    N = generator.N
    howmanyruns = generator.NUM_SAMPLES_test
    low = int(0.2*np.sqrt(N))
    high = int(1.4*np.sqrt(N))
    cliques = np.arange(low,high)
    data = np.empty((howmanyruns,len(cliques)))


    for idx in range(len(cliques)):
        generator.clique_size = cliques[idx]
        logger.args['clique_size'] = cliques[idx]
        for jdx in range(howmanyruns):
            t1 = time.time()
            G, labels = generator.sample_batch(1, is_training=False)
            # print('label clique size: ', labels.sum().item())
            pred = model(G)
            data[jdx,idx] = compute_overlap(pred,labels)
            if ((jdx % 100) == 0):
                print('Mode: Test || K: {:<4} Iter: {:<7} OvLp: {:<10.5f} T: {:<7.2f}'.format(
                    int(torch.sum(labels).item()),
                    jdx,
                    data[jdx,idx],
                    time.time()-t1)
                )
        print('------------------------------------------------------')
    x = cliques/np.sqrt(N)
    y = np.mean(data, axis = 0)
    yerr = np.std(data, axis = 0)
    logger.plot_test_overlap(data, x, y, yerr, epoch)
    model.train() # done with evaluation
