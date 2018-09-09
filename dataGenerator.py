import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import pickle
import os.path
import pdb
import time
from multiprocessing import Pool
from sparsify import Phase2Edges

import matplotlib.pyplot as plt
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class dataGenerator:
    def __init__(self):
        self.NUM_SAMPLES_train = int(10e6)
        self.NUM_SAMPLES_test = int(10e4)
        self.data_train = []
        self.data_test = []
        self.J = 2
        self.N = 1000
        self.Jd = 2
        self.edge_density = 0.5
        self.clique_size = 10
        self.tt = 1.0
        self.ss = 0.1
        self.dd = 'uniform'
        self.M = 2*self.N
        self.extra_ops = 3
        self.line_on = True

    def compute_ops(self, H):
        # To compute Pd, Pm, B(=Wd).
        # input : a sparse graph H
        # output : Wd, P
        #         -  Wd is [M, M]
        #         -  P is  [N, M, 2]
        N = self.N
        edges = H.nonzero().to(device)
        E = self.M
        # Pd = torch.zeros((N,E),device=device)  # oriented
        B = torch.zeros((E,E),device=device)
        row, col = edges.t()[0], edges.t()[1]
        for i in range(E):
            if i == E-1:
                u = row[i]
                mask_plus, mask_minus = (row==u).to(device), (col==u).to(device)
                # Pd[u][mask_plus]  =  1
                # Pd[u][mask_minus] = -1
                loc_target_u = (row ==u).to(device)
                val_target_u = col.clone().to(device)
                val_target_u[~loc_target_u] = 0
                loc_source_v = (torch.eq(row, val_target_u.view(-1,1))[loc_target_u.nonzero().squeeze()]).to(device)
                if len(loc_source_v.shape) == 1:
                    loc_source_v = loc_source_v.byte().to(device)
                else:
                    loc_source_v = torch.sum(loc_source_v,0).byte().to(device)
                val_target_v = col.clone().to(device)
                val_target_v[~loc_source_v] = 0
                mask = (val_target_v == u).to(device)
                loc_source_v[mask]=0
                val_target_v[mask]=0
                B[i] = loc_source_v.int().clone().to(device)
            else:
                if row[i] == row[i+1]:
                    pass
                else:
                    u = row[i]
                    mask_plus, mask_minus = (row==u).to(device), (col==u).to(device)
                    # Pd[u][mask_plus]  =  1
                    # Pd[u][mask_minus] = -1
                    loc_target_u = (row == u).to(device)
                    val_target_u = col.clone().to(device)
                    val_target_u[~loc_target_u] = 0
                    loc_source_v = (torch.eq(row, val_target_u.view(-1,1))[loc_target_u.nonzero().squeeze()]).to(device)
                    if len(loc_source_v.shape) == 1:
                        loc_source_v = loc_source_v.byte().to(device)
                    else:
                        loc_source_v = torch.sum(loc_source_v,0).byte().to(device)
                    val_target_v = col.clone().to(device)
                    val_target_v[~loc_source_v] = 0
                    mask = (val_target_v == u).to(device)
                    loc_source_v[mask]=0
                    val_target_v[mask]=0
                    B[i] = loc_source_v.int().clone().to(device)
        perm = torch.randperm(E)
        B = B[perm].to(device).contiguous()
        # Pm = torch.abs(Pd).to(device)
        # P  = torch.stack((Pm,Pd),1).transpose(1,2)
        # P  = P[:,perm].to(device).contiguous()
        return B#, P

    def compute_ops_B_prime(self, deg, W):
        N, M, bs = self.N, self.M, self.batch_size
        Wd = torch.zeros((bs,M,M),device=device)
        for i in range(bs):
            III = torch.eye(N, device=device)
            BBB = torch.zeros((N,N),device=device)
            BBB = torch.cat((BBB, -III))
            temp = torch.cat(((torch.diagflat(deg[i]).to(device))-III, (W[i]).clone()))
            BBB = torch.cat((BBB, temp), 1)
            Wd[i] = BBB.clone()
        return Wd.to(device)

    def get_maps(self, W, J):
        # compute operators
        N, bs = W.shape[1], self.batch_size
        deg = torch.sum(W.clone(), dim=2, keepdim=True).to(device)
        I = torch.eye((N),device = device).expand(bs, *(N,N))
        A = (W + I).clone().to(device)
        inv_sqrt_DD= torch.sqrt(1/torch.sum(A.clone(), dim=2, keepdim=True)).to(device)

        # filling in ops
        OP = torch.zeros((bs, N, N, J+self.extra_ops), device = device)
        OP[:, :, :, 0] = I.clone()
        W_pow = (W.clone()).to(device)
        for j in range(J-1):
            OP[:, :, :, j+1] = (W_pow.clone()).to(device)
            if j == J-2:
                break
            W_pow = (torch.min(torch.bmm(W_pow,W_pow),torch.ones(*W.size(), device=device))).to(device)
        for k in range(bs):
            OP[k, :, :, J]   = ((torch.diagflat(deg[k])).clone()).to(device)
            OP[k, :, :, J+self.extra_ops-2] = ((torch.diagflat(inv_sqrt_DD[k])).clone()).to(device)
        OP[:, :, :, J+self.extra_ops-1] = ((1.0 / float(N)) * torch.ones((bs, N, N), device = device)).to(device)

        return OP.to(device), deg.unsqueeze(1).to(device)

    def create_dataset(self, is_training=True):
        bs, N, M, p, J, Jd = self.batch_size, self.N, self.M, self.edge_density, self.J, self.Jd

        # erdos_renyi
        W = torch.rand((bs,N,N),device = device)
        W[W >= 1-p] = 1
        W[W < 1-p] = 0
        W = W * (torch.ones((N,N), device=device).triu()).expand(*W.size())
        W = (W + W.transpose(1,2)).clone().to(device)

        # generate clique
        if is_training:
            if self.dd == 'normal':
                mu = float(self.tt * np.sqrt(N))
                sigma = float(self.ss * np.sqrt(N))
                C = torch.round(torch.normal(mu*torch.ones(bs,device=device),sigma*torch.ones(bs,device=device))).to(device, dtype=torch.int)
            elif self.dd == 'uniform':
                mu, bound = float(self.tt * np.sqrt(N)), float(self.ss * np.sqrt(N))
                low, high = int(np.around(mu-bound)), int(np.around(mu+bound))
                C = torch.randint(low=low, high=high, size=(bs,), dtype=torch.int, device=device).to(device, dtype = torch.int)
            elif self.dd == 'delta':
                loc = int(np.around(self.tt * np.sqrt(N)))
                C = (loc * torch.ones(bs, device = device, dtype = torch.int)).to(device, dtype = torch.int)
        else:
            C = (self.clique_size * torch.ones(bs, device=device)).to(device, dtype=torch.int)
        clique_labeling = torch.zeros((bs,N),device=device, dtype = torch.int)
        for i in range(bs):
            clique = torch.zeros((N), device=device)
            clique[torch.randperm(N)[0:C[i].item()]] = 1
            mask1 = torch.ger(clique,clique).byte()
            mask2 = torch.eye((N),dtype=torch.uint8,device=device)
            W[i][mask1]=1   # cliques
            W[i][mask2]=0   # kill diagonals :: self-loops
            clique_labeling[i] = clique.to(device)
        clique_labeling=clique_labeling.to(device, dtype=torch.float)
        OP, deg = self.get_maps(W, J)

        # if self.line_on: # line graph ops

            # for uniform random edge selection
            # sparsifier = UniformEdges(M).to(device)
            # Wd = torch.zeros((bs,M,M),device=device)
            # P = torch.zeros((bs,N,M,2),device=device)
            # for i in range(bs): # sparsify and get ops for line graph GNN
            #     H = sparsifier.sparsify(W[i]).to(device)
            #     Wd[i], P[i] = self.compute_ops(H)
            # P = P.to(device)
            # OPd, degd = self.get_maps(Wd, Jd)


            # # using B' : https://arxiv.org/pdf/1306.5550.pdf
            # M = 2*N
            # self.M = 2*N
            # Wd = self.compute_ops_B_prime(deg, W)
            # P = torch.zeros((bs,N,M,2),device=device).to(device) # will be 0 at all.
            # OPd, degd = self.get_maps(Wd, Jd)
            # # OP [bs, N, N, J+extra_ops]
            # # deg [bs, 1, N, 1]
            # # OPd [bs, M, M, Jd+extra_ops]
            # # P [bs, N, M, 2]
            # # degd [bs, 1, M, 1]
            # # cliq [bs, N]
            # return [OP, deg, OPd, P, degd], clique_labeling

        # sparsifier = Phase2Edges(M).to(device)
        # Wd = torch.zeros((bs,M,M),device=device)
        # for i in range(bs): # sparsify and get ops for line graph GNN
        #     H = sparsifier.sparsify(W[i]).to(device)

        # return below if line_off
        return [OP, deg], clique_labeling

    def sample_batch(self, batch_size, is_training=True):
        self.batch_size = batch_size
        return self.create_dataset(is_training)

    def sparsify_and_sample(self, pred):
        M = self.M
        bs = self.batch_size
        Jd = self.Jd
        sparsifier = Phase2Edges(M)
        Wd = torch.zeros((bs,M,M),device=device)
        for i in range(bs):
            H = sparsifier.sparsify(pred)
            Wd[i] = self.compute_ops(H)
        OPd, degd = self.get_maps(Wd, Jd)
        return [OPd.to(device), degd.to(device)]








# if __name__=='__main__':



    # def create_train_dataset(self):
    #     ### data generation through parallelizing ###
    #     self.data_train = []
    #     pool = Pool()
    #     required = [self.N, self.edge_density, self.J]
    #     inputs = [required for _ in range(self.batch_size)]
    #     self.data_train = pool.map(make_sample,inputs)
    #     pool.close()
    #     pool.join()
