import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import pdb
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class UniformEdges(nn.Module):
    # sampling edges with uniform distribution
    # theoretical reason: https://cs.stanford.edu/~jure/pubs/sampling-kdd06.pdf
    def __init__(self, num_sample_edges):
        super(UniformEdges, self).__init__()
        self.num_sample_edges = num_sample_edges

    def sparsify(self, W):
        # input: W
        # output: H, sparsified.
        H = torch.zeros(*W.size(), device=device).to(device)
        edges = (W.triu().nonzero()).to(device) # index of edges with shape [nE, 2]
        num_edges = edges.shape[0]
        edges_idx = (torch.randperm(num_edges, dtype=torch.int64)[:int(np.around(self.num_sample_edges/2))]).to(device)
        edges = edges[edges_idx].t()
        H[edges[0],edges[1]] = 1
        H = (H + H.t()).clone().to(device)
        return H

class Phase2Edges(nn.Module):
    # y_ij = P((i->j) \in E_{k,k}) = y_i * y_j
    def __init__(self, num_sample_edges):
        super(Phase2Edges, self).__init__()
        self.num_sample_edges = num_sample_edges

    def sparsify(self, pred):
        # input: predictions on nodes from phase 1
        # output: H, sparsified adjacency matrix

        # pred shape =  [1, N]
        N = pred.shape[1]
        pred = pred.squeeze()
        pred = torch.ger(pred, pred).to(device)
        # to kill diagonal entries
        mask = torch.eye(N,device=device).to(device).byte()
        pred[mask] = 0
        pred = pred.view(-1)

        _, ind = torch.sort(-pred)
        ind = ind[:int(np.around(self.num_sample_edges))]

        # now pred = output
        pred = torch.zeros(*(N,N), device=device).to(device)
        pred[(ind // N), (ind %  N)] = 1
        return pred


if __name__=='__main__':
    sparsifier = Phase2Edges(20)
    pred = torch.abs(torch.rand(6))
    H = sparsifier.sparsify(pred)




# sample nodes according to degree distribution
# select edges uniformly

# class DegreeNodesUniformEdges(nn.Module):
#     # sampling nodes with degree distribution
#     # and then sampling edges with uniform distribution.
#     def __init__(self, prop_nodes, num_edges):
#         super(DegreeNodesUniformEdges, self).__init__()
#         self.prob_nodes = prob_nodes
#         self.num_edges = num_edges
#
#     def sample_nodes(self, deg):
#         # deg shape : [bs, N, 1]
#         deg = deg.squeeze() # now [bs, N]
#         deg = torch.multinomial(deg, num_samples=int(self.prob_nodes*deg.shape[1]), replacement=False).to(device)
#         # deg.shape = [ bs, prob_nodes*N ]
#         return deg
#
#     def sample_edges(self, W, deg):
#         # input: W
#         # output: H, sparsified.
#         bs, N, _ = W.shape
#         deg      = self.sample_nodes(deg) # shape [bs, prob_nodes*N]
#         H        = torch.sparse.FloatTensor(bs, self.num_edges, self.num_edges)
#         H        = torch.zeros((bs, self.num_edges, self.num_edges), device=device) # [bs, nE, nE]
#         for i in range(bs):
#             edges = (W[i].triu().nonzero()).to(device) # index of edges with shape [nE, 2]
#             num_edges = edges.shape[0]
#             edges_idx = (torch.randperm(num_edges, dtype=torch.int64)[:self.num_edges]).to(device)
#             edges = edges[edges_idx]
