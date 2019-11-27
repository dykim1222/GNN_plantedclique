# Node Classification for Planted Clique Problem via Graph Neural Networks

* GNN model for planted clique node classification problem
* Algorithm compared to a debatably optimal algorithm based on belief propagation ([Deshpande et al.](https://web.stanford.edu/~montanar/RESEARCH/FILEPAP/clique.pdf) )
* Models consist of variants of GNN and an upgraded version of GNN that also utilizes the edge information by considering line graphs. Graphs in planted clique problem are not sparse, so line graph can be approximated through sparsification.
* This model outperforms BP algorithm in terms of accuracy and test time, but as the number of nodes increase, this performance gap seems to decrease. This might be a sign of BP algorithm being optimal if we consider GNN as an universal approximator of the optimal algorithm.
* The code generates an Erdos-Renyi graph with the prescribed density and plants a clique. The training is done in a supervised fashion.


### Requirements
```
torch
```

### Test plot for 2000 nodes after 500,000 samples
![dd](https://github.com/dykim1222/GNN_plantedclique/blob/master/data/test_overlap_final.png)
