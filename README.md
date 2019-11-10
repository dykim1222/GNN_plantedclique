# Node Classification for Planted Clique Problem via Graph Neural Networks

* GNN model for planted clique node classification problem
* Algorithm compared to a ~~controversially optimal~~ algorithm based on belief propagation ([Deshpande et al.](https://web.stanford.edu/~montanar/RESEARCH/FILEPAP/clique.pdf) )
* Models consist of variants of GNN and an upgraded version of GNN that also utilizes the edge information by considering line graphs. Graphs in planted clique problem are not sparse, so line graph can be approximated through sparsification.

### Requirements
```
torch
```

![dd](https://github.com/dykim1222/GNN_plantedclique/blob/master/data/test_overlap_final.png)
