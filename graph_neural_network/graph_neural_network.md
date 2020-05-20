# Graph Neural Network

Graph consists of node and edge, each of which have different feature.

Graph structure, feature of node and edge are the main problem for Graph Neural Network.

## GNN : Why?
### Why do we need GNN?
- Classification : such as train a classifier of chemical formula to discriminate whether it would mutate.
- Generation : such as train a generator to produce molecular structure to resistance to viruses.
- The data may have underlying structure and relationship.
- A node can learn the structure from its neighbors

## GNN : How?
### Convolution  
How to embed node into a feature space using convolution?  
- Solution 1: Generalize the concept of convolution (corelation) to graph >> Spatial-based convolution
- Solution 2: Back to the definition of convolution in signal processing >> Spectral-based convolution

**Spatial-based Convolution**
|Aggregation|Method|
|--|--|
|Sum|NN4G|
|Mean|DCNN,DGC,GraphSAGE|
|Weighted sum|MoNET,GAT,GIN|
|LSTM|GraphSAGE|
|Max Pooling|GraphSAGE|

**Spectral-based Convolution**  
ChebNet $\rightarrow$ GCN $\rightarrow$ HyperGCN

**Tasks**
- Supervised classification
- Semi-Supervised Learning
- Representaion Learning: Graph InfoMax
- Generation: GraphVAE, MolGAN, etc.

**Benchmark Tasks**
- Semi-supervised node classification
- Regression
- Graph classification
- Graph representation learning
- Link prediction
 
**Common dataset**
- CORA: citation network. 2.7k nodes and 5.4k links
- TU-MUTAG: 188 molecules with 18 nodes on average