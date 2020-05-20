# Spatial-based GNN

## Spatial-based Convolution
Terminology: 
- Aggregate: using feature of neighbor node to update next layer's hidden state.
- Readout: assemble all features of nodes to represent the whole graph

Layer i graph$(h_1^0,h_0^0,h_2^0,h_3^0,h_4^0)$ $\rightarrow$ (Aggregation) $\rightarrow$ Layer i+1 graph$(h_1^1,h_0^1,h_2^1,h_3^1,h_4^1)$ $\rightarrow$ (Readout) $\rightarrow$ $h_G$

### NN4G(Neural Network for Graph)
|Readout|||$y=w_2X_2+w_1X_1+w_0X_0$|
|--|--|--|--|
|Hidden Layer 2||$X_2 = mean(H^2)$||
|Hidden Layer 1(aggregation)|$h_3^1 = \hat{w}_{1,0}(h_0^0+h_2^0+h_4^0) + \overline{w}_1$|$X_1 = mean(H^1)$|(Mean of Nodes Layer i)|
|Hidden Layer 0(embed layer)|$h_3^0 = \overline{w}_0 \times x_3$|$X_0 = mean(H^0)$|
|Input layer|$graph(x_0,x_1,x_2,x_3,x_4)$||

**Note** $h_{node}^{layer}$

### DCNN(Diffusion-Convolution Neural Network)
1. Multiply each hidden layer node $h_{node}^{layer}$ by $w_{node}^{layer} \times mean(d(node,\cdot)=distance)$----the product of the weight and the mean of specified node to another node at a given distance. Such as: 
   - $h_3^1 = w_3^0 \times mean(d(3,\cdot)=1)$
   - $h_3^0 = w_3^1 \times mean(d(3,\cdot)=2)$
   - Each layer is calculated using data from the input layer
2. Put each layer into a matrix $H^0$, $H^1$, $\cdots$, $H^K$
3. Node features
   $$\begin{bmatrix} h_1^k\\ \vdots \\ h_1^1 \\ h_1^0 \end{bmatrix} \times W = y_1$$ 

### DGC(Diffusion Graph Convolution)
The same first two steps as DCNN, and the last step is adding all maxtix $H^K+\cdots+H^1+H^0$. 

### MoNET(Mixture Model Network)
- Define a measure on node 'distances'.such as $u_{node3,node0} = u(x,y)$ 
$$u(x,y) = \left(\frac{1}{\sqrt{deg(x)}},\frac{1}{\sqrt{deg(y)}}\right)^T$$
- Use weighted sum(mean) instead of simply summing up(averaging) neighbor features. 
   
  $w(\cdot)$ is a NN, $\hat{u}$ is a transform from u.
$$h_3^1 = w(\hat{u}_{3,0})\times h_0^0 + w(\hat{u}_{3,2}) \times h_2^0 + w(\hat{u}_{3,4}) \times h_4^0 $$

### GraphSAGE
- Sample and aggregate
  - Aggregation: mean, max-pooling, or LSTM
- Can work on both transductive and inductive setting
- GraphSAGE learns how to embed node features from neighbors

1. Sample neighborhood
2. Aggregate feature information from neighbors
3. Predict graph context and label using aggregated information

### GAT(Graph Attention Network)
- Input: node features $h = \lbrace\vec{h}_1, \vec{h}_2, \cdots, \vec{h}_n\rbrace, \vec{h}_i \in R^F$
- Calculate energy(important for center node): $e_{ij} = a(W\vec{h}_i,W\vec{h}_j)$
  $$f(h_3^0,h_0^0)=e_{3,0}$$
  $$f(h_3^0,h_2^0)=e_{3,2}$$
  $$f(h_3^0,h_4^0)=e_{3,4}$$
  $$h_3^1 = e_{3,0}\times h_0^0 + e_{3,2}\times h_2^0 + e_{3,4}\times h_4^0$$

- Attention score (over the neighbors)
  $$a_{ij} =  \dfrac{\exp\left(LeakyReLU\left(\vec{a}^T[W\vec{h}_i||W\vec{h}_j]\right)\right)}{\sum_{k\in N_i}\exp\left(LeakyReLU\left(\vec{a}^T[W\vec{h}_i||W\vec{h}_j]\right)\right)}$$

### GIN (Graph Isomorphism Network)
- $h_v^{(k)} = MLP^{(k)}\left( (1 + \epsilon^{(k)})\times h_v^{(k-1)} + \sum_{u\in N(v)}h_u^{(k-1)}\right)$
- Sum instead of mean or max 
- MLP instead of 1-layer

      

