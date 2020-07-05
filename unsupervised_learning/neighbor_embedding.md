# Neighbor Embedding

## Manifold Learning : non-linear dimension reduction
Suitable for clustering or following supervised learning

## Locally Linear Embedding(LLE)
We have $x^i$ and its neighbor $x^j$, $w_{ij}$ represents the relation between $x^i$ and $x^j$.  
Find a set of $w_{ij}$ minimizing
$$\sum_i = \left\| x^i - \sum_jw_{ij}x^j \right\|_2$$
Then find the dimension reduction results $z^i$ and $z^j$ based on $w_{ij}$.  
At the same time, keep $w_{ij}$ unchanged.  
Find a set of $z^j$ minimizing  
$$\sum_i = \left\| z^i - \sum_jw_{ij}z^j \right\|_2$$
$z$ is in a low-dim space.

> 在天愿作比翼鸟，在地愿为连理枝。

## Laplacian Eigenmaps
Smoothness Assumption : distance along high density region. If two points have connection along high density region, then they are close.

> Review in semi-super learning :  
> If $x^1$ and $x^2$ are close in a high desity region, $\hat{y}^1$ and $\hat{y}^2$ are probably the same. 
>  
> $L = \sum_{x^r}C(y^r,\hat{y}^r) + \lambda S$  
> $S$ can be acknowledged as a regularization term.  
> 
> $S = \frac{1}{2}\sum_{i,j}w_{ij}(y^i - y^j)^2 = Y^TLY$   
> $S$ evaluates how smooth your label is. 
>    
> $w_{ij}=
>\begin{cases}
>  similarity, & if connected \\
>  0, & otherwise
>\end{cases}$
> 
> $L:(R+U)(R+U)$, Graph Laplacian  
> $L = D - W$

- Graph-based approach
  - Construct the data points as a graph.
  - Distance defined by graph (by connection) approximate the distance on manifold. 

- Dimension Reduction : If $x^1$ and $x^2$ are close in a high density region, $z^1$ and $z^2$ are close to each other.
  $$S = \frac{1}{2}\sum_{i,j}w_{ij}\left\| z^i - z^j \right\|_2$$
  Given some constraints to z: If the dim of $z$ is $M$, span $\lbrace z^1,z^2,\cdots,z^N \rbrace = R^M$  
  
  Actually, here $z$ is the eigenvector of graph laplacian $L$, which is corresponding to the smaller eigenvalue.  

  There is an application named spectral clustering, which clusters on $z$ 


**Problem of the previous approaches : Similar data are close, but different data may collapse.**

## T-distributed Stochastic Neighbor Embedding(t-SNE)
- Computer similarity between all paris of origin $x : S(x^i,x^j)$
  $$P(x^j|x^i) = \dfrac{S(x^i,x^j)}{\sum_{k\neq i}S(x^i,x^k)}$$

- Computer similarity between all paris of dimension reduction $z : S(z^i,z^j)$
  $$P(z^j|z^i) = \dfrac{S^\prime(z^i,z^j)}{\sum_{k\neq i}S^\prime(z^i,z^k)}$$

- Find a set of z making the two distributions as close as possible
  $$L = \sum_iKL\left( p(*|x^i)||Q(*|z^i) \right)=\sum_i\sum_jP(x^j|x^i)\log\dfrac{P(x^j|x^i)}{Q(z^j|z^i)}$$

  How to obtain $z$?   
  Using gradient descent for $z$ on loss function.

As the data point is large, t-SNE's calculation is hard to undertake. So you can perfom PCA to reduce dimension first and then use t-SNE to reduce dimension further. T-SNE is more suitable for visualization.

- t-SNE Similarity Measure
  $$S(x^i,x^j)=\exp(-\left\| x^i - x^j \right\|_2)$$
  $$S^\prime(z^i,z^j) = \dfrac{1}{1 + \left\| z^i - z^j \right\|_2}$$
  It turns out that in high dimensional space, if the origin distance between two different points is very close, then it's still very close after finishing transform; If there is already a gap, after transform they will be pulled far away.






