# Semi-supervised

## Introduction
- Supervised Learning : $\lbrace (x^r,\hat{y}^r) \rbrace_{r=1}^R$
  - E.g. $x^r$ : image,$\hat{y}^r$ : class labels
- Semi-supervised learning : $\lbrace (x^r,\hat{y}^r) \rbrace_{r=1}^R$,$\lbrace x^u\rbrace_{u=R}^{R+U}$
  - A set of unlabeled data,usually U  $\gg$ R
  - Transductive learning : unlabeled data  is the testing data, but trained with its features.
  - Inductive learning : unlabeled data is not the testing data, which need train a model firstly.
- Why semi-supervised learning?
  - Collecting data is easy, but collecting "labeled" data is expensive
  - We do semi-supervised learning in our lives

### Why semi-supervised learning helps?
The distribution of the unlabeled data tell us something usually with some assumptions. The assumption greatly influences the effect of model.  

## Semi-supervised Learning for Generative Model
### Supervised Generative Model
- Given labelled training examples $x^r \in C_1,C_2$
  - Looking for most likely prior probability $P(C_i)$ and class-dependent probability $P(x|C_i)$
  - $P(x|C_i)$ is a Gaussian parameterized by $\mu^i$ and $\Sigma$  
  
With $P(C_1),P(C_2),\mu^1,\mu^2,\Sigma$ : 
$$P(C_1|x) = \dfrac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}$$

### Semi-supervised Generative Model
- Given labelled training example $x^r \in C_1,C_2$
  - Looking for most likely prior probability $P(C_i)$ and class-dependent probability $P(x|C_i)$
  - $P(x|C_i)$ is a Gaussian parameterized by $\mu^i$ and $\Sigma$

The unlabeled data $x^u$ help re-estimate $P(C_1),P(C_2),\mu^1,\mu^2,\Sigma$
- Initialization : $\theta = \lbrace P(C_1),P(C_2),\mu^1,\mu^2,\Sigma\rbrace$
- Step 1 : compute the posterior probability of unlabeled data $P_\theta(C_1|x^u)$, depending on model $\theta$
- Step 2 : update model  
  N : total number of examples   
  $N_1$ : number of examples belonging to $C_1$

$$P(C_1) = \dfrac{N_1+\sum_{x^u}P(C_1|x^u)}{N}$$
$$\mu^1 = \frac{1}{N_1}\sum_{x^r \in C_1}x^r + \dfrac{1}{\sum_{x^u}P(C_1|x^u)}\sum_{x^u}P(C_1|x^u)x^u$$

- Back to step 1

The unlabeled data can tell us how many times $C_1$ occurs. The number of occurrences of $C_1$ is the sum of posterior probability of $C_1$ within all unlabeled data. The unlabeled data is not hard design, saying it must belong to $C_1$ or $C_2$, but depends on its posterior probability. We say what percentage of unlabeled data belongs to $C_1$ and what percentage belongs to $C_2$. Thus we can calculate prior probability of $C_1$ according to unsupervised data which affects your estimate of $C_1$.

The algorithm converges eventually, but the initialization influences the results.

### Math principle
$\theta = \lbrace P(C_1),P(C_2),\mu^1,\mu^2,\Sigma\rbrace$
- Maximum likelihood with labeled data (Closed-form solution)
$$\log L(\theta) = \sum_{x^r}\log P_\theta(x^r,\hat{y}^r)$$
$$P_\theta(x^r,\hat{y}^r) = P_\theta(x^r|\hat{y}^r)P(\hat{y}^r)$$
- Maximum likelihood with labeled + unlabeled data(Solved iteratively)  
  
$$\log L(\theta) = \sum_{x^r}\log P_\theta(x^r,\hat{y}^r) + \sum_{x^u}\log P_\theta(x^u)$$
$$P_\theta(x^u) = P_\theta(x^u|C_1)P(C_1) + P_\theta(x^u|C_2)P(C_2)$$

$x^u$ can come from either $C_1$ and $C_2$

## Semi-supervised Learning Low-density Separation
> Low-density separation : Black-or-White
### Self-training
- Given: labeled data set $\lbrace (x^r,\hat{y}^r) \rbrace_{r=1}^R$, unlabeled data set $\lbrace x^u \rbrace_{u=l}^{R+U}$
- Repeat:
  - Train model $f*$ from labelled data set 
    - Independent to the model 
    - Not for regression
  - Apply $f*$ to the unlabeled data set
    - Obtain $\lbrace (x^u,y^u) \rbrace_{u=l}^{R+U}$ (Pseudo-label)
  - Remove a set of data from unlabeled data set, and add them into the labeled data set
    - You can also provide a weight to each data.

- Similar to semi-supervised learning for generative model
- Hard label v.s. Soft label
  - self-training(hard label): force to assign a piece of training data to a certain category.
  - Generative model(soft label): classification based on posterior probability. Some belong to class 1, others belong to class 2.  
  For neural network, soft label makes no sence. You have to use hard label, which is low-density separation assumption.

### Entropy-based Regularization
$$x^u \rightarrow \theta^* \rightarrow y^u$$
When you use neural network, the output is a distribution. We don't have to draw a arbitrary conclusion, such as saying the output belongs to which class. But the distribution had better become very concentrated.  
Entropy of $y^u$ : evaluate how concentrate the distribution $y^u$ is.
$$E(y^u) = - \sum_{m=1}^ny_m^u\ln (y_m^u)$$
The entropy $E(y^u)$ has to be as small as possible.  
**Loss Function**:
$$L = \sum_{x^r} C(y^r,\hat{y}^r) + \lambda\sum_{x^u}E(y^u)$$

## Outlook: Semi-supervised SVM
Enumerate all possible labels for the unlabeled data. Find a boundary that can provide the largest margin and least erro.

## Semi-supervised Learning Smoothness Assumption
> You are known by the company you keep

- Assumption : "similar" `x` has the same $\hat{y}$
- More precisely:
  - `x` is not uniform
  - If $x^1$ and $x^2$ are close in a high density region. $\hat{y}^1$ and $\hat{y}^2$ are the same.(connected by high density path)

**smooth assumption** : The two samples have a certain form of transition to make them similar.  

## Graph-based Approach
> How to know $x^1$ and $x^2$ are close in a high density region(connected by a high density path)

Represented the data points as a graph. Graph representation is nature sometimes. 

### Graph Construction
- Define the similarity $s(x^i,x^j)$ between $x^i$ and $x^j$
- Add edge:
  - K Nearest Neighbor
  - e-Neighborhood
- Edge weight is proportional to $s(x^i,x^j)$  
  Gaussian Radial Basis Function:
  $$s(x^i,x^j) = \exp(-\gamma\left\|x^i- x^j\right\|_2^2)$$
  $\left\|x^i- x^j\right\|_2^2$ can be seen as the squared Euclidean distance between two feature vectors.  

Because the value of the RBF kernel function decreases with increasing distance and is between 0 (limit) and 1 (when $x^i$ = $x^j$), it is a ready-made similarity measure representation.

### GRBF
Why to use this kind of function? Because its rate of decline is fast for it takes exponential operation. Only when two points are extremely closed, their similarity will be great. If only just a little bit further away, similarity will decline quickly, and become small. In other words, each data point on the graph only connected to very close data point, and ignore data points where are a slightly further away. So connected data points have possiblely same class.

**Construct the graph** : Calculate similarity between each data point `x` and bulid intermediate edges.  

Given two points, when they were connected on this graph, They can be considered as the same category. 
- The labeled data influence their neighbors
- Propagate through the graph

### Graph-based Approach
- Define the smoothness of the labels on the graph
  $$S = \frac{1}{2}\sum_{i,j}w_{i,j}\times(y^i-y_j)$$
  This definition is for all data(no matter   labeled or not)  
  Smaller means smoother

- Redefine the smoothness of the labels on the graph
  $$S = \frac{1}{2}\sum_{i,j}w_{i,j}\times(y^i-y_j)=Y^TLY$$
  Y : (R+U)-dim vector
  $$Y = [\cdots y^i \cdots y^j \cdots]^T$$
  $L : (R+U)\times (R+U) matrix$ (Graph Laplacian)
  $$L = D - W$$
- Loss Function:
  $$L = \sum_{x^r}C(y^r,\hat{y}^r) + \lambda S$$

Label $Y$ depends on network parameter. So in order to add information of graph into training, you can add smoothness item,which make your label outputed from labeled and unlabeled data fit with smoothness assumption. 
