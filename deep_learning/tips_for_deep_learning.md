# Tips for Deep Learning

## Recipe of Deep Learning
*overfitting* is based on testing set, sometimes you have to check the training set. 

|Training Data|Testing Data|
|--|--|
|New Activation|Early Stopping|
|Adaptive Learning Rate|Regularization|
||Dropout|

## Keep in mind  
When you see a method in the literature of deep learning, always think about what kind of problem this method is to solve, because in deep learning, there are two problems.
- The performance on the training set is not good enough
- The performance on the test set is not good enough  
  
When there is a method proposed, it often tends to handle only one of the two problems. So, you have to figure out what the problem is, and then find a targeted method based on this problem.


## Vanishing Gradient Problem
When the inputs of the sigmoid function becomes larger or smaller, the derivative becomes close to zero. When more layers are used, it can cause the gradient to be too small for training to work effectively.  
A small gradient means that the weights and biases of the initial layers will not be updated effectively with each training session. Since these initial layers are often crucial to recognizing the core elements of the input data, it can lead to overall inaccuracy of the whole network. 
 
**Solution**
- ReLU (Rectified Linear Unit)
  - Fast to compute
  - Infinite sigmoid with different biases
  - handle vanishing gradient problem, do not have smaller gradient

$$\sigma(z) =a = \begin{cases}
    z,\qquad  z>0 \\\\
    0,\qquad z\leq0
\end{cases}$$

- Residual networks
  - provide residual connections straight to earlier layers
  - doesn’t go through activation functions that “squashes” the derivatives, resulting in a higher overall derivative of the block
- batch normalization layers

### Maxout
- Given a training data x, we know which z would be the max
  - Arbitrarily group $z_1^1,z_2^1,\cdots$, such as $\{z_1^1,z_2^1\}$
  - $\max \{z_1^1,z_2^1\} \rightarrow a_1^1$
- Learnable Activation Function  
  - Activation function in maxout network can be any piecewise linear convex funtion.
  - How many pieces depending on how many elements in a group.
- Different thin and linear network for different training examples.
- ReLU is a special case of Maxout.
- Output **a** is produced by max operation rather than activation function.

In specific practice, we can first convert the max function to a specific function based on data, and then differentiate the converted thinner linear network.

### Adagrad
If gradient is small, set a larger learning rate; If gradient is large, set a smaller learning rate.  
$$w^{t+1} \leftarrow w^t - \dfrac{\eta}{\sqrt{\sum_{i=0}^{t}(g^i)^2}}g^{t}$$

Adagrad considers the average gradient information of the entire process.

### RMSProp
Adjust the learning rate rapidly according to the gradient at any time (advanced Adagrad)
$$w^{t+1} = w^t - \dfrac{\eta}{\sigma^t}g^t$$
$$\sigma^t = \sqrt{\alpha(\sigma^{t-1})^2 + (1 - \alpha)(g^t)^2}$$
$$\sigma^0 = g^0$$

RMSProp dynamicly tunes incidence of gradient by $\alpha$ at different times.   
(If you set $\alpha$ smaller, RMSProp is tent to believe present gradient whether is slippery or Steep in order to set larger or smaller gradient rather than last gradient)

**Gradient Update**  

$w^1 \leftarrow w^0 - \dfrac{\eta}{\sigma^0} \qquad \sigma^0 = g^0$

$w^2 \leftarrow w^1 - \dfrac{\eta}{\sigma^1}g^1 \qquad \sigma^1 = \sqrt{\alpha(\sigma^0)^2 + (1 - \alpha)(g^1)^2}$

$w^3 \leftarrow w^2 - \dfrac{\eta}{\sigma^2}g^2 \qquad \sigma^2 = \sqrt{\alpha(\sigma^1)^2 + (1 - \alpha)(g^2)^2}$

$\vdots$

$w^{t+1} \leftarrow w^t - \dfrac{\eta}{\sigma^t}g^t \qquad \sigma^t = \sqrt{\alpha(\sigma^{t-1})^2 + (1 - \alpha)(g^t)^2}$

Root Mean Square of the gradients with previous gradients being decayed  


## Momentum  
Movement of last step minus gradient at present
- Start at point $\theta^0$
- Initial movement $v^0 = 0$
- Compute gradient at $\theta^0$
- Movement $v^1 = \lambda v^0 - \eta\nabla L(\theta^0)$
- Move to $\theta^1 = \theta^0 + v^1$
- Compute gradient at $\theta^1$
- Movement $v^2 = \lambda v^1 - \eta\nabla L(\theta^1)$
- Move to $\theta^2 = \theta^1 + v^2$

Movement not just based on gradient,but previous movement.  
$v^i$ is actually the weighted sum of all the previous gradient:
- $v^0 = 0$
- $v^1 = - \eta\nabla L(\theta^0)$
- $v^2 = -\lambda \eta \nabla L(\theta^0) - \eta \nabla L(\theta^1)$

## Momentum V.s. AdaGrad
Momentum adds updates to the slope of our error function and speeds up SGD in turn. AdaGrad adapts updates to each individual parameter to perform larger or smaller updates depending on their importance

## Adam
- $m_0 = 0$, it is previous movement in Momentum
- $v_0 = 0$, it is $\sigma$ in the root mean square of the gradient calculated in RMSProp
- $t = 0$, it means moment.
- calculate gradient $g_t$
$$g_t = \nabla_\theta f_t(\theta_{t-1})$$
- calculate $m_t$ by $m_{t-1}$ and $g_t$ ------ Momentum
$$m_t = \beta_1m_{t-1} + (1 - \beta_1)g_t$$
- calculate $v_t$ by $v_{t-1}$ and $g_t$ ------ RMSProp
$$v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2$$
- bias correction 
$$\hat{m}_t = \dfrac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \dfrac{v_t}{1-\beta_2^t}$$
- Update
$$\theta_t = \theta_{t-1} - \dfrac{\alpha\cdot\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Adam = RMSProp + Momentum**



## Regularization
New loss function to be minimized
- Find a set of weight not only minimizing original cost but also close to zero

> L-P Norm
> $$L_p = \sqrt[^p\!]{\sum_{i=1}^nx_i^p} \quad,x = (x_1,x_2,\cdots,x_n) $$
> 
> L0 Norm : P = 0   
> mainly used to measure the number of non-zero elements in the vector 
>    
> L1 Norm : P =1   
> the solution optimized for L1 is a sparse solution, so the L1 norm is also called the sparse rule operator. L1 can achieve sparse features, remove some features without information.
> $$\left\|\theta\right\|_1 = \sum_i|x_i|$$
> 
> L2 Norm : P = 2  
> The L2 norm is usually used as the regularization term of the optimization goal, to prevent the model from being too complicated in order to cater to the training set and cause overfitting, thereby improving the generalization ability of the model
> $$\left\|\theta\right\|_2 = \sqrt{\sum_ix_i^2}$$


$\theta = \{w_1,w_2,\cdots\}$   

### L2 regularization (L2 Norm)  
$$L^\prime(\theta) = L(\theta) + \lambda\frac{1}{2}\left\|\theta\right\|_2$$

$$\left\|\theta\right\|_2 = (w_1)^2 + (w_2)^2 + \cdots$$

**Gradient update**
$$\dfrac{\partial L^\prime(\theta)}{\partial w} = \dfrac{\partial L(\theta)}{\partial w} + \lambda w$$
$$w^{t+1} \leftarrow w^t - \eta \dfrac{\partial L^\prime(\theta)}{\partial w} = w^t - \eta \left(\dfrac{\partial L(\theta)}{\partial w} + \lambda w^t\right)$$
$$\qquad = (1 - \eta\lambda)w^t - \eta\dfrac{\partial L(\theta)}{\partial w}$$
It is obviously that exits weight decay.

### L1 regularization(L1 Norm)
$$L^\prime(\theta) = L(\theta) + \lambda\frac{1}{2}\left\|\theta\right\|_1$$
$$\left\|\theta\right\|_1 = |w_1| + |w_2| + \cdots$$  

**Gradient Update**
$$\dfrac{\partial L^\prime(\theta)}{\partial w} = \dfrac{\partial L(\theta)}{\partial w} + \lambda\, sgn(w)$$
$$sgn(w) = \begin{cases}
    1,\qquad\quad w>0 \\
    -1,\qquad w<0 \\
\end{cases}$$
$$w^{t+1} \leftarrow w^t - \eta \dfrac{\partial L^\prime(\theta)}{\partial w} = w^t - \eta \left( \dfrac{\partial L(\theta)}{\partial w} + \lambda\, sgn(w^t)\right)$$

$$\qquad = w^t - \eta\dfrac{\partial L(\theta)}{\partial w} - \eta \lambda sgn(w^t)w$$

### L1 V.s. L2
Although they also make the absolute value of the parameter smaller, they actually do slightly different things:
- L1 makes the absolute value of the parameter smaller by subtracting a fixed value for every update
- L2 makes the absolute value of the parameter smaller by multiplying by a fixed value less than 1 for every update.

## Dropout
- During training, each time before updating the parameters, we do sampling for each neuron (also including the input layer). Each neuron has a p% chance that it will be discarded (dropout). If it is lost, the weight connected to it will also be lost.
- The parameters between different networks are shared.
- Assuming that during training, the dropout rate is p%, all weights learned from the training data must be multiplied by (1-p%) to be used as the weight of the testing set.

What Dropout really wants to do is to make your results on the training set worse, but the results on the testing set are better.Dropout is a optimization method for testing set but is used for training .

If the type of network is very close to linear, the performance of dropout will be better, and the network of ReLU and Maxout is relatively close to linear, so we usually use the network with ReLU or Maxout in Dropout.

### Maxout V.s. Dropout  
Maxout is that the network structure corresponding to each data is different, and Dropout is that the network structure is different for each update (each minibatch corresponds to an update, and a minibatch contains many data)

## Parameter Initialization
What should be the scale of this initialization? If we choose large values for the weights, this can lead to exploding gradients. On the other hand, small values for weights can lead to vanishing gradients. There is some sweet spot that provides the optimum tradeoff between these two, but it cannot be known a priori and must be inferred through trial and error.

In general, it is a good idea to avoid presupposing any form of a neural structure by randomizing weights according to a normal distribution.

### Xavier Initialization
Xavier initialization is a simple heuristic for assigning network weights. With each passing layer, we want the variance to remain the same. This helps us keep the signal from exploding to high values or vanishing to zero. In other words, we need to initialize the weights in such a way that the variance remains the same for both the input and the output.

The weights are drawn from a distribution with zero mean and a specific variance. For a fully-connected layer with m inputs:
$$W_{ij} \sim N\left(0,\frac{1}{m}\right)$$
The value m is sometimes called the fan-in: the number of incoming neurons (input units in the weight tensor).

### He Normal Initialization
He normal initialization is essentially the same as Xavier initialization, except that the variance is multiplied by a factor of two.The weights are still random but differ in range depending on the size of the previous layer of neurons. This provides a controlled initialization hence the faster and more efficient gradient descent.

For ReLU units, it is recommended:
$$W_{ij} \sim N\left(0,\frac{2}{m}\right)$$

## Bias Initialization
The simplest and a common way of initializing biases is to set them to zero.

One main concern with bias initialization is to avoid saturation at initialization within hidden units — this can be done, for example in ReLU, by initializing biases to 0.1 instead of zero.

## Pre-Initialization
 This is common for convolutional networks used for examining images, which is to be used on similar data to that which the network was trained on.

## Feature Normalization
manipulate the data itself in order to aid our model optimization
1. min-max normalization  
   rescaling the range of features to scale the range in [0, 1] or [−1, 1].
   As this has a tendency to collapse outliers so they have a less profound effect on the distribution.
   $$x^\prime = \dfrac{x-\min(x)}{\max(x)-\min(x)}$$

2. Feature standardization
   makes the values of each feature in the data have zero-mean and unit-variance,which ameliorate the distortion (uniforming the elongation of one feature compared to another feature)
   $$x^\prime = \dfrac{x - \mu}{\sigma}$$
   
## Batch Normalization
Batch normalization is an extension to the idea of feature standardization to other layers of the neural network.

- To increase the stability of a neural network, batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.  
 $\mu$ is the vector of mean activations across mini-batch, $\sigma$ is the vector of SD of each unit across mini-batch.
$$H^\prime = \dfrac{H - \mu}{\sigma}$$
$$\mu = \dfrac{1}{m}\sum_iH_i$$
$$$$
$$\sigma = \sqrt{\dfrac{1}{m}\sum_i(H - \mu)_i^2+\delta}$$

Batch normalization allows each layer of a network to learn by itself more independently of other layers.

However, after this shift/scale of activation outputs by some randomly initialized parameters, the weights in the next layer are no longer optimal. SGD ( Stochastic gradient descent) undoes this normalization if it’s a way for it to minimize the loss function.

Consequently, batch normalization adds two trainable parameters to each layer, so the normalized output is multiplied by a “standard deviation” parameter (γ) and add a “mean” parameter (β). In other words, batch normalization lets SGD do the denormalization by changing only these two weights for each activation, instead of losing the stability of the network by changing all the weights. ($\gamma,\beta$ are learnable)
$$\gamma H^\prime + \beta$$
This procedure is known as the batch normalization transform.  
During test time, the mean and standard deviations are replaced with running averages collected during training time,which ensures that the output deterministically depends on the input.   
Batch normalization reduces overfitting because it has a slight regularization effect.Similar to dropout, it adds some noise to each hidden layer’s activations.   

There are several advantages to using batch normalization:
- Reduces internal covariant shift.
- Reduces the dependence of gradients on the scale of the parameters or their initial values.
- Regularizes the model and reduces the need for dropout, photometric distortions, local response normalization and other regularization techniques.
- Allows use of saturating nonlinearities and higher learning rates.