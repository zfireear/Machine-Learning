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

### L1 V.S. L2
Although they also make the absolute value of the parameter smaller, they actually do slightly different things:
- L1 makes the absolute value of the parameter smaller by subtracting a fixed value for every update
- L2 makes the absolute value of the parameter smaller by multiplying by a fixed value less than 1 for every update.



