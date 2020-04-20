# Deep Learing

## Neural Network
Network parameter $\theta$ : all the weights and biases in the "neurons"

*Backpropagation*  : To compute the gradients efficiently, we use backpropagation.

**Chain Rule**  
$x = g(s)$ , $y = h(s)$ , $z = k(x,y)$
$$\dfrac{\mathrm{d}z}{\mathrm{d}s} = \dfrac{\partial z}{\partial x}\dfrac{\mathrm{d}x}{\mathrm{d}s} + \dfrac{\partial z}{\partial y}\dfrac{\mathrm{d}y}{\mathrm{d}s}$$

## Backpropagation
$$x^n \rightarrow \underset{\theta}{NN} \rightarrow y^n \underset{C^n}{\Longleftrightarrow}\hat{y}^n$$
$$L(\theta) = \sum_{n=1}^N C^n(\theta)$$
$$\dfrac{\partial L(\theta)}{\partial w} = \sum_{n=1}^N \dfrac{\partial C^n(\theta)}{\partial w}$$  

### Forward pass  
$z = x_1w_1 + x_2w_2 + \cdots$  
Compute $\dfrac{\partial z}{\partial w}$ for all parameters

Equal to the value of the input connected by the weight
### Backward pass  
Compute $\dfrac{\partial C}{\partial z}$ for all activation function a inputs z
$$a = \sigma(z)$$
$$\dfrac{\partial C}{\partial z} = \dfrac{\partial C}{\partial a}\dfrac{\partial a}{\partial z}$$
$$\dfrac{\partial a}{\partial z} = \sigma^\prime(z)$$
$$z_i^{l+1} = aw_i^{l+1} + \cdots$$
$$\dfrac{\partial C}{\partial a} = \dfrac{\partial C}{\partial z_i^{l+1}}\dfrac{\partial z_i^{l+1}}{\partial a} + \dfrac{\partial C}{\partial z_j^{l+1}}\dfrac{\partial z_j^{l+1}}{\partial a}$$
$$\dfrac{\partial C}{\partial a} = \sigma^\prime(z)\left[ w_i^{l+1}\dfrac{\partial C}{\partial z_i^{l+1}} + w_j^{l+1}\dfrac{\partial C}{\partial z_j^{l+1}} \right]$$
$\sigma^\prime(z)$ is a constant, because z is already determined in the forward pass

#### Backward Pass From Output Layout
Output layout $y_1$, $y_2$
$$\dfrac{\partial C}{\partial z_1^{l}} = \dfrac{\partial y_1}{\partial z_1^{l}}\dfrac{\partial C}{\partial y_1}$$
$$\dfrac{\partial C}{\partial z_2^{l}} = \dfrac{\partial y_2}{\partial z_2^{l}}\dfrac{\partial C}{\partial y_2}$$

$\dfrac{\partial y_1}{\partial z_1^{l}}$ is differentiation of activation function(softmax) of output layer $y_1$ to $z_1^l$  

$\dfrac{\partial C}{\partial y_1}$ is differentiation of loss to $y_1$, which depends on how you evaluate between output and target,such as cross entropy here.

**Compute $\dfrac{\partial C}{\partial z}$ from the output layer**
$$\dfrac{\partial C}{\partial z} = \sigma^\prime(z)\left[ w_1\dfrac{\partial C}{\partial z_1^{l}} + w_2\dfrac{\partial C}{\partial z_2^{l}}\right]$$

# Summary
Forward Pass : $\dfrac{\partial z}{\partial w} = a$  

Backward Pass : $\dfrac{\partial C}{\partial z}$

**Backpropagation**
$$\dfrac{\partial C}{\partial w} = \dfrac{\partial z}{\partial w}\dfrac{\partial C}{\partial z}$$

