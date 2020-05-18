# Neural Network

Neural networks, mathematical entities capable of representing complicated functions through a composition of simpler functions.

|Formula|$o = \tanh(w \times x + b)$|
|--|--|
|Input x|scale or vector-valued (holding many scale values)|
|Output o|scale or vector-valued (holding many scale values)|
|Weight w|scale or matrix|
|Bias b|scale or vector|

The dimensionality of the inputs and weights must match. The formula is refered to as a layer of neurons because it represents many neurons via the multidimensional weights and biases.  

A multilayer neural network is a composition of the preceding functions. The output of a layer of neurons is used as an input for the following layer.

A big part of the reason why neural networks have nonconvex error surfaces is due to the` activation  function.

To cap the output values, use a simple activation   function which the default ranges are 0 to 1, or -1 to 1. As following:

|Activation Function|PyTorch Module|Formula|scope|
|--|--|--|--|
|Sigmoid|`Torch.nn.Sigmoid`|$\dfrac{1}{1+e^{-z}}$|$(0,1)$|
|Tanh|`Torch.tanh`|$\tanh z = \frac{\sinh z}{\cosh z}$|$(-1,1)$|
|HardTanh|`Torch.tanh`||$[-1,1]$|
|ReLU||$\begin{cases} 0, z \le 0 \\z,z > 0\end{cases}$|$[0,+\infty)$|
|LeakyReLU||$max(0.01z,z)$||
|Softplus||$\log(1+e^z)$|$(0,+\infty)$|

**Activation Function:**
- Are nonlinear, which allows the overall network to approximate more complex functions.
- are differentiable, which even exists point discontinuities such as ReLU.

Deep neural networks allow you to approximate highly nonlinear phenomena without having an explicit model for them. Deep neural networks are families of functions that can approximate a wide range of input/output relationships without necessarily requiring one to come up with an  explanatory model of a phenomenon. So data-driven methods are your only way forward.


## The Basic Steps of Neural Network Implementation 
- Prepare data
- Define the network structure model
- Define the loss function
- Define optimization algorithm optimizer
- training
  - Prepare input data and labels in the form of tensor (optional)
  - Forward propagation computing the network output and computing the loss of loss function
  - Backpropagation updating parameters
      1. Clear the gradient value calculated in the last iteration to 0, `optimizer.zero_grad ()`
      2. Back propagation calculating the gradient value, `loss.backward ()`
      3. Update the weight parameter, `optimizer.step()`
  - Save the loss on the training set and the loss and accuracy on the verification set and print the training information (optional).
- Show the changes of loss and accuracy during training (optional)
- Test on test set

