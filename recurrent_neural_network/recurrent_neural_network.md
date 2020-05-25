# Recurrent Neural Network

The output of hidden layer are stored in the memory. Memory can be considered as another input. Changing the sequence order will change the output.

## Cost Function
$$C = \frac{1}{2}\sum_{n=1}^N\left\|y^n-\hat{y}^n\right\|_2$$
$$C^n = \left\|y^n-\hat{y}^n\right\|_2$$

## Backpropagation
$$\dfrac{\partial C_x}{\partial w_{ij}^l} = \dfrac{\partial z_i^l}{\partial w_{ij}^l}\times\dfrac{\partial C_x}{\partial z_i^l}$$
Error signal $\delta_i^l$: $\dfrac{\partial C_x}{\partial z_i^l}$  
**Backward Pass**
$$\delta^L = \sigma^\prime(z^L)\times\nabla C_x()y$$
$$\delta^{L-1} = \sigma^\prime(z^{L-1})\times (w^L)^T\times \delta^l$$
$$\cdots$$
$$\delta^l = \sigma^\prime(z^l)\times (w^{l+1})^T\times \delta^{l+1}$$
$$\cdots$$

----
$$\nabla C_x(y) = \begin{matrix} 
\dfrac{\partial C_x(y)}{\partial y_1} \\\\
\dfrac{\partial C_x(y)}{\partial y_1} \\\\
\vdots \\\\
\dfrac{\partial C_x(y)}{\partial y_1} 
\end{matrix}$$

## Backpropagation through Time
UNFOLD:
A very deep neural network
- input : init,$x^1$,$x^2$,$\cdots$,$x^n$
- output : $y^n$
- target : $\hat{y}^n$

**Some weights are shared**  
The values of $w_1$,$w_2$ should always be the same.

### Step of Updating
Initialize $w_1$,$w_2$ by the same value 
$$w_1 \rightarrow w_1 - \dfrac{\partial C}{\partial w_1} - \dfrac{\partial C}{\partial w_2}$$
$$w_2 \rightarrow - \dfrac{\partial C}{\partial w_2} - \dfrac{\partial C}{\partial w_1}$$

### RNN Gradient Vanish Problem
A slight change in weight can produce a butterfly effect, which the results are either too big or too small. For example:  
Assuming the activation is linear function. The weight etween neurons is w. The Initial memory is 1 as well as output y.
$$y^n = w^n$$
$$\dfrac{\partial C^n}{\partial w} = \dfrac{\partial C^n}{\partial y^r}\times\dfrac{\partial y^n}{\partial w}$$
$$\dfrac{\partial y^n}{\partial w} \approx \dfrac{\Delta y^n}{\Delta w} = nw^{n-1}$$
if n = 1000:
$$w = 1 \rightarrow y^n = 1$$
$$w = 1.01 \rightarrow y^n \approx 20000$$
$$w = 0.99 \rightarrow y^n \approx 0$$
$$w = 0.01 \rightarrow y^n \approx 0$$

More often the activation is sigmoid function. RMSProp is the often optimizer.



## Long Short-term Memory
Activation function is usually a sigmoid function.  
- Can deal with gradient vanishing(not gradient explode)
  - Memory and input are added
  - The influence never disappers unless forget gate is closed(No gradient vanishing if forget gate is opened)
- Constant Error Carrousel(CEC)
  - Gradient flows backwards constantly with little decay

## Better Initialization
- Vanilla RNN : Initialized with identity matrix + ReLU

## Difference between RNN and LSTM
The most difference between RNN and LSTM is the way dealing with memory. RNN will frush the memory each time and then store new input. Whereas, LSTM will decide whether flush memory each time depending on forget gate. If not,LSTM will add new input to memory.