# Common CNN

## AlexNet

## Residual Networks(ResNets)
Skip connections allow you to take the activation from one layer and suddenly feed it to another layer, even much deeper in the neural network.
$$a^{[l]} \rightarrow Linear1\rightarrow ReLU1\rightsquigarrow a^{[l+1]} \rightarrow Linear2\rightarrow ReLU2\rightsquigarrow a^{[l+2]}$$
skip connect :
$$a^{[l]} \rightarrow ReLU2$$

**Feedforward calculation** :
$$z^{[l+1]} = w^{[l+1]}a^{[l]} + b^{[l+1]}$$
$$a^{[l+1]} = g(z^{[l+1]})$$
$$z^{[l+2]} = w^{[l+2]}a^{[l+1]} + b^{[l+2]}$$
$$a^{[l+2]} = g(z^{[l+2]}) + a^{[l]}$$

Identity function is easy for residual block to learn. ResNet is a lot of use of same convolutions. In case the output and input differ dimension, you can add a parameter $W_s \in R$ $(R^{dim} = input^{dim} \times output^{dim})$, which can be a matrix or parameters by learning.

A skip function reduces the number of times a linear function is used to achieve an output. A skip function creates what is known as a residual block. The ResNets help to eliminate the varnishing and exploding gradient problems.

### 1×1 Convolution Network
$$28\times28\times192 \rightarrow ReLU,Conv 1×1×192,32 filters \rightarrow 28×28×32$$

**What does 1×1 Conv do?**
$$6×6×32 \rightarrow Conv\quad1×1×\#filters \rightarrow 6×6× \#filters$$
A way to shrink $n_c$, whereas pooling layers just to shrinke $n_H$ and $n_W$. If wanting to keep the number of channel, the effect of a 1×1 convolution is that it just adds nonlinearty, even it also can can increase the number of channel.

### Interpretations of the ResNet 
We can actually drop some of the layers of a trained ResNet and still have comparable performance.  
A ResNet architecture with `i` residual blocks has `2**i` different paths (because each residual block provides two independent paths).  
It is quite clear why removing a couple of layers in a ResNet architecture doesn’t compromise its performance too much — the architecture has many independent effective paths and the majority of them remain intact after we remove a couple of layers.  
It is apparent that the distribution of all possible path lengths follows a Binomial distribution.  
The ResNet did not solve the vanishing gradients problem for very long paths, and that ResNet actually enables training very deep network by shortening its effective paths.

## Inception Network
Base idea: concatenate all the outputs, and let the network learn whatever parameters it wants to use, what the combinatins of these filter size are at once. Also can use 1×1 conv within it to reduce computational cost significantly.