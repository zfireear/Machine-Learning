# Convolutional Neural network

## Why CNN
Each neuron in this network structure should represent a basic classifier,filter out some parameters that are not actually used at the beginning by our prior knowledge.
- Parameter sharing : A sharing feature detector.
- Sparsity of connections : In each layer, each output value depends only on a small number of inputs.
- Good at capturing translation invariance

## Three Property for CNN theory base
- Some patterns are much smaller than the whole image. A neuron does not have to see the whole image to discover the pattern. Connecting to small region with less parameters.
- The same patterns appear in different regions. They can also be detected by the same detector with the same neuron and the same parameters(shared parameters)
- Subsampling the pixels will not change the object. We can subsample the pixels to make image smaller. Less parameters for the network to process the image.

## The whole CNN Structure
Input $\rightarrow$ Convolution $\rightarrow$ Max Pooling $\rightarrow \cdots \rightarrow$ Flatting $\rightarrow$ NN $\rightarrow$ Output

## Convolution
Each Filter is actually a matrix. The value of each element in this matrix is ​​the same as the weight and bias of the neuron. It is a network parameter. Their specific values ​​are learned through training data, not designed by humans

