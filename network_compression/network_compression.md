# Network Compression

Resource-limited Devices : limited memory space,limited computing power, etc.

## Common Newtork Compression Approach
- Network Pruning  
  Let the small model learn better by observing the behavior of the large model while learning the task. (let small models extract knowledge of large models)
- Knowledge Distillation  
  Prune the large model that has been learned to make the overall model smaller. It is just to use the logits predicted by the big model as the standard for the small model to train.
- Parameter Quantization  
  Use a better way to represent or access the parameters in the model, thereby reducing the amount of calculation/consumption capacity.
- Architecture Design  
  Represent the original layer with smaller architecture. (E.g. Convolution ￫ Depthwise & Pointwise Convolution)
- Dynamic Computation

## Network Pruning
Base Idea : Netwoks are typically over-parameterized(there is significant redundant weight or neurons which have no effects on training). So we can discard some weights and bias to reduce network. 

- Large pre-trained Network
- Evaluate the Importance
  - Importance of a weight : evaluate the value of weight. Such as approximating zero means unimportant, which we can discard.
  - Importance of a neuron : the number of times the output isn't zero on a given data set
- Remove  
  Remove unimportant weights or neurons accorrding to evaluation. Then we can obtain a smaller newtork  
  After pruning, the accuracy will drop(hopefully not too much)  
- Fine-tune  
  Fine-tuning on training data to update parameters again for recover the accuracy  
- Circulate multiply times 

**TIPS** Don't prune too much at once, or the network won't recover.

### Why Pruning?
- It is widly known that smaller network is more difficult to learn successfully.
- Larger network is easier to optimize

### Network Pruning - Practical Issue
- Weight pruning  
  The pruning network architecture becomes irregular. In this case, it is hard to be trained by GPU  
  Hard to implement, hard to speed up
- Neuron pruning  
  The pruning network architecture is regular  
  Easy to implement, easy to speed up

### Pruning weight off v.s. Pruning neuron off
The difference between weight and neuron pruning is that pruning off a neuron is equivalent to cutting off the entire column of a matrix. But as a result, the speed will be faster. Because the overall matrix becomes smaller after neuron pruning. But the weight pruning remains the same size, with many holes.

## Knowledge Distillation
We firstly train a large/Teacher network ,and then train a small/Student network to mimic large network. For example, train a large network to recognize image. We got 70% for "1", 20% for "7", 10% for "9". Together train a small networn with same training data to produce same output.
- Temperature
  $$y_i = \dfrac{\exp(x_i)}{\sum_j \exp(x_j)} \quad\underrightarrow{T = 100} \quad \dfrac{\exp(x_i / T)}{\sum_j \exp(x_j / T)}$$
  When we classify image, in final step we have a softmax layer,$x_i$ is the output data of last hidden layer, $y_i$ is the output label of final output layer.    
  Before:  
  |$x_1=100$|$y_1=1$|
  |--|--|
  |$x_2=10$|$y_2 \approx 0$|
  |$x_3=1$|$y_3 \approx 0$|
  After KD:
  |$x_1/T=1$|$y_1=0.56$|
  |--|--|
  |$x_2/T=0.1$|$y_2=0.23$|
  |$x_3/T=0.01$|$y_3=0.21$| 
  The difference tells us that student network can learn about more information.

### Why does it gain a better result when student network learns from teacher network？
In fact, the teacher network provides more information than the label. For example, not only label but also information that "1" is similar to "7" does teacher network teache to student network.    
More advanced usage : Knowledge Distillation helps to merge your ensemble models into a model. It would learn from all ensemble network. Usually we average outputs of all ensemble network, and student network just to learn to produce the same output.

## Parameter Quantization
1. Using less bits to trepresent a value
2. Weight clustering
3. Represent frequent clusters by less bits, represent rare clusters by more bits. e.g. Huffman encoding
4. Binary connect  
   Binary weights : your weights are always $+1$ or $-1$.  
   At first, randomly initial weight , and then search for nearest binary weight, calculate its gradient as gradient for updating. 

## Architecture Design
- Low rank approximation  
  Intervene a linear layer between network layers to reduce parameters
- Depthwise Separable Convolution
  1. Depthwise Convolution
     - filter number = input channel number
     - Each filter only considers one channel
     - The filters are $k×k×1$ matrices
     - There is no interaction between channels
     - Using these filters to do one-on-one convolution operation onto different channel of origin image respectively
  2. Pointwise Convolution  
    To consider relationship between channels, using several `1×1×＃channel` filters to perfom convolution.    
    The each value of output of each neuron coms from pointwise convolutin onto the intermediate output of filter(step 1).    
    The function of intermediate output is similar to low rank approximation.       
    At last, the parameters will reduce a lot.

### Depthwise Separable Convolution v.s. General Convolution
> $I$ : number of input channels  
> $O$ : number of output channels  
> $k×k$ : kernel size of a filter  

For general convolution, the whole number of parameters is $k×k×I×O$.
- The depth of filter$(k×k)$ is the same as the number of input channel $I$
- The number of filters is the same as the number of output channel $O$


For depthwise separable convolution, the number of parameters is $k×k×I＋I×O$
- The number of filter for depthwise convolution is the same as the number of input channel $I$ $(k×k×I)$
  - However different filters are irrelated and their depth are only one. Each is responsible for its own channel
- The depth of filter$(1×1)$ for pointwise convolution is the same as the number of filter for depthwise convolution $I$
- The number of filters is the same as the number of output channel $O$

Divide the number of parameters :
$$\dfrac{k\times k\times I + I\times O}{k\times k\times I \times O} = \dfrac{1}{O} + \dfrac{1}{k\times k}$$
You can see that the number of parameters has been reduced by a $\dfrac{1}{k^2}th$

### Some Applicaton of Depthwise Separable Convolution
- SqueezeNet
- MobileNet
- ShuffleNet
- Xception