# Explainable Machine Learning

## Explainable/Interpretable ML
**Local Explanation** : Explain the Decision  
such as why do you think this image is a cat?  
**Global Explanation** : Explain the whole model    
such as what do you think a cat looks like?

## Why we need Explainable ML
We can improve ML model based on explanation.

**NOTE**  It is also possible to attack interpretation. 

## Explainable ML
**Basic Idea** : **Object x** $\rightarrow$ **Components** : {$x_1,\cdots,x_n,\cdots,x_N$}  
Components can be image(pixel,segment,etc) or text(a word)  

We want to know the importance of each components for making the decision.  
**Idea** : Removing or modifying the values of the components, observing the change of decision.  
Large decision change $\rightarrow$ Importance component

**Method**:
- Gray box cover
- Saliency Map
  $$\lbrace x_1,\cdots,x_n,\cdots,x_N \rbrace \rightarrow \lbrace x_1,\cdots,x_n + \Delta x,\cdots,x_N \rbrace$$
  $$y_k \rightarrow y_k + \Delta y$$
  $y_k$ : the prob of the predicted class of the model   
  We observe $|\dfrac{\Delta y}{\Delta x}| \rightarrow |\dfrac{\partial y_k}{\partial x_n}|$. The lighter spot is, The more important component is.

**NOTE  Limitation of Gradient Based Approaches**  
When we use gradient to evaluate the importance of features, it may encounter gradient **saturation**, which the gradient becomes zero but actually it plays an important role on recognition. Such as according to the length of nose, the gradient of length of nose to elephant approachs zero with the length of nose increasing continuously.  
To deal with this problem:
- Integrated gradient
  $$(x_i - \overline{x}_i) \times\int_{\alpha=0}^1\dfrac{\partial S_c(\widetilde{x})}{\partial(\widetilde{x}_i)}\,\mathrm{d}x \Bigg|_{\widetilde{x}=\widetilde{x}+\alpha(x-\overline{x})}$$
- DeepLIFT
  1. Set a baseline
  2. Reduce with baseline
- Global attribution  
  Spirit : attribution sum up to output  
  - Layer-wise relevance propagation(LRP)
    1. Locate immediate parent nodes
    2. Calculate contributions  
       Contribution = activation ×　weight
    3. Redistribute the output
    4. Backward propagation until reachcing input

Another case is that when origin image is added some noise, the gradients will be influence greatly while the image remains the same. To solve the problems of noisy gradient:
- SmoothGrad
  - Add noise for calculate gradient
  - Add noise during training
  $$\hat{M}_c(x) = \frac{1}{b}\sum_1^nM_c(x+\mathcal{N}(0,\sigma^2))$$
  $\mathcal{N}(0,\sigma^2)$ : Gaussian noise  
  $M_c$ : gradient of saliency map



## Global Explaination
### Actvation Maximization
Find a image to maximize a certain output of some filters or neurons. We take out the output of a certain neuron of output layer. We assume it is $y_i$.  
Find the image that maximizes ith-class probability
$$x^* = \arg \underset{x}{\max} y_i$$
We hope that input $x$ can  maximize the $y_i$, and $x^*$ is the maximum of the output(value) of $y_i$

Deep Neural Networks are easily fooled.

Taking handwritten digit recognition, we hope that the image also looks like a digit.
$$x^* = \arg \underset{x}{\max} y_i + R(x)$$
$R(x)$ : How likely $x$ is a digit
$$R(x) = -\sum_{i,j}|x_{i,j}|$$
The pixel only occurs on the digit region
 
### Constraint from Generator
- Training a generator  
  low-dim vector $z\rightarrow$ Image Generator (by GAN,VAE,etc.) $\rightarrow$ Image x ($x=G(z)$)
  $$x^* = \arg\underset{x}{\max}y_i \rightarrow z^* = \arg\underset{z}{\max}y_i$$
  show image: $x^* = G(z^*)$  

  $z \rightarrow$ Image Generator $\rightarrow$ Image x $\rightarrow$ Image Classifier $\rightarrow y$

  We look forward to finding a best $z$ to produces image from image generator and the image classified by image classifier leads to maximizing probability of a certain class.

## Using a Model to Explain Another
- Using an interpretable model to mimic the behavior of an uninterpretable model.  
  $x^1,x^2,\cdots,x^N \rightarrow$ Black Box    $\rightarrow y^1,y^2,\cdots,y^N$  
  We use an interpretable model such as linear model to try to explain.

  - Linear model    
    $x^1,x^2,\cdots,x^N \rightarrow$ Linear Model $\rightarrow \hat{y}^1,\hat{y}^2,\cdots,\hat{y}^N$  
    Problem:Linear model cannnot mimic neural network.  
    However,it can mimic a local region.

  - Decision Tree  
    $x^1,x^2,\cdots,x^N \rightarrow$ Decision Tree($T_{\theta}$) $\rightarrow \hat{y}^1,\hat{y}^2,\cdots,\hat{y}^N$  
    Problem: We don't want the tree to be too large. We want small $O(T_{\theta})$

The goal is to simulate $y$ by $\hat{y}$ as close as possible.

### Local Interpretable Model-Agnostic Explanations(LIME)
1. Given a data point you want to explain.
2. Sample at the nearby
3. Fit with linear model(or other interpretable models)
4. Interpret the linear model.

**LIME Example--Image Explained by Linear model**
- step 1
- step 2  
  Each image is represented as a set of superpixels(segments)  
  Randomly delete some segements $\rightarrow$ Black Box $\rightarrow$ Compute the probability of result by black box  
- step 3  
  Segements $\rightarrow$ Extract $\rightarrow (x_1,\cdots,x_m,\cdots,x_M) \rightarrow$ Linear $\rightarrow$ Compute the probability  
  $$x_m =   
  \begin{cases}
  0, & Segment\quad m\quad is\quad deleted \\
  1, & Segment\quad m\quad exists
  \end{cases}$$  
  M is the number of segments  
  The key of this step is to **extract** features of segments 
- step 4  
  train the model  
  $y = w_1x_1 + \cdots + w_mx_m + \cdots + w_Mx_M +b$  
  We can know the importance of each segment depending on its weight.  
  That's to say, if $w_m \approx 0 \rightarrow$ segment m is not related to target 
  Otherwise, if $w_m$ is positive $\rightarrow$ segement m indicates the image is target  
  If $w_m$ is negative $w_m$ is negative $\rightarrow$ segment m indicates the image is not target 

**LIME Example--Image Explained by Decision Tree**
- step 3  
  Train a network that is easy to be interpreted by decision tree.  
  $$\theta^* = \arg\underset{\theta}{\min}L(\theta) + \lambda O(T_{\theta})$$
  $T_{\theta}$ : tree mimicking network with parameters $\theta$ (inference from network parameters)  
  $O(T_{\theta})$ : how complex $T_{\theta}$ is (tree regularization).



