# Anomaly Detection

## Problem Formulation
- Given a set of training data $\lbrace x^1,x^2,\cdots,x^N \rbrace$
- We want to find a function detecting input $x$ is similar to training data or not
  - $x$ is similar to training data $\rightarrow$ Anomaly Detector $\rightarrow$ normal
  - $x$ is different from training data $\rightarrow$ Anomaly Detector $\rightarrow$ anomaly(or outlier, novelty, exceptions)
- Different approaches use different ways to determine the similarity
- Categories  
  We have training data $\lbrace x^1,x^2,\cdots,x^N \rbrace$, there are two main categories to classify data
  - Open-set Recognition  
    Classifier With labels $\lbrace \hat{y}^1,\hat{y}^2,\cdots,\hat{y}^N \rbrace$. The classifier can output "unknown" (none of the training data is labelled)
  - Without Labels  
    Clean : All the training data is normal.  
    Polluted : A little bit of training data is anomaly.

### Case 1 : With Classifier
input $x \rightarrow$ Classifier $\rightarrow$ Class $\hat{y}$ and its **Confidence Socre** $c$  
Anomaly Detection :
$$f(x) = \begin{cases}
  normal, & c(x) > \lambda \\
  anomaly, & c(x) \leq \lambda
\end{cases}$$
$\lambda$ is the threshold, Confidence is the maximum or negative entropy.

#### Example Framework
- Training Set: Images $x$ of characters from specified class. Each image $x$ is labelled by its label $\hat{y}$.  
  Train a classifer, and then we can obtain confidence score $c(x)$ from the classifier.
- Dev Set: Images $x$ and label each image $x$ is from training set class or not.(including not only from training set class but also another source class)  
  We can computer the performance of $f(x)$ using dev set to determine $\lambda$ and other hyperparameters.
- Testing Set : Images $x \rightarrow$ from specified or not 
- Evaluation  
  Accuracy is not a good measurement as the number of normal and anomaly data is disparate.A system can have high accuracy but do nothing. We can introduce a cost table of anomaly and normal detection to evaluate.  

  Distrubition table : False alarm and missing are what we should pay attention to.
  ||Anomaly|Normal|
  |--|--|--|
  |Detected|1|1|
  |Not Det|4|99|

  False alarm is that regard normal data as anomaly data  
  Missing is that neglect anomaly data

  Cost Table : Assign points 
  |Cost|Anomaly|Normal|
  |--|--|--|
  |Detected|0|1|
  |Not Det|100|0|

  Multipy distribution with corresponding points to evaluate $\lambda$

  Some evaluation metrics consider the ranking. For example, area under ROC curve

#### To learn more
- Learn a classifier giving low confidence score to anomaly
- Generating anomaly by generative models

### Case 2 : Without Labels
- Given a set of training data, probability model $P(x)$ is probability of certain behaviour.
- We want to find a function detecting input $x$ is similar to training data or not.  
  $P(x) \geq \lambda \rightarrow$ Anomaly Detector $\rightarrow$ normal  
  $P(x) \lt \lambda \rightarrow$ Anomaly Detector $\rightarrow$ anomaly  
  $\lambda$ is decided by developers

#### Example framework
*Maximum Likelihood*
- Assuming the data points is sampled from a probability density function $f_{\theta}(x)$
  - $\theta$ determines the shape of $f_{\theta}(x)$ 
  - $\theta$ is unknown, to be found from data
  
  $$L(\theta) = f_{\theta}(x^1)f_{\theta}(x^2)\cdots f_{\theta}(x^N)$$

    Our goal is to find : 
    $$\theta^* = \underset{\theta}{\argmax}L(\theta)$$

    The common distribution is **Gaussian Distribution**

    $$f_{\mu,\Sigma}(x) = \dfrac{1}{(2\pi)^{\frac{D}{2}}}\dfrac{1}{|\Sigma|^{\frac{1}{2}}}exp\left \lbrace -\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu) \right \rbrace$$

    Input : vector x  
    output : probability density of sampling x  
    $\theta$ which determines ths shape of the function of the function are mean $\mu$ and covariance matrix $\Sigma$  
    $D$ is the dimension of x

    So now we want to maximum likelihood of $f(\theta)$

    $$L(\theta) \Rightarrow L(\mu,\Sigma) = f_{\mu,\Sigma}(x^1)f_{\mu,\Sigma}(x^2)\cdots f_{\mu,\Sigma}(x^N)$$

    $$\theta^* \Rightarrow \mu^*,\Sigma^* = \underset{\mu,\Sigma}{\argmax}L(\mu,\Sigma)$$

    $$\mu^* = \dfrac{1}{N}\sum_{n=1}^Nx^n$$

    $$\Sigma^* = \dfrac{1}{N}\sum_{n=1}^N(x - \mu^*)(x - \mu^*)^T$$

    To evaluate anomaly data, we set a threshold $\lambda$

    $$f(x) = \begin{cases}
    normal, & f_{\mu^*,\Sigma^*}(x) \gt \lambda \\
    anomaly, & f_{\mu^*,\Sigma^*}(x) \leq \lambda
    \end{cases}$$

### outlook : Auto-encoder
- Using training data to learn an autoencoder on training process.
- Using testing data to reconstructe
  - Large reconstructin loss means anomaly (can't be reconstructed)

## Applications
- Fraud Detection
  - Training data : normal Credit card behavior, $x$ : fraudulent charge?
- Network Intrusion Detection
  - Training data : normal connection, $x$ : attack behaviour?
- Cancer Detection
  - Training data : normal cell, $x$ : cancer cells

## Summary
- Classic Method
  - With Classifier
  - GMM(Gaussian Mixture Model)
  - Auto-Encoder
  - PCA
  - Isolation Forest 
  