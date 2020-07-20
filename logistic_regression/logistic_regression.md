# Logistic Regression

## Problem Definition
Assume the data is generated based on $P_{w,b}(C_1|x) = \sigma(z)$ ,$z = w \times x + b = \sum_iw_ix_i+b$  
If $P_{w,b} \geq 0.5$, output $C_1$; otherwise ,output $C_2$, we definde function set $f_{w,b}(x) = P_{w,b}(C_1|x)$

**Loss Function(Likelihood Function)**  
$L(w,b) = f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3)\cdots$  

**Goal**  
$w^*,b^* = \underset{w,b}{argmax}L(w,b) \Rightarrow w^*,b^* = \underset{w,b}{argmin}-lnL(w,b)$

|Logistic Regression|Linear Regression|
|--|--|
|$f_{w,b}(x) = \sigma\left(\sum_iw_ix_i + b\right)$|$f_{w,b}(x) = \sum_i w_ix_i + b$|
|Output:between 0 and 1|Output:any value|
|Training data:$(x^n,\hat{y}^n)$ <br> $\hat{y}^n$:  for $C_1$, 0 for $C_2$|Training data:$(x^n,\hat{y}^n)$ <br> $\hat{y}^n:a real number$|
|$L(f) = \sum_nC(f(x^n),\hat{y}^n)$|$L(f)=\frac{1}{2}\sum_n(f(x^n)-\hat{y}^n)^2$|

**Cross entropy**  
we hope $C(f(x^n),\hat{y}^n)$ be as small as possible in order to make them has more similar distribution to attain better result
$$C(f(x^n),\hat{y}^n) = - [\hat{y}^nlnf(x^n)+(1-\hat{y}^n)ln(1-f(x^n)]$$

**Label Smoothing**
$$H(y,p) = \sum_{k=1}^K-y_k\log (p_k)$$
$$y_k^{LS} = y_k(1 - \alpha) + \alpha / K $$

Where $K$ is the number of categories, $\alpha$ is the hyperparameter introduced by label smoothing

**New Goal**  
We hope minimize cross entropy of $C(f(x^n),\hat{y}^n)$  
$$\underset{w,b}{argmix}L(w,b) = \underset{w,b}{argmix} C(f(x^n),\hat{y}^n) \\ 
= - [\hat{y}^nlnf(x^n)+(1-\hat{y}^n)ln(1-f(x^n)]
$$

**Gradient Descent**
$$\dfrac{-\partial lnL(w,b)}{\partial w_i} = \dfrac{\partial\sum -[\hat{y}^nlnf(x^n)+(1-\hat{y}^n)ln(1-f(x^n)]}{\partial w_i}$$
$$\because\dfrac{\partial lnf_{w_i,b}(x)}{\partial w_i} = \dfrac{\partial lnf_{w,b}(x)}{\partial z} \dfrac{\partial z}{\partial w_i}$$
$$\because \dfrac{\partial ln\sigma(z)}{\partial z}= \dfrac{1}{\sigma (z)}\dfrac{\partial \sigma(z)}{\partial z} $$
$$\because \dfrac{\partial \sigma(z)}{\sigma z} = \sigma(z)(1-\sigma(z))$$
$$\because\dfrac{\partial z}{\partial w_i} = x_i$$
$$\therefore \dfrac{-\partial lnL(w,b)}{\partial w_i} = \sum-[\hat{y}^n(1-f_{w,b}(x^n))x_i^n + (1-\hat{y}^n)(-f_{w,b}(x^n))x_i^n] \\
= \sum_n -(\hat{y}^n - f_{w,b}(x^n))x_i^n
$$

Both Logistic Regression and Linear Regression have the same gradient descent result after doing partial differential calculations
$$w_i \leftarrow w_i - \eta\sum_n-(\hat{y}^n-f_{w,b}(x^n))x_i^n$$  
$\hat{y}^n-f_{w,b}(x^n)$ : Larger Difference ,Larger Update

## Generative v.s. Discriminative
Navie Bayes does not care about correlation of features between different dimensions. Navie Bayes believes that different dimensions are independent.  
Generative mode is based on some assumption ,such as some probability distribution.But generative is based on nothing.  
Discriminative model is affected by numbers of sample data, but generative model will not.Generative model can separate prior and class-dependent probability to calculate.

## Multi-class Classification
**Softmax**  
Given three classes: $C_1$，$C_1$，$C_1$
- step 1 : exponetial, we obtain $e^{z^1}$, $e^{z^2}$, $e^{z^2}$
- step 2 : summarize, $sum = \sum_j^3e^{z^j}$
- step 3 : normalize, we attain $y^1 = \dfrac{e^{z^1}}{sum}$, $y^2 = \dfrac{e^{z^2}}{sum}$, $y^2 = \dfrac{e^{z^3}}{sum}$ (between 0 and 1)  

## Feature Transformation
Cascade more logistic regression

**Cross Entropy** : How close is the two distribution? If they are same,the result is 0.

Distribution p:  $p(x=1) = \hat{y}^n$, $p(x=0) = 1 - \hat{y}^n$ 

Distribution q:  $q(x=1) = f(x^n)$, $q(x=0) = 1 - f(x^n)$  

Cross Entropy between two Bernoulli Distribution.  

$$H(p,q) = -\sum_xp(x)ln(q(x)) = -\sum_x[\hat{y}^nlnf(x^n)+(1 - \hat{y}^n)ln(1 - f(x^n))]$$ 


## Regression v.s. Classification

The regression problem is different from the classification problem. The classification problem is to determine what kind of object is in a fixed n categories. The regression problem is the prediction of specific values. For example, house price prediction and sales volume prediction are regression problems. These problems that need to predict are not a pre-defined category but an arbitrary real number. The neural network to solve the regression problem generally has only one output node, and the output value of this node is the predicted value. The most commonly used loss function for regression problems is the mean square error MSE.

