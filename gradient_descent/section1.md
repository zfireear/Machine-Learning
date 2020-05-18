# Gradient Descent 
Suppose that $\theta$ has two variables {$\theta_{1}$,$\theta_{2}$} ,randomly start at $\theta^{0}$ 
> $$\theta^{0} = \begin{bmatrix} 
\theta_{1}^{0}\\\\ \theta_{2}^{0} 
\end{bmatrix}$$  
> $$\theta^{1} = \theta^{0} - \eta\nabla L(\theta^{0})$$
> $$\nabla L(\theta)=\begin{bmatrix}
    \frac{\partial L(\theta_{1}^{0})}{\partial\theta_{1}} \\\\
    \frac{\partial L(\theta_{2}^{0})}{\partial\theta_{2}} 
\end{bmatrix}$$

**Gradient Descent** : Loss is the summation over all training examples
> $$L = \sum_{n} (\hat{y} - (b + \sum w_ix_i^n))^2$$

## 1.Tuning your learning rates
Rule : Reduce your learning rate by some factor every few epochs.(Fisrt large ,later small)  
eg. 1/t decay:  $\eta^{t} = \eta / \sqrt{t+1}$  

### 1.1.Adagrad
Divide the learning rate of each parameter by the root mean square of its privious derivaties of 
$\theta$

The best step is ` |First Derivative| / Second Derivative`  

> $$\eta \leftarrow \eta^{t} / \sigma^{t}$$
> $$\sigma^{t} = \sqrt{\frac{1}{t+1} \sum_{i=0}^{t} (\nabla L(\theta^{i}))^{2}}$$
> $$\eta^{t} = \eta / \sqrt{t+1}$$

Gradient Descent of weight
> $$w^{t+1} \leftarrow w^t - \frac{\eta^t}{\sigma^t}\nabla L(w^t)$$
> $$w^{t+1} = w^{t} - \eta / \sqrt{\sum_{i=0}^{t} (\nabla L(w^{i}))^{2}}$$

```python
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
```

## 2.Stochastic Gradient Descent
>  The term stochastic comes from the fact that the gradient is typically obtained by averaging over a random subset of all input samples, called a minibatch. The optimizer itself, however, doesn’t know whether the loss was evaluated on all the samples (vanilla) or a random subset thereof (stochastic), so the algorithm is the same in the two cases.

Just pick an example $x^n$ ,Loss for only one example
> $$L^n = ( \hat{y}^n - (b+\sum w_ix_i^n) )^2$$
> $$\theta^i = \theta^{i-1}- \eta\nabla L^n(\theta^{i-1})$$

## 3.Feature Scale
Make different features have same scaling
- Normalize : make the mean of all dimensions are 0 and the variances are all 1.
  > $$x_i^n \leftarrow \frac{x_i^n - \mu_i}{\sigma_i}$$
  > $$\mu_i : mean$$
  >$$\sigma_i : standard deviation$$

```python
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
```

## 4. Mathematical principle
**Taylor series** : Let h(x) be any function infinitely differentiable around $x = x_0$
> $$ h(x) = \sum_{k=0}^{\infty}\frac{h^{(k)}(x_0)}{k!}(x-x_0)^k$$ 

When x is close to $x_0$ ,we have 
> $$ h(x) \approx h(x_0) + h^\prime(x_0)(x-x_0)$$

**Multivariable Taylor Series**
> $$h(x,y) = h(x_0,y_0) + \frac{\partial h(x_0,y_0)}{\partial x}(x - x_0) + \frac{\partial h(x_0,y_0)}{\partial y}(y - y_0) + \cdots $$

When x and y is close to $x_0$ and $y_0$ ,we have 
> $$h(x,y) \approx h(x_0,y_0) + \frac{\partial h(x_0,y_0)}{\partial x}(x - x_0) + \frac{\partial h(x_0,y_0)}{\partial y}(y - y_0) $$

### Back to Formal Derivation
Base on Taylor Series ,if the red circle(learning rate) is small enough ,in the red circle of (a,b):
> $$L(\theta) \approx L(a,b) + \frac{\partial h(a,b)}{\partial \theta_1}(\theta_1 - a) + + \frac{\partial h(a,b)}{\partial \theta_2}(\theta_2 - b)$$

We simplify $L(\theta)$ with $s = L(a,b)$ , $u = \frac{\partial h(a,b)}{\partial \theta_1}$ , $v = \frac{\partial h(a,b)}{\partial \theta_2}$ ,we have

> $$L(\theta) = s + u(\theta_1 - a) + v(\theta_2 -b)$$
> $$\Delta\theta_1 = \theta_1 - a$$
> $$\Delta\theta_2 = \theta_2 - b$$

Find $\theta_1$ and $\theta_2$ in the circle with radius r,minimizing $L(\theta)$ :
> $$(\theta_1 - a)^2 + (\theta_2 - b)^2 \leq r^2$$

To minimizing $L(\theta)$ 
> $$ \begin{bmatrix}
    \Delta\theta_1 \\\\ \Delta\theta_2
\end{bmatrix} = -\eta \begin{bmatrix}
    u \\\\ v
\end{bmatrix} \Longrightarrow \begin{bmatrix}
    \theta_1 \\\\ \theta_2
\end{bmatrix} = \begin{bmatrix}
    a \\\\ b
\end{bmatrix} - \eta \begin{bmatrix}
    u \\\\ v
\end{bmatrix}
$$

This is **Gradient Descent**.

## 5.More Limitation of Gradient Descent
- Stuck at local minima
- Stuck at saddle point
- Very slow at plateau
