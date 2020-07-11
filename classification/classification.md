# Classification

**Loss Function** : The number of times f get incorrect results on training data
> $$L(f) = \sum_n\delta(f(x^n) \neq \hat{y}^n)$$

## Naive Bayes Classifier
All the dimension is independent

## Math Principle
**Conditional Probability**
> $$P(A|B) = \dfrac{P(AB)}{P(B)}$$

**Covariance**  
> $$\Sigma = Cov(X,Y) = E[(X-\mu_x)(Y-\mu_y)^T]$$

**Correlation coefficient**
> $$\rho = \dfrac{Cov(X,Y)}{\sigma_x\sigma_y}$$
> $$\sigma = \sqrt{E((X-\mu)^2)}$$

**Gaussian Distribution**
> $$f(x) = \frac{1}{\sqrt{2\pi}\sigma}exp\left(- \frac{(x - \mu)^2}{2\sigma^2} \right)$$

**Standard Normal Distribution**
> $$f(x) = \frac{1}{\sqrt{2\pi}}exp\left(- \frac{x^2}{2} \right)$$

**Multivariate Normal Distribution**  
Given $x \in R^m$ ,X obeys Multivariate Normal Distribution ,we have mean $\mu$ ,covariance $\Sigma$
> $$f_{\mu,\Sigma}(x) = \frac{1}{(2\pi)^\frac{m}{2} }\frac{1}{|\Sigma|^{\frac{1}{2}}}exp \left( -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) \right)$$

## Probabilistic Generative Model
We have two kinds of classes ,$C_1$ : Class 1 , $C_2$ : Class 2 and sample x 
> **Posterior Probability**
> $$P(x) = P(x|C_1)P(C_1)+P(x|C_2)P(C_2)$$
> $$P(C_1|x) = \frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)} $$
> $$= \dfrac{1}{1+\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}}$$
> **Sigmoid Function**
> $$=\frac{1}{1+exp(-z)} = S(z)$$
> $$z = ln \frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}$$


**Maximun Likelihood function**  
Probility of Likelihood $f$ (Probility Dense Function) of Gaussian distribution(assume)  sampling $x^1,x^2,\cdots,x^n$ with mean $\mu$ and covariance $\Sigma$
> $$L(\mu,\Sigma) = f_{\mu,\Sigma}(x^1)f_{\mu,\Sigma}(x^2) \cdots f_{\mu,\Sigma}(x^n)$$
> $$\mu^*,\Sigma^* = arg max_{\mu,\Sigma}L(\mu,\Sigma)$$
> $$\mu^* = \dfrac{1}{n}\sum_{i=1}^nx^n$$
> $$\Sigma^* = \frac{1}{n}\sum_{i=1}^n (x^n - \mu^*)(x^n - u^*)^T $$

**Using Covariance**  
Find $\mu^1,\mu^2,\Sigma$ maximizing the likelihood $L(\mu^1,\mu^2,\Sigma)$ on samples $x^1,x^2,\cdots,x^n，x^{n+1},x^{n+2},\cdots,x^{2n}$
> $$L(\mu^1,\mu^2,\Sigma) = f_{\mu^1,\Sigma}(x^1)\cdots f_{\mu^1,\Sigma}(n) \times f_{\mu^2,\Sigma}(x^{n+1})\cdots f_{\mu^2,\Sigma}(x^{2n})$$
> $$\Sigma \leftarrow \dfrac{n}{2n}\Sigma^1 + \frac{n}{2n}\Sigma^2$$

With the same covariance matrix, **the boundary is linear.**

## Additional Math Principal 
**Posterior Probability**
> $$P(C_1|x) =  S(z)$$
> $$z = ln \left( \frac{P(x|C_1)P(C_1)}{P(x|C_2)p(C_2)} \right)$$

We assume $x\sim N(\mu,\Sigma)$

> $$z = ln \frac{P(x|C_1)}{P(x|C_2)} + ln\frac{P(C_1)}{P(C_2)}$$
> $$\frac{P(C_1)}{P(C_2)}\Rightarrow \frac{\frac{N_1}{N_1+N_2}}{\frac{N_2}{N_1+N_2}} = \frac{N_1}{N_2}$$
> $$P(x|C_1) = \frac{1}{2\pi^{\frac{m}{2}}} \frac{1}{|\Sigma^1|^{\frac{1}{2}}} exp \left(-\frac{1}{2}(x-\mu^1)^T (\Sigma^1)^{-1} (x-\mu^1) \right)$$
> $$P(x|C_2) = \frac{1}{2\pi^{\frac{m}{2}}} \frac{1}{|\Sigma^2|^{\frac{1}{2}}} exp \left(-\frac{1}{2}(x-\mu^1)^T (\Sigma^2)^{-1} (x-\mu^1) \right)$$
> $$ln\frac{P(x|C_1)}{P(x|C_2)} = ln\frac{|\Sigma_2|^{\frac{1}{2}}}{|\Sigma_1|^{\frac{1}{2}}} - \frac{1}{2}\left[(x-\mu^1)^T\frac{1}{\Sigma^1}(x-\mu^1) - (x-\mu^2)^T \frac{1}{\Sigma^2} (x-\mu^2)\right]$$  
> $$(x- \mu^1)^T\frac{1}{\Sigma^1}(x-\mu^1)$$
> $$= x^T\frac{1}{\Sigma^1}x - x^T\frac{1}{\Sigma^1}\mu^1 - (\mu^1)^T\frac{1}{\Sigma^1}x + (\mu^1)^T\frac{1}{\Sigma^1}\mu^1$$
> $$= x^T\frac{1}{\Sigma^1}x - 2(\mu^1)^T\frac{1}{\Sigma^1}x + (\mu^1)^T\frac{1}{\Sigma^1}\mu^1$$
> $$(x- \mu^2)^T\frac{1}{\Sigma^2}(x-\mu^2)$$
> $$= x^T\frac{1}{\Sigma^2}x - 2(\mu^2)^T\frac{1}{\Sigma^2}x + (\mu^2)^T\frac{1}{\Sigma^2}\mu^2$$
We know $P(C_1|x)=S(z)$ ,let $\Sigma^1 = \Sigma^2 =\Sigma$ we have
> $$ln\frac{P(x|C_1)}{P(x|C_2)}  = (\mu^1-\mu^2)^T\frac{1}{\Sigma}x - \frac{1}{2\Sigma}(\mu^1)^T\mu^1+\frac{1}{2\Sigma}(\mu^2)^T\mu^2$$
> $$z = (\mu^1-\mu^2)^T\frac{1}{\Sigma}x - \frac{1}{2}(\mu^1)^T\frac{1}{\Sigma}\mu^1+\frac{1}{2}(\mu^2)^T\frac{1}{\Sigma}\mu^2 + ln\frac{N_1}{N_2}$$
We can see that $z = w^Tx + b$ ,this is a linear model. In generative model,we estimate $N_1,N_2,\mu^1,\mu^2,\Sigma$ ,then we have w and b.