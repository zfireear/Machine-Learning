# Basic Theory

> Recommemd Reading Material:  
> [机器之心GitHub项目：GAN完整理论推导与实现，Perfect！](https://www.jiqizhixin.com/articles/2017-10-1-1)

## Generation  
We want to find data distribution $P_{data}(x)$  

Traditional Method : Maximum Likelihood Estimation  
- Given a data distribution $P_{data}(x)$ (We can sample from it.)
- We have a distribution $P_G(x;\theta)$ parameterized by $\theta$  
  - We want to find $\theta$ such that $P_G(x;\theta)$ close to $P_{data}(x)$
  - E.g. $P_G(x;\theta)$ is a Gaussian Mixture Model, $\theta$ are means and variances of the Gaussians Mixture Model, $\theta$ are means and variances of the Gaussians.  
  Sample $\lbrace x^1,x^2,\cdots,x^m \rbrace$ from $P_{data}$  
  We can compute $P_G(x^i;\theta)$  
  Likelihood of generating the samples  
  $$L = \prod_{i=1}^mP_G(x^i;\theta)$$
  Find $\theta^*$ maximizing the likelihood.

Maximum Likelihood Estimation = Minimize KL Divergence  
$$\theta^* = \underset{\theta}{\argmax}\prod_{i=1}^mP_G(x^i;\theta) \\
= \underset{\theta}{\argmax}\log \prod_{i=1}^mP_G(x^i;\theta) \\
= \underset{\theta}{\argmax}\sum_{i=1}^m\log P_G(x^i;\theta), x^i \sim P_{data}(x) \\
\approx \underset{\theta}{\argmax}\mathbb{E}_{x \sim P_{data}}[\log P_G(x;\theta)] \\
= \underset{\theta}{\argmax}\int_xP_{data}(x)\log P_G(x;\theta)dx - \int_xP_{data}(x)\log P_{data}(x)dx \\
= \underset{\theta}{\argmax} KL(P_{data || P_G})
$$ 

## Generator
A generator G is a network. The network defines a probability distribution $P_G$  
Normal Distribution z $\rightarrow$ Generator G $\rightarrow P_G(x)\quad \underleftrightarrow{close} \quad P_{data}(x)$  

Goal : $G^* = \underset{G}{\argmin}Div(P_G,P_{data})$

Although we do not know the distributions of $P_G$ and P_{data}, we can sample from them.

## Discriminator 
Object function for D:
$$V(G,D)= \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{x \sim P_G}[\log(1 - D(x))]$$
Goal : $D^* = \underset{D}{\argmax}V(D,G)$  
The maximum objective value is related ot JS divergence.  
- Small divergence : hard to discriminate(can't make objective large)
- Large divergence : easy to discriminate

**$\underset{D}{\max}V(G,D)$**

- Given G, what is the optimal $D^*$ maximizing  
  $$V = \int_x P_{data}(x)\log D(x)dx + \int_xP_G(x)\log(1- D(x))dx \\
  = \int_x [P_{data}(x)\log D(x) + P_G(x)\log(1- D(x))]dx$$
- Given x, the optimal $D^*$ maximizing  
  $F(D) = P_{data}(x)\log D(x) + P_G(x)\log(1- D(x))$
- Find $D^*$ maximizing by $\frac{dF(D)}{dD}$: 
  $$D^*(x) = \frac{P_{data}(x)}{P_{data}(x)+P_G(x)}$$
- calculate 
  $$\underset{D}{V(G,D)} = V(G,D^*)\\
  = \mathbb{E}_{x \sim P_{data}}\left[\log \frac{P_{data}(x)}{P_{data}(x)+P_G(x)}\right] + \mathbb{E}_{x \sim P_G}\left[\log \frac{P_G(x)}{P_{data}(x)+P_G(x)}\right]$$

  $$= \mathbb{E}_{x \sim P_{data}}\left[\log \frac{\dfrac{1}{2}P_{data}(x)}{\dfrac{P_{data}(x)+P_G(x)}{2}}\right] + \mathbb{E}_{x \sim P_G}\left[\log\frac{\dfrac{1}{2}P_G(x)}{\dfrac{P_{data}(x)+P_G(x)}{2}}\right]
  $$

  $$= -2log2 + \int_xP_{data}(x)\log \frac{P_{data}(x)}{\dfrac{P_{data}(x)+P_G(x)}{2}}dx + \int_xP_G(x)\log \frac{P_{data}(x)}{\dfrac{P_{data}(x)+P_G(x)}{2}}dx$$

  We expected that $P_G=P_{data}$ is the best result. And The integral of $P_G$ and $P_data$ in their integral domain is equal to 1.

  $$= -2log2 + KL(P_{data}||\dfrac{P_{data}+P_G}{2}) + KL(P_G||\dfrac{P_{data}+P_G}{2}) \\ 
  = -2\log2 + 2JSD(P_{data}||P_G)
  $$

Besides, to derive max function is actually to calculate which function is the largest in region, and to differentiate the largest function. 

