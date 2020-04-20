# New Optimazation For Deep Learning

**Some Notations**
- $\theta_t$ : model parameters at time step t
- $\nabla L(\theta_t)$ or $g_t$ : gradient at $\theta_t$ , used to compute $\theta_{t+1}$
- $m_{t+1}$ : momentum accumulated from time step 0 to time step t, which is used to compute $\theta_{t+1}$

## What is Optimization about?
- Find a $\theta$ to get the lowest $\Sigma_x L(\theta:x)$
- Or Find a $\theta$ to get the lowest $L(\theta)$

**On-line vs Off-line**  
- On-line : one pair of $(x_t,\hat{y}_t)$ at a time step
- Off-line : pour all $(x_t,\hat{y}_t)$ into the model at every time step 

The rest of this lecture will focus on the off-line cases.

## SGD
- Start at posisiton $\theta^0$
- Compute gradient at $\theta^0$
- Move to $\theta^1 = \theta^0 - \eta \nabla L(\theta^0)$
- Compute gradient at $\theta^1$
- Move to $\theta^2 = \theta^1 - \eta \nabla L(\theta^1)$
- $\vdots$
- Stop until $\nabla L(\theta^t) \approx 0$

## SGD with Momentum(SGDM)
Movement: movement of last step minus gradient at present  
Movement not just based on gradient, but previous.
- Start at point $\theta^0$
- Movement $v^0 = 0$
- Compute gradient at $\theta^0$
- Movement $v^1 = \lambda v^0 - \eta \nabla L(\theta^0)$
- Move to $\theta^1 = \theta^0 + v^1$
- Compute gradient at $\theta^1$
- Movement $v^2 = \lambda v^1 - \eta \nabla L(\theta^1)$
- Move to $\theta^2 = \theta^1 + v^2$

> $v^i$ is actually the weighted sum of all the previous gradient: $\nabla L(\theta^0)$, $\nabla L(\theta^1)$, $\cdots$, $\nabla L(\theta^{i-1})$
> $$v^0 = 0$$
> $$v^1 = -\eta \nabla L(\theta^0)$$
> $$v^2 = -\lambda \eta \nabla L(\theta^0) - \lambda \eta \nabla L(\theta^1)$$

Real Movement = Negative of Gradient + Momentum

## Adagrad 
$$\theta_t = \theta_{t-1} - \dfrac{\eta}{\sqrt{\sum_{i=0}^{t-1}(g_i)^2}}g_{t-1}$$

## RMSProp
$$\theta_t = \theta_{t-1} - \dfrac{\eta}{\sqrt{v_t}}g_{t-1}$$
$$v_1 = g_0^2$$
$$v_t = \alpha v_{t-1} + (1 - \alpha)g_{t-1}^2$$

Exponential moving average(EMA) of squared gradients is not monotonnically increasing.

## Adam
- SGDM
  - $\theta_t = \theta_{t-1} - \eta m_t$
  - $m_t = \beta_1m_{t-1} + (1 - \beta_1)g_{t-1}$
- RMSProp
  - $\theta_t = \theta_{t-1} - \dfrac{\eta}{\sqrt{v_t}}g_{t-1}$
  - $v_1 = g_0^2$
  - $v_t = \beta_2 v_{t-1}+(1-\beta_2)g_{t-1}^2$

**Adam = SGDM + RMSProp**
$$\theta_t = \theta_{t-1} - \dfrac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t$$
$$\hat{m}_t = \dfrac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \dfrac{v_t}{1-\beta_2^t}$$
Recommend : $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

## Adaptive learning rate
- Adagrad
- RMSProp
- Adam

## Non-adaptive learning rate
- SGD
- SGD with momentum(SGDM)

## Optimizers : Real Application
- Adam
  - Bert
  - Tranformer
  - Tacotron
  - Big-GAN
  - MEMO
- SGDM
  - Mask R-CNN
  - YOLO
  - ResNet

## Adam vs SGDM
- Adam : fast training,large generalization gap,unstable
- SGDM : stable,little generalization gap,better convergence

## Towards Improving Adam
**Trouble Shooting**  
- In the final stage of training, most gradients are small and non-informative, while some mini-batches provide large informative gradient rarely.
- Maximun movement distance for one single update is roughly upper bounded by $\sqrt{\dfrac{1}{1-\beta_2}}\,\eta$  
Non-informative gradients contribute more than informative gradients.
- AMSGrad, AdaBound

## Towards Improving SGDM
- SGD-type algorithms : fix learing rate for all updates, too slow for small learing rates and bad result for large learning rates.
- LR range test,Cyclical LR, SGDR, One-cycle LR

## K step forward, 1 step back
- Lookahead  
  - 1 step back: avoid too dangerous exploration(MOre stable)
  - Look for a more flatten minimum(Better generalization)
  
Universal wrapper for all optimizers
> For t = 1,2,$\ldots$ (outer loop)  
> $\qquad$$\theta_{t,0}$ = $\phi_{t-1}$  
> $\qquad$$\qquad$For i = 1,2,$\ldots$,k (inner loop)  
> $\qquad$$\qquad$$\qquad$$\theta_{t,i}$ = $\theta_{t,i-1} + Optim(Loss,data,\theta_{t,i-1}$)
> $\qquad\qquad$$\phi_t = \phi_{t-1} + \alpha(\theta_{t,k} - \phi_{t-1})$

## Look Into the Future
- Nesterov accelerated gradient(NAG)
> $\theta_t = \theta_{t-1} - m_t$  
> $m_t = \lambda m_{t-1} + \eta\nabla L(\theta_{t-1} - \lambda m_{t-1})$  

*No need to maintain a duplication of model parameters* 
> let $\theta_t^\prime = \theta_t - \lambda m_t$    
> $\qquad\,= \theta_{t-1} - m_t - \lambda m_t$  
> $\qquad\,= \theta_{t-1} - \lambda m_t - \lambda m_{t-1} - \eta\nabla L(\theta_{t-1} - \lambda m_{t-1})$  
> $\qquad\,=\theta_{t-1}^\prime - \lambda m_t - \eta\nabla L(\theta_{t-1}^\prime)$  
> $m_t = \lambda m_{t-1} + \eta\nabla L(\theta_{t-1}^\prime)$

- SGDM
  - $\theta_t = \theta_{t-1} - m_t$
  - $m_t = \lambda m_{t-1} + \eta\nabla L(\theta_{t-1})$
  - or 
  - $\theta_t = \theta_{t-1} - \lambda m_{t-1} - \eta\nabla L(\theta_{t-1})$
  - $m_t = \lambda m_{t-1} + \eta\nabla L(\theta_{t-1})$

## AdamW & SGDW with momentum(2017)
*weight decay*
- SGDWM 
  - $\theta_t = \theta_{t-1} - m_t - \gamma\,\theta_{t-1}$
  - $m_t = \gamma\,m_{t-1} + \eta(\nabla L(\theta_{t-1}))$
- AdamW 
  - $m_t = \beta_1m_{t-1} + (1 - \beta_1)\nabla L(\theta_{t-1})$
  - $v_t = \beta_2v_{t-1} + (1 - \beta_2)(\nabla L(\theta_t-1))^2$
  - $\theta_t = \theta_{t-1} - \eta\left( \dfrac{1}{\sqrt{\hat{v}_t}+ \epsilon}\hat{m}_t + \gamma\,\theta_{t-1} \right)$

## Some thing helps optimization
- Increase randomness
  - Shuffling
  - Dropout
  - Gradient noise
- Adjust Learning rate
  - Warm-up
  - Curriculum learning
  - Fine-tuning
- Normalization
- Regularization










