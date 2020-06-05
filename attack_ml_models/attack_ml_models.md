# Attack ML Models

Artificial factors are added to mislead the machine to recognize other results  

Original Image $x \rightarrow$ $\begin{bmatrix} x_1\\x_2\\x_3\\\vdots\end{bmatrix} + \begin{bmatrix} \Delta x_1\\ \Delta x_2\\ \Delta x_3\\\vdots\end{bmatrix} \rightarrow$ Attacked Image $x^\prime = x + \Delta x \rightarrow$ Network $\rightarrow$ Something ELse(Error result)

## Loss Function for Attack
- Training (input $x$ fixed) 
   
  $x \rightarrow$ Network $f_{\theta} \rightarrow y = f_{\theta}(x)\quad \underleftrightarrow{close} \quad y^{true}$

  Loss Function:  
  $$L_{train}(\theta) = C(y,y^{true})$$

- None-targeted Attack (parameters $\theta$ fixed)  
    
  $x^\prime \rightarrow$ Network $f_{\theta} \rightarrow y^\prime = f_{\theta}(x^\prime)\quad \underleftrightarrow{far} \quad y^{true}$

  Loss Function:  
  $$L(x^\prime) = -C(y^\prime,y^{true})$$

- Targeted Attack:
  
  $x^\prime \rightarrow$ Network $f_{\theta} \rightarrow y^\prime = f_{\theta}(x^\prime)\quad \underleftrightarrow{far} \quad y^{true}\quad$ **&** $\quad f_{\theta}(x^\prime)\quad \underleftrightarrow{close} \quad y^{false}$  

  Loss Function:  
  $$L(x^\prime) = -C(y^\prime,y^{true}) + C(y^\prime,y^{false})$$

- Constarint  
  Input $x$ should be similar to Attack $x^\prime$ ,which are hard to distinguish

  $x\quad\underleftrightarrow{close}\quad x^\prime$  

  Function:  
  $$d(x,x^\prime)\leq \epsilon$$

  $$\begin{bmatrix} x_1^{\prime}\\x_2^{\prime}\\x_3^{\prime}\\\vdots\end{bmatrix} - \begin{bmatrix} x_1\\x_2\\x_3\\\vdots\end{bmatrix} = \begin{bmatrix} \Delta x_1\\ \Delta x_2\\ \Delta x_3\\\vdots\end{bmatrix}$$ 
  $$x^{\prime} - x = \Delta x$$

  - L2-norm
    $$d(x,x^{\prime}) = \left\|x - x^{\prime}\right\|_2 = (\Delta x_1)^2 + (\Delta x_2)^2  + (\Delta x_3)^2 +\cdots$$
  - L-infinity  
    $$d(x,x^{\prime}) = \left\|x - x^{\prime}\right\|_{\infin} = \max \lbrace \Delta x_1,\Delta x_2,\Delta x_3,\cdots \rbrace$$

    Using L-infinity for image attack.

## How to attack
Just like traing a neural network, but network parameter $\theta$ is replaced with input $x^{\prime}$
$$x^* = \underset{d(x,x^{\prime})\leq\epsilon}{\argmin} L(x^{\prime})$$
- Gradient Descent(Modified Version)
  > Start from original image x  
  > For t=1 to T:  
  >     $\qquad x^t \leftarrow x^{t-1} - \eta\nabla L(x^{t-1})$    
  >     $\qquad d(x,x^t) > \epsilon$  
  >         $\qquad\qquad x^t \leftarrow fix(x^t)$  
  > def fix($x^t$):  
  > $\qquad$ For all x fulfill $d(x,x^t) > \epsilon$  
  > $\qquad$ Return the one closest to $x^t$ 
  
## What happens to attack
In hige dimension, once we change a pixel in a certain direction, it will turn out to a unrelated class because the scale of confidence of certain class for recognition is very narrow.

## Attack Approaches
FGSM, Basic iterative method, L-BFGS, Deepfool, JSMA, C&W, Elastic net attack, Spatially Transformed, One Pixel Attack

Key difference : different optimization methods and constraints 

**Fast Gradient Sign Method(FGSN)**
$$x^{adv} = x + \epsilon \cdot sign(\nabla_x J(x,y^{true}))$$
where $x$ is the input(origin) image, $x^{adv}$ is the perturbed adversarial image, $J$ is the classification loss function, $y^{true}$ is true label for the input $x$.

## White Box v.s. Black Box
- In the previous attack, we fix network parameters $\theta$ to find optimal $x^{\prime}$
- To attack, we need to know network parameters $\theta$
  - This is called **White Box Attack**
- Are we safe if we do not release model?
  - You cannot obtain model parameters in most on-line API
  - No, because **Black Box Attack** is possible. For example, you can gain a lot of paired image-label from online API, and then train a similar network to attack.
  
### Black Box Attack
If you have the traing data of the target network, train a proxy network yourself. Using the proxy network to generate attacked object. Otherwise, obtaining input-output pairs from target network. And then using the attacked object to attack target network, generally speaking, you could achieve a nice result you desire. 

**Universal adversarial attack also work fine**.  
You could use just one attacked object to attack the whole input and would get a nice result you desire.

**Adversarial Reprogramming**  
Adding adversaial program to origin data change its behavior to reproduce another result.

## Beyond Images
- You can attack audio
- You can attack text

## Defense
- Adversarial attack can't be defended by weight regularization,dropout and model ensemble.
- Two types of defense:
  - Passive defense : finding the attached image without modifying the model
    - Special case of anomaly detection
  - Proactive defense : traing a model that is robust to adversarial attack
  
### Passive defense
Original Image + Attack Signal $\rightarrow$ **Filter** (e.g. Smoothing) $\rightarrow$ Image + attack (less harmful,do not influence classification) $\rightarrow$ Network $\rightarrow$ correct result

Why does filter work?  
Attack signals only in a certain direction or kind can make the attack successful, although these signals can invalidate many machine learning model simultaneously. Once we add a filter, such as smoothing, it wil change the attack signal which leads to invalidating the attack. Meanwhile, the filter doesn't influence origin image.  

**NOTE** This kind of defence should keep filters a secret. Otherwise, it still could be attacked successfully while filters were attack includely.

off-the-shelf model such as **Feature Squeeze**, **Randomization at Inference Phase**

### Proactive Defense
Given training data $X=\lbrace (x^1,\hat{y}^1),(x^2,\hat{y}^2),\cdots,(x^N,\hat{y}^N)$  
Using $X$ to train your modle  
For $t=1$ to T:  
$\qquad$ For $n = 1$ to N:  
$\qquad\qquad$ Find adversarial input $\widetilde{x}^n$ given $x^n$ by an attack algorithm  
$\qquad$ We have new trainging data $X^{\prime}=\lbrace (\widetilde{x}^1,\hat{y}^1),(\widetilde{x}^2,\hat{y}^2),\cdots,(\widetilde{x}^N,\hat{y}^N)$  
$\qquad$ Using both $X^{\prime}$ to updata your model

**NOTE** The process of finding adversarial input and updating model should be circulate multiple times. You also should keep attack algorithm secrets. Even present model would stop algorithm A, but is still vulnerable for algorithm B.

Future : Adaptive Attack / Defense