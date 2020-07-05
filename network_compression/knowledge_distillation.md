# Knowledge Distillation

## Why does it work?
- For example, when the data is not very clean, it is a noise for the general model, which will only interfere with learning. It is better to learn logits predicted by other big models.
- There may be a relationship between labels, which can guide small models to learn. For example, the number 8 may be related to 6,9,0.
- Weaken the target which has been learned well. Lest the gradient interfere with other tasks that have not yet learned well.

## Main Question : Distill what?
- Logits(output value)
  - Match logits directly
  - Learn logits distribution from a batch
- Feature(intermidiate value) 
  - Match featue directly
  - learn how the feature transforms

In general, there are some hidden information in relationship between categories.

## Overfitting in model
- Incompleteness  
  There are not only one individual on the image, but loss function asks to recognized only as a single object
- Inconsistency  
  By crop operaion according to augment, sometimes it would mislabeled label which was labeled on whole image

### Solution
- Label Refinery  
  - Train a base model,and use its output as subsequent input of refinery model.
  - Repeat the training several times until the accuracy is no longer improved    
  
  Each time refined model can reduce incompleteness and inconsistency, and also learn the relationship between labels

## Logits Distillation
### Baseline KD  
Teacher network:  
$x_i \rightarrow$ train big Model(s) $\rightarrow$ divided by $T \rightarrow$ softmax $\rightarrow$ soft target  

Student network:  
$x_i \rightarrow$ small model $\rightarrow$  

$\begin{cases}
  \rightarrow divided \quad by \quad T \rightarrow softmax \rightarrow cross\quad entropy \quad loss\quad \overset{\lambda}{\longrightarrow}
  \\
  \rightarrow softmax \rightarrow cross\quad entropy \quad loss\quad \overset{1- \lambda}{\longrightarrow}
\end{cases}$  

$\overset{sum}{\longrightarrow}\quad$ total loss

Through soft target, small models can learn the relationship between classes.

**Loss Function**  
$$Loss = \alpha T^2 \times KL(\dfrac{Teacher's Logits}{T} || \dfrac{Student's Logits}{T}) + (1-\alpha)(origin\quad Loss)$$

### Distill Logits - Deep Mutual Learning
Train two networks at the same time and learn each other's logits.

### Distill Logits - Born Again Neural Networks
Similar to laber refinery
- initial model with KD
- iteration by Cross Entropy
- At last, ensemble all student model

## Hidden Problem in Pure Logits KD
Small network somehow is hard to learn from too big network.  
Solution : use a TA with a proper number of parameters between Teacher and Student as a middleman to help student learn, so as to avoid the model gap from being bad for learing.

## Feature Distillation
### Distill Feature - FitNet
Let the student learn how to generate the teacher's intermediate feature first, and then use baseline KD.
- Student network applies a weight matrix $W_r$ inside inner hidden layer to fit the intermidiate feature of teacher network in 2-norm distance
- The closer the architecture, the better the effect

#### Problem in FitNet
- Different size of model capacity is different
- There's lots of redundancy in teacher network, which should be avoid to learn.

To sovle this problem, we can apply **knowledge compression** on feature map. We can attain a more compact model which reduces redundacy, and it is beneficial to train.

### Distill Feature - Attention
Let the student network study the teacher's attention map
- How to generate attention map?
  - Square the weights of $(W, H, C)$ and then add to the matrix $T$ of $(W,H)$
  - Attention map = $\dfrac{T}{norm(M)}$
- What target function?
  - L2 distance between teacher's attention map and student's attention map

### Group Convolution Layer--Another Attention 
GC (Group Convolution Layer) is to group feature maps. Let them pass convolution layer and then concat again. It is a compromise between the general convolution and depthwise Convolution. So, when group convolution's group feautures number is equal to input number, it will be depthwise convolution (because each Channel is independent), and Group=1 will be general Convolution (because there is no Group)

## Relationship Distillation
Individual KD  
- Distilling knowledge based on each sample. 
- Student learn teacher's output 

Relational KD  
- Distilling knowledge based on the relationship between samples.
- Student learn model's represent
- Distance-wise KD
  - L1 or L2 loss
- Angle-wise KD
  - Cosine Similarity

### Distill relation between feature
- Similarity-preserving KD  
  Extract relational information on feature as cosine similarity table. Let student network imitate to learn relationship

# Math 
## Kullback–Leibler divergence
The Kullback–Leibler divergence (also called relative entropy) is a measure of how one probability distribution is different from a second, reference probability distribution.  
The smaller the KL divergence, the better the match between the true distribution and the approximate distribution.
A Kullback–Leibler divergence of 0 indicates that the two distributions in question are identical.
For discrete probability distributions 
$P$ and $Q$ defined on the same probability space, ${\mathcal {X}}$, the Kullback–Leibler divergence from $Q$ to $P$ is defined to be
$$D_{KL}(P||Q) = \sum_{x\in {\mathcal {X}}}P(x)\log \left( \dfrac{P(x)}{Q(x)} \right)$$
which is equivalent to 
$$D_{KL}(P||Q) = - \sum_{x\in {\mathcal {X}}}P(x)\log \left( \dfrac{Q(x)}{P(x)} \right)$$
For distributions $P$ and $Q$ of a continuous random variable, the Kullback–Leibler divergence is defined to be the integral:
$$D_{KL}(P||Q) = \int_{-\infin}^{\infin}p(x)\log \left( \dfrac{q(x)}{p(x)} \right)dx$$
where $p$ and $q$ denote the probability densities of $P$ and $Q$.

More generally, if $P$ and $Q$ are probability measures over a set $\mathcal {X}$, and $P$ is absolutely continuous with respect to $Q$, then the Kullback–Leibler divergence from $Q$ to $P$ is defined as
$$D_{KL}(P||Q) = \int_{\mathcal {X}}\log \left( \dfrac{dP}{dQ} \right)dP$$
where $\frac{dP}{dQ}$ is the Radon–Nikodym derivative of $P$ with respect to $Q$, and provided the expression on the right-hand side exists. Equivalently (by the chain rule), this can be written as
$$D_{KL}(P||Q) = \int_{\mathcal {X}}\log \left( \dfrac{dP}{dQ} \right)\dfrac{dP}{dQ}dQ$$
Various conventions exist for referring to ${\displaystyle D_{\text{KL}}(P\parallel Q)}$ in words. Often it is referred to as the divergence between $P$ and $Q$, but this fails to convey the fundamental asymmetry in the relation. Sometimes, as in this article, it may be found described as the divergence of P from Q or as the divergence from $Q$ to $P$. This reflects the asymmetry in Bayesian inference, which starts from a prior $Q$ and updates to the posterior $P$. Another common way to refer to ${\displaystyle D_{\text{KL}}(P\parallel Q)}$ is as the relative entropy of $P$ with respect to $Q$.