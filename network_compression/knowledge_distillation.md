# Knowledge Distillation

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
