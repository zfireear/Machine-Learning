# Conditional Generation by RNN & Attention

## Generation 
Generating a structured object component-by-component
- Sentences are composed of characters/words
  - Generating a character/word at each time by RNN
- Images are composed of pixels 
  - Generating a pixel at each time by RNN 
  
## Conditional Generation
- We don't want to simply generate some random sentences.
- Generate sentences based on conditions.
- Represent the input condition as a vector, and consider the vector as the input of RNN generator.
- Sequence-to-sequence learning
  - Encoder : represent the input condition
  - Decoder : generate the conditional output
  - Jointly train

Problem : how to consider longer context during chatting

## Attention - Dynamic Conditional Generation
### Why need dynamic conditional generation?
- Sometimes our input features are complex and hard to be encoded into a vector
- Let every single decode considers essential encoded information which is required at different single time 
- Focus on local dependent input feature 

## Machine Translation
### Attention-based model 
Base Idea : 
1. Introduce a **match** function to match output $h^n$ of hidden layer of input features through RNN with another vector $z^m$(initial by a NN or simply a vector) to produce an according output $\alpha_m^n$, and then let them through a softmax layer.
2. At last, we combine all the output $\hat{\alpha}_n^i$ of softmax layer 
   $$c^k = \sum \hat{\alpha}_k^nh^n$$
   which is our decoder input 

    |input feature|RNN|hidden output|match vector|match function||softmax layer||matrix multiplication||add up|decoder input|
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    |机|$\rightarrow$|$h^1$|$z^0$|$\rightarrow$|$\alpha_0^1$|$\rightarrow$|$\hat{\alpha}_0^1$|$\rightarrow$|$\hat{\alpha}_0^1h^1$|$\rightarrow$|$c^0$|
    |器|$\rightarrow$|$h^2$|$z^0$|$\rightarrow$|$\alpha_0^2$|$\rightarrow$|$\hat{\alpha}_0^2$|$\rightarrow$|$\hat{\alpha}_0^2h^3$|$\rightarrow$||
    |学|$\rightarrow$|$h^3$|$z^0$|$\rightarrow$|$\alpha_0^3$|$\rightarrow$|$\hat{\alpha}_0^3$|$\rightarrow$|$\hat{\alpha}_0^3h^3$|$\rightarrow$||
    |习|$\rightarrow$|$h^4$|$z^0$|$\rightarrow$|$\alpha_0^4$|$\rightarrow$|$\hat{\alpha}_0^4$|$\rightarrow$|$\hat{\alpha}_0^4h^4$|$\rightarrow$||

    From above, we could attain effective information block which remains its origin information makes decode perfom better. Such as we have $\hat{\alpha}_1^1 = 0.5, \hat{\alpha}_1^2 = 0.5, \hat{\alpha}_1^3 = 0.0, \hat{\alpha}_1^4 = 0.0$. We have then $c^0 = 0.5h^1+0.5h^2$, saying '机器' as an information block. This will produce a better encoded information.
3. Use $c^0$ feed into RNN to produce $z^1$ and then decode first translation with them. The same process individuals decodes with subsequent $c^1,\cdots$ repeat.

### What is macth function?
It designs by yourself, you can try the following:
- Consine similarity of $z$ and $h$
- Small NN whose input is $z$ and $h$, output a scala. It is worth noting that its parameters are jointly learned with other part of NN training.
- $\alpha = h^TWz$

## Memory Network
### Reading Comprehension
- Document is composed of many sentences. Assume we have $N$ sentences $x^1,x^2,\cdots,x^N$ learned by some sentence to vector methods.  
- Query $q$ is question-represented vector.  
- Calculate **match** score between $q$ and each single sentence $x^N$ to produce $\alpha_N$  
  $q,x^1 \overset{Match}{\longrightarrow} \alpha_1$
- Weighted sum all sentences with respective $\alpha$ for extracting information, it can match and select related sentences from document for query while unrelated information will be excluded as its weight equals to zero. 
  $$Extracted\quad Information = \sum_{n=1}^N\alpha_nx^n$$
- At last, feed extracted information and query into DNN to output answer.

**TIPS** Sentence to vector can be jointly trained with the other part of NN.

### Another version 
Extracted informations don't need to equal to match score, which has better performence. Represent same sentences with two kinds of different vectors ($x$ and $h$ vector) which can jointly learn. One to calculate match score and the another to perform weighted sum with $\alpha$. In addition, extracted information can add up to query vector to update. The afore mentioned process can repeat multiple times. At last, feed the extracted information into DNN to produce output.

## Neural Turing Machine
Neural Turing Machine can not only read information from memory but also modify the memory through attention.  
- Given a set of memory $m_0^1,m_0^2,m_0^3$, a vector sequence.
- Given a sef of Attention weight $\hat{\alpha}_0^1,\hat{\alpha}_0^2,\hat{\alpha}_0^3,\hat{\alpha}_0^4$
- Weighted sum to extract information
  $$r^0 = \sum \hat{\alpha}_0^im_0^i$$
- Feed $r_0$ and first input $x^1$ into **controller** function and then it will output three vectors $k^1,e^1,a^1$ which can controll memory and attention.
  - $k^1$ : the function of this vector is to generate attention  
    $$\alpha_1^i = \cos(m_0^i,k^1)$$  
    We generate new match score $\alpha$ after previous step such as $\alpha_1^1,\alpha_1^2,\alpha_1^3,\alpha_1^4$. And then let new match score through a softmax layer to generate distribution of attention $\hat{\alpha}_1^1,\hat{\alpha}_1^2,\hat{\alpha}_1^3,\hat{\alpha}_1^4$.

  - $e^1$ : the function of this vector is to empty memory, often value between 0 and 1.
  - $a^1$ : the function of this vector is to update memory, $a^1$ is the new vector.
- And then update memory :
  $$m_1^i = m_0^i - \hat{\alpha}_1^ie^1\odot m_0^i + \hat{\alpha}_1^ia^1$$

## Tips for Generation
### Attention
$a_t^i$ : i for component, t for time  
Good Attention : each input component has approximately the same attention weight. For emample, let attention cover all the frame and should not be too big.  
e.g. Regularization term: $\sum_i(\tau - \sum_t\alpha_t^i)^2$
- i for each component
- t over the generation
- $\tau$ is the sum of all weights of attention for each frame in the process of generation

### Mismatch between Train and Test
Taking RNN for training, and generation for testing. There is exposure bias.    
Testing : output of model is the input of the next step, which reference aren't known.    
Training : the inputs are reference.  
So one step wrong in training, may be totally wrong in testing as it never explore this case.  
Soluction : Scheduled Sampling

## Beam Search
Basic idea : keep several best path at each step

## Pointer Network
Pointer Network also apply attention mechanism, but the key difference is that in pointer network it uses **argmax** from distribution of attention weight as output rather than the weighted sum on attention weight. In this case, what decoder can output depends on the input.  
Pointer Network can apply on machine translation and chat-bot because it can identify proper noun such as person's name even there are not sample in training data.