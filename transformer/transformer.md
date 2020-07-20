# Transformer

> Recommend Reading material :   
> 1. [illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
> 
> 2. [基于注意力机制，机器之心带你理解与训练神经机器翻译系统](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650742155&idx=1&sn=137825a13a4c31fffb6b2347c0304366&chksm=871ad9f5b06d50e31e2857a08a4a9ae9f57fd0191be580952d80f1518779594670cccc903fbe&scene=21#wechat_redirect)


Transformer, the model extends attention to accelerate training, and the biggest advantage is that it can be parallelized.

Transformer is proposed in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)", where TF application is a sub-module of [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor). Harvard's NLP team specially produced the corresponding PyTorch's [Guide Notes](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

Check the 
[transformer structure image](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9icbicicqO5VlBLqWvjSQuDrbDsyoBUIGjbxocOs31HbHpPQhwOl0hOiaVrkKQHUjuUI3Fe7mNxGOhvg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## A High_Level Look
We first treat the entire model as a black box. For example, in machine translation, a sentence in one language is received as input, and then it is translated into other languages ​​for output. ![image-20200226184149942](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226184150-335044.png)

Take a closer look, it consists of encoding components, decoding components and the connection layer between them.

![image-20200226185543585](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226185545-765586.png)

The encoded part of the model is a stack of encoders, and the decoded part of the model is a stack of the same number of decoders

![image-20200226191121778](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226191123-12424.png)

The input of the encoder first enters a **self-attention** layer, which helps the encoder to see other words in the input sequence when encoding a word

The output of self-attention flows to a feedforward network, and the feedforward network corresponding to each input position is independent of each other.

The decoder also has these sub-layers, but an attention layer is added between the two sub-layers, which helps the decoder to pay attention to the relevant part of the input sentence.

![image-20200226191352743](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226191354-3363.png)

## Bringing The Tensors Into The Picture

Let's analyze the most important part of the model, starting with Vectors/Tensors, and observe how they flow through each component and output.

As a common example of NLP application, first convert the input word into a vector using [embedding algorithm](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)

![image-20200226213047935](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226213049-312489.png)

The vectorization of words only occurs at the input of the lowest encoder, so that each encoder will receive a list (each element is a 512-dimensional word vector), but the input of other encoders is the output of the previous encoder. The size of the list is a super parameter that can be set, usually the length of the longest sentence in the training set.

After vectorizing the input sequence, they flow through the following two sublayers of the encoder.

![image-20200226231909842](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226232657-877655.png)

A key feature of Transformer can be seen here, the word at each position only flows through its own encoder path. In the self-attention layer, these paths are interdependent. **The forward network layer does not have these dependencies**, but these paths can be executed in parallel when flowing through the forward network

Next, we give a simple example, and then look at what happens in each sublayer of the encoder

### *Now We’re Encoding!*

As mentioned earlier, the encoder receives the input of the vector list and feeds it into self-attention, then feeds it into the forward network, and finally passes the input to the next encoder

![image-20200226233452340](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226233453-347194.png)

The word vectors at each position in the figure above are fed into the self-attention module, followed by the forward network (the exact same network structure for each vector).

## Self-Attention at a High Level

Take the following sentence as an example, as the input sentence "The animal didn’t cross the street because it was too tired" that we want to translate. What does "it" mean in the sentence? Does "it" mean "street" or "animal"? Very simple question for humans, but not simple for algorithms

When the model deals with the word "it", self-attention allows the connection between "it" and "animal".

While the model works with words in each position, self-attention allows the model to focus on other position information in the sentence as auxiliary information (cues) to better encode the current word

Transformer uses Self-attention to encode the "understanding" of related words in the current word

![image-20200226233537955](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226233539-739966.png)

As we use #5 encoder to encode The word 'it', part of the Attention will focus on The word 'The Animal' and incorporate some of its representation into The encoding of the word 'it'

### Attentional mechanisms
In the original paper, Google showed a general definition of attention mechanism, that is, it is a coding sequence scheme like RNN or CNN. An attention function can be described as mapping query and a set of key value pairs to output, where query, key, value and output are vectors. The output can be calculated by the weighted sum of the values, and the weight assigned to each value can be calculated by query and the compatibility function of the corresponding key.

In translation tasks,Query can be regarded as the original word vector sequence, while Key and Value can be regarded as the target word vector sequence. The general attention mechanism can be interpreted as calculating the similarity between Query and Key, and using this similarity to determine the attention relationship between Query and Value. In order wors, the operation between Query and Key is equivalent to calculating the internal similarity of the input sequence, and paying attention to the internal connection of the sequence itself (Value) based on this similarity or weight. This internal connection may be that the subject notices the information of the predicate and object or other structures hidden in the sentence.

**Scaled Dot Product Attention**  
The input is composed of Query and Key vector whose dimensions are $d_k$, and Value vector whose dimension is $d_v$. We will first calculate the dot product of Query and all Keys, and divide each by `squre_root(d_k)` to prevent the product result from being too large, and then feed it to the Softmax function to obtain the weight corresponding to Value. With these weights, we can configure the Value vector to get the final output.

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

> **More detials about Scaled Dot Product Attention**  
> In the above formula, the dot product of Q and K will be divided by squre_root($d_k$) to achieve scaling. The author of the original paper found that when the dimension $d_k$ of each Query is relatively small, the performance of dotted attention and additive attention is similar, but as $d_k$ increases, the performance of additive attention will exceed the dotted attention mechanism. However, dot multiplication attention has a powerful property, that is, it can use the parallel operation of matrix multiplication to greatly speed up training.  
> The author of the original paper thought that the reason of the poor effect of dot product attention was that when $d_k$ is relatively large, the product result will be very large, so it will cause Softmax to quickly saturate and only provide a very small gradient to update the parameters. So they adopted the square root of $d_k$ to reduce the dot product result and prevent the Softmax function from saturating.  
> In order to prove why the magnitude of the dot product becomes very large, we assume that the vector $Q$ and $K$ are both independent random variables with mean 0 and variance 1, and their dot product $Q⋅K=\sum_iq_i \cdot k_i$ has 0 mean and the variance $d_k$. To counteract this effect, we can normalize the dot product result by dividing by squre_root($d_k$).


## Self-Attention in Detail
Let's first look at how to calculate the self-attention vector, and then look at how to calculate it in a matrix.

**First step**, according to the input vector of the encoder (in this example, each word Embedding Vector), generate three vectors, for example, for each word vector, generate Query-vector, Key-vector , Value-vector. The generation method is to multiply the three matrices $W^Q, W^K, W^V$, these three matrices need to be learned during the training process. Note: Not every word vector has 3 matrices, but all inputs share 3 conversion matrices;

Note: the dimensions of these new vectors are smaller than those of the input word vector (512 dimension is converted to 64 dimension), which is not necessary, but to make the calculation of multiheaded attention more stable

![image-20200226220455482](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226220456-552343.png)

- q : query(to match others)  
  $q^i = W^qx^i$
- k : key(to be matched)  
  $k^i=W^kx^i$
- v : information to be extracted  
  $v^i = W^vx^i$

Multiplying $X_1$ by the $W^Q$ weight matrix will result in $q_1$, which is the query vector associated with the word. We finally created a "Queries", a "Keys" and a "Values" projection for each word in the input sentence.

**Second step**, in fact, we calculate Self-attention is to calculate a score. For example, for the sentence "Thinking Matchines", when we calculate the score for "Thinking", we need to calculate the "evaluation score"* of each word in the input sentence for the current word "Thinking", * this score determines the encoding " Thinking", how much attention each input word needs to focus on the current word set.

The score of the current word is obtained by performing a dot product of the query-vector corresponding to "Thinking" and the key-vecor of all words in the sentence. So when we deal with the word at position #1 ("Thinking") in the sentence, the first score is the dot product of q1 and k1, and the second score is the dot product of q1 and k2.

![image-20200226224449759](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226224451-504078.png)

**Steps 3 and 4**, divided by 8 ($=\sqrt{d i m_{k e y}}$), so the gradient will be more stable. Then add the softmax operation and normalize the scores so that all are positive and the sum is 1.

$$a_{1,i}=\dfrac{q^1\times k^i}{\sqrt{d}}$$

![image-20200226225145518](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226225146-937330.png)

The softmax score determines the degree of expression (attention) of each word in the sentence at this position. Obviously, the word in this position should have the highest softmax score.

**Step 5**, multiply the softmax score by value-vec bit by bit. Keep the value of the word of interest and weaken the value of non-related words (for example, multiply them by decimals such as 0.001).

**Sixth step**, add all the weighted vectors to produce the output of self-attention at that position.

$$z^1 = \sum_i\hat{a}_{1,i}v^i$$

![image-20200226230901466](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226230902-736522.png)

The above is the calculation process of self-attention, and the generated vectors flow into the forward network. In practical applications, the above calculations are performed in the form of a matrix which speeds up. Below we look at the matrix calculation at the word level.

### Matrix Calculation of Self-Attention

**First step**, calculate **Query/Key/Value** matrix, and merge **all input embedding vector** into input matrix X (each row of input matrix X represents a word vector of input sentence), And multiply it by our weight matrix after training**$W^Q,W^K,W^V$. (In the figure below, we can see the input embedding vector size X [512 dimensions, represented by four small squares], and the output vector Q/K/V [64 dimensions, represented by three small squares])

![image-20200226231303179](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231304-175221.png)

Finally, since we are dealing with matrices, we can combine steps 2 to 6 into a formula for calculating the output of the self-attention layer, as shown below

![image-20200226231328566](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231330-58687.png)

**TIPS** All $z^1,z^2,\cdots$ can be parallelly computed. Merge all input vectors $x^1,x^2,\cdots$ to an ensemble vector $I$. And then multiply it with $W^q,W^k,w^v$ to generate $Q,K,V$ matrix for query, key, value matrix.
$$Q = I \times W^q$$
$$K = I \times W^k$$
$$V = I \times W^v$$
$$A = K^T\times Q$$
$$\hat{A} = \operatorname{softmax}(A)$$
$$Z = V \times \hat{A}$$

## Multi-head Attention Mechanism
The following figure shows the Multi-head Attention structure used in Transformer, which is actually multiple dot product attentions that are processed in parallel and finally stitched together. Generally speaking, we can perform h different linear transformations on the three input matrices $Q, V, K$, and then put them into h dot product attention functions and concatenate all the output results.

[Multihead Attention](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9icbicicqO5VlBLqWvjSQuDrbHo1OE3fZZkPY8CiaCORuia4BZEgA4Luxx0cXovNOWJsV43pM5FNN2nvw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Multi-head Attention allows the model to jointly pay attention to different representation subspace information at different positions. We can understand it as performing dot product attention multiple times without sharing parameters. The expression of Multi-head Attention is as follows:

$$MultiHead(Q,K,V) = Concat(Head_1,\cdots,Head_h)W$$
$$Head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)$$

Where $W$ is the weight matrix corresponding to the linear transformation, and Attention() is the dot product attention function mentioned above.

> In the original paper, the researcher used $h=8$ parallel dot product attention layers to complete Multi-head Attention. For each attention layer, the dimension used in the original paper is $d_k=d_v=\dfrac{d_{model}}{h}=64$. Since the dimensionality of each parallel attention layer is reduced, the total computational cost is very similar to the cost of a single dot multiplication attention in the full dimension.

Transformer uses Multi-head Attention in three different ways. First of all, in the encoder to decoder level, the Query comes from the output of the previous decoder, and the memorized Key and Value come from the output of the encoder. This allows every position in the decoder to pay attention to all positions in the input sequence, so it actually mimics the typical encoder-decoder attention mechanism in the sequence-to-sequence model.

Secondly, the encoder layer contains the self-attention layer, and all Value, Key, and Query in this layer are the same input matrix, which is the output of the previous layer of the encoder. Finally, the self-attention layer in the decoder allows every position in the decoder to notice all legal positions including the current position. This can be achieved by the Mask function, thereby preventing left-direction information flow to maintain autoregressive properties.

## Multi-head Attention in Detail
By adding a mechanism called multi-headed Attention, this paper further improves self-attention, which can improve the performance of Attention in two ways: 

1. Multi-headed expands the model’s ability to focus on different positions in the sentence. In the above example, $z_1$ contains very little information about other words, and is only determined by the actual word. In other cases, such as the translation of "The animal didn’t cross the street because it was too tired", we want to know what the word "it" refers to.
2. Multi-headed gives the attention layer multiple representation subspaces. As shown in the following example, there are multiple sets of Query/Key/Value-matrix under the multi-headed attention layer, not just one set (8-heads are used in the paper, each head has a set of Q/K/V matrix). Each group is randomly initialized, and after training, the input vector can be mapped into different sub-expression spaces.

![image-20200226231351517](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231351-823354.png)

If we perform the same self-attention calculation above, there are eight different sets of weight matrices (Q/K/V), and finally we will get eight different Z matrices.

![image-20200226231411585](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231412-54883.png)

This is a bit of a hassle, because forward networks cann't accept eight matrices at a time, but want the input to be one matrix, so there's a way to deal with combining eight matrices into one matrix.

What shall we do? We merge the matrix, and then multiply the combined matrix by another weight matrix $W_o$

![image-20200226231426268](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231426-438565.png)

The above is the content of the multi-head self-attention mechanism. I think it is only a part of the matrix. Let's try to visualize them as follows.

![image-20200226231440268](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231440-78655.png)

Now that we know about "Attention Heads," let's revisit the example from the previous section to see what the different "Attention Heads" focus on when encoding the word "it" in the example sentence

![image-20200226231450239](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231454-575423.png)

When encoding "it", one attention head is concentrated on "the animal" and the other head is concentrated on "tired". In a sense, the expression of the model for "IT" synthesizes animal and tired.. If we put all the attention heads into the picture, it is difficult to explain intuitively

![image-20200226231502275](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231515-165008.png)

## Representing The Order of The Sequence Using Positional Encoding

So far, we have not discussed how to understand the order of words in input sentences

To solve the problem of word order, Transformer add a new vector for each word. These vectors follow a specific pattern for the model to learn to determine the position of the word or the distance between different words in the sequence. An intuitive understanding, after adding these vectors to the embedding vector, once they are projected into the Q/K/V vector. In dot-product attention, they can provide meaningful distance or location information between the embedding vectors.

In order for the model to keep the word order, we add a positional encoding vector that follows a particular pattern.

![image-20200226231519980](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231524-843898.png)

如果我们假设嵌入的维数为4，则实际的位置编码应如下所示：

![image-20200226231538393](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231538-840441.png)

What does a designated pattern look like?

In the figure below, each line corresponds to the position code of a vector. Therefore, the first line will be the embedded vector that we want to add to the first word in the input sequence. Each row contains 512 values. Each value is between 1 and -1. We have color-coded them to visualize the pattern. (Figure: A real example has 20 words, each with 512 dimensions. You can observe the significant separation in the middle, because the left side is generated with the sine function and the right side is generated with cosine)

![image-20200226231556397](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231556-650814.png)

There is not only one method for position coding. Note that the encoding method must be able to handle sequences of unknown length.

In the original paper, researchers use sine and cosine functions of different frequencies :

$$PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}})$$                                     
$$PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})$$

Where `pos` is the position of the word, and `i` is the i-th element of the position encoding vector. Given the position `pos` of the word, we can map the word to a position vector of $d_{model}$ dimension, and the i-th element of the vector is calculated by the above two formulas. In other words, each dimension of the position code corresponds to a sine curve, and the wavelength constitutes a geometric sequence from $2π$ to $10000⋅2π$.

The position vector of the absolute position is constructed above, but the relative position of the word is also very important. This is the subtlety of the trigonometric function used by Google researchers to represent the position. The sine and cosine functions allow the model to learn the relative position based on two transformations:

$$sin(α+β)=sinα cosβ+cosα sinβ $$
$$cos(α+β)=cosα cosβ−sinα sinβ $$

For a fixed offset $k$ between words, the position vector $PE(pos+k)$ can be represented by the combination of $PE(pos)$ and $PE(k)$, which also represents the relative position between languages.

## The Residuals Connection

One detail worth noting in the encoder architecture is that in each sub-layer (FFNN), there are residual joins, and these are closely followed by [layer-normalization](https://arxiv.org/abs/1607.06450)

![image-20200226231609916](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231746-929475.png)

If we visualize vector and layer-norm operations, it will look like this:

![image-20200226231730087](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231730-99810.png)

## Feed-forward network by Position

In order to pay attention to the sub-layers, each encoder and decoder module finally contains a fully connected feedforward network, which is applied independently and identically to each location. This feedforward network contains two linear transformations and a nonlinear activation function, and we can add a Dropout method between the two layers of the network during the training process:

$$FFN(x) = \max(0, xW_1 + b_1 )W_2 + b_2$$

If we combine these two fully connected networks with residual connection and layer normalization, then it is the last necessary sub-layer for each encoder and decoder module. We can express this sub-layer as : 

$$LayerNorm(x + max(0,xW_1+b_1)W_2+b_2)$$

Although the linear transformation is the same in all different positions, it uses different parameters in different layers. This transformation can actually be described as two convolutions with a kernel size of 1.



## Batch Norm v.s. Layer Norm
- Batch Norm : same dimension at different channels  
  Batch normalization is to consider the input data of a neuron in mini-batch, calculate the average value and variance, and then use this data to normalize the input of each training sample.

- Layer Norm : different dimension at same channels of a layer  
  The input data of the same layer are considered, and the mean value and variance are calculated to normalize the input data of each layer.  
  The formula for calculating the layer normalized statistics for all hidden cells in the same layer is shown below :
  $$h^t = f\lbrack \frac{g}{\sigma^t} \odot (a^t -\mu^t) + b\rbrack$$
  $$\mu^t = \dfrac{1}{H}\sum_{i=1}^Ha_i^t$$
  $$\sigma^t = \sqrt{\dfrac{1}{H}\sum_{i=1}^H(a_i^t - \mu^t)^2}$$

  $h^t$ represents the output of hidden layer t, $a_i^t$ represents the i-th input of the hidden layer t, $g$ is the gain parameter that scales the normalized activation before the nonlinear activation function. $\beta$ is bias.

  Layer normalization can greatly reduce the covariance deviation problem by correcting the mean and variance of the activation values in each layer.

Batch Norm is applied on batch and normalizes on NHW, which is more often used on CV task. As different channels have different important features, different channels perfom different normalization. Same batches in the same channel do the same normalization. However, Layer Norm is applied on channal and normalizes on CHW, which more frequently used on NLP task. It can be regarding as sequence length. So layer norm normalizes on the words among in a sequence of sentence, while batch norm normalized on same locatoin amone a channel. More importantly, the length of sequence can vary and channel can't be consistent, so Layer norm can apply it to sentence by sentence.
 
The sub-layer of the decoder is similar to this structure. Assume that our model is a Transformer composed of 2 stacked encoders and decoders, as shown in the following figure:

![image-20200226231730087](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200226231730-99810.png)

## The Decoder Side

Now that we have understood most of the concepts on the encoder side, we also have a basic understanding of how the decoder works. Let's see how they work together.

The encoder starts from the processing of the input sequence, and the output of the last encoder is converted into a set of attention vectors K and V, which are used by the "encoder-decoder atttention" layer of each decoder to help the decoder focus on the appropriate position of the input sequence

After encoding, it is the decoding process; each step of decoding outputs an element as the output sequence.

![1](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200227122050-255783.gif)

The following steps are repeated until a special symbol appears to indicate that the decoder has completed the translation output. The output of each step is fed to the bottom decoder in the next time step, and the decoder outputs the corresponding decoding result at each step like the encoder. In addition, just as the input of the encoder does, a position vector is added to the input of the decoder to indicate the position of each word.

The self-attention layer in the decoder is slightly different from the self-attention layer in the encoder:

In the decoder, before performing softmax operation, unknown output is achieved by occluding future positions (setting them to -inf). The self-attention layer only allows paying attention to the position earlier than the current output.
The "Encoder-Decoder Attention" layer works in the same way as multi-headed self-attention, except that it gets the output from the front layer and is converted into a Query matrix. And it receives the key and value matrix of the last layer encoder as the key and value matrix of current decoder.

## The Final Linear and Softmax Layer

The decoder finally outputs a floating-point vector, how to convert it into words? This is the main work of the final linear layer and softmax layer.

The linear layer is a simple full connection layer that maps the final output of the decoder to a very large logits vector

Assuming that the model knows 10,000 words (output word list) learned from the training set, the logits vector has 10,000 dimensions, each value representing the possible tendency value of a certain word. The softmax layer converts these scores into probability values (all positive, and the sum is 1), and the word on the dimension corresponding to the highest value is the output word of this step.

---

Compared with the scores in the range of $(-\infty,+\infty)$, the probability is naturally better interpretable, so that subsequent threshold operations can be completed smoothly (dimensionality is not involved after normalization).

Through the full connection layer, we get the score within the range of K categories $(-\infty,+\infty)$ logits, which is called $Z_j$. In order to get the probability of each category, we first map the score to $(0,+\infty)$ through $e^{Z_j}$, and then normalize it to (0,1). Softmax thought is to rely on the calculation dependency of the loss function

$$\hat{y}_j = \operatorname{softmax} \left(z_j \right) = \frac{e^{z_j}}{\sum_{k} e^{z_j}}$$

---

![2](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200227114656-299496.png)

The figure starts at the bottom and uses the resulting vector as the output of the decoder part, and then converts it to an output word

## Recap Of Training

Now that we have understood the forward process of a fully trained Transformer, during training, the model will undergo the forward process described above. When we train on the labeled training set, we can compare the predicted output with the actual output.

For visualization purposes, assume that the output has a total of six words (a, am, I, thanks, student, <eso>)

The output vocabulary of our model is created during the preprocessing stage, even before training has started.

![2](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200227114514-361222.png)

Once the word list is defined, we can construct a vector of the same dimension to represent each word, such as one-hot encoding. The following example encodes "am"

![4](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200227114717-192245.png)

We discuss the loss of the model, the indicators used to optimize in the training process, and guide the learning to get a very accurate model.

## The Loss Function

Let's demonstrate the training with a simple example, such as translating "merci" to "thanks". That means that the probability distribution of the output points to the word "thanks", but since the model is not trained and initialized randomly, it is unlikely to be the expected output.

This means that we want the output to be a probability distribution representing the word "thanks". However, since the model has not been trained, it is unlikely to happen at present.

![](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200227114616-436191.png)

Since the model parameters are initialized randomly, the untrained model outputs random values. We can compare with the real output, and then use error backward to adjust the model weights so that the output is closer to the real output.

How to compare two probability distributions? Simplely adopt cross-entopy or [Kullback-Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained). Since this is an extremely simple example, the more real situation is to use a sentence as input. For example, the input is "je suis étudiant" and the expected output is "i am a student". In this example, we expect the model to output a continuous probability distribution that meets the following conditions:
- Each probability distribution has the same dimensions as the vocabulary. (The dimension in the current example is 6, but in practice it is generally 3000 or 10000)
- The first probability distribution has the highest predicted probability value for "i"
- The second probability distribution has the highest predicted probability value for "am"
- Until the fifth output points to the "end of sentence" tag, the symbol also has cells related to the 10,000 element vocabulary

We will train a sample sentence for the target probability distribution in the training example.

![6](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200227115003-813369.png)

After training the model on a large enough data set for enough time, we hope to generate the probability distribution as follows:

After training, the output of the model is the translation we expect. Of course, this does not mean that the process is coming from the training set. Note that every location has a value, even if it has little to do with the output

![7](https://cdn.jsdelivr.net/gh/chenhaishun/test_pic@master/typora20200227115011-349509.png)

Now, because the model only produces one set of output per step, it is assumed that the model chooses the highest probability and discards the other parts. This is a method of generating prediction results called greedy decoding.

Another way to do it would be to hold on to, say, the top two words (say, ‘I’ and ‘a’ for example), then in the next step, run the model twice: once assuming the first output position was the word ‘I’, and another time assuming the first output position was the word ‘a’, and whichever version produced less error considering both positions #1 and #2 is kept. We repeat this for positions #2 and #3…etc. This method is called “*beam search*”, where in our example, beam_size was two (because we compared the results after calculating the beams for positions #1 and #2), and top_beams is also two (since we kept two words). These are both hyperparameters that you can experiment with.

Another method is beam search. Each step only keeps two outputs with the highest probability. According to these two outputs, the next step is predicted, and new two outputs with high probability are retained. Repeat until the end of prediction.

Beam search is only used for prediction (if greedy is used for prediction, one misstep could ruin all). Only use greedy when training, because in the training process, the output of each decoder has the corresponding correct answer as a reference, so there is no need for beam search to increase the accuracy of the output.

## Summary

### Advantages:
1. Although Transformer ultimately does not escape the traditional learning routine, the transformer is only a combination of full connection (or one-dimensional convolution) and Attention. However, its design is innovative enough, because it has abandoned the most fundamental RNN or CNN in NLP and achieved very good results. 
2. The key of Transformer's design to bring the greatest performance improvement is to set the distance between any two words to 1, which is very effective for solving the difficult long-term dependence problem in NLP. 
3. The parallelism of the algorithm is very good, which is in line with the current hardware (mainly GPU) environment.

### Disadvantages: 
1. Although rudely abandoning RNN and CNN is very dazzling, it also makes the model lose the ability to capture local features. The combination of RNN + CNN + Transformer may bring better results. 
2. The location information lost by Transformer is actually very important in NLP, and adding Position Embedding to the feature vector in the paper is only an expedient measure, and it does not change the inherent defects of Transformer structure.



