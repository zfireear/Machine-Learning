# Bert 

Feed large amount of text without annotation into a pre-train model and then Fine-tune with task-specfific data with annotation.

## Pre-train Model  
**Contextualized Word Embedding**  
Represent each token by an embedding vector based on the whole sentences 
- Each token has its own embedding vector
- Similar tokens have embedding vector with similar distance

### Traning of BERT
- Approach 1 : Masked LM  
  Predicting the masked word by Linear Multi-class Classifier. In technical terms, the prediction of the output words requires:  
  1. Adding a classification layer on top of the encoder output.
  2. Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
  3. Calculating the probability of each word in the vocabulary with softmax.

- Approach 2 : Next Sentence Prediction  
  The model receives pairs of sentences as input and learns to predict if the second sentence in the pair is the subsequent sentence in the original document.  
  - `[SEP]` : the boundary of two sentences  
  - `[CLS]` : the position that outputs classification results. To decide whether two sentences are to be connected together by Linear Binary Classifier.
    
  To help the model distinguish between the two sentences in training, the input is processed in the following way before entering the model:  
  1. A `[CLS]` token is inserted at the beginning of the first sentence and a `[SEP]` token is inserted at the end of each sentence.
  2. A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings are similar in concept to token embeddings with a vocabulary of 2.
  3. A positional embedding is added to each token to indicate its position in the sequence. 

  To predict if the second sentence is indeed connected to the first, the following steps are performed:
  1. The entire input sequence goes through the Transformer model.
  2. The output of the `[CLS]` token is transformed into a 2×1 shaped vector, using a simple classification layer (learned matrices of weights and biases).
  3. Calculating the probability of IsNextSequence with softmax.

Approaches 1 and 2 are used at the same time.

**TIP1** The location of `[CLS]` is not important, but is usually placed at the beginning of the sentences to be classified. Because the structure of encoder of transformer can keep trace of every character in different in a sentence or else, we just simply place it foremost.

**TIP2**  Word Masking  
Training the language model in BERT is done by predicting 15% of the tokens in the input, that were randomly picked. These tokens are pre-processed as follows — 80% are replaced with a `[MASK]` token, 10% with a random word, and 10% use the original word.

### How to use BERT? Or how to fine-turn for downstream task?
**Case 1**  
- Input : single sentence, Output : class  
- Train `[CLS]` to output an embedding to a linear classifier which trained from scratch to classify When performing downstream tasks. And fine-tune on BERT. Both they are trained together.
- Example : Sentiment Analysis, Document Classification

**Case 2**
- Input : single sentence, Output : class of each word
- Train on characters to output their embeddings to respective linear classifiers to classify
- Example : Slot filling

**Case 3**  
- Input : two sentences, output : class
- Using `[SEP]` to connect two sentences, and train `[Cls]` to a linear classifier to classify
- Example : Natural Language inference, for instance given a "premise", determining whether a "hypothesis" is T/F/unknown

**Case 4**  
- Extraction-based Question Answering (QA)  
  Input :   
  Document : $D = \lbrace d_1,d_2,\cdots,d_N\rbrace$  
  Query : $Q = \lbrace q_1,q_2,\cdots,d_M \rbrace$  

  Output : two integers $(s,e)$, $s$ is the answer location of start and $e$ is the answer location of end

  $D,Q \rightarrow$ QA Model $\rightarrow s,e$

  Answer : $A = \lbrace d_s,\cdots,d_e \rbrace$

- Example : SQuAD  

**TIP** Fine-tune the whole pre-trained model and task-specific layer together has better effect. An advanced soluction is **Adaptor Structure**

## Smaller Model
- Distill BERT
- Tiny BERT
- Moblile BERT
- Q8BERT
- ALBERT

[All The Ways You Can Compress BERT
](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)

### New Network Architecture
- Transformer-XL : Segement-Level Recurrence with State Reuse
  [Transformer-XL – Combining Transformers and RNNs Into a State-of-the-art Language Model](https://www.lyrn.ai/2019/01/16/transformer-xl-sota-language-model/)
- Reduce the complexity of self-attention
  - Reformer
  - Longformer





