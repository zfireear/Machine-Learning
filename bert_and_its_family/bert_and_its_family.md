# Bert and its Family

A word can have multiple senses.

**Contextualized Word Embedding**
- Each word token has its own embedding (even though it hase the same word type)
- The embeddings of word tokens also depend on its context.

## Embeddings from Language Model (ELMO)
- RNN-based language models(trained from lots of sentences)
- Bidirectional RNN
- Each layer in deep LSTM can generate a latent representation.
- Weighited sum over all latent representation. Weights are learned with the down stream tastk.

## Bidirectional Encoder Representations from Transformers (BERT)
- BERT = Encoder of Transformer  
  Learned from a large amount of text without annotation

## MASS/BART
- The pre-train model is a typical seq2seq model

## Efficiently Learning an Encoder that Classifies Token Replacements Accurately(ELECTRA)
- Predicting yes/no rather than reconstruction

## Enhanced Representation through Knowledge Integration (ERNIE)
- Designed for Chinese
  
## Generative Pre-Training(GPT)
- Decoder of Transformer

## XLNET
- Transformer-XL

## Robustly Optimized BERT Approach(RoBERTa)
- SOP : Sentence order prediction

## ALBERT
- Reduce parameters
  - Factorize embedding matrix
  - Shared same parameters across layer
- Modify pre-training task
  - Sentence order Prediction

## Reformer
- Find a small set of candidates by hash function (Reduce complexity from $o(N^2)$ to $o(N \log N)$)
- Reversible layer (Save memory by backstepping)

## Sentence Level
Representation for whole sequence
- Skip Thought
- Quick Thought