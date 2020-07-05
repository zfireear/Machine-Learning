# Introduction

## Language Model
A good trained language model:    
- P(I am studying nlp) > P(I studying nlp am) 

**How to calculate $P(\cdot)$ :  Markov hypothesis**
- Uni-gram : predict on present single word  
  - P(I am studying nlp) = P(I)P(am)P(studying)P(nlp) 
- Bi-gram : predict based on last word  
  - P(I am studying nlp) = P(I)P(am|I)P(studying|am)P(nlp|studying)
- Tri-gram : predict based on last two words
  - P(I am studying nlp) = P(I)P(am|I)P(studying|I, am)P(nlp|am, studying)
- N-gram : predict based on last N words

## Statistical MT : Three Problems
- Language Model
  - Given an English sentence, calculate its probability $P(e)$
  - $P(e)$ will be high if it is grammatical.
  - $P(e)$ will be low if it is a random statement.
- Translator Model
  - Given a pair of <c, e>, calculate its probability $P(c|e)$
  - $P(c|e)$ will be high If the semantic similarity is high
  - Otherwise, $P(c|e)$ will be slow
- Decoding Algorithm
  - Given language model, translator and origin sentence c, find the optimal option which maximises $p(e)p(c|e)$

## The four dimensions of natural language processing
1. Semantic
   - NLU
   - MT
2. Syntax
   - Syntactic parsing
   - Dependency Parsing
3. Morphology
   - Word Segmentation
   - Part-of-Speech
   - Name Entity Recognition
4. Phonetics

## Minimum Edit Distance -(DP)  
Edit distances find applications in natural language processing, where automatic spelling correction can determine candidate corrections for a misspelled word by selecting words from a dictionary that have a low distance to the word in question. 
