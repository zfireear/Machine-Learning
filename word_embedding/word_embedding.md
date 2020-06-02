# Word Embedding

- Dimension Reduction
- Unsupervised approach : machine learns the meaning of words from reading a lot of documents without supervision.
- A word can be understood by its context
  
## How to exploit the context
- Count-based 
  - If two words $w_i$ and $w_j$ frequently co-occur, $V(w_i)$ and $V(w_j)$ would be close to each other  
  E.g. Glove Vector
  $$V(w_i) \times V(w_j) \leftrightarrow N_{i,j}$$
  Inner product $\leftrightarrow$ Number of times $w_i$ and $w_j$ in the same document.
- Prediction-based
  - one-hot encoding of word $w_{i-1} \rightarrow$ NN $\rightarrow$ The probability for each word as the next word $w_i$
  - Take out the input of the neurons in the first layer, Use it to represent a word w
  - Word vector, word embedding feature: $V(w)$ 
  - Sharing parameters : same words in different location should have same parameters.
  - **CBOW**(Continuous bag of word) model : predicting the word given its context
  - **Skip-gram** model : predicting the context given a word
  - **Language modeling**
    $$P("wreck \quad a \quad nice \quad beach") = P(wreck|START)P(a|wreck)P(nice|a)P(beach|nice)$$
    i-th output = $P(w_i = i|context)$ : the probability of NN predicting the next word.

## Word Embedding
- Characteristics  
  𝑉(𝑅𝑜𝑚𝑒) − 𝑉(𝐼𝑡𝑎𝑙𝑦) ≈ 𝑉(𝐵𝑒𝑟𝑙𝑖𝑛) − 𝑉(𝐺𝑒𝑟𝑚𝑎𝑛)
- Solving analogies  
  Rome : Italy = Berlin : ?  
  Compute 𝑉(𝐵𝑒𝑟𝑙𝑖𝑛) − 𝑉(𝑅𝑜𝑚𝑒) + 𝑉(𝐼𝑡𝑎𝑙) ≈ 𝑉(𝐺𝑒𝑟𝑚𝑎𝑛𝑦)

## Word Embedding Application
- Multi-lingual embedding
- Document embedding : the vector representing the meaning of the word sequence
- Semantic embedding
- Beyond bag of word : to understand the meaning of a word sequence, the order of the words can not be ignored.

