# Deep Auto-encoder

Deep auto-encoder is an auto-encoder with multi-layer network structure

Assuming we have a image recognition task
- Input an image to a NN Encoder to produce a code
  - Compact representation of the input object
- Input a code to NN Decoder to produce an image
  - Can reconstruct the original object
- Auto-encoder is to combine the above procedure to learn together: Input an image into NN Encoder, then feed the output into NN Decoder to reconstruct the original object.

In general, we hope the dimension of code is smaller than origin dimension, but if the dimension of code is larger than origin dimension, auto-encoder may learn little. So when your hidden layer is larger than origin dimension, you should add a strong regularization such as L1 regularization.

## Better Auto-encoder
- De-noising Auto-encoder  
  Add noise to origin input $x$ to obtain noisy $x^\prime$, and then use $x^\prime$ to perform auto-encoder to obtain $y$. It should be noted that the general auto-encoder is to make the input and output as close as possible, but in de-noising auto-encoder it is to make the output as close input before adding noise as possible, which is robust.

## Application
- Text Retrieval
  - Encode document or query into vector by bag-og-word
- Similar Image Search
- Pre-training DNN
  - Use that target NN structure to perform auto-encoder to initial. Each time fix the weight learned by auto-encoder as initial weight, and greedy layer-wise pre-training again. At last fine-tune by backpropagation
  - Pre-training is still useful when you have a lot of unlabeled data

# More about auto-encoder
What we called code before can now be called **Embedding, Latent Representation, or latent code**.

**What is good embedding?**  
An embedding should represent the object.  

## Beyond Reconstruction  
We have a pair of image and its embedding, feed them into a **Discriminator** to discriminate whether the encoder is good or not.
- Loss of the classification task is $L_D$, the parameters of discriminator are $\phi$
- Train $\phi$ to minimize $L_D$ 
  $$L_D^* = \underset{\phi}{\min}L_D$$
  - Small $L_D^* \rightarrow$ The embeddings are representative.
  - Large $L_D^* \rightarrow$ The embeddings are not representative. 

How to evaluate an encoder? More than minimizing reconstruction error.  
- The parameter of encoder is $\theta$
- Train $\theta$ to minimize $L_D^*$
  $$\theta^* = \underset{\theta}{\argmin} L_D^* = \arg \underset{\theta}{\min} \underset{\phi}{\min}L_D$$
- Train the encoder $\theta$ and discriminator $\phi$ to minimize $L_D$

### Typical auto-encoder is a special case 
We can assume that our discriminator consists of a NN Decoder. 
- Feed into an image and its vector into auto-encoder.
- NN Decoder of Discriminator reconstructes image by decode vector
- Subtract the original image and the image produced by discriminator to calculate the difference score between them. The score is just reconstruction error.

## More interpretable embedding
- Feature Disentangle  
  An object contains multiple aspect information, there are many applications.
  - Voice Conversion
  - Adversarial Training
- Discrete Representation
  - Easier to interpret or clustering
    - Using a One-hot/Binary vector to represent
  - Vector Quantized Variational Auto-encoder(VQVAE)
  - Sequence as Embedding
    - Seq2seq2seq auto-encoder : Using a sequence of words as latent representation
    - Adding a Discriminator

## Conclusion
- More than minimizing reconstruction error
  - Using Discriminator
  - Sequential Data
- More interpretable embedding
  - Feature Disentangle 
  - Discrete and Structured

