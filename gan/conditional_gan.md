# Conditional GAN

To control what should be generated.

## Basic Idea
The conditional GAN consists of two parts :

- Condition c and distribution z $\rightarrow$ Generator $\rightarrow x = G(c,z)$
- Condition c and generator x $\rightarrow$ Discriminator $\rightarrow$ scalar  
  To discriminate whether $x$ is realistic or not and $c$ and $x$ are matched or not

## Other GAN
- Stack GAN
- Patch GAN

# Unsupervised Conditional Generation
Transform an object from one domain to another without paired data.  

Domain X $\rightarrow$ G $\rightarrow$ Domain Y  

- Approach 1 : Direct Transformation  
  Structure 1 :    
  Domain X $\rightarrow G_{X\rightarrow Y} \rightarrow$ become similar to Domain Y $\rightarrow D_Y \leftarrow$ Domain Y   
  - Generator is to generate output which is similar to domain Y  
  - Using Discriminator to judge whether input from domain x belongs to domain Y or not.
  - Feed input and output to pre-trained Network and let the output of encoder network as close as possible

  Structure 2:  
  Domain X $\rightarrow G_{X\rightarrow Y} \rightarrow$ become similar to Domain Y $\rightarrow G_{Y\rightarrow X} \rightarrow$ reconstruction   
  Domain Y $\rightarrow G_{Y\rightarrow X} \rightarrow$ become similar to Domain X $\rightarrow G_{X\rightarrow Y} \rightarrow$ reconstruction
  Similar generated Domain X/Y of middle layer $\rightarrow$ D: Belongs to domain X/Y or not  
  This is namely Cycle GAN, cycle consistency(input X is as close as possible as reconstruction)
  
  Structure 3 :  
  StarGAN : For multiple domains, considering starGAN  

  Depth-wise concatenation of Target domain and Input Image $\rightarrow$ G $\rightarrow$ Fake Image $\rightarrow$ Depth-wise concatenation of Fake Image and Original domain $\rightarrow$ Reconstructed Image $\underleftrightarrow{close}$ Input Image

  On the one side, Discriminator is to judge whether generated result is real or not. On the other side, Discriminator is to judge whether generated result comes from specific domain.

- Approach 2 : Projection to Common Space  
  Domain X $\rightarrow$ Encoder of domain x $\rightarrow$ lantent $\rightarrow$ Decoder of domain Y $\rightarrow$ similar Domain Y
  - Couple GAN
  - UNIT
  - ComboGAN
  - DTN
  - XGAN