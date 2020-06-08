# Network Pruning

## Main Question : Prune What?
### Which is most important? -- How to evaluate importance
- Evaluate by weight  
  - Caculate sum of L1-norm
  - FPGM, choose GM and prune others
- Evaluate by activation
- Evaluate by gradient

### Other parametgers we can use
- Eval by Batch Normalization - Network Slimming
  $$y = \dfrac{x-E(x)}{\sqrt{Var(x)+\epsilon}}\times \gamma + \beta$$
  $\gamma$ is a learnable vector. We can just use the parameters to evaluate importance  
  Without constraint, $\gamma$'s distribution may hard to prune(Because lots of $\gamma$ is non-trivial). To solve this problem ,we add L1-penalty on y, and then $\gamma$'s distribution is good enough to prune.
  $$L = \sum l(f(x,W),y) + \lambda \sum g(\gamma)$$
  $g*$ is L1-norm
- Eval by 0s after ReLU - APoZ  
  Calculate APoZ (avg % of zeros) in each feature maps

### After Evalutation
- Sort by importance and prune by rank
- prune by handcrafted threshold
- prune by generated threshold

## More About Lottery Ticket Hypothesis
- Use L1-norm to prune weight
- Under same architecture, init sign is important.
- Large final weight, same sign
- learning rate must be small

## Rethinking the value of network pruning
Pruning algorithms doesn't learn "network weight", they learn "network structure".
