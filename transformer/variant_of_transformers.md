# Variant of Transformers

## Sandwich Transformer
Question : Could we increase the performace just by reorder the sublayer module?  
Answer : Of couse!

### Designing a better transformer
- models with more self-attention toward the bottom and more feedforward sublayers toward the top tend to perform better in general.
- No extra parameters, memory requirement.

### Combination way of Sandwich Transformer 
$sssssssfsfsfsfsfsfsfffffff$ 

Length of Sandwich Coefficient is 6, which leads to better performance.

## Universal Transformer


## Residual Shuffle Exchage Network
- Fewer parameters compare to other models for the same tasks.
- Sequence processing in 0(n log n) Time, specialize application on long sequence.
- Shuffle & Exchange operators capture distant informations replace attention.