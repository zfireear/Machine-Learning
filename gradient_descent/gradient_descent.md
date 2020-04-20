# Gradient Descent

## Definition
**Loss Function**
> $$L(w,b) = \sum_{n=1}^{10}(\hat{y}^n - (b + w \cdot x_{feature}^n) )^2$$  

**Gradient Descent**
> $$w^*,b^* = arg min_{a,b}L(w,b)$$
> $$= arg min_{a,b} \sum_{n=1}^{10}(\hat{y}^n - (b + w \cdot x_{feature}^n) )^2$$

## Where does the error come from?
**Bias and Variance of Estimator**  
- Estimate the mean of a variable x
    - assume the mean of x is $\mu$
    - assume the variance of x is $\sigma^2$
- Estimator of mean $\mu$
  - Sample N points : { $x^1,\ldots,x^n$}
  
> $$m = \frac{1}{N}\sum_nx^n$$
> $$s^2 = \frac{1}{N}\sum_n(x^n - m)^2$$
> $$m = \frac{1}{n}\sum_nx^n \neq \mu$$
> $$E[m] = E[\frac{1}{n}\sum_nx^n] = \frac{1}{n}\sum_nE[x^n] = \mu$$
> $$Var[m] = \frac{\sigma^2}{N}$$

Variance depends on the number of samples ,But best estimator 
> $$E[s^2] = \frac{N-1}{N}\sigma^2 \neq \sigma^2$$

## Bias vs Variance
- Underfitting
    - Large Bias
    - Small Variance
- Overfitting
    - Small Bias
    - Large Variance 

## Diagnoisis
- If your model cannot even fit the training examples, then you have large bias(Underfitting)
- If you can fit the training data, but large error on testing data, then you probably have large variance(Overfitting)

## What To Do With Large Bias
For large bias, redesign your model:
- Add more features as input
- A more complex model  

## What to do with large variance
- More data,very effective,but not always practical
- Regularization

## Model Selection
- N-fold Cross Validation



