# Training, validation, and overfitting

## Training
The training loss, tells you whether your model can fit the training set at all. In other words, whether your model has enough capacity to process the relevant information in the data. 

If the training loss isn’t decreasing, chances are: 
- The model is too simple for the data.
- The data doesn’t contain meaningful information for it to explain the output.

## Validation
The validation set is used to provide an independent evaluation of the accuracy of the model’s output on data that wasn’t used for training

## Overfitting
Make sure that the model that’s capable of fitting the training data is as regular as possible between the data points. You have several ways to achieve this goal:
- Add so-called *penalization terms* to the loss function to make it cheaper for the model to behave more omoothly and change more slowly(up to a point).
- Add noise to the input samples, to artificially create new data points between training data samples and force the model to try to fit them too. 

## Tradeoffs 
The process for choosing the right size of a neural network model, in terms of parameters, is based on two steps: 
1. Increase the size until it fits 
2. And then scale it down until it stops overfitting

## Split data set
Shuffling the elements of a tensor amounts to finding a permutation of its indices.The `randperm` function does this:
```python
torch.randperm(n) # n (int) - the upper bound(exclusive) 
```
Demo of Split data into traing set and validation set.
```python
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
```
Separate tensors have been run through the same  functions, `model` and `loss_fn`, generating separate computation graphs without affecting each other.

## Switch autograd off
Tracking history comes with additional costs that you could forgo during the validation pass. To address this situation, PyTorch allows you to switch off autograd when you don’t need it by  using the `torch.no_grad` context manager.
```python
# All requires_grad args are forced to be False inside this block
with torch.no_grad():
    val_t_p = model(val_t_u,*params)
    val_loss = loss_fn(val_t_p,val_t_c)
```
Using the related `set_grad_enabled` context, you can also condition code to run with autograd enabled or disabled, according to a Boolean expression.
```python
with torch.set_grad_enabled(is_train):        
    t_p = model(t_u, *params)        
    loss = loss_fn(t_p, t_c)
```
Context managers such as with `torch.no_grad()`: can be used to control autograd behavior.
