# real world data representation

PyTorch tensors are homogeneous, information in PyTorch is encoded as a number, typically floating-point.

## One-hot Encoding 
Achieve one-hot encoding by using the `scatter_` method, which fills the tensor with values from a source tensor along the indices provided as arguments
```python
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
```
The arguments for `scatter_` are
- The dimension along which the following two arguments are specified
- A column tensor indicating the indices of the elements to scatter
- A tensor containing the elements to scatter or a single scalar to scatter(1,in this case)

The second argument of `scatter_`, the index tensor, is required to have the same number of dimensions as the tensor you scatter into. Because target_onehot has two dimensions(such as 4898x10),  you need to add an extra dummy dimension to target by using unsqueeze.unsqueeze adds a singleton dimension, without changing its contents. No  elements were added(That is, you accessed the first element of target as target[0] and the first element of its unsqueezed counterpart as target_unsqueezed[0,0])

## Tensor API
use the `torch.le` function to determine which rows in target correspond to a score less than or equal 
```python
bad_indexes = torch.le(target, 3)
```

## Advanced Indexing
use a binary tensor to index the data tensor.This tensor essentially filters data to be only items (or rows) that correspond to 1 in the indexing  tensor.
```python
bad_data = data[bad_indexes]
```

## Tips and tricks
```python
for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):    
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
```
1. `zip()` is to map the similar index of multiple containers so that they can be used just using as single entity.A zip object, which is an iterator of tuples where the first item in each passed iterator is paired together, and then the second item in each passed iterator are paired together etc.It evaluates the iterables left to right.  
2. The `enumerate()` function is used to combine a iterable data object (such as a list, tuple, or string) into an index sequence, which return list data and data subscripts at the same time. It is generally used in a for loop.

## Time Series
1. use `torch.sort` to order data appropriately.
2. concatenate your matrix to your original data set,  using the `cat` function
    ```python
    torch.cat((bikes[:24], weather_onehot), 1)
    ```

3. rescaling variables to the `[0.0,  1.0]` interval or the `[-1.0, 1.0]` interval is something that you’ll want to do for all quantitative variables
> If the variable were drawn from a Gaussian distribution, 68 percent of the samples would sit in the [-1.0, 1.0] interval.