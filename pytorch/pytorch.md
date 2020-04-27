# Pytorch

## Basis 
`torch.nn`, which provides common neural network layers and other architectural components.Fully  connected  layers,  convolutional  layers,  activation  functions,  and  loss  functions.

Utilities for data loading and handling can be found in `torch.util.data`

`DataLoader`, which can spawn child processes to loaddata from a Dataset in the background 

`torch.nn.DataParallel` and `torch.distributed` can be employed to leverage the additional hardware available

`torch.optim` provides standard ways of updating the model 

## Tensor
1. use zeros or ones to initialize the tensor, providing the size as a tuple
2. What you got back instead was a different view of the same underlying data (tensor),limited to specified row or column.
3. Values  are  allocated  in  contiguous  chunks  of  memory,  managed  by `torch.Storageinstances` , the storage under the hood is a contiguous array.
4. The `size` (or shape, in NumPy parlance) is a tuple indicating how many elements across each dimension the tensor represents. The storage `offset` is the index in the storage that corresponds to the first element in the tensor. The `stride` is the number of elements in the storage that need to be skipped to obtain the next element along each dimension.
5. The storage holds the elements in the tensor sequentially row by row.Accessing an element i, j in a 2D tensor results in accessing the `storage_offset +stride[0] * i + stride[1] * j` element in the storage.
6. `clone()` the subtensor intoa new tensor,not affecting original storage.
7. `torch.arange()` 

## Numeric type
1. `Tensor` is an alias class for `torch.FloatTensor`
2. `dtype` for a tensor by accessing the corresponding attribute
3. cast the output of a tensor-creation function to the right type by using the corresponding casting method(such as .double() method)
4. or the more convenient `to` method
   `double_points = torch.zeros(10, 2).to(torch.double)`
5. cast a tensor of one type as a tensor of another type by using the `type` method
6. Under  the  hood, `type` and `to` perform the same type check-and-convert-if-neededoperation, but the to method can take additional arguments

## Indexing tensor
The same as python list

## zero-copy interoperability with NumPy arrays
1. To get a NumPy array out of your points tensor `torch.ones(3,4).numpy()`. The returned array shares an underlying buffer (CPU RAM) with the tensor storage. Modifying the NumPy array leads to a change in the originating tensor. If the tensor is allocated on the GPU, PyTorch makes a copy of the content of the tensor into a NumPy array allocated on the CPU.
2. obtain a PyTorch tensor from a NumPy array `points = torch.from_numpy(points_np)`

## Serializing tensors
1. PyTorch uses `pickle` under the hood to serialize the tensor object, as well asdedicated serialization code for the storage,for saving it to a file and loading it back at some poin. 
`torch.save()` and `torch.load()`
2. save tensors interoperably for other library such as hdf5,save your tensor by converting it to a NumPy array.
   ```python
   import h5py

   f = h5py.File('ourpoints.hdf5', 'w')
   dset = f.create_dataset('coords', data=points.numpy())

   f = h5py.File('ourpoints.hdf5', 'r')
   dset = f['coords']
   last_points = dset[1:]

   f.close()
   ```
## Moving tensors to the GPU
1. `device`, which is where onthe  computer  the  tensor  data  is  being  placed
   ```python
   points_gpu = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]],device='cuda')
   ```
2. Copy  a  tensor  created  on  the  CPU  to  the  GPU  by  using  the `to` method.
   ```python
   points_gpu = points.to(device='cuda')
   ```
   Returns a new tensor that has the same numerical data but is stored in the RAM of the GPU rather than in regular system RAM.
3. Pass a zero-based integer identifying  the  GPU  on the machine.
   ```python
   points_gpu = points.to(device='cuda:0')
   ```
4. Points_gpu tensor performed  on  the  GPU isn’t brought back to the CPU when the result has been computed.
   - The tensor was copied to the GPU
   - A new tensor was allocated on the GPU and used to store the result of the mul-tiplication
   - A handle to that GPU tensor was returned 
   
   To move the tensor back to the CPU, you need to provide a `cpu` argument to the `to` method
   ```python
   points_cpu = points_gpu.to(device='cpu')
   ```
   use the shorthand methods `cpu` and `cuda` instead of the to method to achievethe same goal
   ```python
   points_gpu = points.cuda()
   points_cpu = points_gpu.cpu()
   ```
   When you use the `to` method, you can change the placement and the data type simultaneously by providing `device` and `dtype` as arguments

## The tensor API
A small number  of  operations  exist  only  as  methods  of  the  tensor object. They’re recognizable by the trailing underscore in their name, such as `zero_`, which  indicates  that  the  method  operates in-place by modifying the input instead ofcreating a new output tensor and returning it.   
Any method without the trailing underscore leavesthe source tensor unchanged and returns a new tensor

## Online Document Struction
- **Creation ops**—Functions for constructing a tensor,such as `ones` and `from_numpy`
- **Indexing,  slicing,  joining,  and  mutating  ops**—Functions  for  changing  the  shape,stride, or content of a tensor,such as `transpose`
- **Math ops**—Functions for manipulating the content of the tensor through com-putations:
  - *Pointwise ops*—Functions for obtaining a new tensor by applying a function toeach element independently, such as `abs` and `cos`
  - *Reduction ops*—Functions for computing aggregate values by iterating throughtensors, such as mean, std, and norm
  - *Comparison ops*—Functions for evaluating numerical predicates over tensors,such as `equal` and `max`
  - *Spectral  ops*—Functions for transforming in and operating in the frequency domain, such as `stft` and `hamming_window`
  - *Other ops*—Special functions operating on vectors, such as `cross`, or matrices,such as `trace`
  - *BLAS and LAPACK ops*—Functions that follow the BLAS (Basic Linear Algebra Subprograms)  specification for scalar,  vector-vector,  matrix-vector,  and matrix-matrix operations
- *Random sampling ops*—Functions for generating  values by drawing randomly from probability distributions, such as `randn` and `normal`
- *Serialization ops*—Functions for saving and  loading tensors, such as `load` and `save`
- *Parallelism ops*—Functions for controlling the  number of threads for parallel CPU execution, such as `set_num_threads`
























































