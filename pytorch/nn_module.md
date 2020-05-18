# NN Module

PyTorch has a whole submodule dedicated to neural networks: `torch.nn`. This submodule contains the building blocks needed to create all sorts of  neural network architectures. Those building blocks are called modules in PyTorch parlance (and layersin other frameworks)  

A PyTorch module is a Python class deriving from the `nn.Module` base class. A `Module` can have one or more Parameter instances as attributes, which are tensors whose values are optimized during the training process.

**NOTE** The submodules must be top-level attributes, not buried inside list or dict instances! Otherwise, the optimizer won’t be able to locate the submodules (and, hence, their parameters). For situations in which your model requires a list or dict of submodules, PyTorch  provides `nn.ModuleList` and `nn.ModuleDict`.

All PyTorch-provided subclasses of `nn.Module` have their `call` method defined, which allows you to instantiate an `nn.Linear` and call it as though it were a function.

## Torch.nn.Linear
`CLASS torch.nn.Linear(in_features, out_features, bias=True)`

`nn.Linear`, a subclass of `nn.Module`, which applies an affine transformation to its input (via the parameter attributes `weight` and `bias`)

*Parameters* :
- in_features –  the number of input features ,or say, size of each input sample(or tensor) (i.e. size of x)
- out_features – the  number  of  output  features, or say, size of each output sample(or tensor) (i.e. size of y)
- bias – whether the linear model includes a bias. If set to False, the layer will not learn an additive bias. Default: True  

Note that the weights `W` have shape `(out_features, in_features)` and biases `b` have shape `(out_features)`. They are initialized randomly and can be changed later (e.g. during the training of a Neural Network they are updated by some optimization algorithm).

Calling an instance of `nn.Module` with a set of arguments ends up calling a method named `forward` with the same arguments. The `forward` method executes the forward computation; `call` does other rather important chores before and after calling `forward`. So it’s technically possible to call forward directly, and it produces the same output as call, but it **shouldn’t be done** from user code.

`weight`, which obtains from `nn.Module`, has the shape of `[out_features,in_features]`. 
- `out_features` is the number of network neurons in this layer.
- `in_features` is the number of neurons in the previous network layer.

`bias`, which also comes from `nn.Module`, has the shape of `[out_features]`

**Note**  To accommodate multiple samples, modules expect the zeroth dimension of the input to be the number of samples in the batch. You can provide an input tensor of size `B x Nin`, where B is the size of the batch and Nin thenumber of input features.

## Optimizer
```python
linear_model = nn.Linear(1,1)
optimizer = optim.SGD(linear_model.parameters(),lr=1e-2)
```
You can ask any `nn.Module` for a list of parameters owned byit or any of its submodules by using the `parameters` method. This call recurses into submodules defined in the module’s `init` constructor and returns a flat list of all parameters encountered.

Now you don’t pass params explicitly to model because the `model` itself holds its `Parameters` internally.Loss functions in nn are stillsubclasses  of nn.Module,  so  create  an  instance  and  call it  as  a  function. As following :

**Demo of NN Module**

```python
# train 
def training_loop(n_epochs,optimizer,model,loss_fn,t_un_train,t_un_val,t_c_train,t_c_val):
    for epoch in range(1,n_epochs):
        t_p_train = model(t_un_train)
        loss_train = loss_fn(t_p_train,t_c_train)
        
        t_p_val = model(t_un_val)
        loss_val = loss_fn(t_p_val,t_c_val)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 200 == 0:
            print("Epoch {}, Train Loss {}, Validation Loss {}".format(epoch,float(loss_train),float(loss_val)))

linear_model = nn.Linear(1,1)
optimizer = optim.SGD(linear_model.parameters(),lr=1e-2)
training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model = linear_model,
    loss_fn = nn.MSELoss(),
    t_un_train = t_un_train,
    t_un_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val
)
print()
print(f"weight : {linear_model.weight}")
print(f"bias : {linear_model.bias}")
```

## Hidden Layer
`nn` provides a simple way to concatenate modules through the `nn.Sequential` container.
```python
seq_model = nn.Sequential(
    nn.Linear(1,13),
    nn.Tanh(),
    nn.Linear(13,1)
)
# out of seq_model:
# Sequential(
#   (0): Linear(in_features=1, out_features=13, bias=True)
#   (1): Tanh()
#   (2): Linear(in_features=13, out_features=1, bias=True)
# )
```
The result is a model that takes the inputs expected by the first module specified as an argument of `nn.Sequential`, passes intermediate outputs to subsequent modules, and produces the  output returned by the last module. The model fans  out from 1 input feature to 13 hidden features, passes them through a tanh activation, and linearly combines the resulting 13 numbers into 1 output feature.

Calling `model.parameters()` collects weight and bias from all modules.
```python
[param.shape for param in seq_model.parameters()]
# Out:
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
```
After `model.backward()` iscalled, all parameters are populated with their `grad`, and then the optimizer updates their values accordingly during  the `optimizer.step()` call.

Identify parameters by their names. There’s a method for that, called `named_parameters`.
```python
for name, param in seq_model.named_parameters():
    print(name, param.shape)
# Out:
# 0.weight torch.Size([13, 1])
# 0.bias torch.Size([13])
# 2.weight torch.Size([1, 13])
# 2.bias torch.Size([1])
```
The name of each module in `Sequential` is the ordinal with which the module appeared in the arguments.

`Sequential` also accepts an `OrderedDict` in which you can name each module passed to `Sequential`
```python
from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
    ("hidden_linear",nn.Linear(1,8)),
    ("hidden_activation",nn.Tanh()),
    ("output_linear",nn.Linear(8,1))
]))
```
You can also get to a particular Parameter by accessing submodules as though they were attributes:
```python
seq_model.output_linear.bias
```

**Advanced Demo of NN Module**
```python
# Using neural network rather than fixed linear module as model,such as seq_model
optimizer = optim.SGD(seq_model.parameters(),lr=1e-3)
training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_un_train = t_un_train,
    t_un_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val
)
print("output",seq_model(t_un_val))
print("answer",t_c_val)
print("hidden",seq_model.hidden_linear.weight.grad)
```