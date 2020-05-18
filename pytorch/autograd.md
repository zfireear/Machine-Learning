# Backpropagate all things

Gradient ,which we called `the rate of change of the loss` earlier.

PyTorch tensors can remember where they come from in terms of the operations and parent tensors that originated them, and they can provide the chain of derivatives of such operations with respect to their inputs automatically. Given a forward expression, PyTorch  provides  the  gradient  of  that expression  with  respect  to  its  inputparameters automatically.
```python
params = torch.tensor([1.0, 0.0], requires_grad=True)
```

**In general, all PyTorch tensors have an attribute named `grad`, normally None.**

**Calling `backward` leads derivatives to accumulate at leaf nodes. You need to zero the gradient explicitly after using it for parameter updates.**  
 
The gradient at each leaf is accumulated (summed) on top of the one computed at the preceding iteration. So you need zero the gradient explicitly at each iteration by using the in-place `zero_` method, which could be done at any point in the loop prior to calling `loss.backward()`
```python
if params.grad is not None:
    params.grad.zero_()
```

Detach the new params tensor from the computation  graph  associatedwith its update expression by calling `.detatch()`  

You can reenable tracking by calling `.requires_grad_()`,  an `in_place` operation (see the trailing _) that reactivates autograd for the  tensor. At that time, you can release the memory held by old versions of params and need to backpropagate through only your current weights.

**NOTE** The usage of `.detach()` is more secure
```python
# Q: What is the value of  x  that minimizes  f ?
x = torch.tensor([5.0],requires_grad=True)
step_size = 0.25
print("iter,\tx,\tf(x),\tf\'(x),\tf\'(x) pytorch")
for i in range(15):
    y = f(x)
    y.backward()
    
    print("{},\t{:.3f},\t{:.3f},\t{:.3f},\t{:.3f}".format(
            i,x.item(),f(x).item(),fp(x).item(),x.grad.item()))
    # .data 和.detach只取出本体tensor数据，舍弃了grad，grad_fn等额外反向图计算过程需保存的额外信息。
    # The .data and .detach operation only withdraw ontological data from tensor, and discard the extra information that needs to be saved in the calculation process of back propagation such as grad and grad_fn.
    x.data = x.data - step_size*x.grad
    # zero the grad
    # the detach_() is for efficiency
    x.grad.detach_()
    x.grad.zero_()
    #print(x.data) 
    #.data取出本体tensor后仍与原数据共享内存，在使用in-place操作后，会修改原数据的值，
    #而如果在反向传播过程中使用到原数据会导致计算错误，而使用.detach后，如果在反向传播过程中发现原数据被修改过会报错。更加安全
    # The .data operation takes out the ontology tensor and still shares memory with the original data, which the value of the original data will be modified after using the in-place operation
    # If the original data is used in the process of back propagation, it will cause calculation errors. After using .detach operation, if the original data had been modified in the process of back propagation, an error will be reported. It results in safer checking.
```

## Demo of Pytorch's autograd
```python
# Prepare the data
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_un = t_u * 0.1

# definition of model
def model(t_u,w,b):
    return w * t_u + b

# definition of loss function
def loss_fn(t_p,t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

# training using pytorch's autograd 
# update params with grad
def training_loop(n_epochs,learning_rate,params,t_u,t_c):
    for epoch in range(1,n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()
        t_p = model(t_u,*params)
        loss = loss_fn(t_p,t_c)
        loss.backward()
        params = (params - learning_rate * params.grad).detach().requires_grad_()    
    return params

loss = loss_fn(model(t_u,*params),t_c)
loss.backward()

training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0,0.0],requires_grad=True),
    t_u = t_un,
    t_c = t_c
)
```

## Optimizer
The `torch` module has an `optim` submodule where you can find classes that implement different optimization algorithms. 
```python
import torch.optim as optim

dir(optim)

# output:
 ['ASGD',  
 'Adadelta',
 'Adagrad',
 'Adam',
 'AdamW',
 'Adamax',
 'LBFGS',
 'Optimizer',
 'RMSprop',
 'Rprop',
 'SGD']
 ```
Every optimizer constructor takes a list of parameters (aka PyTorch tensors, typically with `requires_grad` set to `True`) as the first input.  
```python
params = torch.tensor([1.0,0.0],requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params],lr=learning_rate)
```

All parameters passed to the optimizer are retained inside the optimizer object so that the optimizer can update their values and access their `grad` attribute.

Each optimizer exposes two methods: `zero_grad` and `step`. The former zeros the grad attribute of all the parameters passed to the optimizer upon construction. The latter updates the value of those parameters according to the optimization strategy implemented by the specific optimizer.
```python
t_p = model(t_u,*params)
loss = loss_fn(t_p,t_c)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Optimizer `Adam` in which the learning rate is set adaptivel is a lot less sensitive to the scaling of the parameters.
```python
params = torch.tensor([1.0,0.0],requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params],lr=learning_rate)
```

Neural networks allow you to remove your arbitrary assumptions about the shape of the function you should be approximating.