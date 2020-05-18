# Subclassing NN Module

In order to have the ability to control the flow of data through the network. To take full control of the processing of input data by subclassing `nn.Module`.  

**Demo of subclassing `nn.Module`**
```python
class SubclassModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden_linear = nn.Linear(1,13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13,1)
        
    def forward(self,input):
        hiddent_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t
```
Typically, you want to use the constructor of the module to define the submodules that we call in the `forward` function so that they can hold their parameters throughout the lifetime of your module. Assigning an instance  of `nn.Module` to an attribute in a `nn.Module` automatically registers the module as a submodule, which gives modules access to the parameters of its submodules.

**Note**  PyTorch uses a dynamic graph-based autograd, gradients would flow properly through the sometimes-present activation.

the `named_parameters()` call delves into all submodules assigned as attributes in the constructor and recursively calls `named_parameters()` on them. No matter how nested the submodule is, any nn.Module can access the list of all  child  parameters.

**NOTE** Child modules contained inside Python `list` or `dict` instances won’t be registered automatically!  Subclasses can register those modules manually with the `add_module(name,  module)` method of `nn.Module` or can use the provided `nn.ModuleList` and `nn.ModuleDict` classes (which provide automatic registration for contained instances)

Calling parameters of submodules directly in the `forward` function without rigistering unneccessary submodules, such as `nn.Tanh()`.  
PyTorch has `functional` counterparts of every `nn` module. `torch.nn.functional` provides many of the same modules you find in `nn`, but with all eventual parameters moved as an argument to the function  call.

**Demo of subclassing `nn.Module` using Functional**

```python
class SubclassFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden_linear = nn.Linear(1,14)
        self.output_linear = nn.Linear(14,1)
        
    def forward(self,input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.Tanh(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t
```
The functional version is a bit more concise and fully equivalent to the non-functionalversion.

**TIP** Although general-purpose scientific functions like tanh still exist in `torch.nn.functional` in **version 1.0**, those entry points are deprecated in favor of ones in the **top-level torch namespace**. More niche functions remainin `torch.nn.functional`.