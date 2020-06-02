# Image Classification

## Introduction
### The `Dataset` class
> a subclass of `torch.utils.data.Dataset`

A PyTorch Dataset is an object that is required to implement two methods:`__len__` and `__getitem__`. The former should return the number of items in the dataset, while the latter should return the item, consisting of a sample and its corresponding label(an integer index).

`torchvision.transforms` method can be used to convert the PIL image to a PyTorch tensor.
```python
from torchvision import transforms
to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
```
Once transforms is instanitiated, it can be called like a function with the PIL images as the argument. It also can be passed directly as a argument.  
`ToTensor()` transform method will turn the data into 32-bit floating point per channel,scalling values down from 0.0 to 1.0.

`permute()` method can be used to change the order of the axes from $C\times H \times W$ to $H\times W \times C$ 
```python
img_tensor.permute(1,2,0)
```