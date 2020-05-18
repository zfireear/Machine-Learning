# Pre-trained Network

## TorchVision
The pre-defined models can be found in `torchvision.models`.
```python
from torchvision import models
dir(models)
```
The uppercase names correspond to classes that implement popular architectures for computer vision.  
The lowercase names are functions that instantiate models with pre-defined number of layers and units and optionally download and load pre-trained weights into them.

## Inference
The process of running a trained model on new data is called *inference* in deep learning circles. In order to do inference, we need to put the network in **eval** mode. 

## Demo of Pre-trained ResNet 
```python
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

resnet = models.resnet101(pretrained=True)

# preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485,0.456,0.406],
        std = [0.229,0.224,0.224]
    )])

img = Image.open("data/bobby.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t,0)

# important step
resnet.eval()

# training
out = resnet(batch_t)

# labels
with open("data/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# max possible index
_,index = torch.max(out,1)

# max possible label
labels[index[0]]

# percentage of max possible label
percentage = torch.nn.functional.softmax(out,dim=1)[0]*100
percentage[index[0]].item()

# top 5 possible lable
_,indices = torch.sort(out,descending=True)
[(labels[idx],percentage[idx].item()) for idx in indices[0][:5]]

```

**Tip** If we present an image containing a subjuect outside the training set, it's quite possible that the network will come up with a wrong answer with pretty high confidence.