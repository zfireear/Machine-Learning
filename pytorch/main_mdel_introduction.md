# Main Model Introduction

- Dataset Class  
  - `torch.utils.data`
  - The bridge between custom data (in whatever format it might be) and a standardized PyTorch tensor 

- DataLoader Class  
  - Easy, efficient, parallel processing
  - Its instances can spawn child processes to load data from a data-set in the background
  - To load our data and assemble tensors into batches

- Criterion or Loss Function
  - `torch.nn`
  - Compare the outputs of our model to the desired out-put (the targets) 

- Optimizer
  - `torch.optim`
  - To better resemble the target

-  Distributed Training
   - `torch.nn.parallel.DistributedDataParallel`
   - `torch.distributed` submodule

- TorchScript
  - Compile models ahead of time
  - can serialize a model into a set of instructions that can beinvoked independently from Python
  - Allow to export model, either as TorchScript to be used with the PyTorch runtime, or in a standardized format called ONNX.

- Model module  
  The uppercasenames correspond to classes that implement popular architectures. The lowercase names, on the other hand, are functions that instantiate models with predefined numbers of layers and units and optionally download and load pre-trained weights into them.  Note that there’s nothing essential about  using one of these functions: they just make it convenient to instantiate the model with a numberof layers and units that matches how the pretrained networks were built.