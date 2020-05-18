# Problem Set of PyTorch

1. `grad` can be implicitly created only for scalar outputs, so when we deal with loss, we need return a scalar value.
    ```python
    def loss_fn(t_p,t_c):
        squared_diffs = (t_p - t_c)**2
        return squared_diffs.mean()
    ```

2. "nvidia-smi has failed because it couldn't communicate with the nvidia driver. make sure that the latest nvidia driver is installed and running."  
   Solution: It may happen after your linux kerner update, if you entered this error, you can rebuild your nvidia driver using the following command to fix:  
   1. Firstly, you need have `dkms`,  which can automatically regenerate new modules after kernel version changes.   
   `sudo apt-get install dkms`
   2. Secondly, build your nvidia driver. Here my nvidia driver version is 440.82, if you have installed befor, you could check your installed version on `/usr/src`.  
   `sudo dkms build -m nvidia -v 440.82`
   3. Lastly,reinstall nvidia driver. And then reboot your computer.  
   `sudo dkms install -m nvidia -v 440.82`

   Now you can check to see if you can use it by `sudo nvidia-smi`. 




