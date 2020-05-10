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
1. `zip()` is to map the similar index of multiple containers so that they can be used just using as single entity.A zip object, which is an iterator of tuples where the first item in each passed iterator is paired together, and then the second item in each passed iterator are paired together etc. It evaluates the iterables left to right.  
2. The `enumerate()` function is used to combine a iterable data object (such as a list, tuple, or string) into an index sequence, which return list data and data subscripts at the same time. It is generally used in a for loop.

## Time Series
1. use `torch.sort` to order data appropriately.
2. concatenate your matrix to your original data set,  using the `cat` function
    ```python
    torch.cat((bikes[:24], weather_onehot), 1)
    ```

3. rescaling variables to the `[0.0,  1.0]` interval or the `[-1.0, 1.0]` interval is something that you’ll want to do for all quantitative variables
> If the variable were drawn from a Gaussian distribution, 68 percent of the samples would sit in the [-1.0, 1.0] interval.

## Text One-Hot Encoding
One-hot encoding is a useful technique for representing categorical data in tensors.  
one-hot encoded sentence:
```python
line = '“Impossible, Mr. Bennet, impossible, when I am not acquainted with him'

letter_tensor = torch.zeros(len(line),128)

for i,letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_tensor[i][letter_index] = 1
```

one-hot encoded sentens in whole text:
```python
word_list = sorted(set(clean_words(text)))
word2index_dict = {word:i for (i,word) in enumerate(word_list)}

word_tensor = torch.zeros(len(words_in_line),len(word2index_dict))
for i,word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_tensor[i][word_index] = 1
```

## Text Embedding
Words used in similar contexts map to nearby regions of the embedding.

## Images
```python
img_arr = imageio.imread("bobby.jpg")
img_arr.shape
```
Img is a NumPy array-like object with three dimensions: two spatial dimensions (width and height) and a third dimension corresponding to the channelsred, green, and blue.

>PyTorch modules that deal with image data require tensors to be laid out as **C x H x W** (channels, height, and width respectively) 

You can use the transpose function to get to an appropriate layout. Given an inputtensor W x H x C, you get to a proper layout by swapping the first and last channels
 ```python
 img = torch.from_numpy(img_arr)
 out = torch.transpose(img,0,2)
 ```

Neural networks exhibit the best training performance when input data ranges from roughly 0 to 1 or –1 to 1 (an effect of how their building blocks are defined),Such as compute mean and standard deviation of the input data and scale it  so that the output has zero mean and unit standard deviation.

##  Volumetric data
You have an extra dimension, depth, after the channel dimension, leading to a 5D tensor of shape N x C x D x H x W.  
Load a sample CT scan by using the volread function in the imageio module.
```python
vol_arr = imageio.volread(dir_path,'DICOM')
vol = torch.from_numpy(vol_arr).float()
vol = torch.transpose(vol,0,2)
vol = torch.unsqueeze(vol,0)
```

## Summary
1. Neural networks require data to be represented as multidimensional numericaltensors, often 32-bit floating-point
2. Spreadsheets can be straightforward to convert to tensors
3. Text or categorical data can be encoded to a one-hot representation throughthe use of dictionaries
4. Volumetric data is similar to 2D image data,  with the exception of adding athird dimension: depth
5. Many images have a per-channel bit depth of 8, though 12 and 16 bits per chan-nel are not uncommon. These bit-depths can be stored in a 32-bit floating-pointnumber without loss of precision
6. Convolutional networks, we would need to lay out the tensor as `N x C x L`, where `N` is the number of sounds in a dataset, `C` the number of channels and `L` the number of samples in time.
7. Recurrent networks we mentioned for text, data needs to be laid out as `L x N x C` - sequence length comes first. Intuitively, this is because the latter architectures take one set of `C` values at a time

## Audio
In order to load the sound we resort to SciPy, specifically `scipy.io.wavfile.read`, which has the nice property to return data as a NumPy array:
```python
import scipy.io.wavfile as wavfile

freq, waveform_arr = wavfile.read('1-100038-A-14.wav')
waveform = torch.from_numpy(waveform_arr).float()
```
**Fourier transform** : converte a signal in the time domain into its frequency content

We import the `signal` module from SciPy,
then provide the `spectrogram` function with the waveform and the sampling frequency that we got previously.
The return values are all NumPy arrays, namely frequency `f_arr` (values along the Y axis), time `t_arr` (values along the X axis) and the actual spectrogra `sp_arr` as a 2D array
```python
from scipy import signal

f_arr, t_arr, sp_arr = signal.spectrogram(waveform_arr, freq)

sp_mono = torch.from_numpy(sp_arr)
```
Dimensions are `F x T`, where `F` is frequency and `T` is time.

Stack the two tensors along the first dimension to obtain a two channels image of size `C x F x T`, where `C` is the number channels:
```python
sp_t = torch.stack((sp_left_t, sp_right_t), dim=0)
```

## Video
When it comes to the shape of tensors, video data can be seen as equivalent to volumetric data, with `depth` replaced by the `time` dimension. The result is again a 5D tensor with shape `N x C x T x H x W`.
```python
import imageio

reader = imageio.get_reader('cockatoo.mp4')
meta = reader.get_meta_data()
n_channels = 3
n_frames = meta['nframes']
video = torch.empty(n_channels, n_frames, *meta['size'])
for i, frame_arr in enumerate(reader):
    frame = torch.from_numpy(frame_arr).float()
    video[:, i] = torch.transpose(frame, 0, 2)
```






