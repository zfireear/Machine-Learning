# CNN Network Structure

## Padding 
An `n×n×３` image is convolued with a filter of `f×f×３`， the numbers in the convolution kernel are multiplied with the corresponding samples, and then sum them up, resulting in a `(n-f+1)(n-f+1)×１` output. In practice, there are often several filters being used. So it will turn out to `(n-f+1)(n-f+1)×#filters`.

**Shortcoming of Traditional Convolution**
- shrink output
- throw away information from edge
  
**Solution**
Padding with an additional border of one pixel all around the edges. By convention,padded with zeros. And if `p` is the padding amounts, because we are padding all around with an extra border of one pixels, here p=1. The the output becomes `(n+2p-f+1)(n+2p-f+1)`.

**Valid and Same Convolutions**
- *valid convolution* $\rightarrow$ no padding : `n×n * f×f` $\rightarrow$ `(n-f+1)(n-f+1)` 
- *same convolution* : padding, so the size of output is the same as the size of input : `(n+2p-f+1)(n+2p-f+1)` $\rightarrow$ `n×n`

**Tip** the `f` is usual odd. And $p = \dfrac{f-1}{2}$

## Strided Convolutions
Assuming we have `n×n` images, `f×f` filters, Padding `P` and strided `S`  
Output : 
$$\left\lfloor \dfrac{n+2p-f}{s} + 1 \right\rfloor \times \left\lfloor \dfrac{n+2p-f}{s} + 1 \right\rfloor$$
$\lfloor z \rfloor$ : The floor of z means taking z and rounding down to nearest integer (The largest integer smaller than itself)

## Convolutions over Volumes
RGB images and filters:
$$n_{height}×n_{width}×n_c * f×f×n_c \rightarrow （n_{height}-f+1）×(n_{width}-f+1)×n_c^\prime$$
**NOTE**  The number of channels $n_c$ has to be equal


**Summary of Notation**  
If layer L is a convolutional layer:
- $f^{[l]}$ : filter size as an `f` by `f` filter in the layer L. Each filter is $f^{[l]} \times f^{[l]} \times n_c^{[l]}$
- $p^{[l]}$ : padding
- $s^{[l]}$ : stride
- Input : $n_h^{[l-1]} \times n_w^{[l-1]} \times n_c^{[l-1]}$
- Output  : $n_h^{[l]} \times n_w^{[l]} \times n_c^{[l]}$
- dim of $n_{h/w}^{[l]}$ : $\left\lfloor \dfrac{n_{h/w}^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}} + 1 \right\rfloor$
- Activation :   
  $$a^{[l]} \rightarrow n_h^{[l]} \times n_w^{[l]} \times n_c^{[l]}$$  
  $$A^{[l]} \rightarrow m \times n_h^{[l]} \times n_w^{[l]} \times n_c^{[l]}$$
- Weight : $n_c^{[l]}$


## Single layer of Convolution NN
$$Z^{[l]} = w^{[l]}a^{[l-1]}+b^{[l]}$$
$$a^{[l]} = g(Z^{[l]})$$

**Types of Layer In a Convolutional Network**
- Convolutional(Conv)
- Pooling(Pool)
- Fully Connected(FC)

**Example of ConvNet**  
|Process|Layer 1|Layer 2|Layer 3|Vectorized|Logistic Function|output|
|--|--|--|--|--|--|--|
|filter $f^{[n]}$|3|5|5||||
|stride $s^{[n]}$|1|2|2||||
|padding $p^{[n]}$|0|0|0||||
|number of filter|10|20|40||||
|result(origin from 39×39×3)|(37×37×10)|(17×17×20)|(7×7×40)|7×7*40=1960|softmax|$\hat{y}$|

## Pooling layer : *Max or Average Pooling*  
Max Pooling
|1|3|2|1|                  
|--|--|--|--|
|2|9|1|1|
|1|3|2|2|
|5|6|1|2|

max pooling with `f=2,s=2`, output :
|9|2|
|--|--|
|6|3|

A large number means that it's maybe detected a paticular feature. What the max pooling does is so long as the feature is detected anywhere in those quadrants, it then remains preserved in the output of max pooling. Maybe this feature doesn't exist, then the max of all those numbers is still itself quite small.

**Summary of Pooling**  
Hyperparameters :
- f : filter size
- s : stride

(often `f=2,s=2` which reduces H and W by a factor of 2)

One thing to note about pooling is that there are no parameters to learn. In other words, there are no parameters that backprop will adapt to max pooling. Only set once. When people calculate how many layers there are in a neural network, they usually only count the layers with weights and parameters.

**Classifier of Activation Function**  
The *logistic classifier* is modeled on the Bernoulli distribution, which can be used to distinguish between two categories; the *softmax classifier* is modeled on a polynomial distribution, which can distinguish between multiple mutually exclusive categories.

## Convolutional Neural Network Example
|Process|Conv 1|Pool 1|Conv 2|Pool 2|Flatting|FC 3|FC 4|Activation Function|Output|
|--|--|--|--|--|--|--|--|--|--|
|filter $f^{[n]}$|5|2|5|2||||||
|stride $s^{[n]}$|1|2|1|2||||||
|number of filter|6||10|||||||
|weight $w^{[l]}$||||||(120,400)||||
|bias $b^{[l]}$||||||(120)||||
|result(origin from 32×32×3)|(28×28×6)|(14×14×6)|(10×10×16)|5×5×16|400×1 vector|120 vector|84|softmax output|$\hat{y}$(n output)|

