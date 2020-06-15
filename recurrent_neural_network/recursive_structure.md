# Recursive Structure

Recurrent newtorks is a special case of recursive structure. Recursive network are general structure.  
Recursive networks consider the relationship between input features. 

Basic idea : given four input sampls $x^1,x^2,x^3,x^4$, devide them into two groups such as $x^1,x^2$ and $x^3,x^4$ according to the task. Let each group passes through hidden layer function to output $h^1,h^2$. Next, stack the network structure and let $h^1,h^2$ pass through the same hidden layer function to attain $h^3$ as last hidden output unlike the order of recurrent networks.  
For example, in sentiment analysis task, we have sample vector $V("not"),V("very"),V("good")$. The syntactic structure is $V("very","good")$ and $V("not")$ by composing the $V("very")$ and $V("good")$ together firstly. How to stack function f of the recursive structure is already determined. And then consider $V("not","very\quad good")$ by same function f to output $V("not\quad very\quad good")$. At last, feed the output vector to another function g to generate sentiment. With training data, it is able to backpropage throughout the whole network to update parameter of function f and g in order to attain a good network.

Problem : general neural stucture,   
$$\sigma(w(x^1,x^2))$$
$x^1$ stacks with $x^2$, has little interaction between $x^1$ and $x^2$, but it is important in recusive neural network. So the function f should be a little trick.

## Recursive Neural Tensor Network
$$f = \sigma\left(\sum_{i,j}W_{i,j}x_i^1x_j^2 + W(x^1,x^2)\right)$$
This network structure introduce a structure $X^TWX$ to represent relationship between sample vectors. It's noted that these two parts should have same dimensions in order to add up.

## Matrix-Vector Recursive Network
It believes a word vector consist of two parts : inherent meaning and how it changes the others.

## Tree LSTM
replace function f with LSTM

## More Applications
- Sentence relatedness