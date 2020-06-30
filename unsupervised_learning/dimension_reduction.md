# Dimension Reduction

## Clustering
- K-means
  - Clustering $X = \lbrace x^1,\cdots,x^n,\cdots,x^N \rbrace$ into K clusters
  - Initialize cluster center $c^i,i=1,2,\cdots,K$ (K random $x^n$ from $X$)
  - Repeat
    - For all $x^n$ in $X$ :   
      $b_i^n = \begin{cases}
        1, & x^n\quad is\quad most\quad "close"\quad to\quad c^i \\
        0, & Otherwise
        \end{cases}$
    - Updating all $c^i$ :  
      $c^i = \dfrac{\sum_{x^n}b_i^nx^n}{\sum_{x^n}b_i^n}$  

- Hierarchical Agglomerative Clustering(HAC)
  - Step 1 : build a tree according to their degree of similarity
  - Step 2 : pick a threshold to cluster node

## Clustering v.s. Distributed representation
- Clustering : an object must belong to one cluster
- Distributed representation : a probability distribution representation

## Distributed Representation
$$Highdimvectorx \rightarrow function \rightarrow Lowdimvectorz$$
- Feature selection  
  discard certain dim of feature to reduce dimesion
- Principle component analysis(PCA)

## Principle component analysis(PCA)
$$z = Wx$$  
*Reduce to 1-D* :  
$$z_1 = w^1 \times x$$  
Project all the data points $x$ onto $w^1$, and obtain a set of $z_1$. We want the variance of $z_1$ as large as possible.

$$Var(z_1) = \frac{1}{N}\sum_{z_1}(z_1 - \overline{z_1})^2$$  
  
Constaint : $\left\|w^1\right\|_2=1$  

If want to *reduce to 2-D*, just add another $z_2$ :
$$z_2 = w^2 \times x$$  
We also want the variance of $z_2$ as large as possible

$$Var(z_2) = \frac{1}{N}\sum_{z_2}(z_2 - \overline{z_2})^2$$  
  
Contraint : $\left\|w^2\right\|_2=1$ and $w^1\times w^2 = 0$  

And then concatenate $z_2$ with $z_1$ as Z.

Repeate above operation to attain any dimension we want to reduce to. Finally, concatenate all $w$ as $W$ : 
$$W = \begin{bmatrix} (w^1)^T\\(w^2)^T \\ \vdots \end{bmatrix}$$
W is a Orthogonal matrix

## Math part -- Lagrange multiplier
> $a,b$ are vectors
> $$(a \cdot b)^2 = (a^Tb)^2=a^Tba^Tb=a^Tb(a^Tb)^T=a^Tbb^Ta$$


**Matrix W Calculation of PCA**

$$z_1 = w^1 \cdot x$$
$$\overline{z_1}=\frac{1}{N}\sum z_1=\frac{1}{N}\sum w^1 \cdot x = w^1 \cdot \frac{1}{N}\sum x=w^1\cdot \overline{x}$$
$$Var(z_1)=\frac{1}{N}\sum_{z^1}(z_1-\overline{z_1})^2=\frac{1}{N}\sum (w^1\cdot x - w^1\cdot \overline{x})^2$$
$$=\frac{1}{N}\sum (w^1\cdot (x - \overline{x}))^2=\frac{1}{N}\sum (w^1)^T(x-\overline{x})(x-\overline{x})^Tw^1$$
$$=(w^1)^T\frac{1}{N}\sum(x-\overline{x})(x-\overline{x})^Tw^1=(w^1)^TCov(x)w^1$$  
$S=Cov(x)$ is symmetric and positive-semidefinite(non-negative eigenvalues) 

Goal : Find $w^1$ maximizing $(w^1)^T\cdot S\cdot (w^1)$, and constraint $\left\|w^1\right\|_2=(w^1)^Tw^1=1$

Solution : Using Lagrange multipier
$$g(w^1)=(w^1)^TSw^1-\alpha((w^1)^Tw^1-1)$$

Differentiate with respect to every element of $w^1$,$w^1$ is a vector, and let the gradient to be zero :  
$$\dfrac{\partial g(w^1)}{\partial w_1^1}=0$$  
$$\dfrac{\partial g(w^1)}{\partial w_1^2=0}=0$$  
$$\qquad \vdots$$  

After above calculation,We attain :  
$$Sw^1-\alpha w^1=0$$ 
$$Sw^1 = \alpha w^1$$  
$$(w^1)^TSw^1 = \alpha(w^1)^Tw^1=\alpha$$  
$w^1$ : eigenvector  
$\alpha$ is to be the maximun one

Conclusioin : $w^1$ is the eigenvector of the covariance matrix $S$ correpondind to the largest eigenvalue $\lambda_1$ (above $\alpha$)

Find $w^2$ maximizing $(w^2)^TSw^2$, and constraint $\left\|w^2\right\|_2 = 1,(w^2)^Tw^2=1 , (w^2)^Tw^1=0$
$$g(w^2)=(w^2)^TSw^2 - \alpha((w^2)^Tw^2-1) - \beta((w^2)^Tw^1 - 0)$$

Differentiate with respect to every element of $w^2$,$w^2$ is a vector, and let the gradient to be zero :  
$$\dfrac{\partial g(w^2)}{\partial w_1^2}=0$$  
$$\dfrac{\partial g(w^2)}{\partial w_1^2=0}=0$$  
$$\qquad \vdots$$

After above calculation,We attain :  
$$Sw^2 - \alpha w^2 - \beta w^1 = 0$$  

Multiply both sides of above equation by $w^1$:  
$$(w^1)^TSw^2 - \alpha(w^1)^Tw^2 - \beta(w^1)^Tw^1 = 0$$  

$\because (w^2)^Tw^2=1 , (w^2)^Tw^1=0$ and $S$ is symmetric. $Sw^1=\lambda_1w^1$.So we attain:

$$=(w^1)^TSw^2 - \beta =((w^1)^TSw^2)^T - \beta = (w^2)^TS^Tw^1$$  
$$=(w^2)^TSw^1 - \beta = \lambda_1(w^2)^Tw^1 - \beta = 0- \beta = 0$$  

$\beta = 0$ : $Sw^2 - \alpha w^2 =0 \rightarrow Sw^2=\alpha^2$

Conclusion : $w^2$ is the eigenvector of the covariance matrix $S$ corresponding to the $2^{nd}$ largest eigenvalue $\lambda_2$

## PCA - Decorrelation
$$z = Wx$$
Covariance of $z$ is a diagonal matrix.
$$Cov(z) = D$$
After PCA, it is equivalent to do decorrelation, which makes covariance of different dimension to become zero. So there is not correlation in the feature of different dimension after PCA and then your parameters reduce. It is also benefit to reduce overfitting.

$$Cov(z)=\frac{1}{N}\sum (z-\overline{z})(z-\overline{z})^T=WSW^T$$
$$=WS[w^1 \cdots w^k] = W[Sw^1 \cdots Sw^k]$$
$$= W[\lambda_1w^1 \cdots \lambda_kw^k] = [\lambda_1Ww^1 \cdots \lambda_kWw^k]$$
$$= [\lambda_1e_1 \cdots \lambda_ke_k] = D$$

## PCA-Another Point of View
Assuming we have some basic components $u^1,u^2,\cdots,u^k$, we can use a vector $[c_1,c_2,\cdots,c_k]$ with basic components to represent a digit image. $x$ is pixels in a digit image.
$$x \approx c_1u^1 + c_2u^2 + \cdots + c_ku^k + \overline{x}$$
$$x - \overline{x} \approx c_1u^1 + c_2u^2 + \cdots + c_ku^k = \hat{x}$$

Reconstruction error: $\left\|(x - \overline{x}) - \hat{x}\right\|_2$ 

Goal : Find $u^1,\cdots,u^k$ minimizing the error.

$$L = \underset{\lbrace u^1,\cdots,u^k \rbrace}{\min}\sum \left\| (x - \overline{x}) - \left(\sum_{k=1}^kc_ku^k\right) \right\|_2$$

we know that PCA : $z = Wx$

$$\begin{bmatrix} z_1\\z_2 \\ \vdots\\z_k \end{bmatrix} = \begin{bmatrix} (w^1)^T\\(w^2)^T \\ \vdots \\(w_k)^k\end{bmatrix}x$$

Actually, $\lbrace w^1,w^2,\cdots,w^k\rbrace$ (from PCA) is the component $\lbrace u^1,u^2,\cdots,u^k\rbrace$