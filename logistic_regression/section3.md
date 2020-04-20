# Section3

**Cross Entropy** : How close is the two distribution? If they are same,the result is 0.  
Distribution p:  $p(x=1) = \hat{y}^n$, $p(x=0) = 1 - \hat{y}^n$  
Distribution q:  $q(x=1) = f(x^n)$, $q(x=0) = 1 - f(x^n)$  
Cross Entropy between two Bernoulli Distribution.  
$H(p,q) = -\sum_xp(x)ln(q(x)) = -\sum_x[\hat{y}^nlnf(x^n)+(1 - \hat{y}^n)ln(1 - f(x^n))]$ 