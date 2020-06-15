# Markdown Math Syntax

# Introduction

# Math数学公式

## 1.行间插入
> \\(a+b\\)
## 2.另取一行
>$a+b$
> $$a + b$$
## 3.基本类型的插入
### 3.1 上下标
> $$x_1$$
> $$x_1^2$$
> $$x^2_1$$
> $$x_{22}^{(n)}$$
> $$x_22^(n)$$
> $${}^*x^*$$
> $$x_{balabala}^{bala}$$  
可以看到 x 元素的上标通过 ^ 符号后接的内容体现，下表通过 _ 符号后接的内容体现，多于一位是要加 {} 包裹的

### 3.2 分式
> $$\frac{x+y}{2}$$
> $$\frac{1}{1+\frac{1}{2}}$$  
这里就出现了一个 `frac{}{} `函数的东西，同样，为了区分这是函数不是几个字母，通过` \frac `转义，于是 `frac` 被解析成函数，然后第一个 {} 里面的被解析成分子，第二个 {} 被解析成分母

### 3.3 根式
> $$\sqrt{2}<\sqrt[3]{3}$$
> $$\sqrt{1+\sqrt[n]{1+a^2}}$$
> $$\sqrt{1+\sqrt[^n\!]{1+a^2}}$$  
语法就是 `sqrt[]{}` 。[] 中代表是几次根式，{} 代表根号下的表达式。第二和第三个的区别在于为了美观微调位置

### 3.4 空格
> 紧贴 $a\!b$  
> 没有空格 $ab$  
> 小空格 $a\,b$  
> 中等空格 $a\;b$  
> 大空格 $a\ b$  
> quad空格 $a\quad b$  
> 两个quad空格 $a\qquad b$

### 3.5 求和、积分
> $$\sum_{k=1}^{n}\frac{1}{k}$$
> $\sum_{k=1}^{n}\frac{1}{k}$
> $$\int_a^b f(x)dx$$
> $\int_a^b f(x)dx$  
>$$\int_a^b f(x)\,\mathrm{d}x$$
>$$\int_a^b f(x)\,\mathrm{d}x\Bigg|_{x = x_0}$$

求和函数表达式 **sum_{起点}^{终点}** 表达式，积分函数表达式 **int_下限^上限 被积函数d被积量**

### 3.6 公式界定符号
主要符号有 ( ) [ ] \{ \} | || ，那么如何使用呢？ 通过 \left 和 \right 后面跟界定符来对同时进行界定
> $$\left( \sum_{k=\frac{1}{2}}^{n^2\frac{1}{k}} \right)$$

### 3.7 矩阵
> $$\begin{matrix} 1&2\\\\3&4 \end{matrix}$$
> $$\begin{pmatrix} 1&2\\\\3&4 \end{pmatrix}$$
> $$\begin{bmatrix} 1&2\\\\3&4 \end{bmatrix}$$
> $$\begin{Bmatrix} 1&2\\\\3&4 \end{Bmatrix}$$
> $$\begin{vmatrix} 1&2\\\\3&4 \end{vmatrix}$$
> $$\left|\begin{matrix} 1&2\\\\3&4 \end{matrix}\right|$$
> $$\begin{Vmatrix} 1&2\\\\3&4 \end{Vmatrix}$$

### 3.8 排版数组
$$\mathbf{x} = 
\left( \begin{matrix}
x_{11} & x_{12} & \ldots \\\\
x_{21} & x_{22} & \ldots \\\\
\vdots & \vdots & \ddots \\\\
\end{matrix} \right)$$

## 4.常用公式举例
### 4.1 分段函数
$$ 
y=\begin{cases}
-x,\quad x\leq 0 \\\\
x,\quad x>0
\end{cases}
$$
使用 cases 块表达式，每行 \\结尾，每个元素 & 分隔。
$$
p(x) = 
\begin{cases}
  p, & x = 1 \\
  1 - p, & x = 0
\end{cases}
$$


## 5.常用希腊字幕
$$
\begin{array}{|c|c|c|c|c|c|c|c|}
\hline
{\alpha} & {\backslash alpha} & {\theta} & {\backslash theta} & {o} & {o} & {\upsilon} & {\backslash upsilon} \\\\
\hline
{\beta} & {\backslash beta} & {\vartheta} & {\backslash vartheta} & {\pi} & {\backslash pi} & {\phi} & {\backslash phi} \\\\
\hline
{\gamma} & {\backslash gamma} & {\iota} & {\backslash iota} & {\varpi} & {\backslash varpi} & {\varphi} & {\backslash varphi} \\\\
\hline
{\delta} & {\backslash delta} & {\kappa} & {\backslash kappa} & {\rho} & {\backslash rho} & {\chi} & {\backslash chi} \\\\
\hline
{\epsilon} & {\backslash epsilon} & {\lambda} & {\backslash lambda} & {\varrho} & {\backslash varrho} & {\psi} & {\backslash psi} \\\\
\hline
{\varepsilon} & {\backslash varepsilon} & {\mu} & {\backslash mu} & {\sigma} & {\backslash sigma} & {\omega} & {\backslash omega} \\\\
\hline
{\zeta} & {\backslash zeta} & {\nu} & {\backslash nu} & {\varsigma} & {\backslash varsigma} & {\partial} & {\backslash partial} \\\\
\hline
{\eta} & {\backslash eta} & {\xi} & {\backslash xi} & {\tau} & {\backslash tau} & {\Sigma} & {\backslash Sigma} \\\\
\hline
{\Gamma} & {\backslash Gamma} & {\Lambda} & {\backslash Lambda} & {\Sigma} & {\backslash Sigma} & {\Psi} & {\backslash Psi} \\\\
\hline
{\Delta} & {\backslash Delta} & {\Xi} & {\backslash Xi} & {\Upsilon} & {\backslash Upsilon} & {\Omega} & {\backslash Omega} \\\\
\hline
{\Omega} & {\backslash Omega} & {\Pi} & {\backslash Pi} & {\Phi} & {\backslash Phi} & {\infty } & {\backslash Infty} \\\\
\hline 
{\mathcal{N}} & {\backslash mathcal} & {} & {} & {} & {} & {} & {} \\\\
\hline 
\end{array}
$$

## 6.特殊符号
> $$\vec{a}$$
> $$\overline{a}$$
> $$\hat{a}$$
> $$\widetilde{a}$$ 
> $$\dot{a}$$
> $$\ddot{a}$$
> $$\log_{2}(a+b)$$
> $$\lim_{n\rightarrow0}n$$
> $$\prod_{i=0}^n\frac{1}{x^2}$$
> $$x^{\prime}$$
> $$x^{\prime\prime}$$
> $$\approx$$
> $$\dfrac{\mathrm{d}x}{\mathrm{d}y}$$
> $$\left\|\theta\right\|_2$$
> $$\sim$$
> $$\lfloor x \rfloor $$
> $$\lceil x \rceil$$
> $$\overline{x}$$
> $$\lbrace x \rbrace$$
> $$\lbrack x \rbrack$$
> $$\gg \\ \ll$$
> $$\underleftrightarrow{close}$$
> $$\overset{\lambda}{\longrightarrow}$$

## 7.Dots
|type|shape|
|--|--|
|vdots|$\vdots$|
|cdots|$\cdots$|

## 8.Square grid
$$\begin{array}{|c|c|c|c|}
\hline
 {} & {} & {} & {} \\
\hline
 {} & {} & {} & {} \\
\hline
 {} & {} & {} & {} \\
\hline
 {} & {} & {} & {} \\
\hline 
\end{array}$$

9.数学运算符号
>> $$\odot$$
