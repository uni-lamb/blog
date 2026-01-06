---
math: typst
---

# 数学基础

主要来自《概率导论》，《机器学习：概率视角》三部曲，《统计学习方法》等书籍。

## 矩母函数

为描述概率分布围绕某个点的分布情况，我们引入矩Moment。常用的有原点矩和中心矩。

例如，一阶原点矩 $E(X)=mu$ 描述均值，二阶中心矩 $E((X-mu)^2)=sigma^2$ 描述方差，三阶中心矩 $E((X-mu)^3)$ 描述偏度，四阶中心矩 $E((X-mu)^4)$ 描述峰度。

为了一次性概括所有阶的矩特征，我们想到泰勒展开的形式，引入矩母函数：

$
    M_X(t) = E[e^(t X)] = integral_( - infinity )^( + infinity ) e^(t x) f_X(x) d x\
    = sum_(n=0)^( + infinity ) frac(t^n)(n!) E(X^n)
$

那么通过求导的方式就可以获得所有中心矩。如果希望求取中心矩，可以使用累积矩母函数

$
    K_X(t) = ln(M_X(t)) = sum_(n=1)^( + infinity ) frac(t^n,n!) k_n，quad k_n = E((X - mu)^n)
$

## 正态分布

中心极限定理说明，在一定条件下，大量独立同分布的随机变量的和趋向于正态分布。这使得正态分布在今后的学习中具有重要地位。

> 设 $X_1, X_2, dots, X_n$ 为独立同分布的随机变量，且均值为 $mu$，方差为 $sigma^2$，则当 $n$ 足够大时，随机变量 $Z_n = frac(X_1 + X_2 + dots + X_n - n mu,sqrt(n) sigma)$ 近似服从标准正态分布 $N(0, 1)$。

为了证明中心极限定理，我们需要用到特征函数，即概率密度函数的傅里叶变换。（和矩母函数很相似）

$
    phi_X (t) = E[e^(i t X)] = integral_( - infty )^( + infty ) e^(i t x) f_X(x) d x
$

这里取 $Y_k=X_k-mu,E(Y_K)=0,V\ar(Y_k)=sigma^2$ 考虑独立同分布的特征函数均为 $phi_Y(t)$ ，由卷积定理得

$
    phi_Z (t)=Pi_k phi_(Y_k) (t/(sigma sqrt(n)))= (phi_Y (t/(sigma sqrt(n))))^n
$

将特征函数展开至二阶，利用与矩母函数的相似性： $phi(0)=1,phi'(0)=0,phi''(0)=-sigma^2$

$
    &phi_Y (t/(sigma sqrt(n))) = 1 + 0 - frac(t^2,2n) + o(t^2)\
    &phi_Z= (1 - frac(t^2,2n) + o(t^2))^n -> e^(-t^2/2)~N(0,1)
$

## 多元正态分布及其导出

推广到多维特征，正态分布写作：

$
    p_(vb(X))(x)=frac(1,2pi Sigma^(1/2)) exp(-1/2 vb((x-mu))^T Sigma^(-1) vb((x-mu)))
$

## 指数族模型
