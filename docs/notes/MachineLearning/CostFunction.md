---
math: typst
---

# 代价函数

## 基础

从基础的多变量线性回归开始回顾。

对于具有 $n$ 个特征变量 $x_0=1,x_1,dots,x_n$ 的数据集，其线性的启发式函数（Heuristic Function）表达为

$$
h_theta (vb(x))=theta_0 x_0+theta_1 x_1+ dots +theta_n x_n=vb(theta)^T vb(x)
$$

假设数据集大小为 $m$ ，代价函数如下。我们的目标就是不断优化参数使 $J$ 最小

$$
J(vb(theta))=1/(2m) sum_(i=1)^m (h_theta (vb(x_i))-y_i)^2
$$

### BGD

机器学习最基础的方法是批量梯度下降 (batch gradient descent)，此处batch指训练时每次都用上了所有数据

$$
"Repeat until Convergence"{\
    vb(theta) arrow vb(theta)-alpha grad J\
}
$$

其中 $alpha$ 称为学习率，一般取 $alpha=0.01,0.03,0.1,0.3,1,3,10$

#### 特征缩放

由于各个特征数据的尺度不同，很有可能等值线图像很扁。这时候梯度下降会难以收敛。

一般的特征缩放方法为： $x_n arrow frac(x_n-mu_n,s_n),mu_n "为平均值" ,s_n "为标准差"$

### Normal Equation

或者，当 $n<10000$ 时，使用正则方程法的计算开销足以cover，可以一步到位

$$
"let" vb(X)=mat(
    x_0^((0)),dots,x_n^((0));
    dots.v,dots.,dots.v;
    x_0^((n)),dots,x_n^((n))
),
"最优的参数为" vb(theta)=vb((X^T X)^(-1)X^T y)
$$

#### 证明正则方程法

$$
J(vb(theta))&=frac(1,2m) vb((X theta-Y)^T (X theta-Y))=frac(1,2m) vb((theta^T X^T -Y^T)(X theta-Y))\
&=frac(1,2m)vb(theta^T X^T X theta+Y^T Y-Y^T X theta-theta^T X^T Y)
$$

本质就是要做一个矩阵对向量的偏导， $pdv(,theta)J=0$

此处使用引理 $pdv(,theta)(theta^T A theta)=(A^T+A)theta$

$$
&pdv(J,theta)=frac(1,2m)vb((2X^T X theta-2X^T Y))=frac(1,m)vb(X^T X theta-X^T Y)=0\
<=>&X^T X theta=X^T Y,theta=(X^T X)^(-1)X^T Y
$$

## 逻辑回归

线性回归处理的是连续数据，而对于广泛的离散数据归类问题就不大适配。

为了继续使用线性参数做特征处理，我们引入Sigmoid作为逻辑函数。常用的这个被称为对数几率函数 $g(z)=frac(1,1+e^(-z))$

使用对数几率函数有许多良好性质：

- 将线性函数 $h_theta (vb(x))=vb(theta^T x)$ 限制在 $[0,1]$ 上
- 在 $g:RR arrow [0,1]$ 的映射中唯一保持线性可解释性，与概率联系紧密
- 符合广义线性模型理论（General Linear Model）

### GLM理论

广义线性模型通过三个核心组成部分来描述响应变量与预测变量之间的关系：随机成分、系统成分和链接函数。

1. 随机成分：响应变量 $Y$ 的概率分布，GLM理论要求该分布必须是指数族型
2. 系统成分：预测变量 $vb(x)$ 的概率组合 $eta_i=vb(theta^T x_i)$ ，其中 $eta_i$ 称为线性预测器
3. 链接函数： $g:g(mu_i)=eta_i$,将均值与线性预测器相连接的映射

$$
"指数族形如"f(y;theta)=h(y)exp(eta(theta)T(y)-A(theta)),"特别的，均值"E(T(y))=pdv(A,eta),"方差"Var(T(y))=pdv(A,eta,2)
$$

下面我们分别对Gauss分布和Bernoulli分布进行探讨，后者正是逻辑回归的基础：

#### Gauss分布

GLM中简化处理Gauss分布的方差 $sigma^2$ 是常量

$$
f(y;mu)=frac(1,sqrt(2pi sigma^2))exp(-frac((y-mu)^2,2sigma^2))=exp(-frac(y^2,2sigma^2)-1/2 log(2pi sigma^2))dot exp(frac(mu,sigma^2)y-frac(mu^2,2sigma^2))\
eta(mu)=frac(mu,sigma^2),T(y)=y,E(y)=pdv(,eta)1/2 eta^2 sigma^2=mu,Var(y)=pdv(,eta,2)1/2 eta^2 sigma^2=sigma^2
$$

#### Bernoulli分布

$$
f(y;pi)=pi^y (1-pi)^(1-y)=exp(y log(pi/(1-pi))+log(1−pi))\
eta(pi)=log(pi/(1-pi)),T(y)=y,E(y)=pdv(,eta)(-log(1-pi))=pi,Var(y)=pdv(,eta,2)(-log(1-pi))=pi(1-pi)
$$

特别注意到此处链接函数 $g(t)=log frac(t,1-t)$ 其逆就是上面的对数几率函数

### 逻辑回归梯度下降

## 一对多问题

## 特征工程

如何选取合适的特征，是炼丹的重要话题。普遍而言有三种特征选择方式，即Filter,Wrapper,Embedded。

- Filter:先验的处理，在炼丹前直接筛去低价值特征进行降维。简单常用。
- Wrapper：后验的反馈，遍历特征集的子集炼丹来找到最佳特征组合。计算开销巨大。
- Embedded：结合两者，在炼丹的同时进行特征选择。最常见的就是正则化惩罚。

### 正则化


