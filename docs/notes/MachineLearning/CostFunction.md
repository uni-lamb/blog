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

假设数据集大小为 $m$ ，平方代价函数如下。我们的目标就是不断优化参数使 $J$ 最小

$$
J(vb(theta))=1/(2m) sum_(i=1)^m (h_theta (vb(x_i))-y_i)^2
$$

### 多项式回归

本质是当特征拟合不线性时，对输入特征进行非线性变换升维 $Phi: RR^n arrow RR^(k times n), k$ 为阶数

虽然对原特征不再线性，但不妨碍对参数的线性性，所以保持了凸性

### BGD

机器学习最基础的方法是批量梯度下降 (batch gradient descent)，此处batch指训练时每次都用上了所有数据

$$
&"Repeat until Convergence"{\
&quad vb(theta) = vb(theta)-alpha grad J\
&}
$$

其中 $alpha$ 称为学习率，一般取 $alpha=0.01,0.03,0.1,0.3,1,3,10$

#### 特征缩放

由于各个特征数据的尺度不同，很有可能等值线图像很扁。这时候梯度下降会难以收敛。

一般的特征缩放方法为： $x_n arrow.l frac(x_n-mu_n,s_n),mu_n "为平均值" ,s_n "为标准差"$

### Normal Equation

或者，当 $n<10000$ 时，使用正则方程法的计算开销足以cover，可以一步到位

$$
"let" vb(X)=mat(
    x_0^((0)),dots,x_n^((0));
    dots.v,dots.down,dots.v;
    x_0^((n)),dots,x_n^((n))
),
"then" vb(theta)=vb((X^T X)^(-1)X^T y)=^Delta X^+ y
$$

此处 $X^+$ 称为Moore-Penrose伪逆（Pseudo Inverse），当矩阵 $X$ 满秩时等价于逆，非满秩时使用奇异值分解（SVD）计算：

$$
X=  U Sigma V^T,X^+=V Sigma^(-1) U^T
$$

#### 证明正则方程法

$$
J(vb(theta))&=frac(1,2m) vb((X theta-Y)^T (X theta-Y))=frac(1,2m) vb((theta^T X^T -Y^T)(X theta-Y))\
&=frac(1,2m)vb((theta^T X^T X theta+Y^T Y-Y^T X theta-theta^T X^T Y))
$$

本质就是要做一个矩阵对向量的偏导， $pdv(,theta)J=0$

此处使用引理 $pdv(,theta)(theta^T A theta)=(A^T+A)theta$

$$
&pdv(J,theta)=frac(1,2m)vb((2X^T X theta-2X^T Y))=frac(1,m)vb((X^T X theta-X^T Y))=0\
<=>&vb(X^T X theta=X^T Y),vb(theta=(X^T X)^(-1)X^T Y)
$$

## 感知机

线性回归处理的是连续数据，而对于广泛的离散数据归类问题就不大适配。不过作为基础，感知机（线性分类器）还是值得一提。

假设数据集被分为两类 $y in {0,1}$ , 且我们希望使用一个超平面 $vb(w^T x) + b = 0$ 分割两类数据点。即 $forall y_i=-1,1$ , $vb(w^T x_i) + b >= 0$ 表示 $y_i=1$ , $vb(w^T x_i) + b < 0$ 表示 $y_i=-1$ 。能这样分割的数据集称为线性可分（linearly separable）。

注意到 $-y_i (vb(w^T x)+b)>0$ 恒成立，取代价函数具有点到超平面距离的形式：

$$
L(vb(w),b)= sum_i 1/(||w||) abs(vb(w^T x_i)+b) = sum_i 1/(||w||) -y_i (vb(w^T x_i)+b)\
J(vb(w),b)=sum_i -y_i (vb(w^T x_i)+b)
$$

### PLA

感知机算法（Perceptron Learning Algorithm）不同于一般的BGD，而是接近于一种特殊的batch=1的 SGD。准确来说，PLA不支持梯度下降，因为阶跃函数不可导，无法视为连续参量问题。其特殊性在于，只有当分类错误时才更新参数。

1. 随机初始化参数 $vb(w_0),b_0$ ,一般来说是 $vb(0),0$
2. 在一个epoch内随机/顺序遍历所有的样本，对于每一个样本 $vb(x_i),y_i$ ：
   1. 若分类正确 $y_i (vb(w^T x_i)+b)>0$ 则跳过
   2. 否则，更新参数：$w_k=w_(k-1)+eta y_i x_i,b_k=b_(k-1)+eta y_i, k$ 为迭代次数
3. 重启一个epoch，直到收敛（一整次遍历无误分类）

### Novikoff定理

做一个简化，令 $vu(w_i)=(vb(w_i^T),b)^T,vu(x_i)=(vb(x_i),1)$

!!! note "Novikoff定理"

    - 假设数据集线性可分，存在超平面 $vu(w_i dot x_i)=0,||vu(w#sub(opt))||=1$ 分割数据集，且存在 $gamma>0,forall i,y_i vu(w_i dot x_i)>=gamma$
    - 令 $R=max_i ||x_i||$ ，感知机算法在数据集上的误分类次数 $k<=(R^2)/(gamma^2)$

证明：

(1) 由于数据集有限，直接取 $gamma=min_i (vb(w_i dot x_i)+b)>=0$ 即可

(2) PLA算法给出 $vu(w_k)=vu(w_(k-1))+eta y_i vu(x_i)$

- 由于 $vu(w_k dot w#sub(opt))=vu(w_(k-1) dot w#sub(opt)) + eta y_i vu(x_i dot w#sub(opt))>=vu(w_(k-1)dot w#sub(opt))+eta gamma$，所以有

$$
vu(w_k dot w#sub(opt))>=vu(w_0 dot w#sub(opt))+k eta gamma = k eta gamma
$$

- 范数 $||w_k||^2=||w_(k-1)||^2+2 eta y_i vu(w_(k-1)dot x_i)+eta^2 ||x_i||^2<=||w_(k-1)||^2+eta^2 ||x_i||^2<=||w_(k-1)||^2+eta^2 R^2$, 所以 $||w_k||^2<=k eta^2 R^2$

以上两不等式又可得到 $k eta gamma <=vu(w_k dot w#sub(opt))<=sqrt(k eta^2 R^2)$ ，即 $k<=(R^2)/(gamma^2)$ 

Novigoff定理说明，误差迭代次数是有上限的，在有限轮迭代后一定能收敛。

## 逻辑斯谛回归

显然用直线切割类适用面很窄，因此我们引入Sigmoid作为逻辑函数。常用的这个被称为对数几率函数 $g(z)=1\/(1+e^(-z))$

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
"指数族形如" f(y;theta)=h(y)exp(eta(theta)T(y)-A(theta)),"特别的，均值" E(T(y))=pdv(A,eta),"方差" V\ar(T(y))=pdv(A,eta,2)
$$

下面我们分别对Gauss分布和Bernoulli分布进行探讨，后者正是逻辑回归的基础：

#### Gauss分布

GLM中简化处理Gauss分布的方差 $sigma^2$ 是常量

$$
f(y;mu)=frac(1,sqrt(2pi sigma^2))exp(-frac((y-mu)^2,2sigma^2))=exp(-frac(y^2,2sigma^2)-1/2 log(2pi sigma^2))dot exp(frac(mu,sigma^2)y-frac(mu^2,2sigma^2))\
eta(mu)=frac(mu,sigma^2),T(y)=y,E(y)=pdv(,eta)(1/2 eta^2 sigma^2)=mu,V\ar(y)=pdv(,eta,2)(1/2 eta^2 sigma^2)=sigma^2
$$

#### Bernoulli分布

$$
f(y;pi)=pi^y (1-pi)^(1-y)=exp(y log(pi/(1-pi))+log(1−pi))\
eta(pi)=log(pi/(1-pi)),T(y)=y,E(y)=pdv(,eta)(-log(1-pi))=pi,V\ar(y)=pdv(,eta,2)(-log(1-pi))=pi(1-pi)
$$

特别注意到此处链接函数 $g(t)=log frac(t,1-t)$ 其逆就是上面的对数几率函数

### 逻辑回归梯度下降

若继续使用最小二乘法形式的代价，那将是非凸的，极易停在局部最值。

重新定义代价函数为独立统计的联合分布概率的负对数（cost最小就是似然概率最大）

$$
J(vb(theta))=-1/m sum_(i=1)^m (y_i log h_theta (vb-(x_i))+(1-y_i)log (1- h_theta (vb(x_i))))\
grad J=1/m sum_(i=1)^m (h_theta (vb(x_i))-y_i)vb(x_i)
$$

### 一对多问题

接下来将简单的0/1归类问题扩展到任意多种类的离散归类问题。

假设能被分为 $n$ 类，朴素的想法就是分别以一个类为正类，其他为负类建立 $n$ 个分类器，最终的判断以给出概率最高的分类器为准。

延申方向包括无监督学习的归类（多次K-means，甚至不需指定类），softmax回归等等

## 特征工程引论

如何选取合适的特征，是炼丹的重要话题。普遍而言有三种特征选择方式，即Filter,Wrapper,Embedded。

- Filter:先验的处理，在炼丹前直接筛去低价值特征进行降维。简单常用。
- Wrapper：后验的反馈，遍历特征集的子集炼丹来找到最佳特征组合。计算开销巨大。
- Embedded：结合两者，在炼丹的同时进行特征选择。最常见的就是正则化惩罚。

### 正则化

如果炼丹过拟合了，首先可以进行Filter降维（如[LaplaceScore](https://papers.nips.cc/paper_files/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html)），但这不是任何时候都实用的。更多的时候，过拟合是因为多项式拟合的阶数过高，拟合性有余，预测性不足。

一个自然的想法是，降低高阶项参数的权重。具体来说就是在cost函数中附加想降低的参数项的绝对正项（如L1正则化为绝对值，L2正则化为平方）。如果不确定哪些参数需要被抑制，就直接全部正则化（按惯例忽略 $theta_0$）

$$
J(vb(theta))=1/(2m) [sum_(i=1)^m (h_theta (vb(x_i))-y_i)^2 + lambda sum_(j=1)^n theta_j^2 ]
$$

在梯度下降的意义上，这相当于在步长里添加了一项portion项，是简易的pid思想啊（大雾）

延申发散并添加integral和derivative，就会得到Adam优化器。不过这是后文了。

另外也可以使用正规方程解出最优参数
$$
vb(theta)=vb((X^T X+lambda underbrace(dmat(0,1,dots.down,1),(n+1)*(n+1)))^(-1) X^T y)
$$

### 正则化的解空间

以L2惩罚函数为例，我们有以下断言：

!!! note "断言"

    含惩罚函数的无约束优化问题，与一个无惩罚函数的有约束优化问题等价。（依赖KKT条件）

$$
min_theta J(theta) = J_0 (theta) + lambda dot ||theta||^2 <=> min_theta J_0 (theta),"subject to" ||theta||^2 <=C
$$

### KKT条件

KKT条件将等式极值的拉格朗日乘数法推广到不等式 $min f(vb(x)),"subject to"g(vb(x))<=0$ 。由此定义“可行域” $Set(vb(x)in RR,g(vb(x))<=0)$

- $g(vb(x^*))<0$ 内部解，约束条件inactive，退化为无约束优化问题。$grad f=0,lambda=0$

- $g(vb(x^*))=0$ 边界解，约束条件active，转换为一般拉格朗日问题

$grad f=-lambda grad g$ 由于要求 $f$ 最小化（指向内部），且 $g$ 指向外部，所以 $lambda>=0$

结合合两种情况，发现最优解具有互补松弛性 $lambda g(vb(x^*))=0$ ，

综上，扩展到多个约束情形给出KKT条件：$min f(vb(x)),"subject to"g_i (vb(x))=0;h_j (vb(x))<=0$

!!! note "KKT"

  - 定常方程式 $vb(nabla_x) L = grad (f + sum_i (lambda_i g_i) +sum_j (mu_j h_j)) =0$
  - 原始可行性 $g_i (vb(x))=0,h_j (vb(x))<=0$
  - 对偶可行性 $mu_j>=0$
  - 互补松弛性 $mu_j h_j (vb(x^*))=0$

回过头来看，两个定常方程并无差别。也就是说正则惩罚是一个“软约束”，和硬约束效果类似。

$$
vb(nabla_theta)L_1=vb(nabla_theta)J_0+2lambda theta=0\
vb(nabla_theta)L_2=vb(nabla_theta)J_0+2mu theta=0
$$
