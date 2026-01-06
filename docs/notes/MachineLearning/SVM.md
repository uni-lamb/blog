---
math: typst
---
# 分类器：从感知机到支持向量机

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

    - 假设数据集线性可分，存在超平面 $vu(w_i dot x_i)=0,||vu(w#sub("opt"))||=1$ 分割数据集，且存在 $gamma>0,forall i,y_i vu(w_i dot x_i)>=gamma$
    - 令 $R=max_i ||x_i||$ ，感知机算法在数据集上的误分类次数 $k<=(R^2)/(gamma^2)$

证明：

(1) 由于数据集有限，直接取 $gamma=min_i (vb(w_i dot x_i)+b)>=0$ 即可

(2) PLA算法给出 $vu(w_k)=vu(w_(k-1))+eta y_i vu(x_i)$

- 由于 $vu(w_k dot w#sub("opt"))=vu(w_(k-1) dot w#sub("opt")) + eta y_i vu(x_i dot w#sub("opt"))>=vu(w_(k-1)dot w#sub("opt"))+eta gamma$，所以有

$$
vu(w_k dot w#sub("opt"))>=vu(w_0 dot w#sub("opt"))+k eta gamma = k eta gamma
$$

- 范数 $||w_k||^2=||w_(k-1)||^2+2 eta y_i vu(w_(k-1)dot x_i)+eta^2 ||x_i||^2<=||w_(k-1)||^2+eta^2 ||x_i||^2<=||w_(k-1)||^2+eta^2 R^2$, 所以 $||w_k||^2<=k eta^2 R^2$

以上两不等式又可得到 $k eta gamma <=vu(w_k dot w#sub("opt"))<=sqrt(k eta^2 R^2)$ ，即 $k<=(R^2)/(gamma^2)$ 

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
f(y;theta)=h(y)exp(eta(theta)T(y)-A(theta)), E(T(y))=pdv(A,eta), V\ar(T(y))=pdv(A,eta,2)
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

## 支持向量机

支撑向量机（SVM）算法在分类问题中有着重要地位，其主要思想是最大化两类之间的间隔。按照数据集的特点：

1. 线性可分问题，如之前的感知机算法处理的问题
2. 线性可分，只有一点点错误点，如感知机算法发展出来的 Pocket 算法处理的问题
3. 非线性问题，完全不可分，如在感知机问题发展出来的多层感知机和深度学习

这三种情况对于 SVM 分别有下面三种处理手段：

1. hard-margin SVM
2. soft-margin SVM
3. kernel Method

## 引入：从逻辑回归出发

在LR中，我们使用一个线性模型 $h(vb(x))=vb(w^T x)+b$ 计算每个点的得分，并代入Sigmond函数映射为概率 $P(y=1|x)$ ，决策边界为 $h(vb(x))=0$

由于LR的参数受到所有样本的影响，鲁棒性和泛化性能可能不佳（在引入新的样本点时原模型很容易过拟合失败）。这等价于另一个问题：“若有多条直线都能正确分割数据，哪条最好？”

SVM解决了这个问题，通过一个新的思维方式：**不仅要分割数据，还要让两个数据集分开的最远**，这样的鲁棒性和泛化性能是最好的

> SVM的基本方法是新引入了两个平行超平面 $vb(w^T x)+b=1$ 和 $vb(w^T x)+b=-1$，以它们构成的长条取代原本的一条线，要求转化为 $y_i (vb(w^T x_i)+b)>=1$ 。
> 所有正样本都在上方超平面之上，所有负样本都在下方超平面之下，最终的超平面是 $vb(w^T x)+b=0$ 即两者的中值，而两个超平面之间的距离被称为间隔（margin）。
> 可以证明两个超平面的间隔为 $frac(2,||w||)$ ，因此最大化间距等价于最小化 $1/2 ||w||^2$ 。

我们寻找，优化问题 $min_(vb(w),b) 1/2 ||w||^2$  s.t. $y_i (vb(w^T x_i)+b)>=1$ 的对偶问题，使用Lagrange方法：

构造Lagrange函数形如 $L(vb(w),b,vb(alpha))=1/2 ||w||^2 - sum(alpha_i(y_i(vb(w^T x_i)+b)-1))$ ，其中 $alpha_i>=0$ 是拉格朗日乘子。

分别求偏导，并令为0：

$
    &pdv(L,vb(w))=vb(w)-sum(alpha_i y_i vb(x_i))=0  => &&vb(w)=sum(alpha_i y_i vb(x_i))\
    &pdv(L,b)=-sum(alpha_i y_i)=0  => &&sum(alpha_i y_i)=0
$

整理得到Stationarity ，即我们最终得到的对偶问题

$
    max_(vb(alpha)) (-1/2 sum_i sum_j alpha_i alpha_j y_i y_j vb(x_i^T x_j) + sum_i alpha_i) \
$

当然还有Primal Feasibility $y_i (vb(w^T x_i)+b) >= 1$ 和 Dual Feasibility $alpha_i >= 0$ 以及 Complementary Slackness $alpha_i (y_i (vb(w^T x_i)+b)-1)=0$ 。

> Complementary Slackness 告诉我们，若 $alpha_i > 0$ ，则 $y_i (vb(w^T x_i)+b)-1=0$ ，即该样本点在间隔边界上，这些点称为支持向量（Support Vectors）。只有支持向量对最终的分类结果有影响，其他点对分类结果没有影响。

下面构造合页损失（Hinge Loss）优化问题，其中第一项是L2正则化，第二项是合页损失：

$
    min_(vb(w),b) 1/2 ||w||^2 + C sum_i max(0,1 - y_i (vb(w^T x_i)+b))
$

## Hard-margin SVM

使用较强的约束条件，假设数据线性可分，所有样本点都能被正确分类，即合页项取0。这等价于合页损失的参数 $C -> infinity$ 

此时合页优化直接等价于原问题，KKT略

## Soft-margin SVM

软间隔问题允许一些样本点不满足硬约束条件，但会有相应的惩罚函数。这使我们引入了松弛变量 $xi_i >=0$:

约束放宽为 $forall (x_i,y_i) : y_i (vb(w^T x_i)+b) >= 1 - xi_i$ ，当 $xi_i=0$ 为硬约束，当 $0<xi_i<1$ 时，点在间隔内但被正确分类，当 $xi_i>1$ 时，点被错误分类。

那么，原问题为 $min_(vb(w),b) 1/2 ||w||^2 + C sum_i xi_i,s.t. y_i (vb(w^T x_i)+b) >= 1 - xi_i,xi_i>=0$。

由于这两个约束导出 $xi_i>=max(0,1 - y_i (vb(w^T x_i)+b))$ ，原问题与合页损失问题等价，KKT略

## Kernel Method

为了处理严格线性不可分数据集，我们引入核方法。核方法依赖于CoversTheorem，即在高维特征空间中几乎所有数据集都是线性可分的。这启示我们将低维特征映射为高维数据以简化问题。

映射后，同样回到SVM的对偶问题：

$
    L_D=sum_i alpha_i - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j vb(phi(x_i)^T phi(x_j)),quad (vb(w)=sum_i alpha_i y_i phi(vb(x_i)))
$

解函数为 $f(vb(x))=vb(w^T)h(vb(x))+b=sum_i alpha_i y_i iprod(phi(vb(x_i)),phi(vb(x_j)))$

我们发现数据点仅以内积形式出现 $vb(x_i^T x_j)$ 。因此，问题不在于映射 $phi$ 本身。我们只需要研究映射后元素间的内积，也就是正定核函数 $K(vb(x_i),vb(x_j))=iprod(phi(vb(x_i)),phi(vb(x_j)))$ 。此处正定是考虑能生成内积。

（甚至很多时候映射是隐式的，如Gaussian核 $K(vb(x_i),vb(x_j))=exp(-||vb(x_i)-vb(x_j)||^2/(2sigma^2))$ 没有显式映射函数。我们不需要真的映射到高维，只需要在低维计算核函数）

总结：构造核函数，并验证内积的存在性（即核函数满足对称性和正定性），就能用SVM处理非线性问题。

## 序列最小最优化算法

我们知道，支持向量机的学习问题可以形式化为求解凸二次规划问题。这样的凸二次规划问题具有全局最优解，并且有许多最优化算法可以用于这一问题的求解。但是当训练样本容量很大时，这些算法往往变得非常低效，以致无法使用。所以，如何高效地实现支持向量机学习就成为一个重要的问题。

已知优化问题对偶为以下问题：

$
    &min_(vb(alpha)) 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(vb(x_i),vb(x_j)) - sum_i alpha_i \
    &s.t. sum_i alpha_i y_i = 0, C>= alpha_i >= 0 \
$

在这个问题中，变量是拉格朗日乘子，一个变量 $alpha_i$ 对应于一个样本点 $(x_i,y_i)$

SMO算法是一种启发式算法，其基本思路是：

1. 如果所有变量的解都满足此最优化问题的KKT条件，那么这个最优化问题的解就得到了。因为KKT条件是该最优化问题的充分必要条件。

2. 否则，选择两个变量，固定其他变量，针对这两个变量构建一个二次规划问题。这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。重要的是，这时子问题可以通过解析方法求解，这样就可以大大提高整个算法的计算速度。子问题有两个变量，一个是违反KKT条件最严重的那一个，另一个由约束条件 $sum_i alpha_i y_i = 0$ 自动确定。

3. 如此，SMO算法将原问题不断分解为子问题并对子问题求解，进而达到求解原问题的目的。
