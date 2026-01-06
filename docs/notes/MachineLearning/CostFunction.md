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

### 线性回归的概率解释

为什么选择平方代价函数？实际上这基于线性拟合的高斯噪声假设 $y=vb(w^T x)+b+epsilon,quad epsilon ~ N(0,sigma^2)$ 。那么给定输入 $vb(x)$ ，输出 $y$ 的似然为

$
  P(y|x)=frac(1,sqrt(2 pi) sigma) exp(-frac((y-vb(w^T x)-b)^2,2 sigma^2))
$

最大似然估计MLE指出，最大化似然等价于最小化负对数似然，因此优化目标是 $min_(vb(w),b) (y-vb(w^T x)-b)^2$

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

$
vb(theta)=vb((X^T X+lambda underbrace(dmat(0,1,dots.down,1),(n+1)*(n+1)))^(-1) X^T y)
$

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

$
vb(nabla_theta)L_1=vb(nabla_theta)J_0+2lambda theta=0\
vb(nabla_theta)L_2=vb(nabla_theta)J_0+2mu theta=0
$

## 大规模机器学习

如果我们有一个低方差的模型，增加数据集的规模可以帮助你获得更好的结果，代价是BGD方法变得难以承受。

### SGD和Mini BGD

随机梯度下降 (Stochastic Gradient Descent, SGD) 每次只用一个样本来更新参数，而小批量梯度下降 (Mini-Batch Gradient Descent) 则是每次用 $b$ 个样本来更新参数， $1<b<m$ 。

这两种方法都可以有效地减少计算开销，并且在大规模数据集上表现良好（引入噪声，避免陷入局域最优），此外，随机思想使问题Online化了，更适合流数据处理。实践上，SGD被应用于凸优化等需要精调的场景较多。

但SGD还是有三大问题：

1. 固定的学习率 $alpha$ 不合适
   1. 平坦区域期望大步长
   2. 陡峭区域期望小步长
   3. 不同参数期望不同步长
2. 梯度噪声带来振荡，可以进一步优化
3. 不同参数更新速度应当不同
   1. 稀疏特征需要更大步长
   2. 密集特征需要更小步长

### Momentum系列

动量法发挥惯性的作用，使得平坦区域加速，陡峭区域减速。另外，在“峡谷”中缓解了振荡问题。

$
f\or vb(theta_i) \in S\et: \
vb(v)_t=beta vb(v)_(t-1)+(1-beta) vb(nabla_theta) J(theta) \
vb(theta)_(t+1)=vb(theta)_t-eta vb(v)_t
$

或者更进一步，加一个预测（Nesterov Accelerated Gradient，NAG），修正小样本导致的梯度方向不佳：

$
f\or vb(theta_i) \in S\et: \
vb(v)_t=beta vb(v)_(t-1)+eta vb(nabla_theta) J(vb(theta) - beta vb(v)_(t-1)) \
vb(theta)_(t+1)=vb(theta)_t-vb(v)_t
$

动量思想足以应付大多数深度学习任务了。

### Ada系列

另一条修正方向是Adaptive Learning Rate方法，针对历史梯度调整每个参数的学习率。

#### AdaGrad

核心思想是根据**历史梯度的平方和**调整每个参数的学习率，历史梯度平方和越大，学习率越小

$
G_t=G_(t-1)+[vb(nabla_theta) J(vb(theta))]^2\
vb(theta)_(t+1)=vb(theta)_t-frac(eta,sqrt(G_t+epsilon)) vb(nabla_theta) J(vb(theta))
$

AdaGrad适合处理稀疏特征的问题，被广泛应用于NLP等领域。

#### RMSProp

AdaGrad考虑全历史的平方和，会积分饱和（甚至由于平方非负不能脱离饱和？！废物啊）

RMSProp引入指数加权移动平均，使较古老的贡献衰减，避免了饱和（和Momentum类似）

$
  E[g^2]_t=beta E[g^2]_(t-1)+(1-beta)[vb(nabla_theta) J(vb(theta))]^2\
  vb(theta)_(t+1)=vb(theta)_t-frac(eta,sqrt(E[g^2]_t+epsilon)) vb(nabla_theta) J(vb(theta))_t
$

RMSProp广泛应用于循环神经网络（RNN）的训练中。

### Adam

Adaptive Momentum Estimation是目前最流行的优化算法，它结合了动量法和RMSProp的优点，同时考虑一阶矩估计和二阶矩估计

$
  vb(m)_t=beta_1 vb(m)_(t-1)+(1-beta_1) vb(nabla_theta) J(vb(theta))\
  vb(v)_t=beta_2 vb(v)_(t-1)+(1-beta_2)[vb(nabla_theta) J(vb(theta))]^2\
  hat(vb(m))_t=frac(vb(m)_t,1-beta_1^t),hat(vb(v))_t=frac(vb(v)_t,1-beta_2^t)\
  vb(theta)_(t+1)=vb(theta)_t-frac(eta,sqrt(hat(vb(v))_t)+epsilon) hat(vb(m))_t
$

Adam实际上属于伪二阶，它使用了梯度的平方来近似Hessian矩阵的对角线，避免了计算二阶导的高昂开销。

#### Adam与高斯牛顿法

Adam的处理思想与单纯形法、高斯牛顿法一脉相承。

**一、单纯形法**：对于一个线性问题 $f(x;theta)=vb(A theta)$ ，寻找一个 $theta^*$ 使得 $vb(A theta^*) arrow y$ 。
- 单纯数学处理的视角：我们构造了自伴矩阵 $vb(A^T A)$ ，便可得到正规方程 $vb(theta^*)=vb((A^T A))^(-1) A^T y$ 。
- 几何的视角： $vb(A^T A)$ 描述了参数空间的曲率（线性问题的Hessian矩阵恒定），从而一次性跳到最优解。
- 代数的视角：对 L2-norm $L(vb(theta))=1/2 ||vb(r(theta))||^2 =1/2 (vb(A theta -y))^T (vb(A theta -y))$ 求导， $nabla L(vb(theta))=vb(r^T pdv(r,theta))=0$ ，也就是梯度 $vb(g)=vb(nabla L)^T=vb(A^T r)=0$ （为了转为列向量）

**二、高斯牛顿法**：如果问题非线性 $f(x;theta)=vb(r(theta))$ ，不妨局部以直代曲，线性化处理： $r(theta)~r(theta_t)+vb(J)Delta theta$ ，其中 $vb(J)$ 为雅可比矩阵 $vb(J)_(i j)=pdv(r_i,theta_j)$ 表征局部曲率

带回单纯形法，即得 $Delta theta =- vb((J^T J))^(-1) vb(J^T r)=- vb(J^T J)^(-1) vb(g)$

高斯牛顿法的高明之处在于，假设了 $L=1/2 ||r||^2$ 的二阶小量展开中，残差一阶项的二次占主导，残差二阶项的一次贡献可忽略，从而将Hessian矩阵 $vb(H)=vb(nabla^2 L)=vb(J^T J)+sum_i r_i pdv^2 (r_i,theta)$ 简化为 $vb(H)~vb(J^T J)$ 。这个假设在残差较小 / 线性化程度高的情况下很优秀。

**三、迭代器**的二阶矩估计：参数较多时计算 $vb(J^T J)$ 也工作量巨大。不如假设该矩阵对角， $(vb(J^T J)^(-1))~d\ia\g(1/(vb(J^T J)_(i i)))$ 。再由统计上随机采样的结论 $E[vb(g g^T)]=E[vb((J^T r)(r^T J))]=vb(J^T J) sigma^2$ ，在参数互不关联时很好的用梯度的平方估计了Hessian。

实际上在CNN时代，Adam是远不如SGD的，因为Hessian矩阵相对“均匀”，同质性；而在Transformer/MLP等网络，Hessian矩阵表现出强烈对角性质（甚至分块对角），异质性，因此Adam得到了推广。

### 战无不胜的控制思想（大雾）

各种优化的本质不过是一个PI控制（×）。~~D还是算了，开销太大。~~

- 梯度下降就是一个比例控制
- 动量法相当于于一个积分控制
- NAG引入了预测控制，有点MPC的意思
- AdaGrad是一个变比例控制，但本质更像积分控制
- RMSProp纠正了AdaGrad的积分饱和问题，算是滑动窗口法
- Adam结合了动量和RMSProp
