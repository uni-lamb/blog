# 恒等式证明

用更严谨的方法证明恒等式，特别是矢量恒等式，尤其是含 $\nabla$ 不等式！

**基本定义和约定：**

1.  **爱因斯坦求和约定：** 当一个指标（角标）在一个项中重复出现一次时，表示对该指标进行求和。例如，$A_i B_i = \sum_{i=1}^n A_i B_i = A_1 B_1 + A_2 B_2 +\ldots+ A_n B_n$。
2.  **偏导数算子：** $\partial_i = \frac{\partial}{\partial x_i}$。
3.  **克罗内克 Delta (Kronecker Delta) $\delta_{ij}$：**
    $\delta_{ij} = \begin{cases} 1, & \text{if } i = j \\ 0, & \text{if } i \neq j \end{cases}$
4.  **列维-奇维塔符号 (Levi-Civita Symbol) $\epsilon_{ijk}$：**

    $\epsilon_{ijk} = \begin{cases} +1, & \text{if } (i,j,k) \text{ is } (1,2,3), (2,3,1), \text{ or } (3,1,2) \text{ (偶排列)} \\ -1, & \text{if } (i,j,k) \text{ is } (3,2,1), (1,3,2), \text{ or } (2,1,3) \text{ (奇排列)} \\ 0, & \text{if any index is repeated} \end{cases}$

5.  **重要的 $\epsilon-\delta$ 恒等式：**
    $\epsilon_{ijk} \epsilon_{klm} = \delta_{il} \delta_{jm} - \delta_{im} \delta_{jl}$

## $\epsilon_{ijk} \epsilon_{klm} = \delta_{il} \delta_{jm} - \delta_{im} \delta_{jl}$

$LHS:k$ 是哑指标，根据 $\epsilon$ 的定义，非零项满足 $\set{i,j}=\set{l,m}=\set{1,2,3}\setminus \set{k}$

$RHS:$ 非零项满足 $l=i,m=j$ 或 $l=j,m=i$ ，否则为0。非零条件一致

然后验证分别对应了正负号，证毕。

## $\vec{A} \times (\vec{B} \times \vec{C})= \vec{B}(\vec{A} \cdot \vec{C}) - \vec{C}(\vec{A} \cdot \vec{B})$

$[\vec{A} \times (\vec{B} \times \vec{C})]_i=\epsilon_{ijk} A_j [\vec{B} \times \vec{C}]_k=\epsilon_{ijk} A_j \epsilon_{klm} B_l C_m=(\delta_{il} \delta_{jm} - \delta_{im} \delta_{jl}) A_j B_l C_m$

然后简化，得到 $[\vec{A} \times (\vec{B} \times \vec{C})]_i=A_j B_i C_j-A_j B_j C_i=[\vec{B}(\vec{A} \cdot \vec{C}) - \vec{C}(\vec{A} \cdot \vec{B})]_i$

## $\nabla \times (\nabla \times \vec{E}) = \nabla(\nabla \cdot \vec{E}) - \nabla^2 \vec{E}$

$[\nabla \times (\nabla \times \vec{E})]_i =\epsilon_{ijk} \partial_j [\nabla \times \vec{E}]_k =\epsilon_{ijk} \partial_j \epsilon_{klm} \partial_l E_m=(\delta_{il}\delta_{jm}-\delta_{im}\delta_{jl})\partial_j\partial_lE_m$

然后简化，得到 $[\nabla \times (\nabla \times \vec{E})]_i=\partial_i(\partial_j E_j) - \partial_j\partial_j E_i=[\nabla(\nabla \cdot \vec{E}) - \nabla^2 \vec{E}]_i$

## $\nabla(\vec{A} \cdot \vec{B})=(\vec{A} \cdot \nabla) \vec{B} + (\vec{B} \cdot \nabla) \vec{A}+\vec{A} \times (\nabla \times \vec{B}) + \vec{B} \times (\nabla \times \vec{A})$

$[\nabla(\vec{A} \cdot \vec{B})]_i=\partial_i (A_j B_j)=B_j \partial_i A_j + A_j \partial_i B_j$

从右边开始证明

- $[(\vec{A} \cdot \nabla) \vec{B}]_i = A_j \partial_j B_i$
- $[(\vec{B} \cdot \nabla) \vec{A}]_i = B_j \partial_j A_i$
- $[\vec{A} \times (\nabla \times \vec{B})]_i = \epsilon_{ijk} A_j \epsilon_{klm}\partial_l B_m=(\delta_{il}\delta_{jm}-\delta_{im}\delta_{jl})A_j\partial_l B_m=A_j \partial_i B_j - A_j \partial_j B_i$
- $[\vec{B} \times (\nabla \times \vec{A})]_i = \epsilon_{ijk} B_j \epsilon_{klm}\partial_l A_m=(\delta_{il}\delta_{jm}-\delta_{im}\delta_{jl})B_j\partial_l A_m=B_j \partial_i A_j - B_j \partial_j A_i$

$RHS=A_j \partial_i B_j + B_j \partial_i A_j=LHS,Q.E.D.$

## $\nabla \cdot (\vec{A} \times \vec{B}) = \vec{B} \cdot (\nabla \times \vec{A}) - \vec{A} \cdot (\nabla \times \vec{B})$

## $\nabla \times (\psi \nabla \phi)=\nabla \psi \times \nabla \phi$

## $\nabla \times (\psi \vec{A})=\psi \nabla \times \vec{A}+(\nabla \psi)\times \vec{A}$

## $\nabla \times (\vec{A} \times \vec{B})=\vec{A}(\nabla \cdot \vec{B})-\vec{B}(\nabla \cdot \vec{A})+(\vec{B}\cdot\nabla)\vec{A}-(\vec{A}\cdot\nabla)\vec{B}$
