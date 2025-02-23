---
layout: default
---

# 时间序列卷积

<p align="center"><span style="color:gray">一种用于增强时间序列趋势建模和时间模式挖掘的卷积核方法，可利用傅里叶变换进行快速计算。同时，可解释的机器学习模型（如稀疏回归）为更好地捕捉现实世界时间序列的长期变化和时间模式提供了途径。</span></p>

<br>

本文将讨论近期研究中的基本建模思路：

- Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). [Laplacian convolutional representation for traffic time series imputation](https://doi.org/10.1109/TKDE.2024.3419698). *IEEE Transactions on Knowledge and Data Engineering*. 36 (11): 6490-6502.
- Xinyu Chen, Xi-Le Zhao, Chun Cheng (2024). [Forecasting urban traffic states with sparse data using Hankel temporal matrix factorization](https://doi.org/10.1287/ijoc.2022.0197). *INFORMS Journal on Computing*.
- Xinyu Chen, HanQin Cai, Fuqiang Liu, Jinhua Zhao (2024). [Correlating time series with interpretable convolutional kernels](https://arxiv.org/abs/2409.01362). arXiv:2409.01362.


**内容：**

在本系列的**第一部分**中，我们介绍了具有全局和局部趋势的时间序列建模动机。这些时间序列趋势对于提高时间序列插补的性能非常重要。如果有一个合适的可解释机器学习模型，还可以量化时间序列的周期性。在基本动机之后，我们在**第二部分**详细阐述了几个关键概念，如循环卷积、卷积矩阵、循环矩阵和离散傅里叶变换。

**第三部分**和**第四部分**给出了循环矩阵核范数最小化和拉普拉斯卷积表示的建模思想，解决了时间序列插补任务中的关键挑战。这两种模型的优化算法利用快速傅里叶变换，具有对数线性时间复杂度。**第五部分**提出了一种可解释的卷积核方法，其中卷积核的稀疏性通过<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>范数诱导的稀疏约束进行建模。

<br>

## I. 动机

过去十年中，机器学习模型的发展确实令人瞩目。卷积是应用数学和信号处理中最常用的操作之一，已广泛应用于多个机器学习问题。本文的目的是重新审视循环卷积的核心思想，并为时间序列数据建模奠定基础。

如今，尽管我们手头有许多机器学习算法，但在时间序列建模中仍然需要重新思考以下问题：

- 如何表征全局时间序列趋势？
- 如何表征局部时间序列趋势？
- 如何学习可解释的局部和非局部模式作为卷积核？

有时，如果[时间序列不是平稳的](https://otexts.com/fpp2/stationarity.html)，它们会表现出复杂的趋势。

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/time_series_global_trends.png" width="350" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
(a) 全局趋势（例如，长期日/周周期性）
</p>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/time_series_local_trends.png" width="350" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
(b) 局部趋势（例如，短期时间序列趋势）
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>图1.</b> 时间序列趋势示意图。
</p>

<br>

## II. 预备知识

在本研究中，我们基于信号处理领域的几个关键思想构建了拉普拉斯卷积表示（LCR）的建模概念，包括循环卷积、离散傅里叶变换和快速傅里叶变换。在接下来的部分中，我们将讨论：

- 什么是循环卷积、卷积矩阵和循环矩阵？
- 什么是卷积定理？
- 如何使用快速傅里叶变换计算循环卷积？

### II-A. 循环卷积

卷积是许多经典深度学习框架中的强大操作，例如卷积神经网络（CNN）。在离散序列（通常是向量）的上下文中，循环卷积指的是两个离散数据序列的卷积，它在最大化某些常见滤波操作的效率中起着重要作用（参见维基百科上的[循环卷积](https://en.wikipedia.org/wiki/Circular_convolution)）。

根据定义，对于任何向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,\cdots,x_T)^\top\in\mathbb{R}^{T}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(y_1,y_2,\cdots,y_\tau)^\top\in\mathbb{R}^{\tau}"/>，其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau\leq T"/>，这两个向量的循环卷积（用符号 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\star"/> 表示）定义如下：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}\in\mathbb{R}^{T}"/></p>

逐元素地，我们有

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;z_{t}=\sum_{k=1}^{\tau}x_{t-k+1} y_{k},\,\forall t\in\{1,2,\ldots,T\}"/></p>

其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_t"/> 表示 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> 的第 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> 个元素。对于循环操作，当 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t+1\leq k"/> 时，取 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{t-k+1}=x_{t-k+1+T}"/>。虽然循环卷积的定义对初学者来说可能看起来很复杂，但当你考虑卷积矩阵或循环矩阵的概念时，它会变得更加清晰，如下面的例子所示。

如上所述，向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 的长度为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/>，向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> 的长度为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/>。根据循环卷积的定义，结果向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> 的长度也将为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/>，与原始向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 的长度匹配。

给定任何向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,x_3,x_4)^\top\in\mathbb{R}^4"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(y_1,y_2,y_3)^\top\in\mathbb{R}^3"/>，它们之间的循环卷积可以表示为：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\begin{bmatrix}
\displaystyle\sum_{k=1}^{3}x_{1-k+1}y_k \\
\displaystyle\sum_{k=1}^{3}x_{2-k+1}y_k \\
\displaystyle\sum_{k=1}^{3}x_{3-k+1}y_k \\
\displaystyle\sum_{k=1}^{3}x_{4-k+1}y_k \\
\end{bmatrix}
=\begin{bmatrix}
x_1y_1+x_4y_2+x_3y_3 \\ x_2y_1+x_1y_2+x_4y_3 \\ x_3y_1+x_2y_2+x_1y_3 \\ x_4y_1+x_3y_2+x_2y_3 \\
\end{bmatrix} \\"/></p>

其中根据定义，<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{0}=x_4"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{-1}=x_3"/>。

在这种情况下，结果向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> 的每个元素 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_t"/> 是通过向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> 和向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 的反转和截断版本的内积计算得到的。具体来说，元素 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_1"/> 是通过计算 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;(x_1,x_4,x_3)^\top"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;(y_1,y_2,y_3)^\top"/> 的内积得到的，即：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;z_1=\begin{bmatrix} x_1 & x_4 & x_3 \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix}=x_1y_1+x_4y_2+x_3y_3"/></p>

对于后续的元素 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_2,z_3,z_4"/>，向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 被循环反转并截断（仅保留前 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/> 个元素），然后与 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> 计算内积。图2展示了计算循环卷积每个元素的基本步骤。

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/circular_convolution_steps.png" width="450" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>图2.</b> 向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,x_3,x_4)^\top"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(y_1,y_2,y_3)^\top"/> 之间的循环卷积示意图。(a) 计算 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_1"/> 涉及 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{0}=x_4"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{-1}=x_3"/>。(b) 计算 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_2"/> 涉及 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{0}=x_4"/>。该图灵感来自 [Prince (2023)](https://udlbook.github.io/udlbook/)。
</p>

<br>

可以看出，两个向量之间的循环卷积本质上可以看作是一个线性操作。这种视角允许我们将循环卷积重新表述为使用卷积矩阵（或当这两个向量长度相同时的循环矩阵）的线性变换。

<br>

---

<span style="color:gray">
<b>示例1.</b> 给定向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(2,-1,3)^\top"/>，循环卷积 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=(5,14,3,7,11)^\top"/> 的元素如下：</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{cases} z_1=x_1y_1+x_5y_2+x_4y_3=0\times 2+4\times(-1)+3\times 3=5 \\ z_2=x_2y_1+x_1y_2+x_5y_3=1\times 2+0\times(-1)+4\times 3=14 \\ z_3=x_3y_1+x_2y_2+x_1y_3=2\times 2+1\times (-1)+0\times 3=3 \\ z_4=x_4y_1+x_3y_2+x_2y_3=3\times 2+2\times (-1)+1\times 3=7 \\ z_5=x_5y_1+x_4y_2+x_3y_3=4\times 2+3\times (-1)+2\times 3=11 \end{cases}"/></p>

<span style="color:gray">
其中根据定义，<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{0}=x_5"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{-1}=x_4"/>。
</span>

---

<br>

### II-B. 卷积矩阵

使用上述符号，对于任何向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{\tau}"/>，其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau<T"/>，循环卷积可以表示为线性变换：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}_{\tau}(\boldsymbol{x})\boldsymbol{y}"/></p>

其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}:\mathbb{R}^{T}\to\mathbb{R}^{T\times \tau}"/> 表示卷积算子。卷积矩阵可以表示为：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}(\boldsymbol{x})=\begin{bmatrix} x_1 & x_T & x_{T-1} & \cdots & x_{T-\tau+2} \\ x_2 & x_1 & x_{T} & \cdots & x_{T-\tau+3} \\ x_3 & x_2 & x_1 & \cdots & x_{T-\tau+4} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ x_{T} & x_{T-1} & x_{T-2} & \cdots & x_{T-\tau+1} \\ \end{bmatrix}\in\mathbb{R}^{T\times\tau}"/></p>

在信号处理领域，这种线性变换是循环卷积的基本性质，突出了其在高效实现滤波操作中的作用。

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/linear_conv_mat.png" width="320" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>图3.</b> 循环卷积作为卷积矩阵的线性变换示意图。
</p>

<br>

---

<span style="color:gray">
<b>示例2.</b> 给定向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(2,-1,3)^\top"/>，循环卷积 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}"/> 可以表示为：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}_{3}(\boldsymbol{x})\boldsymbol{y}=(5,14,3,7,11)^\top"/></p>

<span style="color:gray">
其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{3}(\boldsymbol{x})"/> 是具有 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau=3"/> 列的卷积矩阵。具体来说，卷积矩阵的结构如下：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{3}(\boldsymbol{x})=\begin{bmatrix} 0 & 4 & 3 \\ 1 & 0 & 4 \\ 2 & 1 & 0 \\ 3 & 2 & 1 \\ 4 & 3 & 2 \end{bmatrix}"/></p>

<span style="color:gray">
因此，结果为</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\mathcal{C}_{3}(\boldsymbol{x})\boldsymbol{y}=\begin{bmatrix} 0 & 4 & 3 \\ 1 & 0 & 4 \\ 2 & 1 & 0 \\ 3 & 2 & 1 \\ 4 & 3 & 2 \end{bmatrix}\begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}=\begin{bmatrix} 5 \\ 14 \\ 3 \\ 7 \\ 11 \end{bmatrix}"/></p>

<span style="color:gray">
这种表示表明，循环卷积等价于矩阵-向量乘法，使其更容易理解，特别是在信号处理应用中。
</span>

---

<br>

在这篇文章中，我们旨在通过结合Python编程代码、直观的图示和公式的详细解释，使概念清晰易懂。为了演示如何计算循环卷积，我们将使用Python的`numpy`库。首先，我们将在向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 上构建卷积矩阵，然后执行循环卷积，如下所示。

<br>

```python
import numpy as np

def conv_mat(vec, tau):
    n = vec.shape[0]
    mat = np.zeros((n, tau))
    mat[:, 0] = vec
    for i in range(1, tau):
        mat[:, i] = np.append(vec[-i :], vec[: n - i], axis = 0)
    return mat

x = np.array([0, 1, 2, 3, 4])
y = np.array([2, -1, 3])
mat = conv_mat(x, y.shape[0])
print('Convolution matrix of x with 3 columns:')
print(mat)
print()
z = mat @ y
print('Circular convolution of x and y:')
print(z)
```

<br>

### II-C. 循环矩阵

回想一下，卷积矩阵 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}(\boldsymbol{x})"/> 指定了 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/> 列，对应于向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> 的长度。在 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^{T}"/> 且列数相同的情况下，卷积矩阵变为一个方阵，称为**循环矩阵**。在本研究中，我们强调循环矩阵及其性质的重要性，例如它们与循环卷积和离散傅里叶变换的紧密联系，即使我们不直接使用循环矩阵。

对于任何向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^{T}"/>，循环卷积可以表示为线性变换，使得

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}(\boldsymbol{x})\boldsymbol{y}"/></p>

其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}:\mathbb{R}^{T}\to\mathbb{R}^{T\times T}"/> 表示循环算子。循环矩阵定义为：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} x_1 & x_T & x_{T-1} & \cdots & x_{2} \\ x_2 & x_1 & x_{T} & \cdots & x_{3} \\ x_3 & x_2 & x_1 & \cdots & x_{4} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ x_{T} & x_{T-1} & x_{T-2} & \cdots & x_{1} \\ \end{bmatrix}\in\mathbb{R}^{T\times T}"/></p>
它形成一个方阵。始终有 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\mathcal{C}(\boldsymbol{x})\|_F=\sqrt{T}\cdot\|\boldsymbol{x}\|_2"/>，其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\cdot\|_F"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\cdot\|_2"/> 分别是矩阵的Frobenius范数和向量的 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_2"/>-范数。

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/linear_circ_mat.png" width="320" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>图4.</b> 循环卷积作为循环矩阵的线性变换示意图。
</p>

<br>

---

<span style="color:gray">
<b>示例3.</b> 给定向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(2,-1,3,0,0)^\top"/>，循环卷积 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}"/> 等同于
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}(\boldsymbol{x})\boldsymbol{y}=(5,14,3,7,11)^\top"/></p>

<span style="color:gray">
其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})"/> 是由 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 形成的循环矩阵，定义为：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} 0 & 4 & 3 & 2 & 1 \\ 1 & 0 & 4 & 3 & 2 \\ 2 & 1 & 0 & 4 & 3 \\ 3 & 2 & 1 & 0 & 4 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}"/></p>

<span style="color:gray">
因此，结果可以写成如下形式：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\mathcal{C}(\boldsymbol{x})\boldsymbol{y}=\begin{bmatrix} 0 & 4 & 3 & 2 & 1 \\ 1 & 0 & 4 & 3 & 2 \\ 2 & 1 & 0 & 4 & 3 \\ 3 & 2 & 1 & 0 & 4 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}\begin{bmatrix} 2 \\ -1 \\ 3 \\ 0 \\ 0 \end{bmatrix}=\begin{bmatrix} 5 \\ 14 \\ 3 \\ 7 \\ 11 \end{bmatrix}"/></p>

<span style="color:gray">
该示例表明，<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;(2,-1,3)^\top"/> 的循环卷积等同于 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> 的循环卷积，其中最后两个元素填充为零。因此，当 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> 大于 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/> 时，可以通过在 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> 的末尾附加 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T-\tau"/> 个零来执行循环卷积。
</span>

---

<br>

### II-D. 离散傅里叶变换

离散傅里叶变换（参见[维基百科](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)）是数学和信号处理中的基本工具，广泛应用于机器学习。离散傅里叶变换是用于傅里叶分析的关键离散变换，能够将信号分解为其组成频率。快速傅里叶变换是计算离散傅里叶变换的高效算法（参见[离散傅里叶变换与快速傅里叶变换的区别](https://math.stackexchange.com/q/30464/738418)），将时间复杂度从 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{O}(T^2)"/> 显著降低到 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{O}(T\log T)"/>，其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> 是数据点的数量。这种效率使得快速傅里叶变换在处理大规模问题时至关重要。

信号处理中的一个关键概念是卷积定理，它指出时域中的卷积是频域中的乘法。这意味着可以使用快速傅里叶变换高效地计算循环卷积。离散傅里叶变换的卷积定理总结如下：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{F}(\boldsymbol{x}\star\boldsymbol{y})=\mathcal{F}(\boldsymbol{x})\circ\mathcal{F}(\boldsymbol{y})"/></p>

或

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\star\boldsymbol{y}=\mathcal{F}^{-1}(\mathcal{F}(\boldsymbol{x})\circ\mathcal{F}(\boldsymbol{y}))"/></p>

其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{F}(\cdot)"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{F}^{-1}(\cdot)"/> 分别表示离散傅里叶变换和逆离散傅里叶变换。符号 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\circ"/> 表示Hadamard积，即逐元素乘法。

事实上，这一原理是信号处理和数据分析中许多高效算法的基础，允许在频域中高效执行复杂操作。

<br>

---

<span style="color:gray">
<b>示例4.</b> 给定向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(2,-1,3,0,0)^\top"/>，循环卷积 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}"/> 可以通过`numpy`中的快速傅里叶变换计算如下：
</span>

```python
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y = np.array([2, -1, 3, 0, 0])
fx = np.fft.fft(x)
fy = np.fft.fft(y)
z = np.fft.ifft(fx * fy).real
print('Fast Fourier transform of x:')
print(fx)
print('Fast Fourier transform of y:')
print(fy)
print('Circular convolution of x and y:')
print(z)
```

其中输出为：

```python
Fast Fourier transform of x:
[10. +0.j         -2.5+3.4409548j  -2.5+0.81229924j -2.5-0.81229924j
 -2.5-3.4409548j ]
Fast Fourier transform of y:
[ 4.        +0.j         -0.73606798-0.81229924j  3.73606798+3.4409548j
  3.73606798-3.4409548j  -0.73606798+0.81229924j]
Circular convolution of x and y:
[ 5. 14.  3.  7. 11.]
```

<br>

---

<br>

### II-F. Hankel矩阵分解与离散傅里叶变换

Hankel矩阵在应用数学和信号处理的许多领域中起着基础作用。根据定义，Hankel矩阵是一个方阵或矩形矩阵，其中每个上升的斜对角线（从左到右）具有相同的值。给定向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/>，Hankel矩阵可以构造如下：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{H}(\boldsymbol{x})=\begin{bmatrix} x_1 & x_2 & \cdots & x_{T-n+1} \\ x_2 & x_3 & \cdots & x_{T-n+2} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n} & x_{n+1} & \cdots & x_{T} \end{bmatrix}"/></p>

其中Hankel矩阵有 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> 行和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T-n+1"/> 列。该矩阵通常用于表示信号或时间序列数据，捕捉它们的顺序依赖性和结构，参见例如我们工作中的 [Chen et al. (2024)](https://doi.org/10.1287/ijoc.2022.0197)。

在Hankel矩阵 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{H}(\boldsymbol{x})"/> 上，如果它可以近似为两个矩阵 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}\in\mathbb{R}^{n\times R}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Q}\in\mathbb{R}^{(T-n+1)\times R}"/> 的乘积，则可以计算Hankel矩阵分解的逆如下：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \mathcal{H}(\boldsymbol{x})\approx&\boldsymbol{W}\boldsymbol{Q}^\top \\ \Rightarrow\quad\tilde{\boldsymbol{x}}=&\mathcal{H}^{\dagger}(\boldsymbol{W}\boldsymbol{Q}^{\top}) \\ \Rightarrow\quad\tilde{x}_t=&\frac{1}{\rho_t}\sum_{a+b=t+1}\boldsymbol{w}_{a}^\top\boldsymbol{q}_{b} \\ \Rightarrow\quad\tilde{x}_{t}=&\frac{1}{\rho_t}\sum_{r=1}^{R}\underbrace{\sum_{a+b=t+1}w_{a,r}q_{b,r}}_{\text{\color{red}循环卷积}} \end{aligned}"/></p>

其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}_a"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{q}_b"/> 分别是 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Q}"/> 的第 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;a"/> 行和第 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;b"/> 行。这里，<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{H}^{\dagger}(\cdot)"/> 表示Hankel矩阵的逆算子。对于任何大小为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n\times (T-n+1)"/> 的矩阵 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}"/>，逆算子定义为：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;[\mathcal{H}^{\dagger}(\boldsymbol{Y})]_{t}=\frac{1}{\rho_{t}}\sum_{a+b=t+1}y_{a,b}"/></p>

其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t\in\{1,2,\cdots, T\}"/>。<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\rho_t"/> 是 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n\times (T-n+1)"/> 矩阵的第 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> 条反对角线上的元素数量，满足：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\rho_t=\begin{cases} t, & t\leq\min\{n, T-n+1\} \\ T-t+1, & \text{否则} \end{cases}"/></p>

根据上述公式，可以很容易地将Hankel分解与循环卷积和离散傅里叶变换联系起来（[Cai et al., 2019](https://arxiv.org/abs/1910.05859); [Cai et al., 2022](https://arxiv.org/abs/2204.03316)），即：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\tilde{x}_{t}=\frac{1}{\rho_t}\sum_{r=1}^{R}[\tilde{\boldsymbol{w}}_r\star\tilde{\boldsymbol{q}}_r]_{t}=\frac{1}{\rho_t}\sum_{r=1}^{R}[\mathcal{F}^{-1}(\mathcal{F}(\tilde{\boldsymbol{w}}_r)\circ\mathcal{F}(\tilde{\boldsymbol{q}}_r))]_{t}"/></p>

其中我们定义了两个向量：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{cases} \tilde{\boldsymbol{w}}_{r}=(w_{1,r},w_{2,r},\cdots,w_{t,r})^\top \\ \tilde{\boldsymbol{q}}_{r}=(q_{1,r},q_{2,r},\cdots,q_{t,r})^\top \end{cases}"/></p>

长度为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>。符号 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;[\cdot]_{t}"/> 表示向量的第 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> 个元素。值得注意的是，它们与向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}_r\in\mathbb{R}^{n}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{q}_r\in\mathbb{R}^{T-n+1}"/> 不同。如果 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t\leq n"/>，则向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\boldsymbol{w}}_r"/> 由 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}_r"/> 的前 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> 个元素组成。如果 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t> n"/>，则向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\boldsymbol{w}}_r"/> 由向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}_r"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t-n"/> 个零组成，即：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\boldsymbol{w}}_r=(w_{1,r},w_{2,r},\cdots,w_{n,r},\underbrace{0,\cdots,0}_{t-n})^\top\in\mathbb{R}^{t}"/></p>

这一原理非常适合向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\boldsymbol{q}}_r"/> 的构造。为了计算每个 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{x}_t,\forall t\in\{1,2,\ldots,n\}"/>，上述循环卷积的时间复杂度为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{O}(R\cdot t\log t)"/>，逐元素乘法的时间复杂度为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(R\cdot t)"/>。

<br>

---

<span style="color:gray">
<b>示例 5.</b> 给定向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,\cdots,x_6)^\top\in\mathbb{R}^{6}"/>，设 Hankel 矩阵的行数为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;4"/>，则 Hankel 矩阵为
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{H}(\boldsymbol{x})=\begin{bmatrix} x_1 & x_2 & x_3 \\ x_2 & x_3 & x_4 \\ x_3 & x_4 & x_5 \\ x_4 & x_5 & x_6 \end{bmatrix}"/></p>

<span style="color:gray">
如果对其进行秩一近似，使得 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{H}(\boldsymbol{x})\approx\boldsymbol{w}\boldsymbol{q}^\top"/>，其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}\in\mathbb{R}^{4}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{q}\in\mathbb{R}^{3}"/>，则 Hankel 矩阵分解的逆可以写成如下形式：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \tilde{x}_1&=\frac{1}{1}\sum_{a+b=2}w_aq_b=w_1q_1 \\ \tilde{x}_2&=\frac{1}{2}\sum_{a+b=3}w_aq_b=\frac{1}{2}(w_1q_2+w_2q_1) \\ &=\frac{1}{2}[(w_1,w_2)\star(q_1,q_2)]_{2} \\ \tilde{x}_3&=\frac{1}{3}\sum_{a+b=4}w_aq_b=\frac{1}{3}(w_1q_3+w_2q_2+w_3q_1) \\ &=\frac{1}{3}[(w_1,w_2,w_3)\star(q_1,q_2,q_3)]_{3} \\ \tilde{x}_4&=\frac{1}{3}\sum_{a+b=5}w_aq_b=\frac{1}{3}(w_2q_3+w_3q_2+w_4q_1) \\ &=\frac{1}{3}[(w_1,w_2,w_3,w_4)\star(q_1,q_2,q_3,0)]_{4} \\ \tilde{x}_5&=\frac{1}{2}\sum_{a+b=6}w_aq_b=\frac{1}{2}(w_3q_3+w_4q_2) \\ &=\frac{1}{2}[(w_1,w_2,w_3,w_4,0)\star(q_1,q_2,q_3,0,0)]_{5} \\ \tilde{x}_6&=\frac{1}{1}\sum_{a+b=7}w_aq_b=w_4q_3 \\ &=[(w_1,w_2,w_3,w_4,0,0)\star(q_1,q_2,q_3,0,0,0)]_{6} \end{aligned}"/></p>

<span style="color:gray">
这可以转化为循环卷积。通过这种方式，计算过程可以通过快速傅里叶变换实现。
</span>

<br>

---

<b>致谢.</b> 感谢 @<a href='https://github.com/yangjm67'>Jinming Yang</a> 纠正了本示例中循环卷积的符号错误。

---

<br>

---

<span style="color:gray">
<b>时间复杂度.</b> 给定 Hankel 矩阵分解公式为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\boldsymbol{x}}=\mathcal{H}^{\dagger}(\boldsymbol{w}\boldsymbol{q}^\top)"/>，其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}\in\mathbb{R}^{n}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{q}\in\mathbb{R}^{n}"/>，假设 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T=2n-1"/>，计算向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\boldsymbol{x}}"/> 的逐元素乘法操作次数为

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;1+2+\cdots+n=\frac{n(n-1)}{2}"/></p>

<span style="color:gray">
导致时间复杂度为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{O}(n^2)"/>。根据示例 5 中的 Hankel 矩阵分解逆运算，操作次数为
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;2(1+2+\cdots+\operatorname{floor}(\frac{n}{2}))=\operatorname{floor}(\frac{n}{2})(1+\operatorname{floor}(\frac{n}{2}))"/></p>

<span style="color:gray">
如果 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> 为奇数。否则，操作次数为
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;2(1+2+\cdots+\frac{n}{2})-\frac{n}{2}=\frac{n}{2}(1+\frac{n}{2})-\frac{n}{2}"/></p>

---

<br>

图 5 展示了分别使用逐元素乘法和循环卷积的 Hankel 矩阵分解逆运算的经验时间复杂度。如果使用快速傅里叶变换进行循环卷积，则逆运算的计算成本大约是逐元素乘法的 100 倍。结果表明，在这种情况下，逐元素乘法比循环卷积更高效。

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/empirical_time_complexity_hankel.png" width="550" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>图 5.</b> Hankel 矩阵分解逆运算 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\boldsymbol{x}}=\mathcal{H}^{\dagger}(\boldsymbol{w}\boldsymbol{q}^\top)"/> 的经验时间复杂度，其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}\in\mathbb{R}^{n}"/> 和 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{q}\in\mathbb{R}^{T-n+1}"/>。注意，我们将向量长度设置为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n\in\{2^5,2^6,\ldots,2^{13}\}"/> 且 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T=2n-1"/>。我们对每个向量长度重复了 100 次数值实验。
</p>

<br>

```python
import numpy as np

def inverse_hankel(w, q, fft = False):
    dim1 = w.shape[0]
    dim2 = q.shape[0]
    dim = min(dim1, dim2)
    T = dim1 + dim2 - 1
    x_tilde = np.zeros(T)
    for t in range(T):
        w_new = np.zeros(t + 1)
        if t < dim1:
            w_new[: t + 1] = w[: t + 1]
        elif t >= dim1:
            w_new[: dim1] = w
        q_new = np.zeros(t + 1)
        if t < dim2:
            q_new[: t + 1] = q[: t + 1]
        elif t >= dim2:
            q_new[: dim2] = q
        if t < dim:
            rho = t + 1
        else:
            rho = T - (t + 1) + 1
        if fft == True:
            vec = np.fft.ifft(np.fft.fft(w_new) * np.fft.fft(q_new)).real
            x_tilde[t] = vec[t] / rho
        elif fft == False:
            x_tilde[t] = np.inner(w_new, np.flip(q_new)) / rho
    return x_tilde
```

<br>

完整实现请查看 [附录](https://spatiotemporal-data.github.io/posts/ts_conv_supp/)。

<br>

---

<span style="color:gray">
<b>示例 6.</b> 对于任意向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/>，始终成立以下等式：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\mathcal{H}(\mathcal{D}(\boldsymbol{x}))\|_F=\|\boldsymbol{x}\|_2"/></p>

<span style="color:gray">
其中，运算符 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{D}(\cdot)"/> 定义如下：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{D}(\boldsymbol{x})=\bigr(x_1,\frac{1}{\sqrt{\rho_2}}x_2,\frac{1}{\sqrt{\rho_3}}x_3,\cdots,\frac{1}{\sqrt{\rho_{T-1}}}x_{T-1},x_T\bigl)^\top\in\mathbb{R}^{T}"/></p>

<span style="color:gray">
验证此性质在向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,\cdots,x_6)^\top"/> 上是否成立，设 Hankel 矩阵的行数为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;4"/>。
</span>

<br>

<span style="color:gray">
根据定义，我们有：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{D}(\boldsymbol{x})=\bigr(x_1,\frac{1}{\sqrt{2}}x_2,\frac{1}{\sqrt{3}}x_3,\frac{1}{\sqrt{3}}x_4,\frac{1}{\sqrt{2}}x_{5},x_6\bigl)^\top"/></p>

<span style="color:gray">
Hankel 矩阵为：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{H}(\mathcal{D}(\boldsymbol{x}))=\begin{bmatrix} x_1 & \frac{1}{\sqrt{2}}x_2 & \frac{1}{\sqrt{3}}x_3 \\ \frac{1}{\sqrt{2}}x_2 & \frac{1}{\sqrt{3}}x_3 & \frac{1}{\sqrt{3}}x_4 \\ \frac{1}{\sqrt{3}}x_3 & \frac{1}{\sqrt{3}}x_4 & \frac{1}{\sqrt{2}}x_5 \\ \frac{1}{\sqrt{3}}x_4 & \frac{1}{\sqrt{2}}x_5 & x_6 \end{bmatrix}"/></p>

<span style="color:gray">
因此，该 Hankel 矩阵的 Frobenius 范数等于向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> 的 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_2"/>-范数。
</span>

---

<br>

## III. 循环矩阵核范数最小化

循环矩阵在信号处理和机器学习的许多计算和理论方面具有基础性作用，为实现各种算法（如循环矩阵核范数最小化）提供了高效的框架。根据定义，循环矩阵是一种特殊的方阵，其中每一行都是前一行向右移动一位的结果，最后一行的最后一个元素循环到第一行的第一个位置。正如我们之前讨论的循环矩阵，接下来我们将介绍循环矩阵核范数、其最小化问题及其应用。

### III-A. 定义

核范数是矩阵计算和凸优化中的一个关键概念，常用于低秩矩阵近似和补全问题。对于任意矩阵 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{m\times n}"/>，核范数定义为奇异值的和：

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{X}\|_{*}=\sum_{r=1}^{t}s_{r}"/></p>

其中 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\cdot\|_*"/> 表示核范数。如图 6 所示，奇异值为 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;s_1,s_2,\ldots, s_t"/>，且 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t=\min\{m,n\}"/>。

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/svd_explained.png" width="500" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>图 6.</b> 矩阵 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{m\times n}"/> 的奇异值分解。在分解后的矩阵中，酉矩阵 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}\in\mathbb{R}^{m\times t}"/>（或 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Q}\in\mathbb{R}^{n\times t}"/>）由 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> 个正交的左（或右）奇异向量组成，而 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}"/> 的 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> 个对角元素是奇异值，满足 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;s_1\geq s_2\geq\cdots\geq s_t\geq 0"/>。注意，为方便表示，<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t=\min\{m,n\}"/>。
</p>

<br>

---

<span style="color:gray">
<b>示例 7.</b> 给定向量 <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/>，其循环矩阵为：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} 0 & 4 & 3 & 2 & 1 \\ 1 & 0 & 4 & 3 & 2 \\ 2 & 1 & 0 & 4 & 3 \\ 3 & 2 & 1 & 0 & 4 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}"/></p>

<span style="color:gray">
因此，奇异值为：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{s}=(10, 4.25325404, 4.25325404, 2.62865556, 2.62865556)^\top"/></p>

<span style="color:gray">
因此，核范数为：
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\mathcal{C}(\boldsymbol{x})\|_{*}=\sum_{t=1}^{5}s_t=23.7638"/></p>

<span style="color:gray">
请使用以下 `numpy` 实现复现结果。
</span>

```python
import numpy as np

def circ_mat(vec):
    n = vec.shape[0]
    mat = np.zeros((n, n))
    mat[:, 0] = vec
    for i in range(1, n):
        mat[:, i] = np.append(vec[-i :], vec[: n - i], axis = 0)
    return mat

x = np.array([0, 1, 2, 3, 4])
mat = circ_mat(x)
w, s, q = np.linalg.svd(mat, full_matrices = False)
print('C(x) 的奇异值：')
print(s)
```

<br>

---

<br>

