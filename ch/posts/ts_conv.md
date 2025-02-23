---
layout: default
---

# 时间序列卷积

<p align="center"><span style="color:gray">一种用于增强时间序列趋势建模和时间模式挖掘的卷积核方法，可利用傅里叶变换进行快速计算。同时，可解释的机器学习模型（如稀疏回归）为更好地捕捉现实世界时间序列的长期变化和时间模式提供了途径。</span></p>

<br>

本文将讨论近期研究中的基本建模思路：

- Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). [Laplacian convolutional representation for traffic time series imputation](https://doi.org/10.1109/TKDE.2024.3419698). IEEE Transactions on Knowledge and Data Engineering. 36 (11): 6490-6502.
- Xinyu Chen, Xi-Le Zhao, Chun Cheng (2024). [Forecasting urban traffic states with sparse data using Hankel temporal matrix factorization](https://doi.org/10.1287/ijoc.2022.0197). INFORMS Journal on Computing.
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

