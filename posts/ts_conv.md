---
layout: default
---

# Time Series Convolution

<p align="center"><span style="color:gray">A convolutional kernel approach for reinforcing the modeling of time series trends and interpreting temporal patterns, allowing one to leverage Fourier transforms and learn sparse representations.</span></p>

<br>

In this post, we intend to explain the essential ideas of our latent research work:

- Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). [Laplacian convolutional representation for traffic time series imputation](https://doi.org/10.1109/TKDE.2024.3419698). IEEE Transactions on Knowledge and Data Engineering. Early Access.


Content:
- Motivation
- Preliminaries
- Prior art on global trend modeling with circulant matrix nuclear norm minimization
- Local trend modeling with circular convolution
- LCR: A unified optimization with an efficient implementation
- Python implementation with `numpy`
- Univariate traffic volume/speed imputation
- Multivariate speed field reconstruction

<br>

## I. Motivation

Convolution has been widely used to machine learning problems. We hope to revisit the essential ideas of circular convolution and lays an insightful foundation for modeling time series data.


Though nowadays we have a lot of machine learning algorithms on hand, it is still necessary to address the following challenges:

- How to characterize global time series trends?
- How to characterize local time series trends?
- How to learn interpretable local and nonlocal patterns as convolutional kernels?

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/time_series_global_trends.png" width="350" />
</p>

<p align = "center">
(a) Global trends
</p>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/time_series_local_trends.png" width="350" />
</p>

<p align = "center">
(b) Local trends
</p>

<p align = "center">
<b>Figure 1.</b> Illustration of time series trends.
</p>


<br>

## II. Preliminaries

In this study, we build the modeling concepts of Laplacian convolutional representation (LCR) upon several key ideas from the fields of signal processing and machine learning, including circular convolution, discrete Fourier transform, and fast Fourier transform. In the following sections, we will discuss: 

- What are circular convolution, convolution matrix, and circulant matrix?
- What is the convolution theorem?
- How can fast Fourier transform be used to compute the circular convolution?

### A. Circular Convolution

Convolution is a powerful operation in many classical deep learning frameworks, such as convolutional neural networks (CNNs). In the context of discrete sequences (typically vectors), circular convolution refers to the convolution of two discrete sequences of data, and it plays an important role in maximizing the efficiency of certain common filtering operations (see [circular convolution](https://en.wikipedia.org/wiki/Circular_convolution) on Wikipedia).

By definition, for any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,\cdots,x_T)^\top\in\mathbb{R}^{T}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(y_1,y_2,\cdots,y_\tau)^\top\in\mathbb{R}^{\tau}"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau\leq T"/>, the circular convolution (denoted by the symbol <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\star"/>) of these two vectors is formulated as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}\in\mathbb{R}^{T}"/></p>

Elment-wise, we have

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;z_{t}=\sum_{k=1}^{\tau}x_{t-k+1} y_{k},\,\forall t\in\{1,2,\ldots,T\}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_t"/> represents the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>th entry of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>. For a cyclical operation, it takes <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{t-k+1}=x_{t-k+1+T}"/> when <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t+1\leq k"/>. While the definition of circular convolution might seem complicated for beginners, it becomes much clearer when you consider the concept of a convolution or circulant matrix, as demonstrated in the examples provided below.

As mentioned above, the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> has a length of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/>, and the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> has a length of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/>. According to the definition of circular convolution, the resulting vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> will also have a length of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/>, matching the length of the original vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>.

Given any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,x_3,x_4)^\top"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(y_1,y_2,y_3)^\top"/>, the circular convolution between them can be expressed as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\begin{bmatrix}
\displaystyle\sum_{k=1}^{3}x_{1-k+1}y_k \\
\displaystyle\sum_{k=1}^{3}x_{2-k+1}y_k \\
\displaystyle\sum_{k=1}^{3}x_{3-k+1}y_k \\
\displaystyle\sum_{k=1}^{3}x_{4-k+1}y_k \\
\end{bmatrix}
=\begin{bmatrix}
x_1y_1+x_4y_2+x_3y_3 \\ x_2y_1+x_1y_2+x_4y_3 \\ x_3y_1+x_2y_2+x_1y_3 \\ x_4y_1+x_3y_2+x_2y_3 \\
\end{bmatrix} \\"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{0}=x_4"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{-1}=x_3"/> according to the definition.

In this case, each entry <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_t"/> of the resulting vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> is computed as the inner product between the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> and a reversed and truncated version of the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>. Specifically, the entry <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_1"/> is obtained by computing the inner product between <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;(x_1,x_4,x_3)^\top"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/>. For subsequent entries <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_2,z_3,z_4"/>, the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> is cyclically shifted in reverse and truncated (only preserving the first <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/> entries), and the inner product with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> is calculated. Figure 2 illustrates the basic steps for computing each entry of the circular convolution.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/circular_convolution_steps.png" width="450" />
</p>

<p align = "center">
<b>Figure 2.</b> Illustration of the circular convolution between <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,x_3,x_4)^\top"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(y_1,y_2,y_3)^\top"/>. (a) Computing <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_1"/> involves <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{0}=x_4"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{-1}=x_3"/>. (b) Computing <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_2"/> involves <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{0}=x_4"/>. The figure inspired by [Prince (2023)](https://udlbook.github.io/udlbook/).
</p>

<br>

As can be seen, circular convolution between two vectors can essentially be viewed as a linear operation. This perspective allows one to reformulate the circular convolution as a linear transformation using a convolution matrix (or a circulant matrix when these two vectors have the same length).

<br>

---

<span style="color:gray">
**Example 1.** Given vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(2,-1,3)^\top"/>, the circular convolution <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=(5,14,3,7,11)^\top"/> has the following entries:</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{cases} z_1=x_1y_1+x_5y_2+x_4y_3=0\times 2+4\times(-1)+3\times 3=5 \\ z_2=x_2y_1+x_1y_2+x_5y_3=1\times 2+0\times(-1)+4\times 3=14 \\ z_3=x_3y_1+x_2y_2+x_1y_3=2\times 2+1\times (-1)+0\times 3=3 \\ z_4=x_4y_1+x_3y_2+x_2y_3=3\times 2+2\times (-1)+1\times 3=7 \\ z_5=x_5y_1+x_4y_2+x_3y_3=4\times 2+3\times (-1)+2\times 3=11 \end{cases}"/></p>

<span style="color:gray">
where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{0}=x_5"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{-1}=x_4"/> according to the definition.
</span>

---

<br>

### B. Convolution Matrix

Using the notations above, for any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{\tau}"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau<T"/>, the circular convolution can be expressed as a linear transformation:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}_{\tau}(\boldsymbol{x})\boldsymbol{y}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}:\mathbb{R}^{T}\to\mathbb{R}^{T\times \tau}"/> denotes the convolution operator. The convolution matrix can be represented as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}(\boldsymbol{x})=\begin{bmatrix} x_1 & x_T & x_{T-1} & \cdots & x_{T-\tau+2} \\ x_2 & x_1 & x_{T} & \cdots & x_{T-\tau+3} \\ x_3 & x_2 & x_1 & \cdots & x_{T-\tau+4} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ x_{T} & x_{T-1} & x_{T-2} & \cdots & x_{T-\tau+1} \\ \end{bmatrix}\in\mathbb{R}^{T\times\tau}"/></p>

In signal processing, this linear transformation is a fundamental property of circular convolution, highlighting its role in efficently implementing filtering operations.

<br>

---

<span style="color:gray">
**Example 2.** Given vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(2,-1,3)^\top"/>, the circular convolution <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}"/> can be expressed as:
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}_{3}(\boldsymbol{x})\boldsymbol{y}=(5,14,3,7,11)^\top"/></p>

<span style="color:gray">
where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{3}(\boldsymbol{x})"/> is the convolution matrix with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau=3"/> columns. Specifically, the convolution matrix is structured as follows,
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{3}(\boldsymbol{x})=\begin{bmatrix} 0 & 4 & 3 \\ 1 & 0 & 4 \\ 2 & 1 & 0 \\ 3 & 2 & 1 \\ 4 & 3 & 2 \end{bmatrix}"/></p>

<span style="color:gray">
As a result, it gives</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\mathcal{C}_{3}(\boldsymbol{x})\boldsymbol{y}=\begin{bmatrix} 0 & 4 & 3 \\ 1 & 0 & 4 \\ 2 & 1 & 0 \\ 3 & 2 & 1 \\ 4 & 3 & 2 \end{bmatrix}\begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}=\begin{bmatrix} 5 \\ 14 \\ 3 \\ 7 \\ 11 \end{bmatrix}"/></p>

<span style="color:gray">
This representation shows that the circular convolution is equivalent to a matrix-vector multiplication, making it easier to understand, especially in signal processing applications.
</span>

---

<br>

In this post, we aim to make the concepts clear and accessible by incorporting programming code, intuitive illustrations, and detailed explanations of the formulas. To demonstrate how circular convolution can be computed, we will use Python's `numpy` library. First, we will construct the convolution matrix on the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> and then perform the circular convolution as follows.

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

### C. Circulant Matrix

Recall that the convolution matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}(\boldsymbol{x})"/> is specified with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/> columns, corresponding to the length of vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/>). In the case of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^{T}"/> with same columns, the convolution matrix becoms a square matrix, known as a **circulant matrix**. In this study, we emphasize the importance of circulant matrices and their properties, such as their strong connection with circular convolution and discrete Fourier transform, even through we do not work directly with circulant matrices.

For any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^{T}"/>, the circular convolution can be expressed as a linear transformation such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}(\boldsymbol{x})\boldsymbol{y}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}:\mathbb{R}^{T}\to\mathbb{R}^{T\times T}"/> denotes the circulant operator. The circulant matrix is defined as:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} x_1 & x_T & x_{T-1} & \cdots & x_{2} \\ x_2 & x_1 & x_{T} & \cdots & x_{3} \\ x_3 & x_2 & x_1 & \cdots & x_{4} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ x_{T} & x_{T-1} & x_{T-2} & \cdots & x_{1} \\ \end{bmatrix}\in\mathbb{R}^{T\times T}"/></p>
which forms a square matrix.

<br>

---

<span style="color:gray">
**Example 3.** Given vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(2,-1,3,0,0)^\top"/>, the circular convolution <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}"/> is identical to
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}(\boldsymbol{x})\boldsymbol{y}=(5,14,3,7,11)^\top"/></p>

<span style="color:gray">
where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})"/> is the circulant matrix formed from <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>, defined as:
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} 0 & 4 & 3 & 2 & 1 \\ 1 & 0 & 4 & 3 & 2 \\ 2 & 1 & 0 & 4 & 3 \\ 3 & 2 & 1 & 0 & 4 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}"/></p>

<span style="color:gray">
Thus, the result can be written as follows,
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\mathcal{C}(\boldsymbol{x})\boldsymbol{y}=\begin{bmatrix} 0 & 4 & 3 & 2 & 1 \\ 1 & 0 & 4 & 3 & 2 \\ 2 & 1 & 0 & 4 & 3 \\ 3 & 2 & 1 & 0 & 4 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}\begin{bmatrix} 2 \\ -1 \\ 3 \\ 0 \\ 0 \end{bmatrix}=\begin{bmatrix} 5 \\ 14 \\ 3 \\ 7 \\ 11 \end{bmatrix}"/></p>

<span style="color:gray">
The example shows that the circular convolution of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;(2,-1,3)^\top"/> is equivalent to the circular convolution of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> with its last two entries padded with zeros. Thus, to compute the circular convolution of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{\tau}"/> when <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> is greater than <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/>, one can simply append <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T-\tau"/> zeros to the end of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> and perform the circular convolution.
</span>

---

<br>

### D. Discrete Fourier Transform

Discrete Fourier transform (see [Wikipedia](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)) is a fundamental tool in mathematics and signal processing with widespread applications to machine learning. The discrete Fourier transform is the key discrete transform used for Fourier analysis, enabling the decomposition of a signal into its constituent frequencies. The fast Fourier transform is an efficient algorithm for computing the discrete Fourier tranform (see the [difference between discrete Fourier transform and fast Fourier transform](https://math.stackexchange.com/q/30464/738418)), significantly reducing the time complexity from <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{O}(T^2)"/> to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{O}(T\log T)"/>, where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> is the number of data points. This efficiency makes fast Fourier transform essential for processing large problems.

A crucial concept in signal processing is the convolution theorem, which states that convolution in the time domain is the multiplication in the frequency domain. This implies that the circular convolution can be efficiently computed using the fast Fourier transform. The convolution theorem for discrete Fourier transform is summarized as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{F}(\boldsymbol{x}\star\boldsymbol{y})=\mathcal{F}(\boldsymbol{x})\circ\mathcal{F}(\boldsymbol{y})"/></p>

or

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\star\boldsymbol{y}=\mathcal{F}^{-1}(\mathcal{F}(\boldsymbol{x})\circ\mathcal{F}(\boldsymbol{y}))"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{F}(\cdot)"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{F}^{-1}(\cdot)"/> denote the discrete Fourier transform and the inverse discrete Fourier transform, respectively. The symbol <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\circ"/> represents the Hadamard product, i.e., element-wise multiplication.

In fact, this principle underlies many efficient algorithms in signal processing and data analysis, allowing complex operations to be performed efficiently in the frequency domain.

<br>

---

<span style="color:gray">
**Example 4.** Given vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(2,-1,3,0,0)^\top"/>, the circular convolution <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}"/> can be computed via the use of fast Fourier transform in `numpy` as follows,
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

in which the outputs are

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

## III. Circulant Matrix Nuclear Norm

Circulant matrices are fundamental in many computational and theoretical aspects of signal processing and machine learning, providing an efficient framework for implementating various algorithms such as circulant matrix nuclear norm minimization. By definition, a circulant matrix is a spcial square matrix where which shifts the previous row to the right by one position, with the last entry wrapping around to the first position. As we already discussed the circulant matrix above, we will present the circulant matrix nuclear norm, its minimization problem, and applications.


### A. Definition

Nuclear norm is a key concept in matrix computations and convex optimization, frequently applied in low-rank matrix approximation and completion problems. For any matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{m\times n}"/>, the nuclear norm is defined as the sum of singular values:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{X}\|_{*}=\sum_{r=1}^{t}s_{r}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\cdot\|_*"/> denotes the nuclear norm. As illustrated in Figure 3, the singular values are <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;s_1,s_2,\ldots, s_t"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t=\min\{m,n\}"/>.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/svd_explained.png" width="500" />
</p>

<p align = "center">
<b>Figure 3.</b> Singular value decomposition of matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{m\times n}"/>. In the decomposed matrices, the unitary matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}\in\mathbb{R}^{m\times t}"/> (or <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Q}\in\mathbb{R}^{n\times t}"/>) consists of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> orthogonal left (or right) singular vectors, while the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> diagonal entries of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}"/> are singular values such that <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;s_1\geq s_2\geq\cdots\geq s_t\geq 0"/>. Note that <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t=\min\{m,n\}"/> for notational convenience. 
</p>

<br>

---

<span style="color:gray">
**Example 5.** Given vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/>, the circulant matrix is
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} 0 & 4 & 3 & 2 & 1 \\ 1 & 0 & 4 & 3 & 2 \\ 2 & 1 & 0 & 4 & 3 \\ 3 & 2 & 1 & 0 & 4 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}"/></p>

<span style="color:gray">
Thus, the singular values are
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{s}=(10, 4.25325404, 4.25325404, 2.62865556, 2.62865556)^\top"/></p>

<span style="color:gray">
As a result, we have the nuclear norm as follows,
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\mathcal{C}(\boldsymbol{x})\|_{*}=\sum_{t=1}^{5}s_t=23.7638"/></p>

<span style="color:gray">
Please reproduce the results by using the following `numpy` implementation.
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
print('Singular values of C(x):')
print(s)
```

<br>

---

<br>

### B. Property

One of the most intriguing properties of circulant matrices is that they are diagonalizable by the discrete Fourier transform matrix. The eigenvalue decomposition of circulant matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})\in\mathbb{R}^{T\times T}"/> (constructed from any vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^T"/>) is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\boldsymbol{F}\operatorname{diag}(\mathcal{F}(\boldsymbol{x}))\boldsymbol{F}^H"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{F}"/> is the unitary discrete Fourier transform matrix, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{F}^H"/> is the Hermitian transpose of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{F}"/>, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{F}(\boldsymbol{x})"/> is a diagonal matrix containing the eigenvalues of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})"/>. Due to this property, the nuclear norm of the circulant matrix can be formulated as the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm of the discrete Fourier transform of the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \|\mathcal{C}(\boldsymbol{x})\|_*=&\|\boldsymbol{F}\operatorname{diag}(\mathcal{F}(\boldsymbol{x}))\boldsymbol{F}^H\|_{*} \\ =&\|\operatorname{diag}(\mathcal{F}(\boldsymbol{x}))\|_* \\ =&\|\mathcal{F}(\boldsymbol{x})\|_1 \end{aligned}"/></p>

This relationship draws the strong connection between circulant matrices and Fourier analysis, enabling efficient computation and analysis in various applications.

<br>

---

<span style="color:gray">
**Example 6.** Given vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/>, the circulant matrix is
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} 0 & 4 & 3 & 2 & 1 \\ 1 & 0 & 4 & 3 & 2 \\ 2 & 1 & 0 & 4 & 3 \\ 3 & 2 & 1 & 0 & 4 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}"/></p>

<span style="color:gray">
The eigenvalues of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})"/> and the fast Fourier transform of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> can be computed using `numpy` as follows.
</span>

```python
import numpy as np

x = np.array([0, 1, 2, 3, 4])
mat = circ_mat(x)
eigval, eigvec = np.linalg.eig(mat)
print('Eigenvalues of C(x):')
print(eigval)
print('Fast Fourier transform of x:')
print(np.fft.fft(x))
```

<span style="color:gray">
in which the outputs are
</span>

```python
Eigenvalues of C(x):
[10. +0.j         -2.5+3.4409548j  -2.5-3.4409548j  -2.5+0.81229924j
 -2.5-0.81229924j]
Fast Fourier transform of x:
[10. +0.j         -2.5+3.4409548j  -2.5+0.81229924j -2.5-0.81229924j
 -2.5-3.4409548j ]
```

<span style="color:gray">
In this case, the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm of the complex valued <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{F}(\boldsymbol{x})=(a_1+b_1i, a_2+b_2i, \cdots, a_5+b_5i)^\top"/>---the imaginary unit is defined as <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;i=\sqrt{-1}"/>---is given by
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\mathcal{C}(\boldsymbol{x})\|_{*}=\|\mathcal{F}(\boldsymbol{x})\|_1=\sum_{t=1}^{5}|a_t+b_ti|=\sum_{t=1}^{5}\sqrt{a_t^2+b_t^2}=23.7638"/></p>

---

<br>

### C. Optimization

For any partially observed time series in the form of a vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^T"/> with the observed index set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega"/>, solving the circulant matrix nuclear norm minimization allows one to reconstruct missing values in time series. The problem is formulated as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&\|\mathcal{C}(\boldsymbol{x})\|_* \\ \text{s.t.}\,&\|\mathcal{P}_{\Omega}(\boldsymbol{x}-\boldsymbol{y})\|_2\leq\epsilon \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\epsilon"/> in the constraint represents the tolerance of errors between the reconstructed time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> and the partially observed time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/>.

In this case, one can rewrite the constraint as a penalty term (weighted by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\gamma"/>) in the objective function:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \min_{\boldsymbol{x}}\,\|\mathcal{C}(\boldsymbol{x})\|_*+\frac{\gamma}{2}\|\mathcal{P}_{\Omega}(\boldsymbol{x}-\boldsymbol{y})\|_2^2"/></p>

Since the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> is associated with a circulant matrix nuclear norm and a penalty term, the first impluse is using variable splitting to convert the problem into the following one:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&\|\mathcal{C}(\boldsymbol{x})\|_*+\frac{\gamma}{2}\|\mathcal{P}_{\Omega}(\boldsymbol{z}-\boldsymbol{y})\|_2^2 \\ \text{s.t.}\,&\boldsymbol{x}=\boldsymbol{z} \end{aligned}"/></p>

<br>

### D. Solution Algorithm

The augmented Lagrangian function is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})=\|\mathcal{C}(\boldsymbol{x})\|_*+\frac{\gamma}{2}\|\mathcal{P}_{\Omega}(\boldsymbol{z}-\boldsymbol{y})\|_2^2+\frac{\lambda}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_2^2+\langle\boldsymbol{w},\boldsymbol{x}-\boldsymbol{z}\rangle"/></p>

where the dual variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}\in\mathbb{R}^{T}"/> is an estimate of the Lagrange multiplier, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\lambda"/> is a penalty parameter that controls the convergence rate.

Thus, the variables <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>, and the dual variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> can be updated iteratively as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{cases} \displaystyle\boldsymbol{x}:=\arg\min_{\boldsymbol{x}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ \displaystyle\boldsymbol{z}:=\arg\min_{\boldsymbol{z}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ \boldsymbol{w}:=\boldsymbol{w}+\lambda(\boldsymbol{x}-\boldsymbol{z}) \end{cases}"/></p>

<br>

### E. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>-Subproblem

Solving the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> involves discrete Fourier transform, convolution theorem, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm minimization in complex space. The optimization problem with respect to the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \boldsymbol{x}:=&\arg\min_{\boldsymbol{x}}\,\|\mathcal{C}(\boldsymbol{x})\|_{*}+\frac{\lambda}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_2^2+\langle\boldsymbol{w},\boldsymbol{x}\rangle \\ =&\arg\min_{\boldsymbol{x}}\,\|\mathcal{C}(\boldsymbol{x})\|_{*}+\frac{\lambda}{2}\|\boldsymbol{x}-\boldsymbol{z}+\boldsymbol{w}/\lambda\|_2^2 \end{aligned}"/></p>

Using discrete Fourier transform, the optimization problem can be converted into the following one in complex space:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \hat{\boldsymbol{x}}:=\arg\min_{\hat{\boldsymbol{x}}}\,\|\hat{\boldsymbol{x}}\|_{1}+\frac{\lambda}{2T}\|\hat{\boldsymbol{x}}-\hat{\boldsymbol{z}}+\hat{\boldsymbol{w}}/\lambda\|_2^2 "/></p>

where the complex-valued variables <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\{\hat{\boldsymbol{x}},\hat{\boldsymbol{z}},\hat{\boldsymbol{w}}\}=\{\mathcal{F}(\boldsymbol{x}),\mathcal{F}(\boldsymbol{z}),\mathcal{F}(\boldsymbol{w})\}"/> refer to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\{\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}\}"/> in the frequency domain. The closed-form solution to the complex-valued variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\hat{\boldsymbol{x}}"/> is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \hat{x}_t:=\frac{\hat{h}_t}{|\hat{h}_t|}\cdot\max\{0,|\hat{h}_t|-T/\lambda\},\,t=1,2,\ldots, T "/></p>

with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\hat{h}_t=\hat{z}_t-\hat{w}_t/\lambda"/>. For reference, this closed-form solution can be found in Lemma 3.3 in [Yang et al., (2009)](https://doi.org/10.1137/080730421), see Eq. (3.18) and (3.19) for discussing real-valued variables. In the sparsity-induced norm optimization of machine learning, this closed-form solution is also called as proximal operator or shrinkage operator.

<br>

```python
import numpy as np

def update_x(z, w, lmbda):
    T = z.shape[0]
    h = np.fft.fft(z - w / lmbda)
    temp = 1 - T / (lmbda * np.abs(h))
    temp[temp <= 0] = 0
    return np.fft.ifft(h * temp).real
```


<br>

### F. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>-Subproblem

In terms of the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>, the partial derivative of the augmented Lagrangian function with respect to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{P}_{\Omega}(\boldsymbol{z})"/> is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\mathcal{P}_{\Omega}(\boldsymbol{z})}=&\gamma\mathcal{P}_{\Omega}(\boldsymbol{z}-\boldsymbol{y})+\lambda\mathcal{P}_{\Omega}(\boldsymbol{z}-\boldsymbol{x})-\mathcal{P}_{\Omega}(\boldsymbol{w}) \\ =&(\gamma+\lambda)\mathcal{P}_{\Omega}(\boldsymbol{z})-\mathcal{P}_{\Omega}(\gamma\boldsymbol{y}+\lambda\boldsymbol{x}+\boldsymbol{w}) \end{aligned}"/></p>

while

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{z})}=&\lambda\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{z}-\boldsymbol{x})-\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{w}) \\ =&\lambda\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{z})-\mathcal{P}_{\Omega}^{\perp}(\lambda\boldsymbol{x}+\boldsymbol{w}) \end{aligned}"/></p>

As a result, letting the partial derivative of the augmented Lagrangian function with respect to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> be a zero vector, the least squares solution is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \boldsymbol{z}:=&\Bigl\{\boldsymbol{z}\mid \frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\mathcal{P}_{\Omega}(\boldsymbol{z})}+\frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{z})}=\boldsymbol{0}\Bigr\} \\ =&\frac{1}{\gamma+\lambda}\mathcal{P}_{\Omega}(\gamma\boldsymbol{y}+\lambda\boldsymbol{x}+\boldsymbol{w})+\frac{1}{\lambda}\mathcal{P}_{\Omega}^{\perp}(\lambda\boldsymbol{x}+\boldsymbol{w}) \end{aligned}"/></p>

<br>

```python
import numpy as np

def update_z(y_train, pos_train, x, w, lmbda, gamma):
    z = x + w / lmbda
    z[pos_train] = (gamma * y_train + lmbda * z[pos_train]) / (gamma + lmbda) 
    return z
```


<br>

### G. Time Series Imputation

As shown in Figure 4, we randomly remove 95% observations as missing values, and we only have 14 volume observations (i.e., 14 blue dots) for the reconstruction. The circulant matrix nuclear norm minimization can capture the global trends from partial observations.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/circnnm_volumes_95.png" width="350" />
</p>

<p align = "center">
<b>Figure 4.</b> Univariate time series imputation on the freeway traffic volume time series. The blue and red curves correspond to the ground truth time series and reconstructed time series achieved by the circulant matrix nuclear norm minimization.
</p>

<br>

<span style="color:gray">
Please reproduce the experiments by following the [Jupyter Notebook](https://github.com/xinychen/LCR/blob/main/univariate-models/CircNNM.ipynb), which is available at the [LCR repository](https://github.com/xinychen/LCR) on GitHub. For the supplementary material, please check out [Appendix I(A)](https://spatiotemporal-data.github.io/posts/ts_conv_supp/).
<span>

<br>



## IV. Laplacian Convolutional Representation

## V. Learning Interpretable Convolutional Kernels


<br>
<br>
<br>


## Python Implementation

We implement the LCR model with `numpy`...

<br>

```python
import numpy as np

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def laplacian(n, tau):
    ell = np.zeros(n)
    ell[0] = 2 * tau
    for k in range(tau):
        ell[k + 1] = -1
        ell[-k - 1] = -1
    return ell

def prox(z, w, lmbda, denominator):
    T = z.shape[0]
    temp1 = np.fft.fft(lmbda * z - w) / denominator
    temp2 = 1 - T / (denominator * np.abs(temp1))
    temp2[temp2 <= 0] = 0
    return np.fft.ifft(temp1 * temp2).real

def update_z(y_train, pos_train, x, w, lmbda, eta):
    z = x + w / lmbda
    z[pos_train] = (lmbda / (lmbda + eta) * z[pos_train]
                    + eta / (lmbda + eta) * y_train)
    return z

def update_w(x, z, w, lmbda):
    return w + lmbda * (x - z)

def flip_vec(x):
    return np.append(x, np.flip(x))

def inv_flip_vec(vec):
    dim = vec.shape[0]
    T = int(dim / 2)
    return (vec[: T] + np.flip(vec[T :])) / 2

def LCR(y_true, y, lmbda, gamma, tau, maxiter = 50):
    eta = 100 * lmbda
    data_true = flip_vec(y_true)
    data = flip_vec(y)
    T = data.shape
    pos_train = np.where(data != 0)
    data_train = data[pos_train]
    pos_test = np.where((y_true != 0) & (y == 0))
    y_test = y_true[pos_test]
    z = data.copy()
    w = data.copy()
    ell = np.fft.fft(laplacian(T, tau))
    denominator = lmbda + gamma * np.abs(ell) ** 2
    del y_true, y
    show_iter = 100
    for it in range(maxiter):
        x = prox(z, w, lmbda, denominator)
        z = update_z(data_train, pos_train, x, w, lmbda, eta)
        w = update_w(x, z, w, lmbda)
        if (it + 1) % show_iter == 0:
            print(it + 1)
            print(compute_mape(y_test, inv_flip_vec(x)[pos_test]))
            print(compute_rmse(y_test, inv_flip_vec(x)[pos_test]))
            print()
    return inv_flip_vec(x)
```

<br>

Some simple examples on [this dataset](https://github.com/xinychen/transdim/tree/master/datasets/Portland-data-set).

<br>

```python
import numpy as np
np.random.seed(1)
import time

missing_rate = 0.1
print('Missing rate = {}'.format(missing_rate))

dense_mat = np.load('speed.npy')
d = 3
vec = dense_mat[0, : 96 * 4]
dense_vec = dense_mat[0, : 96 * d]
T = dense_vec.shape[0]
sparse_vec = dense_vec * np.round(np.random.rand(T) + 0.5 - missing_rate)

dense_vec1 = np.append(dense_vec, np.zeros(96))
sparse_vec1 = np.append(sparse_vec, np.zeros(96))
T = dense_vec1.shape[0]

import time
start = time.time()
lmbda = 1e-2 * T
gamma = 5 * lmbda
tau = 2
maxiter = 100
x = LCR(dense_vec1, sparse_vec1, lmbda, gamma, tau, maxiter)
end = time.time()
print('Running time: %d seconds.'%(end - start))
d = 4

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

fig = plt.figure(figsize = (7.5, 2.2))
ax = fig.add_subplot(111)
plt.plot(vec[: 96 * d], 'dodgerblue', linewidth = 1)
plt.plot(np.arange(0, 96 * d), sparse_vec1[: 96 * d], 'o',
         markeredgecolor = 'darkblue', alpha = missing_rate,
         markerfacecolor = 'deepskyblue', markersize = 10)
plt.plot(x[: 96 * d], 'red', linewidth = 4)
plt.xlabel('Time')
plt.ylabel('Speed (mph)')
plt.xlim([0, 96 * d])
plt.ylim([54, 65])
plt.xticks(np.arange(0, 96 * d + 1, 24))
plt.yticks(np.arange(54, 66, 2))
plt.grid(linestyle = '-.', linewidth = 0.5)
ax.tick_params(direction = 'in')

plt.savefig('speeds_{}.pdf'.format(round(missing_rate * 100)),
            bbox_inches = 'tight')
plt.show()

```

Example: Next-day passenger flow prediction

<br>

```python
import numpy as np
np.random.seed(1)
import time

dense_mat = np.load('sept_15min_occupancy_dense_mat.npy')
d = 27
vec = dense_mat[0, : 96 * (d + 1)]
dense_vec = dense_mat[0, : 96 * d]

dense_vec1 = np.append(dense_vec, np.zeros(96))
sparse_vec1 = np.append(dense_vec, np.zeros(96))
T = dense_vec1.shape[0]

import time
start = time.time()
lmbda = 1e-2 * T
gamma = 5 * lmbda
tau = 2
maxiter = 100
x = LCR(dense_vec1, sparse_vec1, lmbda, gamma, tau, maxiter)
end = time.time()
print('Running time: %d seconds.'%(end - start))

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

fig = plt.figure(figsize = (7.5, 2.2))
ax = fig.add_subplot(111)
plt.plot(vec[: 96 * (d + 1)], 'dodgerblue', linewidth = 1)
plt.plot(x[: 96 * (d + 1)], 'red', linewidth = 2)
plt.xlabel('Time')
plt.ylabel('Speed (mph)')
plt.xlim([96 * (d - 3), 96 * (d + 1)])
plt.grid(linestyle = '-.', linewidth = 0.5)
ax.tick_params(direction = 'in')

plt.show()
```

<br>

## LCR-2D without Flipping Operation

```python
import numpy as np

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def laplacian(T, tau):
    ell = np.zeros(T)
    ell[0] = 2 * tau
    for k in range(tau):
        ell[k + 1] = -1
        ell[-k - 1] = -1
    return ell

def prox_2d(z, w, lmbda, denominator):
    N, T = z.shape
    temp1 = np.fft.fft2(lmbda * z - w) / denominator
    temp2 = 1 - N * T / (denominator * np.abs(temp1))
    temp2[temp2 <= 0] = 0
    return np.fft.ifft2(temp1 * temp2).real

def update_z(y_train, pos_train, x, w, lmbda, eta):
    z = x + w / lmbda
    z[pos_train] = (lmbda / (lmbda + eta) * z[pos_train] 
                    + eta / (lmbda + eta) * y_train)
    return z

def update_w(x, z, w, lmbda):
    return w + lmbda * (x - z)

def LCR_2d(y_true, y, lmbda, gamma, tau_s, tau_t, maxiter = 50):
    eta = 100 * lmbda
    if np.isnan(y).any() == False:
        pos_test = np.where((y_true != 0) & (y == 0))
    elif np.isnan(y).any() == True:
        pos_test = np.where((y_true > 0) & (np.isnan(y)))
        y[np.isnan(y)] = 0
    y_test = y_true[pos_test]
    pos_train = np.where(y != 0)
    y_train = y[pos_train]
    z = y.copy()
    w = y.copy()
    ell_s = np.zeros(N)
    ell_s[0] = 1
    # ell_s = laplacian(N, tau_s)
    ell_t = laplacian(T, tau_t)
    ell = np.fft.fft2(np.outer(ell_s, ell_t))
    denominator = lmbda + gamma * np.abs(ell) ** 2
    del y_true, y
    show_iter = 10
    for it in range(maxiter):
        x = prox_2d(z, w, lmbda, denominator)
        z = update_z(y_train, pos_train, x, w, lmbda, eta)
        w = update_w(x, z, w, lmbda)
        if (it + 1) % show_iter == 0:
            print(it + 1)
            print(compute_mape(y_test, x[pos_test]))
            print(compute_rmse(y_test, x[pos_test]))
            print()
    return x
```

<br>

### Large Time Series Imputation

PeMS dataset is available at [our GitHub repository](https://github.com/xinychen/transdim/tree/master/datasets/California-data-set).

Hyperparameters:

- On 30%/50% missing data, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\lambda=10^{-5}NT"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\gamma=10\lambda"/>, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau=1"/>;
- On 70% missing data, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\lambda=10^{-5}NT"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\gamma=10\lambda"/>, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau=2"/>;
- On 90% missing data, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\lambda=10^{-5}NT"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\gamma=10\lambda"/>, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau=3"/>.

<br>

```python
import numpy as np
np.random.seed(1000)

dense_mat = np.load('pems-w1.npz')['arr_0']
for t in range(2, 5):
    dense_mat = np.append(dense_mat, np.load('pems-w{}.npz'.format(t))['arr_0'],
                          axis = 1)
dim1, dim2 = dense_mat.shape

missing_rate = 0.9
sparse_mat = dense_mat * np.round(np.random.rand(dim1, dim2) + 0.5 - missing_rate)
# np.savez_compressed('dense_mat.npz', dense_mat)
# np.savez_compressed('sparse_mat.npz', sparse_mat)

# import cupy as np

# dense_mat = np.load('dense_mat.npz')['arr_0']
# sparse_mat = np.load('sparse_mat.npz')['arr_0']

import time
start = time.time()
N, T = sparse_mat.shape
lmbda = 1e-5 * N * T
gamma = 10 * lmbda
tau_s = 1
tau_t = 3
maxiter = 100
mat_hat = LCR_2d(dense_mat, sparse_mat, lmbda, gamma, tau_s, tau_t, maxiter)
end = time.time()
print('Running time: %d seconds.'%(end - start))
```


<br>
<br>
<br>

```python
import numpy as np


# def sparse_reg(y, A, tau, maxiter = 50):
#     m, n = A.shape
#     x = np.zeros(n)
#     S = np.array([])
#     r = y.copy()
#     for it in range(maxiter):
#         A_tilde = A.copy()
#         ar = A_tilde.T @ r
#         ell = np.argpartition(ar, - 1 * tau)[- 1 * tau :].flatten().tolist()
#         S = np.append(S, ell, axis = 0).astype(int)
#         xS = np.linalg.pinv(A[:, S]) @ y
#         x[S] = xS
#         S = np.argpartition(x, - tau)[- tau :].flatten().tolist()
#         xS = np.linalg.pinv(A[:, S]) @ y
#         x = np.zeros(n)
#         x[S] = xS
#         r = y - A[:, S] @ x[S]
#     return x
```

```python
data = np.load('volume.npy')
y = data[1, :]
mat = circ(y)
A = mat[:, 1 :]
tau = 5
x = sparse_reg(y, A, tau)
print('Non-zero entries of kernel x:')
print(x[x > 0])
print('Index set of the non-zero entries of kernel x:')
print(np.where(x > 0)[0])
print()
```

<br>

## Appendix



<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 17, 2024.)</p>
