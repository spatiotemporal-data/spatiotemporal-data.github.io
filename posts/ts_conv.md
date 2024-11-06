---
layout: default
---

# Time Series Convolution

<p align="center"><span style="color:gray">A convolutional kernel approach for reinforcing the modeling of time series trends and interpreting temporal patterns, allowing one to leverage Fourier transforms and learn sparse representations. The interpretable machine learning models such as sparse regression unlock opportunities to better capture the long-term changes and temporal patterns of real-world time series.</span></p>

<p align="center"><span style="color:gray">(Updated on October 29, 2024)</span></p>

<br>

In this post, we intend to explain the essential ideas of our latent research work:

- Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). [Laplacian convolutional representation for traffic time series imputation](https://doi.org/10.1109/TKDE.2024.3419698). IEEE Transactions on Knowledge and Data Engineering. 36 (11): 6490-6502.
- Xinyu Chen, HanQin Cai, Fuqiang Liu, Jinhua Zhao (2024). [Correlating time series with interpretable convolutional kernels](https://arxiv.org/abs/2409.01362). arXiv:2409.01362.


**Content:**
- Motivation
- Preliminaries
- Prior art on global trend modeling with circulant matrix nuclear norm minimization
- Laplacian convolutional representation
- Learning interpretable convolutional kernels

<br>

## I. Motivation

Convolution is one of the most commonly-used operations in applied mathematics and signal processing, which has been widely used to several machine learning problems. We hope to revisit the essential ideas of circular convolution and lays an insightful foundation for modeling time series data.


Though nowadays we have a lot of machine learning algorithms on hand, it is still necessary to address the following challenges:

- How to characterize global time series trends?
- How to characterize local time series trends?
- How to learn interpretable local and nonlocal patterns as convolutional kernels?

Sometimes, time series exhibit complicated trends if [they are not stationary](https://otexts.com/fpp2/stationarity.html).

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

### II-A. Circular Convolution

Convolution is a powerful operation in many classical deep learning frameworks, such as convolutional neural networks (CNNs). In the context of discrete sequences (typically vectors), circular convolution refers to the convolution of two discrete sequences of data, and it plays an important role in maximizing the efficiency of certain common filtering operations (see [circular convolution](https://en.wikipedia.org/wiki/Circular_convolution) on Wikipedia).

By definition, for any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,\cdots,x_T)^\top\in\mathbb{R}^{T}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(y_1,y_2,\cdots,y_\tau)^\top\in\mathbb{R}^{\tau}"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau\leq T"/>, the circular convolution (denoted by the symbol <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\star"/>) of these two vectors is formulated as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}\in\mathbb{R}^{T}"/></p>

Elment-wise, we have

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;z_{t}=\sum_{k=1}^{\tau}x_{t-k+1} y_{k},\,\forall t\in\{1,2,\ldots,T\}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;z_t"/> represents the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>th entry of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>. For a cyclical operation, it takes <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{t-k+1}=x_{t-k+1+T}"/> when <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t+1\leq k"/>. While the definition of circular convolution might seem complicated for beginners, it becomes much clearer when you consider the concept of a convolution or circulant matrix, as demonstrated in the examples provided below.

As mentioned above, the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> has a length of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/>, and the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/> has a length of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/>. According to the definition of circular convolution, the resulting vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> will also have a length of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/>, matching the length of the original vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>.

Given any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,x_3,x_4)^\top\in\mathbb{R}^4"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}=(y_1,y_2,y_3)^\top\in\mathbb{R}^3"/>, the circular convolution between them can be expressed as follows,

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

### II-B. Convolution Matrix

Using the notations above, for any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{\tau}"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau<T"/>, the circular convolution can be expressed as a linear transformation:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}_{\tau}(\boldsymbol{x})\boldsymbol{y}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}:\mathbb{R}^{T}\to\mathbb{R}^{T\times \tau}"/> denotes the convolution operator. The convolution matrix can be represented as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}(\boldsymbol{x})=\begin{bmatrix} x_1 & x_T & x_{T-1} & \cdots & x_{T-\tau+2} \\ x_2 & x_1 & x_{T} & \cdots & x_{T-\tau+3} \\ x_3 & x_2 & x_1 & \cdots & x_{T-\tau+4} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ x_{T} & x_{T-1} & x_{T-2} & \cdots & x_{T-\tau+1} \\ \end{bmatrix}\in\mathbb{R}^{T\times\tau}"/></p>

In signal processing, this linear transformation is a fundamental property of circular convolution, highlighting its role in efficently implementing filtering operations.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/linear_conv_mat.png" width="320" />
</p>

<p align = "center">
<b>Figure 3.</b> Illustration of circular convolution as the linear transformation with a convolution matrix.
</p>

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

### II-C. Circulant Matrix

Recall that the convolution matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}_{\tau}(\boldsymbol{x})"/> is specified with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/> columns, corresponding to the length of vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/>). In the case of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^{T}"/> with same columns, the convolution matrix becoms a square matrix, known as a **circulant matrix**. In this study, we emphasize the importance of circulant matrices and their properties, such as their strong connection with circular convolution and discrete Fourier transform, even through we do not work directly with circulant matrices.

For any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^{T}"/>, the circular convolution can be expressed as a linear transformation such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}=\boldsymbol{x}\star\boldsymbol{y}=\mathcal{C}(\boldsymbol{x})\boldsymbol{y}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}:\mathbb{R}^{T}\to\mathbb{R}^{T\times T}"/> denotes the circulant operator. The circulant matrix is defined as:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} x_1 & x_T & x_{T-1} & \cdots & x_{2} \\ x_2 & x_1 & x_{T} & \cdots & x_{3} \\ x_3 & x_2 & x_1 & \cdots & x_{4} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ x_{T} & x_{T-1} & x_{T-2} & \cdots & x_{1} \\ \end{bmatrix}\in\mathbb{R}^{T\times T}"/></p>
which forms a square matrix.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/linear_circ_mat.png" width="320" />
</p>

<p align = "center">
<b>Figure 4.</b> Illustration of circular convolution as the linear transformation with a circulant matrix.
</p>

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

### II-D. Discrete Fourier Transform

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


### III-A. Definition

Nuclear norm is a key concept in matrix computations and convex optimization, frequently applied in low-rank matrix approximation and completion problems. For any matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{m\times n}"/>, the nuclear norm is defined as the sum of singular values:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{X}\|_{*}=\sum_{r=1}^{t}s_{r}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\cdot\|_*"/> denotes the nuclear norm. As illustrated in Figure 5, the singular values are <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;s_1,s_2,\ldots, s_t"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t=\min\{m,n\}"/>.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/svd_explained.png" width="500" />
</p>

<p align = "center">
<b>Figure 5.</b> Singular value decomposition of matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{m\times n}"/>. In the decomposed matrices, the unitary matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}\in\mathbb{R}^{m\times t}"/> (or <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Q}\in\mathbb{R}^{n\times t}"/>) consists of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> orthogonal left (or right) singular vectors, while the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> diagonal entries of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}"/> are singular values such that <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;s_1\geq s_2\geq\cdots\geq s_t\geq 0"/>. Note that <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t=\min\{m,n\}"/> for notational convenience. 
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

### III-B. Property

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

### III-C. Optimization

For any partially observed time series in the form of a vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^T"/> with the observed index set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega"/>, solving the circulant matrix nuclear norm minimization allows one to reconstruct missing values in time series. The problem is formulated as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&\|\mathcal{C}(\boldsymbol{x})\|_* \\ \text{s.t.}\,&\|\mathcal{P}_{\Omega}(\boldsymbol{x}-\boldsymbol{y})\|_2\leq\epsilon \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\epsilon"/> in the constraint represents the tolerance of errors between the reconstructed time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> and the partially observed time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/>.

In this case, one can rewrite the constraint as a penalty term (weighted by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\gamma"/>) in the objective function:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \min_{\boldsymbol{x}}\,\|\mathcal{C}(\boldsymbol{x})\|_*+\frac{\gamma}{2}\|\mathcal{P}_{\Omega}(\boldsymbol{x}-\boldsymbol{y})\|_2^2"/></p>

Since the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> is associated with a circulant matrix nuclear norm and a penalty term, the first impluse is using variable separated to convert the problem into the following one:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&\|\mathcal{C}(\boldsymbol{x})\|_*+\frac{\gamma}{2}\|\mathcal{P}_{\Omega}(\boldsymbol{z}-\boldsymbol{y})\|_2^2 \\ \text{s.t.}\,&\boldsymbol{x}=\boldsymbol{z} \end{aligned}"/></p>

<br>

### III-D. Solution Algorithm

The augmented Lagrangian function is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})=\|\mathcal{C}(\boldsymbol{x})\|_*+\frac{\gamma}{2}\|\mathcal{P}_{\Omega}(\boldsymbol{z}-\boldsymbol{y})\|_2^2+\frac{\lambda}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_2^2+\langle\boldsymbol{w},\boldsymbol{x}-\boldsymbol{z}\rangle"/></p>

where the dual variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}\in\mathbb{R}^{T}"/> is an estimate of the Lagrange multiplier, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\lambda"/> is a penalty parameter that controls the convergence rate.

Thus, the variables <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>, and the dual variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> can be updated iteratively as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{cases} \displaystyle\boldsymbol{x}:=\arg\min_{\boldsymbol{x}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ \displaystyle\boldsymbol{z}:=\arg\min_{\boldsymbol{z}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ \boldsymbol{w}:=\boldsymbol{w}+\lambda(\boldsymbol{x}-\boldsymbol{z}) \end{cases}"/></p>

<br>

### III-E. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>-Subproblem

Solving the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> involves discrete Fourier transform, convolution theorem, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm minimization in complex space. The optimization problem with respect to the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \boldsymbol{x}:=&\arg\min_{\boldsymbol{x}}\,\|\mathcal{C}(\boldsymbol{x})\|_{*}+\frac{\lambda}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_2^2+\langle\boldsymbol{w},\boldsymbol{x}\rangle \\ =&\arg\min_{\boldsymbol{x}}\,\|\mathcal{C}(\boldsymbol{x})\|_{*}+\frac{\lambda}{2}\|\boldsymbol{x}-\boldsymbol{z}+\boldsymbol{w}/\lambda\|_2^2 \end{aligned}"/></p>

Using discrete Fourier transform, the optimization problem can be converted into the following one in complex space:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \hat{\boldsymbol{x}}:=\arg\min_{\hat{\boldsymbol{x}}}\,\|\hat{\boldsymbol{x}}\|_{1}+\frac{\lambda}{2T}\|\hat{\boldsymbol{x}}-\hat{\boldsymbol{z}}+\hat{\boldsymbol{w}}/\lambda\|_2^2 "/></p>

where the complex-valued variables <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\{\hat{\boldsymbol{x}},\hat{\boldsymbol{z}},\hat{\boldsymbol{w}}\}=\{\mathcal{F}(\boldsymbol{x}),\mathcal{F}(\boldsymbol{z}),\mathcal{F}(\boldsymbol{w})\}"/> refer to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\{\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}\}"/> in the frequency domain. The closed-form solution to the complex-valued variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\hat{\boldsymbol{x}}"/> is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \hat{x}_t:=\frac{\hat{h}_t}{|\hat{h}_t|}\cdot\max\{|\hat{h}_t|-T/\lambda,0\},\,t=1,2,\ldots, T "/></p>

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

---

<span style="color:gray">
**Shrinkage Operator.** For any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^{n}"/>, the closed-form solution to the optimization problem
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{y}}\,\|\boldsymbol{y}\|_1+\frac{\alpha}{2}\|\boldsymbol{y}-\boldsymbol{x}\|_2^2"/></p>

<span style="color:gray">
can be expressed as
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;y_{i}=\frac{x_i}{|x_i|}\cdot\max\{|x_i|-1/\alpha, 0\},\,i=1,2,\cdots,n"/></p>

<span style="color:gray">
or
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;y_{i}=\begin{cases} x_i-1/\alpha, & \text{if}\,x_i>1/\alpha \\ x_i+1/\alpha, & \text{if}\,x_i<-1/\alpha \\ 0, & \text{otherwise} \end{cases}"/></p>

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/shrinkage_operator.png" width="300" />
</p>

<p align = "center">
<b>Figure 6.</b> Illustration of the shrinkage operator for solving the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm minimization problem.
</p>

<br>

---

<br>

### III-F. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>-Subproblem

In terms of the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>, the partial derivative of the augmented Lagrangian function with respect to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{P}_{\Omega}(\boldsymbol{z})"/> is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\mathcal{P}_{\Omega}(\boldsymbol{z})}=&\gamma\mathcal{P}_{\Omega}(\boldsymbol{z}-\boldsymbol{y})+\lambda\mathcal{P}_{\Omega}(\boldsymbol{z}-\boldsymbol{x})-\mathcal{P}_{\Omega}(\boldsymbol{w}) \\ =&(\gamma+\lambda)\mathcal{P}_{\Omega}(\boldsymbol{z})-\mathcal{P}_{\Omega}(\gamma\boldsymbol{y}+\lambda\boldsymbol{x}+\boldsymbol{w}) \end{aligned}"/></p>

while

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{z})}=&\lambda\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{z}-\boldsymbol{x})-\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{w}) \\ =&\lambda\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{z})-\mathcal{P}_{\Omega}^{\perp}(\lambda\boldsymbol{x}+\boldsymbol{w}) \end{aligned}"/></p>

As a result, letting the partial derivative of the augmented Lagrangian function with respect to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> be a zero vector, the least squares solution is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \boldsymbol{z}:=&\Bigl\{\boldsymbol{z}\mid \frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\mathcal{P}_{\Omega}(\boldsymbol{z})}+\frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\mathcal{P}_{\Omega}^{\perp}(\boldsymbol{z})}=\boldsymbol{0}\Bigr\} \\ =&\frac{1}{\gamma+\lambda}\mathcal{P}_{\Omega}(\gamma\boldsymbol{y}+\lambda\boldsymbol{x}+\boldsymbol{w})+\frac{1}{\lambda}\mathcal{P}_{\Omega}^{\perp}(\lambda\boldsymbol{x}+\boldsymbol{w}) \end{aligned}"/></p>

where the partial derivative of the augmented Lagrangian function with respect to the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/> is the combination of variables <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{P}_{\Omega}(\boldsymbol{z})"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{P}_{\Omega}^\perp(\boldsymbol{z})"/>.

<br>

```python
import numpy as np

def update_z(y_train, pos_train, x, w, lmbda, gamma):
    z = x + w / lmbda
    z[pos_train] = (gamma * y_train + lmbda * z[pos_train]) / (gamma + lmbda) 
    return z
```


<br>

### III-G. Time Series Imputation

As shown in Figure 7, we randomly remove 95% observations as missing values, and we only have 14 volume observations (i.e., 14 blue dots) for the reconstruction. The circulant matrix nuclear norm minimization can capture the global trends from partial observations.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/circnnm_volumes_95.png" width="350" />
</p>

<p align = "center">
<b>Figure 7.</b> Univariate time series imputation on the freeway traffic volume time series. The blue and red curves correspond to the ground truth time series and reconstructed time series achieved by the circulant matrix nuclear norm minimization.
</p>

<br>

<span style="color:gray">
Please reproduce the experiments by following the [Jupyter Notebook](https://github.com/xinychen/LCR/blob/main/univariate-models/CircNNM.ipynb), which is available at the [LCR repository](https://github.com/xinychen/LCR) on GitHub. For the supplementary material, please check out [Appendix I(A)](https://spatiotemporal-data.github.io/posts/ts_conv_supp/).
<span>

<br>



## IV. Laplacian Convolutional Representation

## V. Learning Interpretable Convolutional Kernels

### V-A. Convolutional Kernels

On the univariate time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/> in the form of a vector, the circular convolution between time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/> and convolutional kernel <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}\in\mathbb{R}^{T}"/> can be constructed with certain purposes. The expression is formally defined by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\star\boldsymbol{\theta}\in\mathbb{R}^{T}"/>. When using this kernel to characterize the temporal correlations of time series, one simple yet interesting idea is making the loss of circular convolution, namely, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{x}\star\boldsymbol{\theta}\|_2^2"/>, as small as possible. However, the optimization problem

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{\theta}}\,\|\boldsymbol{x}\star\boldsymbol{\theta}\|_2^2"/></p>

is ill-posed because the optimal solution is all entries of kernel <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}"/> being zeros. To address this challenge, we assume that the first entry of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}"/> is one and the following <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T-1"/> entries are <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;-\boldsymbol{w}"/> in which <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}\in\mathbb{R}^{T-1}"/> is a non-negative vector. Thus, the optimization problem becomes

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{w}\geq 0}\,&\|\boldsymbol{x}\star\boldsymbol{\theta}\|_2^2 \\ \text{s.t.}\,&\boldsymbol{\theta}=\begin{bmatrix} 1 \\ -\boldsymbol{w} \end{bmatrix} \end{aligned}"/></p>

where the constraint is of great significance for minimizing the objective function.

<br>

### V-B. Sparse Linear Regression

Recall that the circular convolution can be converted into a linear transformation with circulant matrix. Considering the kernel <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}=\begin{bmatrix} 1 \\ -\boldsymbol{w} \end{bmatrix}"/>, the loss of circular convolution is equivalent to

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{x}\star\boldsymbol{\theta}\|_2^2=\|\mathcal{C}(\boldsymbol{x})\boldsymbol{\theta}\|_2^2=\|\boldsymbol{x}-\boldsymbol{A}\boldsymbol{w}\|_2^2"/></p>

where the auxiliary matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{T\times (T-1)}"/> is comprised of the last <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T-1"/> columns of the circulant matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})\in\mathbb{R}^{T\times T}"/>, namely,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}=\begin{bmatrix} x_{T} & x_{T-1} & x_{T-2} & \cdots & x_{2} \\ x_{1} & x_{T} & x_{T-1} & \cdots & x_{3} \\ x_{2} & x_{1} & x_{T} & \cdots & x_{4} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ x_{T-2} & x_{T-3} & x_{T-4} & \cdots & x_{T} \\ x_{T-1} & x_{T-2} & x_{T-3} & \cdots & x_{1} \\ \end{bmatrix}"/></p>

where the matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})"/> is separated into the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> (i.e., first column of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})"/>) and the matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> (i.e., last <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T-1"/> columns of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})"/>), see Figure 8 and Figure 9 for illustrations.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/linear_regression_circ.png" width="420" />
</p>

<p align = "center">
<b>Figure 8.</b> Illustration of circular convolution as the linear transformation with a circulant matrix.
</p>

<br>

As can be seen, one of the most intriguing properties is the circular convolution <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\star\boldsymbol{\theta}"/> can be converted into the expression <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}-\boldsymbol{A}\boldsymbol{w}"/> (see Figure 9), which takes the form of a linear regression with the data pair <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\{\boldsymbol{x},\boldsymbol{A}\}"/>.

<br>



<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/linear_regression_w_conv.png" width="420" />
</p>

<p align = "center">
<b>Figure 9.</b> Illustration of circular convolution with a structured vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}"/>.
</p>


<br>

---

<span style="color:gray">
**Example 7.** Given vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}=(1,0,0,0,-1)^\top"/>, the circular matrix is
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})=\begin{bmatrix} 0 & 4 & 3 & 2 & 1 \\ 1 & 0 & 4 & 3 & 2 \\ 2 & 1 & 0 & 4 & 3 \\ 3 & 2 & 1 & 0 & 4 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}"/></p>

<span style="color:gray">
and the auxiliary matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> is
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}=\begin{bmatrix} 4 & 3 & 2 & 1 \\ 0 & 4 & 3 & 2 \\ 1 & 0 & 4 & 3 \\ 2 & 1 & 0 & 4 \\ 3 & 2 & 1 & 0 \end{bmatrix}"/></p>

<span style="color:gray">
Thus, we can compute the circular convolution <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\star\boldsymbol{\theta}"/> by
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}-\boldsymbol{A}\boldsymbol{w}=\begin{bmatrix} 0 \\ 1 \\ 2 \\ 3 \\ 4 \\ \end{bmatrix}-\begin{bmatrix} 4 & 3 & 2 & 1 \\ 0 & 4 & 3 & 2 \\ 1 & 0 & 4 & 3 \\ 2 & 1 & 0 & 4 \\ 3 & 2 & 1 & 0 \end{bmatrix}\begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \\ \end{bmatrix}=\begin{bmatrix} 0 \\ 1 \\ 2 \\ 3 \\ 4 \\ \end{bmatrix}-\begin{bmatrix} 1 \\ 2 \\ 3 \\ 4 \\ 0 \\ \end{bmatrix}=\begin{bmatrix} -1 \\ -1 \\ -1 \\ -1 \\ 4 \end{bmatrix}"/></p>

---

<br>

To reinforce the model interpretability, we impose a sparsity constraint on the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/>. This is extremely important for capturing local and nonlocal correlations of time series. The process of learning interpretable convolutional kernels can be formulated as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{w}\geq 0}\,&\|\boldsymbol{x}-\boldsymbol{A}\boldsymbol{w}\|_2^2 \\ \text{s.t.}\,&\|\boldsymbol{w}\|_0\leq \tau \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau\in\mathbb{Z}^{+}"/> is the sparsity level, i.e., the number of nonzero values.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/sparse_reg_time_series.png" width="420" />
</p>

<p align = "center">
<b>Figure 10.</b> Illustration of learning <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau"/>-sparse vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> from the time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> with the constructed formula as <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\approx\boldsymbol{A}\boldsymbol{w}"/>.
</p>

<br>

---

<span style="color:gray">
**<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-Norm**. For any vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{T}"/>, the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-norm is defined as follows,
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{x}\|_0=\operatorname{card}\{i\mid x_i\neq 0\}=|\operatorname{supp}(\boldsymbol{x})|"/></p>

<span style="color:gray">
where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\operatorname{card}\{\cdot\}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\operatorname{supp}(\cdot)"/> denote the cardinality and the support set of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>, respectively. For instance, given vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/>, the support set is <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\operatorname{supp}(\boldsymbol{x})=\{2,3,4,5\}"/> and the number of nonzero entries is  <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{x}\|_0=4"/>.
</span>

---

<br>

#### V-C. Mixed-Integer Linear Programming

There are often some special-purpose algorithms for solving constrained linear regression problems, see [constrained least squares](https://en.wikipedia.org/wiki/Constrained_least_squares). Some examples of constraints include: 1) Non-negative least squares in which the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> must satisfy the vector inequality <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}\geq 0"/> (each entry must be either positive or zero); 2) Box-constrained least squares in which the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> must satisfy the vector inequalities <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{b}_{\ell}\leq\boldsymbol{w}\leq \boldsymbol{b}_{u}"/>.

<br>

---

<span style="color:gray">
**Example 8.** Given vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(0,1,2,3,4)^\top"/>, the auxiliary matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> (i.e., the last 4 columns of circulant matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{C}(\boldsymbol{x})\in\mathbb{R}^{5\times 5}"/>) is
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}=\begin{bmatrix} 4 & 3 & 2 & 1 \\ 0 & 4 & 3 & 2 \\ 1 & 0 & 4 & 3 \\ 2 & 1 & 0 & 4 \\ 3 & 2 & 1 & 0 \end{bmatrix}"/></p>

<span style="color:gray">
The question is how to learn the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> with sparsity level <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau=2"/> from the following optimization problem:
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{w}\geq 0}\,&\|\boldsymbol{x}-\boldsymbol{A}\boldsymbol{w}\|_2^2 \\ \text{s.t.}\,&\|\boldsymbol{w}\|_0\leq \tau \end{aligned}"/></p>

<span style="color:gray">
In this case, the empirical solution of the decision variable is <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}=(0.44,0,0,0.44)^\top"/>. To reproduce it, the solution algorithm of mixed-integer linear programming is given by
</span>

```python
import numpy as np
import cvxpy as cp

x = np.array([0, 1, 2, 3, 4])
A = circ_mat(x)[:, 1 :]
tau = 2 # sparsity level
d = A.shape[1]

# Variables
w = cp.Variable(d, nonneg=True)
z = cp.Variable(d, boolean=True)

# Constraints
constraints = [cp.sum(z) <= tau, w <= z, w >= 0]

# Objective
objective = cp.Minimize(cp.sum_squares(x - A @ w))

# Problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GUROBI)  # Ensure to use a solver that supports MIP

# Solution
print("Optimal beta:", w.value)
print("Active indices:", np.nonzero(z.value > 0.5)[0])
```

Please install the optimization packages in-ahead, e.g., `pip install gurobipy`. One can also check out the optimization solvers in `cvxpy`:

```python
import cvxpy as cp

print(cp.installed_solvers())
```

<br>

---

<br>


<br>

## VI. Experiments



<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/pickup_dropoff_trips_chicago_2024_april_may.png" width="600" />
</p>

<p align = "center">
<b>Figure 11.</b> Daily average of rideshare trips over 77 zones during the first 8 weeks since April 1, 2024 in the City of Chicago, USA. There are 11,374,540 trips in total, while the daily average is 203,117 trips.
</p>

<br>





<br>

## VII. Concluding Remarks

From a time series analysis perspective, our modeling ideas can guide further research by using the properties of circular convolution and connecting with various signals. Because circular convolution is a core component in many machine learning tasks, this post could provide an example for how to leverage the circular convolution, discrete Fourier transform, and linear regression. While we focused on the time series imputation and the convolutional kernel learning problems, we hope that our modeling ideas will inspire others in machine learning and optimization.

<br>
<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 17, 2024.)</p>
