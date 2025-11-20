---
layout: default
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<script>
window.MathJax = {
  chtml: {
    scale: 0.95,
    minScale: 0.9
  },
  svg: {
    scale: 0.95,
    minScale: 0.9
  },
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>

# Kronecker Product for Machine Learning and Beyond

<p align="center"><span style="color:gray">An introductory post for explaining Kronecker product and its applications to machine learning and beyond.</span></p>

<p align="center"><span style="color:gray">(Updated on November 19, 2025)</span></p>

<br>


In this post, we intend to explain the essential components related to Kronecker product.


**Content:**

In **Part I** of this series, we introduce the fundamental definition of Kronecker product.



<br>

## I. Definition

Kronecker product is a commonly-used operator for connecting matrix computations with tensor computations. By definition, given any matrices $\boldsymbol{X}\in\mathbb{R}^{m\times n}$ and $\boldsymbol{Y}\in\mathbb{R}^{p\times q}$, the Kronecker product (denoted by $\otimes$) between two matrices is

$$
\boldsymbol{X} \otimes \boldsymbol{Y}=\begin{bmatrix}
x_{11} \boldsymbol{Y} & x_{12} \boldsymbol{Y} & \cdots & x_{1 n} \boldsymbol{Y} \\
x_{21} \boldsymbol{Y} & x_{22} \boldsymbol{Y} & \cdots & x_{2 n} \boldsymbol{Y} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m 1} \boldsymbol{Y} & x_{m 2} \boldsymbol{Y} & \cdots & x_{m n} \boldsymbol{Y}
\end{bmatrix} \in \mathbb{R}^{(m p) \times(n q)} \tag{1}
$$
resulting in an $(mp)\times (nq)$ matrix. As can be seen, the matrix $\boldsymbol{Y}$ would multiply with all entries of $\boldsymbol{X}$ independently, demonstrating that the $(i,j)$-th block of $\boldsymbol{X}\otimes\boldsymbol{Y}$ is defined as $x_{ij}\boldsymbol{Y}\in\mathbb{R}^{p\times q}$. If one considers $\boldsymbol{Y}\otimes\boldsymbol{X}$, then the matrix $\boldsymbol{X}$ would multiply with all entries of $\boldsymbol{Y}$ independently. Although $\boldsymbol{X}\otimes\boldsymbol{Y}$ and $\boldsymbol{Y}\otimes\boldsymbol{X}$ are still of same size, they are not identical. In other words, the Kronecker product $\boldsymbol{X}\otimes\boldsymbol{Y}$ is non-commutative.


<br>
<br>


<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on November 19, 2025)</p>
