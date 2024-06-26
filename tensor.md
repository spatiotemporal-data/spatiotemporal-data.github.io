---
layout: default
---

### Tensor Decomposition for Machine Learning

This article summarizes a series of tensor decomposition models and algorithms in the existing literature, providing comprehensive reviews and tutorials from matrix and tensor computations to tensor decomposition techniques across a wide range of scientific areas. In terms of matrix and tensor computations, this article gives a detailed description of algebraic tensor structure, tensor unfolding, Kronecker product, Khatri-Rao product, and norms. In terms of tensor decomposition, this article introduces CP decomposition, Tucker decomposition, tensor-train decomposition, Bayesian tensor factorization, and non-negative tensor factorization, in the meantime presenting the applications of these methods. Since the decomposition of tensors is usually constructed as an optimization problem, this article also gives preliminaries of some methods for solving convex and nonconvex optimization. This work is expected to provide insight into both machine learning and data science communities, drawing strong connections with the core concept “*tensor decomposition*". To make this work reproducible and sustainable, we provide sources such as datasets and Python implementation (mainly relying on NumPy).

<br>

### Content

- **Introduction**
  - Tensor decomposition in the past 10-100 years
  - Tensor decomposition in the past decade
- **What are tensors?**
  - Tensors in algebra & machine learning
    - Basic structures
    - Tensor unfolding
    - Tesor vectorization
  - Tensors in data science
    - Fluid flow
    - Climate & weather
    - International trade
    - Urban human mobility
- **Foundation of Tensor Computations**
  - Norms
    - Vector norms
    - Frobenius norm
    - Nuclear norm
  - Matrix Trace
    - Definition
    - Property
  - Kronecker product
    - Definition
    - Basic property
    - Mixed-product property
    - Inverse of Kronecker product
    - Vectorization
    - Matrix trace
    - Frobenius norm
  - Khatri-Rao product
  - Modal product
    - Definition
    - Property
  - Outer product
    - Definition
    - Property
  - Derivatives
    - Basic principle
    - L2-norm
    - Matrix trace
    - Frobenius norm
    - Kronecker product
    - Modal product


<br>

**Materials & References**

- Yuejie Chi, Yue M. Lu, and Yuxin Chen (2019). [Nonconvex optimization meets low-rank matrix factorization: An overview](https://doi.org/10.1109/TSP.2019.2937282). IEEE Transactions on Signal Processing, 67(20): 5239-5269.
- Simon J.D. Prince (2023). Understanding Deep learning, MIT Press. [[Book website](https://udlbook.github.io/udlbook/)]
- Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020). Mathematics for Machine Learning, Cambridge University Press. [[Book website](https://mml-book.github.io/)]

<br>

**YouTube**

- [Ankur Moitra: "Tensor Decompositions and their Applications (Part 1/2)"](https://youtu.be/UyO4igyyYQA?si=8GvZeeGXp5v80hEv)
- [Ankur Moitra: "Tensor Decompositions and their Applications (Part 2/2)"](https://www.youtube.com/watch?v=npPaMknLJWQ)

<br>
