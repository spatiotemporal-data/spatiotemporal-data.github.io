---
layout: default
---

### Tensor Decomposition for Machine Learning

<p align="center"><span style="color:gray">Xinyu Chen, Dingyi Zhuang, Jinhua Zhao (2024)</span></p>

<br>

This article summarizes the development of tensor decomposition models and algorithms in the literature, providing comprehensive reviews and tutorials from matrix and tensor computations to tensor decomposition techniques across a wide range of scientific areas and applications. Since the decomposition of tensors is usually constructed as an optimization problem, this article also has a preliminary introduction to some popular methods for solving convex and nonconvex optimization. This work is expected to provide insight into both machine learning and data science communities, drawing strong connections with the key concepts relating to *tensor decomposition*. To make this work reproducible and sustainable, we provide sources such as datasets and Python implementation (mainly relying on Python's `numpy` library).

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
  - Matrix trace
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
    - <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_2"/>-norm
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
