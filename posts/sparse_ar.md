---
layout: default
---

# Interpretable Sparse Autoregression

<p align="center"><span style="color:gray">Quantifying periodicity and seasonality of time series with sparse autoregression.</span></p>

<p align="center"><span style="color:gray">(Updated on February 15, 2025)</span></p>

<br>

In this post, we intend to explain the essential ideas of our research work:

- Interpretable time series autoregression.

**Content:**

In **Part I** of this series, we introduce the essential idea of time series autoregression in statistics.

<br>

## I. Univariate Autoregression

### I-A. Time Series & Auto-Correlations

### I-B. Definition of Autoregression

The <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d"/>th-order univariate autoregression of time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,\cdots,x_{T})^\top\in\mathbb{R}^{T}"/> can be written as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;x_{t}=\sum_{k=1}^{d}w_{k}x_{t-k}+\epsilon_{t}"/></p>

There is a closed-form solution to the coefficient vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}=(w_1,w_2,\cdots,w_{d})^\top\in\mathbb{R}^{d}"/> from the optimization problem such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}:=\arg\,\min_{\boldsymbol{w}}\,\sum_{t=d+1}^{T}\Bigl(x_{t}-\sum_{k=1}^{d}w_{k}x_{t-k}\Bigr)^2"/></p>

which is equivalent to

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}:=\arg\,\min_{\boldsymbol{w}}\,\|\tilde{\boldsymbol{x}}-\boldsymbol{A}\boldsymbol{w}\|_2^2=\boldsymbol{A}^{\dagger}\tilde{\boldsymbol{x}}"/></p>

However, the challenges arise if there is a sparsity constraint in the form of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-norm, for instance,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned}\min_{\boldsymbol{w}}\,&\|\tilde{\boldsymbol{x}}-\boldsymbol{A}\boldsymbol{w}\|_2^2 \\ \text{s.t.}\,&\|\boldsymbol{w}\|_0\leq\tau \end{aligned}"/></p>

where the upper bound the constraint is an integer <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau\in\mathbb{Z}"/>, which is supposed to be no greater than the order <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d"/>.


## II. Sparse Autoregression

### II-A. Subspace Pursuit

### II-B. Mixed-Integer Programming

## III. Time-Varying Sparse Autoregression

The optimization problem is formulated as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned}\min_{\boldsymbol{w}_1,\boldsymbol{w}_{2},\ldots,\boldsymbol{w}_{\delta},\boldsymbol{\beta}}\,&\sum_{\gamma=1}^{\delta}\|\tilde{\boldsymbol{x}}_{\gamma}-\boldsymbol{A}_{\gamma}\boldsymbol{w}_{\gamma}\|_2^2 \\ \text{s.t.}\,&\begin{cases} \boldsymbol{\beta}\in\{0,1\}^{d} \\ \boldsymbol{w}_{\gamma}\leq\alpha\cdot\boldsymbol{\beta},\,\forall \gamma\in\{1,2,\ldots,\delta\} \\ \displaystyle \sum_{k=1}^{d}\beta_{k}=\|\boldsymbol{\beta}\|_1\leq\tau \end{cases} \end{aligned}"/></p>


### III-A. Ridesharing Data

### III-B. Formulating Time-Varying Systems

### III-C. Solving the Optimization Problem




<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on February 15, 2025)</p>
