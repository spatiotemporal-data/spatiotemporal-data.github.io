---
layout: default
---

# Interpretable Sparse Autoregression

<p align="center"><span style="color:gray">Quantifying periodicity and seasonality of time series with sparse autoregression.</span></p>

<p align="center"><span style="color:gray">(Updated on February 15, 2025)</span></p>

<br>

In this post, we intend to explain the essential ideas of our research work:

- Interpretable time series autoregression.
- Essential idea of sparse autoregression.

**Content:**

In **Part I** of this series, we introduce the essential idea of time series autoregression in statistics.

<br>

## I. Univariate Autoregression

Time series autoregression is a statistical model used to analyze and forecast time series data. The class of autoregression models is widely used in the fields of economics, finance, weather forecasting, and signal processing.

### I-A. Time Series & Auto-Correlations

The first challenge is extracting the temporal correlations from autoregression coefficients if the coefficient vector is densely valued and has both positive and negative entries.

Does this situation appear if you use naive least square? (due to random noise in the observation?)




### I-B. Definition of Autoregression

The essential idea of time series autoregression is that a given data point of a time series is linearly dependent on the previous data points. Mathematically, the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d"/>th-order univariate autoregression of time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,\cdots,x_{T})^\top\in\mathbb{R}^{T}"/> can be written as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;x_{t}=\sum_{k=1}^{d}w_{k}x_{t-k}+\epsilon_{t}"/></p>

for all <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t\in\{d+1,d+2,\ldots,T\}"/>. The integer <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d\in\mathbb{Z}^{+}"/> is the order. Here, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{t}\in\mathbb{R}"/> is the value of the time series at time <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>. The vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}=(w_1,w_2,\cdots,w_{d})^\top\in\mathbb{R}^{d}"/> represents the autoregressive coefficients. The random error <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\epsilon_t\in\mathbb{R},\,\forall t"/> is assumed to be normally distributed, following a mean of zero and a constant variance.


There is a closed-form solution to the coefficient vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}=(w_1,w_2,\cdots,w_{d})^\top\in\mathbb{R}^{d}"/> from the optimization problem such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}:=\arg\,\min_{\boldsymbol{w}}\,\sum_{t=d+1}^{T}\Bigl(x_{t}-\sum_{k=1}^{d}w_{k}x_{t-k}\Bigr)^2"/></p>


which is equivalent to

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}:=\arg\,\min_{\boldsymbol{w}}\,\|\tilde{\boldsymbol{x}}-\boldsymbol{A}\boldsymbol{w}\|_2^2=\boldsymbol{A}^{\dagger}\tilde{\boldsymbol{x}}"/></p>

However, the challenges arise if there is a sparsity constraint in the form of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-norm, for instance,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned}\min_{\boldsymbol{w}}\,&\|\tilde{\boldsymbol{x}}-\boldsymbol{A}\boldsymbol{w}\|_2^2 \\ \text{s.t.}\,&\|\boldsymbol{w}\|_0\leq\tau \end{aligned}"/></p>

where the upper bound the constraint is an integer <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tau\in\mathbb{Z}^{+}"/>, which is supposed to be no greater than the order <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d"/>.


## II. Sparse Autoregression

### II-A. Subspace Pursuit

### II-B. Mixed-Integer Programming

## III. Time-Varying Sparse Autoregression

The optimization problem is formulated as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned}\min_{\boldsymbol{w}_1,\boldsymbol{w}_{2},\ldots,\boldsymbol{w}_{\delta},\boldsymbol{\beta}}\,&\sum_{\gamma=1}^{\delta}\|\tilde{\boldsymbol{x}}_{\gamma}-\boldsymbol{A}_{\gamma}\boldsymbol{w}_{\gamma}\|_2^2 \\ \text{s.t.}\,&\begin{cases} \boldsymbol{\beta}\in\{0,1\}^{d} \\ -\alpha\cdot\boldsymbol{\beta}\leq \boldsymbol{w}_{\gamma}\leq\alpha\cdot\boldsymbol{\beta},\,\forall \gamma\in\{1,2,\ldots,\delta\} \\ \displaystyle \sum_{k=1}^{d}\beta_{k}=\|\boldsymbol{\beta}\|_1\leq\tau \end{cases} \end{aligned}"/></p>


---

<span style="color:gray">
<b>Example 1.</b> For any vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^{n}"/>, verify that <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}^\top\boldsymbol{y}=\operatorname{tr}(\boldsymbol{y}\boldsymbol{x}^\top)"/>.
</span>

<span style="color:gray">
According to the definition of inner product, we have <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}^\top\boldsymbol{y}=\sum_{i=1}^{n}x_iy_i"/>. In contrast, the outer product is given by
</span>


<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\boldsymbol{x}^\top=\begin{bmatrix} y_1x_1 & y_1x_2 & \cdots & y_1x_n \\ y_2x_1 & y_2x_2 & \cdots & y_nx_2 \\ \vdots & \vdots & \ddots & \vdots \\ y_nx_1 & y_nx_2 & \cdots & y_nx_n \\ \end{bmatrix}"/></p>

<span style="color:gray">
Recall that the trace of a square matrix is the sum of diagonal entries, we therefore have <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\operatorname{tr}(\boldsymbol{y}\boldsymbol{x}^\top)=\sum_{i=1}^{n}y_ix_i=\boldsymbol{y}^\top\boldsymbol{x}=\boldsymbol{x}^\top\boldsymbol{y}"/>.
</span>


---

<br>



### III-A. Ridesharing Data

### III-B. Formulating Time-Varying Systems

### III-C. Solving the Optimization Problem




<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on February 15, 2025)</p>
