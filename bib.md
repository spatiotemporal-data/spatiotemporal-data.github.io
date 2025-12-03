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


## Knowledge Repository

[Xinyu Chen (陈新宇)](https://xinychen.github.io/) created this page since early 2024 at MIT with the purpose of fostering research knowledge, vision, insight, and style from an interdisciplinary perspective. In the meantime, it aims to connect random concepts with mathematics and machine learning.

<br>


### 75th Commit
#### A Convex Binary Reformulation of Sparse Linear Regression




<br>


### 74th Commit
#### Quadratic Programming over Huber Loss Function

---

<p style="font-size: 14px; color: gray">
<b>Reference</b>: <a href="https://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/26-qp/qp.html">Optimization Examples - Quadratic Programming</a>.
</p>

---

For the Huber loss function such that

$$
\phi(u)=\begin{cases} u^2 & \text{if $|u|<\delta$} \\ \delta(2|u|-\delta) &\text{otherwise} \end{cases} \tag{74-1}
$$

with any positive Huber threshold $\delta>0$, minimizing the Huber loss function is equivalent to the following quadratic programming problem (Exercise **6.3(c)** in [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), see page 344):

$$
\begin{aligned}
\min_{\alpha,\gamma}\quad&\gamma^2+\delta\alpha \\
\text{s.t.}\quad&-\alpha\leq u-\gamma\leq\alpha,\quad\alpha>0
\end{aligned} \tag{74-2}
$$

where $\alpha$ and $\gamma$ are continuous decision variables.

<br>


### 73rd Commit
#### Portfolio Selection (Definition)

---

<p style="font-size: 14px; color: gray">
<b>Reference</b>: Dimitris Bertsimas, Ryan Cory-Wright (2022). <a href="https://doi.org/10.1287/ijoc.2021.1127">A Scalable Algorithm for Sparse Portfolio Selection</a>. INFORMS Journal on Computing. 34 (3): 1489–1511.
</p>

---

For a universe of $n$ securities, we estimate a mean $\boldsymbol{\mu}\in\mathbb{R}^{n}$ (or return) and a covariance matrix $\boldsymbol{\Sigma}\in\mathbb{R}^{n\times n}$. The Markowitz model proposed in 1952 selects a portfolio that provides the highest expected return for a given amount of variance. The mean-variance portfolio optimization laid the foundation of modern portfolio theory, which can be formulated as follows,

$$
\begin{aligned}
\min_{\boldsymbol{x}\in\mathbb{R}_{+}^{n}}\quad&\frac{\lambda}{2}\boldsymbol{x}^\top\boldsymbol{\Sigma}\boldsymbol{x}-\boldsymbol{\mu}^\top\boldsymbol{x} \\
\text{s.t.}\quad&\boldsymbol{e}^\top\boldsymbol{x}=1 \\
\end{aligned} \tag{73-1}
$$

where $\lambda$ is a parameter that controls the trade-off between the portfolio's risk and return, and $\boldsymbol{e}\in\mathbb{R}^{n}$ denotes the vector of all ones. As the decision variables, $\boldsymbol{x}$ denotes the portfolio allocation vector.

<br>


### 72nd Commit
#### Local Linear Approximation for Nonconvex Penalties

---

<p style="font-size: 14px; color: gray">
<b>Reference</b>: Jianqing Fan, Jinchi Lv, Lei Qi (2011). <a href="https://doi.org/10.1146/annurev-economics-061109-080451">Sparse high-dimensional models in economics</a>. Annual Review of Economics. 3, 291-317. [<a href="https://faculty.marshall.usc.edu/jinchi-lv/publications/ARE-FLQ11.pdf">PDF</a>]
</p>

---

Nonconvex penalties such as SCAD and MCP for sparse modeling are more flexible than LASSO. Given the decision variables $\boldsymbol{w}\in\mathbb{R}^{d}$, the local linear approximation (LLA, see [Zou, 2008](https://www.jstor.org/stable/20441455)) is

$$
\rho_{\lambda}(|w_k|)\approx\rho_{\lambda}(|w_k^{*}|)+\rho_{\lambda}^{\prime}(|w_k^{*}|)(|w_k|-|w_k^{*}|)\quad\text{for $w_k\approx w_k^{*}$} \tag{72-1}
$$

which is a convex majorant of a concave penalty function $\rho_{\lambda}(\cdot)$ on $[0,\infty)$.

The penalized least-squares (PLS) estimation can therefore be formulated as follows,

$$
\min_{\boldsymbol{w}}\quad\frac{1}{2n}\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\|_2^2+\sum_{k=1}^{d}\underbrace{\alpha_k|w_k|}_{\approx\rho_{\lambda}(|w_k|)} \tag{72-2}
$$

where $\boldsymbol{X}\in\mathbb{R}^{n\times d}$ and $\boldsymbol{y}\in\mathbb{R}^{n}$ are design matrix and response vector, respectively. The weights are

$$
\alpha_k=\rho_{\lambda}^{\prime}(|w_k^{*}|) \tag{72-3}
$$

In the case of SCAD, the derivative is given by

$$
\rho_{\lambda}^{\prime}(w)=\lambda\left(I\{w\leq\lambda\}+\frac{(a\lambda-w)_{+}}{(a-1)\lambda}I\{w>\lambda\}\right) \tag{72-4}
$$

for some $a>2$. Here, $\rho_{\lambda}(0)=0$ and $a=3.7$ is often used.

In the case of MCP, the derivative is given by

$$
\rho_{\lambda}^{\prime}(w)=\frac{(a\lambda-w)_{+}}{a} \tag{72-5}
$$

In fact, if the initial estimate is zero, then $\alpha_k=\lambda$, and the resulting estimate is the LASSO estimate. Nonconvex penalties further reduce the bias problem of LASSO by assigning an adapative weighting scheme.

<br>


### 71st Commit
#### Automatic Lag Selection with Support Vector Regression

---

<p style="font-size: 14px; color: gray">
<b>Reference</b>: Sebastián Maldonado, Agustín González, Sven Crone (2019). <a href="https://doi.org/10.1016/j.asoc.2019.105616">Automatic time series analysis for electric load forecasting via support vector regression</a>. Applied Soft Computing. 83, 105616.
</p>

---

The $\epsilon$-SVR method aims to solve the following optimization problem:

$$
\begin{aligned}
\min_{\boldsymbol{w},b,\boldsymbol{\xi}\geq0,\boldsymbol{\xi}^{*}\geq0}\quad&\frac{1}{2}\|\boldsymbol{w}\|_2^2+\lambda\sum_{i=1}^{n}(\xi_i+\xi_i^{*}) \\
\text{s.t.}\quad&y_i-(\boldsymbol{w}^\top\boldsymbol{x}_{i}+b)-\epsilon\geq\xi_{i},\quad\forall i\in\{1,2,\ldots,n\} \\
&(\boldsymbol{w}^\top\boldsymbol{x}_{i}+b)-y_i-\epsilon\geq\xi_{i}^{*}, \quad\forall i\in\{1,2,\ldots,n\} \\
\end{aligned} \tag{71-1}
$$

taking from the $\epsilon$-insensitive loss function such that

$$
L_{\epsilon}(y,f(x))=\max(0,|y-f(x)|-\epsilon) \tag{71-2}
$$

where loss is zero if the absolute error is within $\epsilon$.


<br>


### 70th Commit
#### High-Dimensional Quantile Tensor Regression

---

<p style="font-size: 14px; color: gray">
<b>Reference</b>: Wenqi Lu, Zhongyi Zhu, Heng Lian (2020). <a href="https://jmlr.org/papers/v21/20-383.html">High-dimensional Quantile Tensor Regression</a>. Journal of Machine Learning Research. 21: 1-31.
</p>

---

Modern machine learning applications involve tensor data in the form of quantile regression such that

$$
Q_{\tau}(y\mid\boldsymbol{\mathcal{X}})=\langle\boldsymbol{\mathcal{A}},\boldsymbol{\mathcal{X}}\rangle \tag{70-1}
$$

for the $\tau$th conditional quantile of response given $\boldsymbol{\mathcal{X}}\in\mathbb{R}^{p_1\times p_2\times \cdots \times p_{K}}$ and $y\in\mathbb{R}$. In the regressor, $\boldsymbol{\mathcal{A}}$ can be parameterized by the Tucker decomposition:

$$
\boldsymbol{\mathcal{A}}=\boldsymbol{\mathcal{G}}\times_1\boldsymbol{U}_1\times_2\boldsymbol{U}_2\cdots\times_{K}\boldsymbol{U}_{K} \tag{70-2}
$$

Considering the loss of quantile regression as follows,

$$
\rho_{\tau}(u)=\begin{cases} \tau u & \text{if $u>0$} \\ (\tau-1)u & \text{otherwise} \end{cases} \tag{70-3}
$$

By integrating tensor decomposition into quantile regression, we have

$$
\begin{aligned}
\min_{\boldsymbol{\mathcal{G}},\boldsymbol{U}_{1},\boldsymbol{U}_2,\cdots,\boldsymbol{U}_{K}}\quad&\sum_{i=1}^{n}\rho_{\tau}(y_i-\langle\boldsymbol{\mathcal{G}}\times_1\boldsymbol{U}_1\times_2\boldsymbol{U}_2\cdots\times_{K}\boldsymbol{U}_{K},\boldsymbol{\mathcal{X}}_{i}\rangle) \\
&\quad +\lambda\|\boldsymbol{U}_K\otimes\cdots\otimes\boldsymbol{U}_1\|_1 \\
\text{s.t.}\quad &\boldsymbol{U}_k^\top\boldsymbol{U}_k=\boldsymbol{I},\quad\forall k\in\{1,2,\ldots,K\}
\end{aligned} \tag{70-4}
$$

with an $\ell_1$-norm induced sparsity penalty and orthogonality constraints on factor matrices.

Estimating these factor matrices can be converted into the following optimization problem:

$$
\begin{aligned}
\min_{\boldsymbol{r},\boldsymbol{U}}\quad&\rho_{\tau}(\boldsymbol{r})+\lambda\|\boldsymbol{U}\|_1 \\
\text{s.t.}\quad&\boldsymbol{r}=\boldsymbol{y}-\boldsymbol{Z}\operatorname{vec}(\boldsymbol{U}),\quad\boldsymbol{U}^\top\boldsymbol{U}=\boldsymbol{I}
\end{aligned} \tag{70-5}
$$


<br>


### 69th Commit
#### [Non-Quadratic Losses](https://ee104.stanford.edu/lectures/losses.pdf)

---

<p style="font-size: 14px; color: gray">
<b>Reference</b>: Sanjay Lall and Stephen Boyd. Non-Quadratic Losses. EE104 at Stanford University.
</p>

---

The Huber loss has linear growth for large residual $r$ that makes fit less sensitive to outliers. Thus, empirical risk minimization (ERM) with Huber loss is called a **robust** prediction method. The log Huber loss is 

$$
\phi(u)=\begin{cases} u^2 & \text{if $|u|<\delta$} \\ \delta^2(1-2\operatorname{log}(\delta)+\operatorname{log}(u^2)) &\text{otherwise} \end{cases} \tag{69-1}
$$

for any positive parameter $\delta>0$.


<br>


### 68th Commit
#### [Regression & Statistical Estimation](https://www.cvxgrp.org/nasa/pdf/lecture5.pdf) ([CVXPY x NASA Course 2024](https://www.cvxgrp.org/nasa/))

Huber penalty function with the Huber threshold $\delta>0$ has linear growth for large $u$ that makes approximation less sensitive to outliers:

$$
\phi(u)=\begin{cases} u^2 & \text{if $|u|<\delta$} \\ \delta(2|u|-\delta) &\text{otherwise} \end{cases} \tag{68-1}
$$

So, it is also called robust penalty.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/quad_vs_huber.png" width="400" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
[Excellent example from page 300 of the <a href="https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf">convex optimization book</a>] Affine function fitted on 42 points (with two outliers) using quadratic (dashed) and Huber (solid) penalty.
</p>

Given $\boldsymbol{A}\in\mathbb{R}^{m\times n}$ and $\boldsymbol{b}\in\mathbb{R}^{m}$ with $m$ measurements and $n$ regressors, when it minimizes residual $\boldsymbol{r}=\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\in\mathbb{R}^{m}$, the optimization can be formulated to the following form:

$$
\begin{aligned}
\min_{\boldsymbol{x},\boldsymbol{r}}\quad&\sum_{i=1}^{m}\phi(r_i) \\
\text{s.t.}\quad&\boldsymbol{r}=\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b} \\
\end{aligned} \tag{68-2}
$$

where $\phi:\mathbb{R}\to\mathbb{R}$ is a convex penalty function. In particular, Huber loss would lead to the robust regression.

As an alternative, quantile regression for any $\tau\in(0,1)$ takes the following form:

$$
\begin{aligned}
\min_{\boldsymbol{x},\boldsymbol{r}\geq0,\boldsymbol{r}^{*}\geq0}\quad&\sum_{i=1}^{m}\left(\tau r_{i}+(1-\tau) r_{i}^{*}\right) \\
\text{s.t.}\quad&\boldsymbol{r}-\boldsymbol{r}^{*}=\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b} \\
\end{aligned} \tag{68-3}
$$

where $\boldsymbol{r}$ and $\boldsymbol{r}^{*}$ are over-estimated and under-estimated residual vectors, respectively.

- For $\tau=\frac{1}{2}$, same penalty for under-estimate and over-estimate.
- For $\tau>\frac{1}{2}$, higher penalty to under-estimate than over-estimate.
- For $\tau<\frac{1}{2}$, higher penalty to over-estimate than under-estimate.


<br>


### 67th Commit
#### [Linear Algebra & Robot Control](https://adamheins.com/blog/linear-algebra-robot-control)

##### Preliminaries

The column space $\mathcal{C}(\boldsymbol{A})$ of a matrix $\boldsymbol{A}$ is the span of its columns, i.e., the set of all linear combinations of the columns of $\boldsymbol{A}$:

$$
\mathcal{C}(\boldsymbol{A})=\{\boldsymbol{A}\boldsymbol{x}\mid \boldsymbol{x}\in\mathbb{R}^{n}\} \tag{67-1}
$$

The nullspace $\mathcal{N}(\boldsymbol{A})$ of a matrix $\boldsymbol{A}$ is the set of all vectors $\boldsymbol{x}$ for which $\boldsymbol{A}\boldsymbol{x}=\boldsymbol{0}$ such that

$$
\mathcal{N}(\boldsymbol{A})=\{\boldsymbol{x}\in\mathbb{R}^{n}\mid \boldsymbol{A}\boldsymbol{x}=\boldsymbol{0}\} \tag{67-2}
$$

The condition number of a square matrix $\boldsymbol{A}$ is defined as the ratio between the largest and smallest singular values, i.e.,

$$
\kappa(\boldsymbol{A})=\frac{\sigma_{\text{max}}(\boldsymbol{A})}{\sigma_{\text{min}}(\boldsymbol{A})} \tag{67-3}
$$

measuring how much the solution $\boldsymbol{x}$ of $\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$ can change when the value of $\boldsymbol{b}$ changes.

- $\boldsymbol{A}$ is well-conditioned: When $\kappa(\boldsymbol{A})$ is low, small changes in $\boldsymbol{b}$ do not cause large changes in $\boldsymbol{x}$.
- $\boldsymbol{A}$ is ill-conditioned: When $\kappa(\boldsymbol{A})$ is high, small changes in $\boldsymbol{b}$ can cause large changes in $\boldsymbol{x}$.

##### Optimization

There are some optimization problems that have been formulated:

- Minimizing the $\ell_2$-norm of $\boldsymbol{x}$ (under-determined system):

$$
\begin{aligned}
\min_{\boldsymbol{x}}\quad&\frac{1}{2}\|\boldsymbol{x}\|_2^2 \\
\text{s.t.}\quad&\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}
\end{aligned} \tag{67-4}
$$

- (Damped) Introducing the slack variable $\boldsymbol{s}$:

$$
\begin{aligned}
\min_{\boldsymbol{x},\boldsymbol{s}}\quad&\frac{\alpha}{2}\|\boldsymbol{x}\|_2^2+\frac{1}{2}\|\boldsymbol{s}\|_2^2 \\
\text{s.t.}\quad&\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}+\boldsymbol{s} \\
\end{aligned} \tag{67-5}
$$

- Finding $\boldsymbol{x}$ which is closest to another vector $\boldsymbol{y}$:

$$
\begin{aligned}
\min_{\boldsymbol{x}}\quad&\frac{1}{2}\|\boldsymbol{x}-\boldsymbol{y}\|_2^2 \\
\text{s.t.}\quad&\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b} \\
\end{aligned} \tag{67-6}
$$

- (Damped) Introducing the slack variable $\boldsymbol{s}$ and another vector $\boldsymbol{y}$:

$$
\begin{aligned}
\min_{\boldsymbol{x},\boldsymbol{s}}\quad&\frac{\alpha}{2}\|\boldsymbol{x}-\boldsymbol{y}\|_2^2+\frac{1}{2}\|\boldsymbol{s}\|_2^2 \\
\text{s.t.}\quad&\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}+\boldsymbol{s}
\end{aligned} \tag{67-7}
$$


<br>


### 66th Commit
#### Temporal Regularity of Wikipedia Consumption

Human life is driven by strong temporal regularities at multiple scales, with events and activities recurring in daily, weekly, month, yearly, or even longer periods. The large-scale and quantitative study of human behavior on digital platforms such as Wikipedia is important for understanding underlying dynamics of platform usage. There are some remarkable findings:

- The consumption habits of individual articles maintain strong diurnal regularities.
- The prototypical shapes of consumption patterns show a particularly strong distinction between articles preferred during the evening/night and articles preferred during working
hours.
- Topical and contextual correlates of Wikipedia articles’ access rhythms show the importance of article topic, reader country, and access device (mobile vs. desktop) for predicting daily attention patterns.



**References**

- Piccardi, T., Gerlach, M., & West, R. (2024, May). [Curious rhythms: Temporal regularities of wikipedia consumption](https://ojs.aaai.org/index.php/ICWSM/article/view/31386). In Proceedings of the International AAAI Conference on Web and Social Media (Vol. 18, pp. 1249-1261).


<br>


### 65th Commit
#### Definition of Granger Causality

Granger causality is primarily based on **predictability** in which predictability implies causality under some assumptions. It claims how well past values of a time series $$y_{t}$$ could predict future values of another time series $$x_{t}$$. Let $$\mathcal{H}_{<t}$$ be the history of all relevant information up to time $$t-1$$ for both time series, and $$\mathcal{P}(x_{t}\mid\mathcal{H}_{<t})$$ the optimal prediction of $$x_{t}$$ given $$\mathcal{H}_{<t}$$. If it always holds that

$$
\operatorname{var}[x_{t}-\mathcal{P}(x_{t}\mid\mathcal{H}_{<t})]<\operatorname{var}[x_{t}-\mathcal{P}(x_{t}\mid \mathcal{H}_{<t}\backslash y_{<t})]
$$

where $$\mathcal{H}_{<t}\backslash y_{<t}$$ indicates excluding the values of $$y_{<t}$$ from $$\mathcal{H}_{<t}$$. The variance of the optimal prediction error of $$x$$ is reduced by including the history information of $y$. Thus, $y$ is "causal" of $x$ if past values of $y$ improve the prediction of $x$.

In the case of bivariate time series, Granger causality corresponds to nonzero entries in the autoregressive coefficients such that

$$
\begin{aligned}
a_{x}^{0}x_{t}&=\sum_{k=1}^{d}a_{xx}^{k}x_{t-k}+\sum_{k=1}^{d}a_{xy}^{k}y_{t-k}+e_{t,x} \\
a_{y}^{0}y_{t}&=\sum_{k=1}^{d}a_{yy}^{k}y_{t-k}+\sum_{k=1}^{d}a_{yx}^{k}x_{t-k}+e_{t,y} \\
\end{aligned}
$$

where time series $y$ is Granger causal for time series $x$ if and only if $a_{xy}^{k}\neq 0$ for $k\in\{1,2,\ldots,d\}$.


**References**

- Shojaie, A., & Fox, E. B. (2022). [Granger causality: A review and recent advances](https://doi.org/10.1146/annurev-statistics-040120-010930). Annual Review of Statistics and Its Application, 9(1), 289-319.

<br>


### 64th Commit
#### [The Heilmeier Catechism](https://www.darpa.mil/about/heilmeier-catechism)

1. What are you trying to do? Articulate your objectives using absolutely no jargon.
2. How is it done today, and what are the limits of current practice?
3. What is new in your approach and why do you think it will be successful?
4. Who cares? If you are successful, what difference will it make?
5. What are the risks?
6. How much will it cost?
7. How long will it take?
8. What are the mid-term and final "exams" to check for success?

<br>


### 63rd Commit
#### Detecting Impact on Wikipedia Page Views

Wikipedia is the 5th most-visited website worldwide. Web traffic is often monitored to understand the user behavior on digital platforms, e.g., popularity of products and pages. Analyzing temporal observations allows one to allocate resources, optimizing energy in data centers, and detecting threat/anomaly. Counterfactual predictions can be used to estimate how various external campaigns or events affect readership on Wikipedia, detecting whether there are significant changes to the existing trends. The core is comparing the counterfactual predictions and actual page views to quantify the causal impact of the intervention.


**References**

- Chelsy Xie, X., Johnson, I., & Gomez, A. (2019, May). [Detecting and gauging impact on Wikipedia page views](https://dl.acm.org/doi/pdf/10.1145/3308560.3316751). In Companion Proceedings of The 2019 World Wide Web Conference (pp. 1254-1261).
- [Structural Time Series](https://structural-time-series.fastforwardlabs.com/).

<br>


### 62nd Commit
#### Time Series Analysis of Activity Patterns

Understanding collective technology adoption patterns (e.g., Wikipedia edit activity logs) can provide valuable inisghts into digital platform activities. Given the objective concerning the continuous usage of Wikipedia, time series analysis methods allow to study Wikipedia's temporal patterns of edit activity in terms of user satisfaction, information quality, and productivity enhancements. The alignment is modeled by the global similarity between two time series, which is actually formulated as a clustering problem. Each Wikipedia time series was represented as a temporally ordered set

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{a}=(a_1,a_2,\ldots, a_n)"/></p>

of edit activity levels at <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> months. In order to segment the time series into sequences of activity and inactivity periods, the data points are marked as either 0 or 1 for active and inactive months, respectively. 

Despite of importance of such analysis, unavailability of large-scale empirical data of Wikipedia life-cycle adoption is a critical bottleneck. In addition, exploring causal relationships of corporate Wikipedia allows to discover a certain temporal activity pattern.

**References**

- Arazy, O., & Croitoru, A. (2010). [The sustainability of corporate wikis: A time-series analysis of activity patterns](https://dl.acm.org/doi/pdf/10.1145/1877725.1877731). ACM Transactions on Management Information Systems (TMIS), 1(1), 1-24.

<br>


### 61st Commit
#### EUV Lithography Techniques

In almost every cutting-edge and high-tech device (e.g., that included CPU, GPU, SoC, DRAM, and SSD), the transistors inside microchips are incredibly small with the tiniest features measuring around 10 nanometers (nm). Each of microchips is made from connecting billions of transistors together, while each individual transistor is only nanometers in size.

The photolithography tools can be thought of as nanoscale microchip photocopiers. The state-of-the-art EUV Photolithography System uses Extreme Ultraviolet Light (EUV) and a set of mirrors to copy the design from photomask onto a silicon wafer, taking about 18 seconds to duplicate the same microchip design around 100 times across the entire area of a 300-millimeter wafer.

**References**

- [How does EUV Lithography Work? Inside the Most Advanced Machine Ever Made 🛠️⚙️🤯](https://www.youtube.com/watch?v=B2482h_TNwg). YouTube.

<br>


### 60th Commit
#### Optimal Sketching Bounds for Sparse Linear Regression



**References**

- Tung Mai, Alexander Munteanu, Cameron Musco, Anup B. Rao, Chris Schwiegelshohn, David P. Woodruff (2023). [Optimal Sketching Bounds for Sparse Linear Regression](https://arxiv.org/abs/2304.02261). arXiv:2304.02261.

<br>


### 59th Commit
#### Natural Gradient Descent

Suppose an objective function <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f(\boldsymbol{x}):\mathbb{R}^{n}\to\mathbb{R}"/> in an optimization problem, the gradient descent is the vector of partial derivatives of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f(\boldsymbol{x})"/> with respect to each variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_i,i=1,2,\ldots,n"/> such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\nabla f(\boldsymbol{x})=\left(\frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \cdots, \frac{\partial f(\boldsymbol{x})}{\partial x_n}\right)^\top\in\mathbb{R}^{n}"/></p>

The essential idea of gradient descent update is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{\ell+1}=\boldsymbol{x}_{\ell}-\alpha\nabla f(\boldsymbol{x}_{\ell})"/></p>

with a certain learning rate <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\alpha\in\mathbb{R}"/>.

Recall that [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) for any function <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f(x)"/> can be defined as

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^{n}"/></p>

where 

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;n!=\prod_{k=1}^{n}k"/> </p>

denotes the factorial of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/>. The function <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f^{(n)}(a)"/> denotes the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/>th derivative of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f"/> evaluated at the point <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;a"/>.

Based on steepest gradient descent, the optimization problem at the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell+1"/> step is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{\ell+1}=\arg\min_{\boldsymbol{x}}\,f(\boldsymbol{x}_{\ell})+\nabla f(\boldsymbol{x}_{\ell})^\top(\boldsymbol{x}-\boldsymbol{x}_{\ell})+D(\boldsymbol{x},\boldsymbol{x}_{\ell})"/> </p>

for any arbitrary metric <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;D(\boldsymbol{x},\boldsymbol{x}_{\ell})"/> that measures a dissimilarity or distance between <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{\ell}"/>. 

In particular, if we use KL divergence <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;D_{KL}(p(y\mid \boldsymbol{x})\mid\mid p(y\mid\boldsymbol{x}_{\ell}))"/> with some data <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;y"/> as the metric, then the update for the natural gradient is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{\ell+1}=\boldsymbol{x}_{\ell}-\alpha\boldsymbol{F}^{-1}\nabla f(\boldsymbol{x}_{t})"/> </p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{F}"/> is the Fisher information matrix.

**References**

- Andy Jones. [Natural gradients](https://andrewcharlesjones.github.io/journal/natural-gradients.html). Technical blog.
- Josh Izaac (2019). [Quantum natural gradient](https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient/). Blog.

<br>


### 58th Commit
#### Sum of Squares (SOS) Technique

SOS is a classical method for solving polynomial optimization. The feasible set with a multivariate polynomial function

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;p(x)=\sum_{\alpha=0}^{n}a_{\alpha}x^{\alpha}\geq 0,\forall x"/></p>

if 

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;p(x)=\sum_{\alpha=0}^{n}q_{\alpha}^{2}(x), q_{\alpha}(x)\in\mathbb{R}[x]"/></p>

becomes an SOS by using semidefinite programming, i.e., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;p(x)=w(x)^\top Qw(x)"/> with a semidefinite matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;Q"/>. Suppose the polynomial function such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;p(x,y)=2x^4+5y^4+2x^3y"/></p>

The SOS is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned}
&\begin{bmatrix}
x^2 & xy & y^2
\end{bmatrix}\begin{bmatrix}
2 & 1 & -1 \\ 1 & 2 & 0 \\ -1 & 0 & 5 \\
\end{bmatrix}\begin{bmatrix}
x^2 \\ xy \\ y^2 \\
\end{bmatrix} \\
=&\begin{bmatrix}
2x^2+xy-y^2 & x^2+2xy & -x^2+5y^2
\end{bmatrix}\begin{bmatrix}
x^2 \\ xy \\ y^2 \\
\end{bmatrix} \\
=&(2x^4+x^3y-x^2y^2)+(x^3y+2x^2y^2)+(-x^2y^2+5y^4) \\
=&2x^4+5y^4+2x^3y
\end{aligned}
"/></p>


SOS can optimize directly over the sum-of-squares cone and its dual, circumventing the semidefinite programming (SDP) reformulation, which requires a large number of auxiliary variables when the degree of sum-of-squares polynomials is large.

**References**

- Pablo A. Parrilo. [Sum of squares techniques and polynomial optimization](https://www.princeton.edu/~aaa/Public/Presentations/CDC_2016_Parrilo_1). [[MIT Course -- 6.7230 Algebraic Techniques and Semidefinite Optimization](https://www.mit.edu/~parrilo/)]
- Dávid Papp, Sercan Yildiz (2019). [Sum-of-Squares Optimization without Semidefinite Programming](https://doi.org/10.1137/17M1160124). SIAM Journal on Optimization. 29 (1).
- Yang Zheng, Giovanni Fantuzzi (2023). [Sum-of-squares chordal decomposition of polynomial matrix inequalities](https://doi.org/10.1007/s10107-021-01728-w). Mathematical Programming. Volume 197, pages 71–108.

<br>


### 57th Commit
#### Subspace-Conjugate Gradient

Solving multi-term linear equations efficiently in numerical linear algebra is still a challenging problem. For any <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}_{k}\in\mathbb{R}^{m\times m},\boldsymbol{B}_{k}\in\mathbb{R}^{n\times n},\,k\in[d]"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{C}\in\mathbb{R}^{m\times n}"/>, the Sylvester equation is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\sum_{k\in[d]}\boldsymbol{A}_k\boldsymbol{X}\boldsymbol{B}_k=\boldsymbol{C}"/></p>

whose closed-form solution is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\operatorname{vec}(\boldsymbol{X})=\Bigl(\sum_{k\in[d]}\boldsymbol{B}_k^\top\otimes\boldsymbol{A}_k\Bigr)^\dagger\operatorname{vec}(\boldsymbol{C})"/></p>

Recent study proposed a new iterative scheme for symmetric and positive definite operators, significantly advancing methods such as truncated matrix-oriented Conjugate Gradients (CG). The new algorithm capitalizes on the low-rank matrix format of its iterates by fully exploiting the subspace information of the factors as iterations proceed.

**References**

- Davide Palitta, Martina Iannacito, Valeria Simoncini (2025). [A subspace-conjugate gradient method for linear matrix equations](https://arxiv.org/abs/2501.02938). arXiv:2501.02938.

<br>


### 56th Commit
#### Characterizing Wikipedia Linking Across the Web

[Common Crawl](https://commoncrawl.org/) maintains a free, open repository of web crawl data that can be used by anyone. Using the dataset with 90 million English Wikipedia links spanning 1.68% of Web domains, recent study ([Veselovsky et al, 2025](https://arxiv.org/abs/2505.15837)) reveals three key findings:

- Wikipedia is most frequently cited by news and science websites for informational purposes, while commercial websites reference it less often.
- The majority of Wikipedia links appear within the main content rather than in boilerplate or user-generated sections, highlighting their role in structured knowledge presentation. 
- Most links (95%) serve as explanatory references rather than as evidence or attribution, reinforcing Wikipedia’s function as a background knowledge provider.

**References**

- Veniamin Veselovsky, Tiziano Piccardi, Ashton Anderson, Robert West, Akhil Arora (2025). [Web2Wiki: Characterizing Wikipedia Linking Across the Web](https://arxiv.org/abs/2505.15837). arXiv:2505.15837.
- Cristian Consonni, David Laniado, Alberto Montresor (2019). [WikiLinkGraphs: A Complete, Longitudinal and Multi-Language Dataset of the Wikipedia Link Networks](https://arxiv.org/abs/1902.04298). arXiv:1902.04298.
- [Interactive Map of Wikipedia - GitHub Pages](https://lmcinnes.github.io/datamapplot_examples/wikipedia/).

<br>


### 55th Commit
#### Random Projections for Ordinary Least Squares

For any <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{n}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\Phi}\in\mathbb{R}^{n\times d}"/> (i.e., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> observations and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d"/> features), the linear regression such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{\theta}\in\mathbb{R}^{d}}\,\|\boldsymbol{y}-\boldsymbol{\Phi}\boldsymbol{\theta}\|_2^2"/></p>

can be replaced by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{\theta}\in\mathbb{R}^{d}}\,\|\boldsymbol{S}\boldsymbol{y}-\boldsymbol{S}\boldsymbol{\Phi}\boldsymbol{\theta}\|_2^2"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}\in\mathbb{R}^{s\times n}"/> is an i.i.d. Gaussian matrix. We typically have <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n>s>d"/> (more observations than the feature dimension), and one of the benefits of sketching is to be able to store a reduced representation of the data (i.e., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathbb{R}^{s\times d}"/> instead of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathbb{R}^{n\times d}"/>).

The matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}\boldsymbol{\Phi}\in\mathbb{R}^{s\times d}"/> is a subspace embedding for <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\Phi}\in\mathbb{R}^{n\times d}"/> if

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;(1-\epsilon)\|\boldsymbol{\Phi}\boldsymbol{\theta}\|_2^2\leq \|\boldsymbol{S}\boldsymbol{\Phi}\boldsymbol{\theta}\|_2^2\leq (1+\epsilon)\|\boldsymbol{\Phi}\boldsymbol{\theta}\|_2^2"/></p>

for all <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}\in\mathbb{R}^{d}"/>.

A sketching matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}\in\mathbb{R}^{d\times s}"/> of random projection can also be introduced to

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{\eta}\in\mathbb{R}^{s}}\,\|\boldsymbol{y}-\boldsymbol{\Phi}\boldsymbol{S}\boldsymbol{\eta}\|_2^2"/></p>

for the case of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d>n>s"/> (i.e., underdetermined). This corresponds to replacing the feature vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\phi(\boldsymbol{x})\in\mathbb{R}^{d}"/> by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}^\top\phi(\boldsymbol{x})\in\mathbb{R}^s"/>.

**References**

- Ethan N. Epperly. [Does Sketching Work?](https://www.ethanepperly.com/index.php/2023/11/13/does-sketching-work/)
- Francis Bach (2025). Learning Theory from First Principles. **Chapter 10.2**. MIT Press. [[PDF](https://www.di.ens.fr/~fbach/ltfp_book.pdf)]
- Yuji Nakatsukasa, Joel A Tropp (2024). Fast and accurate randomized algorithms for linear systems and eigenvalue problems. SIAM Journal on Matrix Analysis and Applications. 45 (2): 1183-1214.
- Christopher Musco. [Tutorial on Matrix Sketching](https://www.birs.ca/workshops/2023/23w5108/files/Christopher%20Musco/sketching_tutorial.pdf).
- Moses Charikar. [Lecture 19: Sparse Subspace Embeddings](https://cs368-stanford.github.io/spring2022/lectures/lec19.pdf). CS369G: Algorithmic Techniques for Big Data.
- [Tensor sketch](https://en.m.wikipedia.org/wiki/Tensor_sketch). Wikipedia.

<br>


### 54th Commit
#### The Challenge of Insuring Vehicles with Autonomous Functions



<br>

**References**

- Oliver Wyman. [The Challenge of Insuring Vehicles with Autonomous Functions](https://www.oliverwyman.com/our-expertise/insights/2017/dec/risk-journal-vol-7/redefining-business-models/the-challenge-of-insuring-vehicles-with-autonomous-functions.html). [[PDF](https://www.oliverwyman.com/content/dam/oliver-wyman/v2/publications/2017/dec/The_Challenge_of_Insuring_Vehicles_with_Autonomous_Functions.pdf)]

<br>


### 53rd Commit
#### Two Decades of Low-Rank Optimization

Semidefinite programming (SDP) is powerful for solving low-rank optimization. Despite the most classical algorithms of SDP, one can use first-order methods to solve bigger SDPs more efficiently, including dual-scaling method, spectral bundle method, nonlinear programming approaches, dual Cholesky approach, chordal-graph approaches, and iterative solver for the Newton system.

Low-rank matrix optimization can be formulated as

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned}\min_{\boldsymbol{X}}\,&\langle\boldsymbol{C},\boldsymbol{X}\rangle \\ \text{s.t.}\,&\langle\boldsymbol{A}_{i},\boldsymbol{X}\rangle=b_i,\,\forall i=1,2,\ldots,m \\ &\boldsymbol{X}\succeq 0 \end{aligned}"/></p>

There exists an optimal solution <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}^{*}"/> with rank <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;r^{*}"/> satisfying <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;r^{*}<\sqrt{2m}"/>. For almost all cost matrices <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{C}"/>, the first- and second-order necessary optimality conditions are sufficiently for global optimality.

One idea to verify the global optimality is summarized as follows. Suppose the original SDP feasible set is compact with interior. Consider the rank-constrained SDP, enforcing <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\text{rank}(\boldsymbol{X})\leq r"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;r\geq \lceil \sqrt{2(m+1)}\rceil"/>. Let <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\bar{\boldsymbol{X}}"/> be a local minimum of the rank-constrained SDP. If <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\text{rank}(\boldsymbol{X})< r"/>, then <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\bar{\boldsymbol{X}}"/> is optimal for the original SDP.

Benign nonconvexity refers to a property of certain nonconvex optimization problems where, despite the lack of global convexity, the problem exhibits characteristics that make it tractable—meaning efficient optimization methods can still find good (often global) solutions.


**References**

- Sam Burer (2023). Two decades of low-rank optimization. [[YouTube](https://www.youtube.com/watch?v=wSauUgRIQDg)]

<br>

### 52nd Commit
#### Matrix Completion and Decomposition in Phase-Bounded Cones

The nuclear norm minimization for matrix completion

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{X}}\,\|\boldsymbol{X}\|_{*}\quad\text{s.t.}\,\mathcal{P}_{\Omega}(\boldsymbol{X})=\mathcal{P}_{\Omega}(\boldsymbol{Y})"/></p>

is equivalent to the semidefinite programming such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{X},\boldsymbol{W}_1,\boldsymbol{W}_2}\,\frac{1}{2}\text{tr}(\boldsymbol{W}_1+\boldsymbol{W}_2)\quad \text{s.t.}\,\begin{bmatrix} \boldsymbol{W}_1 & \boldsymbol{X} \\ \boldsymbol{X}^\top & \boldsymbol{W}_2 \end{bmatrix}\succeq 0,\quad \mathcal{P}_{\Omega}(\boldsymbol{X})=\mathcal{P}_{\Omega}(\boldsymbol{Y})"/></p>

where the feasible set is the positive semidefinite cone. There is a phase-bounded cone for matrix completion with chordal graph pattern, i.e., phase-bounded completions of a completable partial matrix with a block bounded pattern.


**References**

- Ding Zhang, Axel Ringl, and Li Qiu (2025). Matrix Completion and Decomposition in Phase-Bounded Cones. SIAM Journal on Matrix Analysis and Applications. [[DOI](https://doi.org/10.1137/23M1626529)]

<br>


### 51st Commit
#### Revisiting Interpretable Machine Learning (IML)

**Local** IML methods explain individual predictions of ML models. Popular IML methods are Shapley values and counterfactual explanations. Counterfactual explanations explain predictions in the form of what-if scenarios, they are contrastive and focus on a few reasons. The Shapley values provide an answer on how to fairly share a payout among the players of a collaborative game.

**Global** model-agnostic explanation methods are used to explain the expected model behavior, i.e., how the model behaves on average for a given dataset. A useful distinction of global explanations are feature importance and feature effect. **Feature importance** ranks features based on how relevant they were for the prediction. One of the most popular importance measures is permutation feature importance, originated from random forests. **Feature effect** expresses how a change in a feature changes the predicted outcome. 

There are many challenges in IML methods: 1) uncertainty quantification of the explanation, 2) causal interpretation for reflecting the true causal structure of its underlying phenomena, and 3) feature dependence.


**References**

- Christoph Molnar,  Giuseppe Casalicchio,  and Bernd Bischl (2020). Interpretable Machine Learning – A Brief History, State-of-the-Art and Challenges. arXiv preprint arXiv:2010.09337.

<br>


### 50th Commit
#### Orthogonal Procrustes Problem

Ever heard of the Orthogonal Procrustes Problem (OPP)? It might sound complex, but the optimal solution can be achieved in just two simple steps:

- Singular Value Decomposition (SVD)
- Matrix Multiplication

That’s it! This elegant approach helps find the closest orthogonal matrix to a given one, minimizing the Frobenius norm. 

📌 Key Takeaways:

- OPP is a powerful tool for matrix alignment and optimization.
- The solution is computationally efficient with SVD at its core.
- Python makes implementation a breeze—just a few lines of code!

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/opp.png" width="650" />
</p>


<br>


### 49th Commit
#### Amazon Deforestation

[Deforestation in the Amazon: past, present and future](https://infoamazonia.org/en/2023/03/21/deforestation-in-the-amazon-past-present-and-future/) visually analyzed the deforestation rates of recent years, identifying the main threats (e.g., cattle-raising activity, road network, and navigable rivers) in the present and pointing to measures needed to reverse this process. There are some basic data:

- In 2001, the forest cover of the Amazon occupied over **600 million hectares**.
- Between 2001 and 2020, the deforestation in the Amazon totalled about **54.2 million hectares**, the equivalent of 9% of the forest cover.

The Amazon could lose almost half of what it lost in the past two decades.

<br>


### 48th Commit
#### Convergence Rates

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/convergence_rate.png" width="650" />
</p>

<br>

**References**

- [Rates of convergence](https://bookdown.org/rdpeng/advstatcomp/rates-of-convergence.html)

<br>

### 47th Commit
#### [The Art of Linear Programming](https://www.youtube.com/watch?v=E72DWgKP_1Y)

Linear programming is an important technique for solving NP-hard problems such as Knapsack (e.g., packing problem), TSP, and Vertex Cover.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/linear_prog_farmer_profit.png" width="600" />
</p>

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/knapsack.png" width="600" />
</p>

<br>

### 46th Commit
#### Semidefinite Programming

The basic definition of positive definite matrix is that: For any square matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{n\times n}"/>, if it always holds that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}^\top\boldsymbol{A}\boldsymbol{x}>0"/></p>

with any <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{n}"/> not being a vector of zeros, then <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\succ 0"/> is a positive definite matrix. Similarly, we can define a positive semidefinite matrix as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\succeq 0 \Leftrightarrow \boldsymbol{x}^\top\boldsymbol{A}\boldsymbol{x}\geq 0,\forall \boldsymbol{x}\neq\boldsymbol{0}"/></p>

Semidefinite programming is the most exciting development of mathematical programming techniques in 1990s. One can leverage an initial point <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{u}_0"/>, following by the governing equation <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{u}_{t+1}=\boldsymbol{A}\boldsymbol{u}_{t},t=0,1,2,\ldots"/>, to build a dynamical system as shown below.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/dyna_system_Aut_toy.png" width="600" />
</p>

<br>

**References**

- [What Does It Mean For a Matrix to be POSITIVE? The Practical Guide to Semidefinite Programming(1/4)](https://www.youtube.com/watch?v=2uKoKKLgZ4c)
- [The Practical Guide to Semidefinite Programming (2/4)](https://www.youtube.com/watch?v=9z2OtPOi8T0)
- [Stability of Linear Dynamical Systems: The Practical Guide to Semidefinite Programming (3/4)](https://www.youtube.com/watch?v=K2Mw3fUYaHI)
- [Goemans-Williamson Max-Cut Algorithm: The Practical Guide to Semidefinite Programming (4/4)](https://www.youtube.com/watch?v=aFVnWq3RHYU)

<br>

### 45th Commit
#### Physics-Informed Machine Learning

Reviewed machine learning algorithms:
- Deterministic regression
  - Linear regression
  - Decision tree
  - Machine learning algorithms: Sparse Identification of Nonlinear Dynamics (SINDy)
  - Neural networks
- Probabilistic regression (Gaussian process regression)
- Ensemble methods
  - Bagging (Random forest)
  - Boosting (Gradient boosting machine, XGB regressor)

**References**
- Navid Zobeiry. [Physics-Informed Machine Learning](https://www.youtube.com/playlist?list=PLJ6U7kzSli-3h6iMQ7Ww_Madv_TleW75E): A nine-lecture series on Physics-Informed Machine Learning (PIML) delivered by Professor Navid Zobeiry. This course introduces the key techniques of PIML and demonstrates how integrating physics-based constraints with machine learning (ML) can help solve complex multi-physics challenges in engineering.

<br>

### 44th Commit
#### Sparse Linear Regression

For any vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{m}"/> and matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{m\times n}"/>, the sparse linear regression such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{align} \min_{\boldsymbol{w}}\,&\|\boldsymbol{y}-\boldsymbol{A}\boldsymbol{w}\|_2^2 \\ \text{s.t.}\,&\|\boldsymbol{w}\|_0\leq \tau \end{align}"/></p>

There might be two solutions: 

1) Mixed-integer programming such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{align} \min_{\boldsymbol{w},\boldsymbol{\beta}}\,&\|\boldsymbol{y}-\boldsymbol{A}\boldsymbol{w}\|_2^2 \\ \text{s.t.}\,&\begin{cases} \|\boldsymbol{\beta}\|_1\leq \tau,\,\boldsymbol{\beta}\in\{0,1\}^{n}, \\ -M\cdot\boldsymbol{\beta}\leq\boldsymbol{w}\leq M\cdot\boldsymbol{\beta} \end{cases} \end{align}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;M"/> is a sufficiently large constant.

2) Semidefinite programming such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{align} \min_{\boldsymbol{X},\boldsymbol{w},\boldsymbol{\beta}}\,&\|\boldsymbol{y}-\boldsymbol{A}\boldsymbol{w}\|_2^2 \\ \text{s.t.}\,&\begin{cases} \begin{bmatrix} 1 & \boldsymbol{w}^\top \\ \boldsymbol{w} & \boldsymbol{X} \\ \end{bmatrix}  \succeq 0, \\ x_{i}\leq M^2\beta_{i},\forall i\in[n], \\ \|\boldsymbol{\beta}\|_1\leq \tau,\,\boldsymbol{\beta}\in\{0,1\}^{n}, \\ \end{cases} \end{align}"/></p>

MIP provides exact solution, but it scales poorly with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/>. SDP has better scalability, but not exact in most cases.

<br>

### 43rd Commit
#### [Matrix Calculus for Machine Learning and Beyond](https://github.com/mitmath/matrixcalc)


<br>


### 42nd Commit
#### Causal Inference, Causal Discovery, and Machine Learning

Causal inference is a framework to answer causal questions from observational and/or experimental data. It is important to infer underlying mechanisms of data (e.g., climate system), learn correlation networks, and recognize patterns. Pearl's causal inference framework assumes an underlying **structural causal model** with an associated acyclic **graph**.

There are two types of tasks in the causal inference framework: 1) Utilizing qualitative causal knowledge in the form of **directed acyclic graphs**; 2) Learning causal graphs based on general assumption.

**References**

- [Causal Inference, Causal Discovery, and Machine Learning with Jakob Runge](https://www.youtube.com/watch?v=R5JMeEy9koA&t=732s). YouTube.

<br>


### 41st Commit
#### Composable Optimization for Robotic Motion Planning and Control

From a perspective of control as optimization, the objective could be what one wants system to do, given the model of one's robot as constraints. This is exactly an optimal control problem such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{x(t),u(t)}\,&J(x(t),u(t))=\int_{0}^{T}L(x(t),u(t))\,\mathrm{d}t \\ \text{s.t.}\,&\begin{cases} \dot{x}=f(x,u) \\ u_{\text{min}}\leq u\leq u_{\text{max}} \end{cases} \end{aligned}"/></p>


**References**

- [MIT Robotics - Zac Manchester - Composable Optimization for Robotic Motion Planning and Control](https://www.youtube.com/watch?v=eSleutHuc0w). YouTube.

<br>


### 40th Commit
#### Cauchy-Schwarz Regularizers

The main idea is that Cauchy-Schwarz inequality <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;|\langle\boldsymbol{x},\boldsymbol{y}\rangle|\leq\|\boldsymbol{x}\|_2\|\boldsymbol{y}\|_2"/> can be used to binarize neural networks. Cauchy-Schwarz regularizers are a new class of regularization that can promote discrete-valued vectors, eigenvectors of a given matrix, and orthogonal matrices. These regularizers are effective for quantizing neural network weights and solving underdetermined systems of linear equations.

**References**
- Sueda Taner, Ziyi Wang, Christoph Studer (2025). [Cauchy-Schwarz Regularizers](https://arxiv.org/abs/2503.01639). ICLR 2025.

<br>


### 39th Commit
#### Nystrom Truncation of Spectral Features

**Parameterizing policy gradient.** (**Mercer's Theorem**) If <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K:\mathcal{X}\times\mathcal{X}\to\mathbb{R}"/> is a continuous, symmetric, and positive definite kernel, then there exists a sequence of non-negative eigenvalues <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\{\lambda_i\}_{i=1}^{\infty}"/> and corresponding orthonomal basis <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\{\phi_i\}_{i=1}^{\infty}"/> such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;K(s,t)=\sum_{i=1}^{\infty}\lambda_i\phi_i(s)\phi_i(t),\,\forall (s,t)\in\mathcal{X}\times\mathcal{X}"/></p>

In practice, for large datasets, computing the full kernel matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/> and its eigenvalue decomposition (EVD) is in <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{O}(n^3)"/> time and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{O}(n^2)"/> memory. 

The Nystrom method approximates the kernel matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/> by selecting a subset. Let <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K_1"/> be the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;m\times m"/> kernel matrix for the subset and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K_2"/> be the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n\times m"/> kernel matrix between the full dataset and the subset, then the Nystrom approximation of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/> is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;K\approx K_2K_1^{-1}K_2^\top"/></p>

Furthermore, let <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K_1=U\Lambda_1U^\top"/> be the EVD, then

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\Phi}=K_2U\Lambda_1^{-1/2},\quad\tilde{\Lambda}=\Lambda_1"/></p>

corresponding to eigenvectors and eigenvalues. Selecting top-<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;k"/> eigenvalues and eigenvectors, denoted by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\Lambda}_k"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\tilde{\Phi}_k"/>, respectively. The truncation approximation of the kernel matrix becomes

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;K\approx\tilde{\Phi}_k\tilde{\Lambda}_k\tilde{\Phi}_k^\top"/></p>

**References**
- Na Li (Harvard). [Representation-based Learning and Control for Dynamical Systems](https://www.youtube.com/watch?v=Z7QB8-vu8sY). YouTube.

<br>


### 38th Commit
#### Sparse Dictionary Learning

Interpretable machine learning provides a data-driven framework for understanding complicated dynamical systems. One important perspective is sparsity to reinforce the interpretability of several state-of-the-art models. Sparse dictionary learning stems from sparse signal processing, which takes the form of linear regression with sparse parameters.

In a very recent study, researchers developed an interpretable and efficient reinforcement learning model for sparse dictionary learning. The essential idea is iteratively improving control and dynamics on the data <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;(\boldsymbol{X},\boldsymbol{U})"/> where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}"/> is the velocity matrix and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{U}"/> could be angles. Reinforcement learning can train policies in the dictionary approximation. However, one of the challenges is how to address the overfitting issue. In the modeling process, uncertainty quantification is also meaningful to improve the robustness of the system control.

**References**
- Nicholas Zolman, Urban Fasel, J. Nathan Kutz, Steven L. Brunton (2024). [SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning](https://arxiv.org/abs/2403.09110). arXiv:2403.09110.

<br>


### 37th Commit
#### Cardinality Minimization, Constraints, and Regularization

This is a review paper for solving the optimization problem that involves the cardinality of variable vectors in constraints or objective function. The problems can be formulated as follows,

- Cardinality minimization problems:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&\|\boldsymbol{x}\|_0 \\ \text{s.t.}\,&\boldsymbol{x}\in\mathcal{X}\subset\mathbb{R}^{n} \end{aligned}"/></p>

- Cardinality-constrained problems:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&f(\boldsymbol{x}) \\ \text{s.t.}\,&\|\boldsymbol{x}\|_0\leq k,\quad\boldsymbol{x}\in\mathcal{X}\subset\mathbb{R}^{n} \end{aligned}"/></p>

- Regularized cardinality problems:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&\|\boldsymbol{x}\|_0+\rho(\boldsymbol{x}) \\ \text{s.t.}\,&\boldsymbol{x}\in\mathcal{X}\subset\mathbb{R}^{n} \end{aligned}"/></p>


These optimization problems have broad applications such as signal and image processing, portfolio selection, and machine learning. 

**References**

- Andreas M. Tillmann, Daniel Bienstock, Andrea Lodi, Alexandra Schwartz (2024). [Cardinality Minimization, Constraints, and Regularization: A Survey](https://doi.org/10.1137/21M142770X). SIAM Review, 66(3). [[PDF](https://arxiv.org/pdf/2106.09606)]

<br>


### 36th Commit
#### Single-Factor Matrix Decomposition with Sparse Penalty

For any positive semidefinite matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}\in\mathbb{R}^{n\times n}"/>, the optimization problem of rank-one single-factor matrix decomposition with sparse penalty can be formulated as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{x}}\,\frac{1}{2}\|\boldsymbol{Y}-\boldsymbol{x}\boldsymbol{x}^{\top}\|_F^2+\frac{\lambda}{2}\|\boldsymbol{x}\|_1"/></p>

can be solved by the following algorithm:

- Initialize <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> as the unit vector with equal entries;
- Repeat
  - Compute

  <p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}:=\mathcal{S}_{\lambda}(\boldsymbol{Y}\boldsymbol{x})/\|\mathcal{S}_{\lambda}(\boldsymbol{Y}\boldsymbol{x})\|_2"/></p>

- Until convergence
- Compute <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d=\boldsymbol{x}^{\top}\boldsymbol{Y}\boldsymbol{x}"/> (referring to the singular value);
- Compute <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}:=\sqrt{d}\boldsymbol{x}"/>.

In the algorithm, the soft-thresholding operator is defined as

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;[\mathcal{S}_{\lambda}(\boldsymbol{x})]_{i}=\begin{cases} x_{i}-\lambda, & \text{if}\,x_{i}>t \\ x_{i}+\lambda, & \text{if}\,x_{i}<-t \\ 0, & \text{otherwise} \end{cases}"/></p>

for all <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;i\in\{1,2,\ldots,n\}"/>.

**References**

- (**Deng et al., 2021**) Correlation tensor decomposition
and its application in spatial imaging data. Journal of the American Statistical Association. [[DOI](https://doi.org/10.1080/01621459.2021.1938083)] (see Algorithm 2)

- (**Witten et al., 2009**) A penalized matrix
decomposition, with applications to sparse principal components and
canonical correlation analysis. Biostatistics. [[DOI](https://doi.org/10.1093/biostatistics/kxp008)]

<br>


### 35th Commit
#### Iterative Shrinkage Thresholding Algorithm (ISTA)

In machine learning, the closed-form solution to LASSO is defined upon the soft thresholding operator such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{S}_{\lambda t}(\boldsymbol{\beta})=\arg\min_{\boldsymbol{z}}\,\frac{1}{2t}\|\boldsymbol{\beta}-\boldsymbol{z}\|_2^2+\lambda\|\boldsymbol{z}\|_1"/></p>

element-wise, we have

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;[\mathcal{S}_{\lambda t}(\boldsymbol{\beta})]_{i}=\begin{cases} \beta_i-\lambda t, & \text{if}\, \beta_i>\lambda t \\ \beta_i+\lambda t, & \text{if}\, \beta_i<-\lambda t \\ 0, & \text{otherwise} \end{cases}"/></p>

for all <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;i\in\{1,2,\ldots, n\}"/>.

Considering the optimization problem

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{\beta}}\,\frac{1}{2}\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\|_2^2+\lambda\|\boldsymbol{\beta}\|_1"/></p>

The proximal gradient update can be written as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\beta}:=\mathcal{S}_{\lambda t}(\boldsymbol{\beta}+t\boldsymbol{X}^\top (\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}))"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> is the step size, and the gradient of the first component in the objective function is <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;-\boldsymbol{X}^\top(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})"/>.

**References**

- Ryan Tibshirani. [Proximal Gradient Descent (and Acceleration)](https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf).
- Xiaohan Chen, Jialin Liu, Zhangyang Wang, Wotao Yin (2021). [Hyperparameter Tuning is All You Need for LISTA](https://proceedings.neurips.cc/paper/2021/file/60c97bef031ec312b512c08565c1868e-Paper.pdf). NeurIPS 2021.

<br>


### 34th Commit
#### Learning Sparse Nonparametric Directed Acyclic Graphs (DAG)

DAG learning problem: Given a data matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{n\times d}"/> consisting of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> independent and identically distributed observations and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;d"/> column vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\{\boldsymbol{x}_{j}\}_{j=1}^{d}"/>, one can learn the DAG <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{G}(\boldsymbol{X})"/> that encodes the dependency between the variables in <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}"/>. One approach is to learn <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f=f(f_1,f_2,\cdots,f_d)"/> such that <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{G}(f)=\mathcal{G}(\boldsymbol{X})"/> using a well-designed score. Given a loss function <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell(y,\hat{y})"/> such as least squares or the negative log-likelihood, the optimization problem can be summarized as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{f}\,&\frac{1}{n}\sum_{j=1}^{d}\ell(\boldsymbol{x}_j,f_j(\boldsymbol{X})) \\ \text{s.t.}\,&\mathcal{G}(f)\in\text{DAG} \end{aligned}"/></p>

Two challenges in this formulation:

- How to enforce the acyclicity constraint that <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{G}(f)\in\text{DAG}"/>?
- How to enforce sparsity in the learned DAG <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{G}(f)"/>?

If one uses MLP in the optimization, then it becomes

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{\theta}}\,&\frac{1}{n}\sum_{j=1}^{d}\ell(\boldsymbol{x}_j,\text{MLP}(\boldsymbol{X};\boldsymbol{\theta}_j))+\lambda\|\boldsymbol{A}_{j}^{(1)}\|_{1,1} \\ \text{s.t.}\,&h(W(\boldsymbol{\theta}))=0 \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}=\{\boldsymbol{\theta}_{j}\}_{j=1}^{d}"/> denotes all parameters and the parameters of the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;j"/>th MLP are <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\theta}_j=(\boldsymbol{A}_{j}^{(1)},\boldsymbol{A}_{j}^{(2)},\cdots, \boldsymbol{A}_{j}^{(h)})"/>.

**References**

- Xun Zheng, Bryon Aragam, Pradeep Ravikumar, Eric P. Xing (2018). [DAGs with NO TEARS: Continuous Optimization for Structure Learning](https://arxiv.org/abs/1803.01422). arXiv:1803.01422.
- Xun Zheng, Chen Dan, Bryon Aragam, Pradeep Ravikumar, Eric P. Xing (2019). [Learning Sparse Nonparametric DAGs](https://arxiv.org/abs/1909.13189). arXiv:1909.13189.
- Victor Chernozhukov, Christian Hansen, Nathan Kallus, Martin Spindler, Vasilis Syrgkanis (2024). [Applied Causal Inference Powered by ML and AI](https://arxiv.org/abs/2403.02467). arXiv:2403.02467. (See Chapter 7)

<br>


### 33rd Commit
#### Learning Sparse Nonlinear Dynamics via Mixed-Integer Optimization

Discovering governing equations of complex dynamical systems directly from data is a central problem in scientific machine learning. In recent years, the sparse identification of nonlinear dynamics (SINDy, see [Brunton et al., (2016)](https://doi.org/10.1073/pnas.1517384113)) framework, powered by heuristic sparse regression methods, has become a dominant tool for learning parsimonious models. The optimization problem for learning system equations is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{\xi},\boldsymbol{z}}\,&\|\dot{\boldsymbol{X}}_{j}-\boldsymbol{\Theta}(\boldsymbol{X})\boldsymbol{\xi}\|_2^2+\lambda\|\boldsymbol{\xi}\|_2^2 \\ \text{s.t.}\,&\begin{cases} M_i^{\ell}z_{i}\leq \xi_{i}\leq M_i^{u}z_{i} \\ \displaystyle\sum_{i=1}^{D}z_{j}\leq k_j,\,z_{i}\in\{0,1\} \end{cases} \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;M_i^{\ell},M_{i}^{u}"/> are lower and upper bounds on the coefficients. The full system dynamics are <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\xi}"/>.

**References**

- Bertsimas, D., & Gurnee, W. (2023). [Learning sparse nonlinear dynamics via mixed-integer optimization](https://link.springer.com/article/10.1007/s11071-022-08178-9). Nonlinear Dynamics, 111(7), 6585-6604.

<br>


### 32nd Commit

- [Behavioral changes during the COVID-19 pandemic decreased income diversity of urban encounters](https://www.nature.com/articles/s41467-023-37913-y)
- [COVID-19 is linked to changes in the time–space dimension of human mobility](https://www.nature.com/articles/s41562-023-01660-3)
- [Evacuation patterns and socioeconomic stratification in the context of wildfires in Chile](https://arxiv.org/pdf/2410.06017)
- [Fine tune an LLM](https://github.com/phunterlau/code-in-blog/blob/main/understand_x/understand-x.pdf)
- [Learning with Combinatorial Optimization Layers: A Probabilistic Approach](https://arxiv.org/pdf/2207.13513)

<br>


### 31st Commit

#### <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>-Statistic & Student <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>-Distribution

Given population mean <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mu"/>, suppose the sample mean <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\bar{x}"/>, sample standard deviation <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;s"/>, and sample size <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> (small value), the formula of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>-statistic for small sample sizes is written as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;t=\frac{\bar{x}-\mu}{s/\sqrt{n}}"/></p>

A high absolute value of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> suggests a statistically significant difference. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;H_0"/> is the null hypothesis, namely, the population mean is <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mu"/>. Below is the student <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>-distribution with a <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;95%"/> confidence interval.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/student_t_test.png" width="500" />
</p>

<br>

Please check out the details of [the relevance of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>-statistics for small sample sizes](https://xinychen.github.io/slides/t_stat.pdf) as the teaching sample.

**Supporting Materials**

- [Hypothesis Testing Problems - Z Test & T Statistics - One & Two Tailed Tests 2](https://www.youtube.com/watch?v=zJ8e_wAWUzE)
- [Introduction to the t Distribution (non-technical)](https://www.youtube.com/watch?v=Uv6nGIgZMVw)
- [Confidence Intervals for One Mean: Determining the Required Sample Size](https://www.youtube.com/watch?v=7zcbVaVz_P8)
- [Student's T Distribution](https://www.youtube.com/watch?v=32CuxWdOlow)
- [An Introduction to the t Distribution (Includes some mathematical details)](https://www.youtube.com/watch?v=T0xRanwAIiI)

<br>


### 30th Commit
#### Interpretable ML vs. Explainable ML

In the context of AI, there is a subtle difference between terms interpretability and explainability. The interpretability techniques such as sparse linear regression were used to "understand how the underlying AI technology works", while the explainability refer to "the level of understanding how AI-based systems produce with a given result". Main claim from Wikipedia is that "treating the model as a black box and analyzing how marginal changes to the inputs affect the result sometimes provides a sufficient explanation."

**References**

- [Explainable Artificial Intelligence](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence) on Wikipedia.
- W. James Murdoch, Chandan Singh, Karl Kumbier, and Bin Yu (2019). [Definitions, methods, and applications in interpretable machine learning](https://doi.org/10.1073/pnas.1900654116). PNAS.
- Christoph Molnar (2024). [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/).

<br>


### 29th Commit
#### INFORMS 2024 | Optimal k-Sparse Ridge Regression

The classical linear regression with a <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-norm induced sparsity penalty can be written as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{\beta}}\,&\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\|_2^2+\lambda\|\boldsymbol{\beta}\|_2^2 \\ \text{s.t.}\,&\|\boldsymbol{\beta}\|_0\leq k \end{aligned}"/></p>

which is equivalent to

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{\beta},\boldsymbol{z}}\,&\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\|_2^2+\lambda\|\boldsymbol{\beta}\|_2^2 \\ \text{s.t.}\,&\begin{cases} (1-z_j)\beta_{j}=0\quad\text{or}\quad \underbrace{-Mz_{j}\leq \beta_{j}\leq Mz_{j}}_{\text{\color{blue}lower/upper bounds}} \\ \displaystyle\sum_{j=1}^{p}z_{j}\leq k,\,z_{j}\in\{0,1\} \end{cases} \end{aligned}"/></p>

**References**

- Jiachang Liu, Sam Rosen, Chudi Zhong, Cynthia Rudin (2023). [OKRidge: Scalable Optimal k-Sparse Ridge Regression](https://arxiv.org/pdf/2304.06686). NeurIPS 2023.

<br>


### 28th Commit
#### Mixed Integer Linear Programming (Example)

- [Mixed Integer Linear Programming with Python](https://readthedocs.org/projects/python-mip/downloads/pdf/latest/)

<br>

```python
import cvxpy as cp
import numpy as np

# Data
n, d, k = 100, 50, 3  # n: samples, d: features, k: sparsity level
X = np.random.randn(n, d)
y = np.random.randn(n)
M = 1  # Large constant for enforcing non-zero constraint

# Variables
beta = cp.Variable(d, nonneg=True)
z = cp.Variable(d, boolean=True)

# Constraints
constraints = [
    cp.sum(z) <= k,
    beta <= M * z,
    beta >= 0
]

# Objective
objective = cp.Minimize(cp.sum_squares(y - X @ beta))

# Problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GUROBI)  # Ensure to use a solver that supports MIP

# Solution
print("Optimal beta:", beta.value)
print("Active indices:", np.nonzero(z.value > 0.5)[0])
```

<br>

Note that the ["Model too large for size-limited Gurobi license" error](https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error).

<br>


### 27th Commit
#### Importance of Sparsity in Interpretable Machine Learning

Sparsity is an important type of model-based interpretability methods. Typically, the practitioner can impose sparsity on the model by limiting the number of nonzero parameters. When the number of nonzero parameters is sufficiently small, a practitioner can interpret the variables corresponding to those parameters as being meaningfully related to the outcome in question and can also interpret the magnitude and direction of the parameters. Two important methods including <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm penalty (e.g., LASSO) and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-norm constraint (e.g., OMP). Model sparsity is often useful for high-dimensional problems, where the goal is to identify key features for further analysis.

**References**

- W. James Murdoch, Chandan Singh, Karl Kumbier, Reza Abbasi-Asl, and Bin Yu (2019). [Definitions, methods, and applications in interpretable machine learning](https://www.pnas.org/doi/10.1073/pnas.1900654116). PNAS.


<br>


### 26th Commit
#### INFORMS 2024 | Core Tensor Shape Optimization

Recall that the sum of squared singular values of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}_{t}\in\mathbb{R}^{N\times D},\,t=1,2,\ldots,T"/> and outcomes <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}"/> is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\sum_{i=1}^{I}(\sigma_{i})^2=\|\boldsymbol{X}\|_F^2"/></p>
because Frobenius norm is invariant under orthogonal transformations with respect to singular vectors.

This means that we can solve a singular value **packing** problem instead of considering the complement of the surrogate loss. Please reproduce the aforementioned property as follows,

<br>

```python
import numpy as np

X = np.random.rand(100, 100)
print(np.linalg.norm(X, 'fro') ** 2)
u, s, v = np.linalg.svd(X, full_matrices = False)
print(np.linalg.norm(s, 2) ** 2)
```

<br>

Thus, Tucker packing problem on the non-increasing sequences <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{a}^{(n)}\in\mathbb{R}_{\geq 0}^{I_n}"/> (w.r.t. singular values), the optimization problem is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\{R_n\}_{n=1}^{N}}\,&\underbrace{\sum_{n=1}^{N}\sum_{i_n=1}^{R_n}a_{i_n}^{(n)}}_{\text{\color{blue}sum of singular values}} \\ \text{s.t.}\,&\underbrace{\prod_{n=1}^{N}R_n}_{\text{\color{blue}core tensor shape}}+\underbrace{\sum_{n=1}^{N}I_nR_n}_{\text{\color{blue}matrix shapes}}\leq \underbrace{c}_{\text{\color{blue}weight}} \end{aligned}"/></p>

The optimization problem can be implemented by using an integer programming solvers, and its solution quality is competitive with the greedy algorithm.

**References**

- [Mehrdad Ghadiri](https://web.mit.edu/mehrdadg/www/), Matthew Fahrbach, Gang Fu, Vahab Mirrokni (2023). [Approximately Optimal Core Shapes for Tensor Decompositions](https://proceedings.mlr.press/v202/ghadiri23a/ghadiri23a.pdf). ICML 2023. [[Python code](https://github.com/fahrbach/approximately-optimal-core-shapes)]

<br>


### 25th Commit
#### Mobile Service Usage Data

- Orlando E. Martínez-Durive et al. (2023). [The NetMob23 Dataset: A High-resolution Multi-region Service-level Mobile Data Traffic Cartography](https://arxiv.org/abs/2305.06933). arXiv:2305.06933.
- André Zanella (2024). [Characterizing Large-Scale Mobile Traffic Measurements for Urban, Social and Networks Sciences](https://dspace.networks.imdea.org/handle/20.500.12761/1852). PhD thesis.

<br>


### 24th Commit
#### Optimization in Reinforcement Learning

**References**

- Jalaj Bhandari and Daniel Russo (2024). [Global Optimality Guarantees for Policy Gradient Methods](https://doi.org/10.1287/opre.2021.0014). Operations Research, 72(5): 1906 - 1927.
- Lucia Falconi, Andrea Martinelli, and John Lygeros (2024). [Data-driven optimal control via linear programming: boundedness guarantees](https://doi.org/10.1109/TAC.2024.3465536). IEEE Transactions on Automatic Control.

<br>


### 23rd Commit
#### Sparse and Time-Varying Regression

This work addresses a time series regression problem for features <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}_{t}\in\mathbb{R}^{N\times D},\,t=1,2,\ldots,T"/> and outcomes <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}_t\in\mathbb{R}^{N},\,t=1,2,\ldots,T"/>, taking the following expression:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}_t\approx\boldsymbol{X}_t\boldsymbol{\beta}_t"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\beta}_t\in\mathbb{R}^{D},\,t=1,2,\ldots,T"/> are coefficient vectors, which are supposed to represent both sparsity and time-varying behaviors of the system. Thus, the optimization problem has both temporal smoothing (in the objective) and sparsity (in the constraint), e.g.,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{\beta}_1,\boldsymbol{\beta}_2,\ldots,\boldsymbol{\beta}_T}\,&\sum_{t=1}^{T}\bigl(\|\boldsymbol{y}_t-\boldsymbol{X}_t\boldsymbol{\beta}_t\|_2^2+\lambda_{\beta}\|\boldsymbol{\beta}_t\|_2^2\bigr)+\lambda_{\delta}\sum_{(s,t)\in E}\|\boldsymbol{\beta}_t-\boldsymbol{\beta}_s\|_2^2 \\ \text{s.t.}\,&\begin{cases} |\text{supp}(\boldsymbol{\beta}_t)|\leq K_L,\,\forall t \\ \bigl|\bigcup\limits_{t=1}^{T}\text{supp}(\boldsymbol{\beta}_t)\bigr|\leq K_G \\ \sum_{(s,t)\in E}|\text{supp}(\boldsymbol{\beta}_t)\Delta\text{supp}(\boldsymbol{\beta}_s)|\leq K_C \end{cases} \end{aligned}"/></p>

where the constraint is indeed the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-norm of vectors, as the symbol <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\text{supp}(\cdot)"/> denotes the index set of nonzero entries in the vector. For instance, the first constraint can be rewritten as <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;|\text{Supp}(\boldsymbol{\beta}_t)|=\|\boldsymbol{\beta}_t\|_0\leq K_L"/>. Thus, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K_L,K_G,K_C\in\mathbb{Z}^{+}"/> stand for sparsity levels.

The methodological contribution is reformulating this problem as a binary convex optimization problem (w/ a novel relaxation of the objective function), which can be solved efficiently using a cutting plane-type algorithm.

**References**
- Dimitris Bertsimas, Vassilis Digalakis, Michael Lingzhi Li, Omar Skali Lami (2024). [Slowly Varying Regression Under Sparsity](https://doi.org/10.1287/opre.2022.0330). Operations Research. [[arXiv](https://arxiv.org/pdf/2102.10773)]


<br>


### 22nd Commit
#### Revisiting <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-Norm Minimization




**References**
- Lijun Ding (2023). [One dimensional least absolute deviation problem](https://threesquirrelsdotblog.com/2023/06/08/one-dimensional-least-absolute-deviation-problem/). Blog post.
- Gregory Gundersen (2022). [Weighted Least Squares](https://gregorygundersen.com/blog/2022/08/09/weighted-ols/). Blog post.
- stephentu's blog (2014). [Subdifferential of a norm](https://stephentu.github.io/blog/convex-analysis/2014/10/01/subdifferential-of-a-norm.html). Blog post.

<br>


### 21st Commit
#### Research Seminars

- [Computational Research in Boston and Beyond (CRIBB) seminar series](https://math.mit.edu/crib/): A forum for interactions among scientists and engineers throughout the Boston area working on a range of computational problems. This forum consists of a monthly seminar where individuals present their work.
- [Param-Intelligence (𝝅) seminar series](https://sites.google.com/view/paramintelligencelab/seminar-series): A dynamic platform for researchers, engineers, and students to explore and discuss the latest advancements in integrating machine learning with scientific computing. Key topics include data-driven modeling, physics-informed neural surrogates, neural operators, and hybrid computational methods, with a strong focus on real-world applications across various fields of computational science and engineering.

<br>


### 20th Commit
#### Robust, Interpretable Statistical Models: Sparse Regression with the LASSO

First of all, we revisit the classical least squares such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{x}}\,\frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2"/></p>

Putting the Tikhonov regularization together with least squares, it refers to as the Ridge regression used almost everywhere:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{x}}\,\frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2+\alpha\|\boldsymbol{x}\|_2^2"/></p>

Another classical variant is the LASSO:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{x}}\,\frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2+\lambda\|\boldsymbol{x}\|_1"/></p>

with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm on the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>. It allows one to find a few columns of the matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> that are most correlated with the designed outcomes (e.g., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{b}"/>) for making decisions (e.g., why they take actions?).

One interesting application is using sparsity-promoting techniques and machine learning with nonlinear dynamical systems to discover governing equations from noisy measurement data. The only as-sumption about the structure of the model is that there are only a fewimportant terms that govern the dynamics, so that the equations aresparse in the space of possible functions; this assumption holds formany physical systems in an appropriate basis.

**References**
- Steve Brunton (2021). Robust, Interpretable Statistical Models: Sparse Regression with the LASSO. see [YouTube](https://www.youtube.com/watch?v=GaXfqoLR_yI&t=632s). (Note: Original paper by Tibshirani (1996))
- Steve L. Brunton, Joshua L. Proctor, and Nathan Kutz (2016). [Discovering governing equations from data by sparse identification of nonlinear dynamical systems](https://doi.org/10.1073/pnas.1517384113). Proceedings of the National Academy of Sciences. 113 (15), 3932-3937.

<br>


### 19th Commit
#### Causal Inference for Geosciences

Learning causal interactions from time series of complex dynamical systems is of great significance in real-world systems. But the questions arise as: 1) How to formulate causal inference for complex dynamical systems? 2) How to detect causal links? 3) How to quantify causal interactions?

**References**
- Jakob Runge (2017). [Causal inference and complex network methods for the geosciences](https://www.belmontforum.org/wp-content/uploads/2018/04/Causal-Inference-and-Complex-Network-Methods-for-the-Geosciences.pdf). Slides.
- Jakob Runge, Andreas Gerhardus, Gherardo Varando, Veronika Eyring & Gustau Camps-Valls (2023). [Causal inference for time series](https://www.nature.com/articles/s43017-023-00431-y). Nature Reviews Earth & Environment, 4: 487–505.
- Jitkomut Songsiri (2013). [Sparse autoregressive model estimation for learning Granger causality in time series](https://doi.org/10.1109/ICASSP.2013.6638248). 2013 IEEE International Conference on Acoustics, Speech and Signal Processing.

<br>


### 18th Commit
#### Tensor Factorization for Knowledge Graph Completion

Knowledge graph completion is a kind of link prediction problems, inferring missing "facts" based on existing ones. Tucker
decomposition of the binary tensor representation of knowledge graph triples allows one to make data completion.

**References**
- [TuckER: Tensor Factorization for Knowledge Graph Completion](https://github.com/ibalazevic/TuckER). GitHub.
- Ivana Balazevic, Carl Allen, Timothy M. Hospedales (2019). TuckER: Tensor Factorization for Knowledge Graph Completion. arXiv:1901.09590. [[PDF](https://arxiv.org/pdf/1901.09590)]

<br>


### 17th Commit
#### RESCAL: Tensor-Based Relational Learning

Multi-relational data is everywhere in real-world applications such as computational biology, social networks, and semantic web. This type of data is often represented in the form of graphs or networks where nodes represent entities, and edges represent different types of relationships.

Instead of using the classical Tucker and CP tensor decomposition, RESCAL takes the inherent structure of dyadic relational data into account, whose tensor factorization on the tensor variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\mathcal{X}}:=\{\boldsymbol{X}_k\}_k"/> (i.e., frontal tensor slices) is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}_k=\boldsymbol{A}\boldsymbol{S}_{k}\boldsymbol{A}^\top"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{n\times r}"/> is the global entity factor matrix, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}_{k}\in\mathbb{R}^{r\times r},\forall k"/> specifies the interaction of the latent components. Such kind of methods can be used to solve link prediction, collective classification, and link-based clustering.

**References**

- Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel (2011). A Three-Way Model for Collective Learning on Multi-Relational Data. ICML 2011. [[PDF](https://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf)] [[Slides](https://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf)]
- Maximilian Nickel (2013). [Tensor Factorization for Relational Learning](https://edoc.ub.uni-muenchen.de/16056/1/Nickel_Maximilian.pdf). PhD thesis.
- Denis Krompaß, Maximilian Nickel, Xueyan Jiang, Volker Tresp (2013). [Non-Negative Tensor Factorization with RESCAL](https://www.dbs.ifi.lmu.de/~krompass/papers/NonNegativeTensorFactorizationWithRescal.pdf). ECML Workshop 2013.
- Elynn Y. Chen, Rong Chen (2019). Modeling Dynamic Transport Network with Matrix Factor Models: with an Application to International Trade Flow. arXiv:1901.00769. [[PDF](https://arxiv.org/pdf/1901.00769)]
- Zhanhong Cheng. [factor_matrix_time_series](https://github.com/chengzhanhong/factor_matrix_time_series). GitHub.

<br>


### 16th Commit
#### Subspace Pursuit Algorithm

Considering the optimization problem for estimating <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/>-sparse vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{n}"/>:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&\frac{1}{2}\|\boldsymbol{y}-\boldsymbol{A}\boldsymbol{x}\|_2^2 \\ \text{s.t.}\,& \|\boldsymbol{x}\|_0\leq K,\,K\in\mathbb{Z}^{+} \end{aligned}"/></p>

with the signal vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{m}"/> (or measurement vector), the dictionary matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{m\times n}"/> (or measurement matrix), and the sparsity level <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K\in\mathbb{Z}^{+}"/>.

The subspace pursuit algorithm, introduced by [W. Dai and O. Milenkovic
in 2008](https://arxiv.org/pdf/0803.0811), is a classical algorithm in the greedy family. It bears some resemblance with compressive sampling matching
pursuit (CoSaMP by [D. Needell and J. A. Tropp in 2008](https://arxiv.org/pdf/0803.2392)), except that, instead of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;2K"/>, only <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/> indices of largest (in modulus) entries of the residual vector are selected, and that an additional orthogonal projection step is performed at each iteration. The implementation of subspace pursuit algorithm (adapted from [A Mathematical Introduction to Compressive Sensing](https://users.math.msu.edu/users/iwenmark/Teaching/MTH994/Holger_Simon_book.pdf), see Page 65) can be summarized as follows:

- **Input**: Signal vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{m}"/>, dictionary matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{m\times n}"/>, and sparsity level <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K\in\mathbb{Z}^{+}"/>.
- **Output**: <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/>-sparse vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{n}"/> and index set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;S"/>.
- **Initialization**: Sparse vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=\boldsymbol{0}"/> (i.e., zero vector), index set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;S=\emptyset"/> (i.e., empty set), and error vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{r}=\boldsymbol{y}"/>.
- **while** not stop **do**
  - Find <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell"/> as the index set of the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/> largest entries of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;|\boldsymbol{A}^\top\boldsymbol{r}|"/>.
  - <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;S:=S\cup\ell"/>.
  - <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_S:=\boldsymbol{A}_S^{\dagger}\boldsymbol{y}"/> (least squares).
  - Find <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;S"/> as the index set of the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/> largest entries of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;|\boldsymbol{x}|"/>.
  - <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_S:=\boldsymbol{A}_S^{\dagger}\boldsymbol{y}"/> (least squares again!).
  - Set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_i=0"/> for all <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;i\notin S"/>.
  - <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{r}=\boldsymbol{y}-\boldsymbol{A}_S\boldsymbol{x}_S"/>.
- **end**

The subspace pursuit algorithm is a fixed-cardinality method, quite different from [the classical orthogonal matching pursuit algorithm developed in 1993](https://doi.org/10.1109/ACSSC.1993.342465) such that

- **Input**: Signal vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{m}"/>, dictionary matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{m\times n}"/>, and sparsity level <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K\in\mathbb{Z}^{+}"/>.
- **Output**: <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;K"/>-sparse vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{n}"/> and index set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;S"/>.
- **Initialization**: Sparse vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=\boldsymbol{0}"/> (i.e., zero vector), index set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;S=\emptyset"/> (i.e., empty set), and error vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{r}=\boldsymbol{y}"/>.
- **while** not stop **do**
  - Find <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell"/> as the index set of the largest entry of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;|\boldsymbol{A}^\top\boldsymbol{r}|"/>, while <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell\notin S"/>.
  - <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;S:=S\cup\ell"/>.
  - <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_S:=\boldsymbol{A}_S^{\dagger}\boldsymbol{y}"/> (least squares).
  - <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{r}=\boldsymbol{y}-\boldsymbol{A}_S\boldsymbol{x}_S"/>.
- **end**


<br>


### 15th Commit
#### Synthetic Sweden Mobility

The Synthetic Sweden Mobility (SySMo) model provides a simplified yet statistically realistic microscopic representation of the real population of Sweden. The agents in this synthetic population contain socioeconomic attributes, household characteristics, and corresponding activity plans for an average weekday. This agent-based modelling approach derives the transportation demand from the agents’ planned activities using various transport modes (e.g., car, public transport, bike, and walking). The dataset is available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10648077).

Going back to the individual mobility trajectory, there would be some opportunities to approach taxi trajectory data such as

- [1 million+ trips collected by 13,000+ taxi cabs during 5 days in Harbin, China](https://github.com/boathit/deepgtt)
- [Daily GPS trajectory data of 664 electric taxis in Shenzhen, China](https://guangwang.me/#/data)
- Takahiro Yabe, Kota Tsubouchi, Toru Shimizu, Yoshihide Sekimoto, Kaoru Sezaki, Esteban Moro & Alex Pentland (2024). [YJMob100K: City-scale and longitudinal dataset of anonymized human mobility trajectories](https://www.nature.com/articles/s41597-024-03237-9). Scientific Data. [[Data](https://zenodo.org/records/13237029)] [[Challenge](https://wp.nyu.edu/humobchallenge2024/)]
- [The dataset comprises trajectory data of traffic participants, along with traffic light data, current local weather data, and air quality data from the Application Platform Intelligent Mobility (AIM) Research Intersection](https://zenodo.org/records/11396372)


<br>


### 14th Commit
#### Prediction on Extreme Floods

AI increases global access to reliable flood forecasts (see [dataset](https://zenodo.org/doi/10.5281/zenodo.8139379)).

Another weather forecasting dataset for consideration: [Rain forecasts world-wide on an expansive data set with over a magnitude more hi-res rain radar data](https://weather4cast.net/neurips2024/challenge/).

**References**
- Nearing, G., Cohen, D., Dube, V. et al. (2024). [Global prediction of extreme floods in ungauged watersheds](https://doi.org/10.1038/s41586-024-07145-1). Nature, 627: 559–563.

<br>


### 13th Commit
#### Sparse Recovery Problem

Considering a general optimization problem for estimating the sparse vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{n}"/>:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x}}\,&\frac{1}{2}\|\boldsymbol{y}-\boldsymbol{A}\boldsymbol{x}\|_2^2 \\ \text{s.t.}\,& \begin{cases} \boldsymbol{x}\geq 0 \\ \displaystyle\|\boldsymbol{x}\|_0\leq K,\,K\in\mathbb{Z}^{+} \end{cases} \end{aligned}"/></p>

with the signal vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{m}"/> and a dictionary of elementary functions <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{m\times n}"/> (i.e., dictionary matrix). There are a lot of solution algorithms in literature:

- Mehrdad Yaghoobi, Di Wu, Mike E. Davies (2015). [Fast Non-Negative Orthogonal Matching Pursuit](https://doi.org/10.1109/LSP.2015.2393637). IEEE Signal Processing Letters, 22 (9): 1229-1233.
- Thanh Thi Nguyen, Jérôme Idier, Charles Soussen, El-Hadi Djermoune (2019). [Non-Negative Orthogonal Greedy Algorithms](https://doi.org/10.1109/TSP.2019.2943225). IEEE Transactions on Signal Processing, 67 (21): 5643-5658.
- Nicolas Nadisic, Arnaud Vandaele, Nicolas Gillis, Jeremy E. Cohen (2020). [Exact Sparse Nonnegative Least Squares](https://doi.org/10.1109/ICASSP40776.2020.9053295). IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
- Chiara Ravazzi, Francesco Bullo, Fabrizio Dabbene (2022). [Unveiling Oligarchy in Influence Networks From Partial Information](https://doi.org/10.1109/TCNS.2022.3225299). IEEE Transactions on Control of Network Systems, 10 (3): 1279-1290.
- Thi Thanh Nguyen (2019). [Orthogonal greedy algorithms for non-negative sparse reconstruction](https://hal.science/tel-02376895/document). PhD thesis.

The most classical (greedy) method for solving the linear sparse regression is orthogonal matching pursuit (see an introduction [here](https://angms.science/doc/RM/OMP.pdf)).



<br>


### 12th Commit
#### Economic Complexity

**References**

- César A. Hidalgo (2021). [Economic complexity theory and applications](https://doi.org/10.1038/s42254-020-00275-1). Nature Reviews Physics. 3: 92-113.

<br>


### 11th Commit
#### Time-Varying Autoregressive Models

Vector autoregression (VAR) has a key assumption that the coeffcients are invariant across time (i.e., time-invariant), but it is not always true when accounting for psychological phenomena such as the phase transition from a healthy to unhealthy state (or vice versa). Consequently, time-varying vector autoregressive models are of great significance for capturing the parameter changes in response to interventions. From the statistical perspective, there are two types of lagged effects between pairs of variables: **autocorrelations** (e.g., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{t}\to\boldsymbol{x}_{t+1}"/>) and **cross-lagged effects** (e.g., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{t}\to\boldsymbol{y}_{t+1}"/>). The time-varying autoregressive models can be solved by using generalized additive model and kernel smoothing estimation.

**References**

- Haslbeck, J. M., Bringmann, L. F., & Waldorp, L. J. (2021). [A tutorial on estimating time-varying vector autoregressive models](https://doi.org/10.1080/00273171.2020.1743630). Multivariate Behavioral Research, 56(1), 120-149.

<br>


### 10th Commit
#### Higher-Order Graph & Hypergraph

The concept of a higher-order graph extends the traditional notion of a graph, which consists of nodes and edges, to capture more complex relationships and structures in data. A common formalism for representing higher-order graphs is through hypergraphs, which generalize the concept of a graph to allow for hyperedges connecting multiple nodes. In a hypergraph, each hyperedge connects a subset of nodes, forming higher-order relationships among them.

**References**

- [Higher-order organization of complex networks](https://snap.stanford.edu/higher-order/). Stanford University.
- Quintino Francesco Lotito, Federico Musciotto, Alberto Montresor, Federico Battiston (2022). [Higher-order motif analysis in hypergraphs](https://www.nature.com/articles/s42005-022-00858-7). Communications Physics, volume 5, Article number: 79.
- Christian Bick, Elizabeth Gross, Heather A. Harrington, and Michael T. Schaub (2023). [What are higher-order networks?](https://doi.org/10.1137/21M1414024) SIAM Review. 65(3).
- Vincent Thibeault, Antoine Allard & Patrick Desrosiers (2024). [The low-rank hypothesis of complex systems](https://www.nature.com/articles/s41567-023-02303-0). Nature Physics. 20: 294-302.
- Louis Boucherie, Benjamin F. Maier, Sune Lehmann (2024). [Decomposing geographical and universal aspects of human mobility](https://arxiv.org/pdf/2405.08746). arXiv:2405.08746.
- Raissa M. D’Souza, Mario di Bernardo & Yang-Yu Liu (2023). [Controlling complex networks with complex nodes](https://doi.org/10.1038/s42254-023-00566-3). Nature Reviews Physics. 5: 250–262.
- PS Chodrow, N Veldt, AR Benson (2021). [Generative hypergraph clustering: From blockmodels to modularity](https://doi.org/10.1126/sciadv.abh1303). Science Advances, 28(7).

<br>


### 9th Commit
#### Eigenvalues of Directed Cycles

The graph signal processing possesses an interesting property of directed cycle (see Figure 2 in the [literature](https://arxiv.org/pdf/2303.12211)). The adjacency matrix of a directed cycle has a set of unit eigenvalues as follows.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/eigenvalues_directed_cycle.png" width="300" />
</p>

<br>

```python
import numpy as np

## Construct an adjacency matrix A
a = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
n = a.shape[0]
A = np.zeros((n, n))
A[:, 0] = a
for i in range(1, n):
    A[:, i] = np.append(a[-i :], a[: -i])

## Perform eigenvalue decomposition on A
eig_val, eig_vec = np.linalg.eig(A)

## Plot eigenvalues
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Helvetica'

fig = plt.figure(figsize = (3, 3))
ax = fig.add_subplot(1, 1, 1)
circ = plt.Circle((0, 0), radius = 1, edgecolor = 'b', facecolor = 'None', linewidth = 2)
ax.add_patch(circ)
plt.plot(eig_val.real, eig_val.imag, 'rx', markersize = 8)
ax.set_aspect('equal', adjustable = 'box')
plt.xlabel('Re')
plt.ylabel('Im')
plt.show()
fig.savefig('eigenvalues_directed_cycle.png', bbox_inches = 'tight')
```

<br>


### 8th Commit
#### Graph Filter

Defining graph-aware operator plays an important role for characterizing a signal <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{N}"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> vertices over a graph <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{G}"/>. One simple idea is introducing the adjacency matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> so that the operation is <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\boldsymbol{x}"/>. In that case, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> is a simple operator that accounts for the local connectivity of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{G}"/>. One example is using the classical unit delay (seems to be time-shift) such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}=\begin{bmatrix} 0 & 0 & 0 & \cdots & 1 \\ 1 & 0 & 0 & \cdots & 0 \\ 0 & 1 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 0 \end{bmatrix}\in\mathbb{R}^{N\times N}"/></p>

The simplest signal operation as multiplication by the adjacency matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> defines graph filters as matrix polynomials of the form

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;p(\boldsymbol{A})=p_0\boldsymbol{I}_N+p_1\boldsymbol{A}+\cdots+p_{N-1}\boldsymbol{A}^{N-1}"/></p>

For instance, we have

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}=\begin{bmatrix} 0 & 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \end{bmatrix}\quad \boldsymbol{A}^2=\begin{bmatrix} 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \end{bmatrix}"/></p>

On the signal <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2,x_3,x_4,x_5)^\top"/>, it always holds that
<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\boldsymbol{x}=\underbrace{(x_5,x_1,x_2,x_3,x_4)^\top}_{\text{\color{red}one-hop neighbors}}\quad \boldsymbol{A}^2\boldsymbol{x}=\underbrace{(x_4,x_5,x_1,x_2,x_3)^\top}_{\text{\color{red}two-hop neighbors}}"/></p>


When applying the polynomial filter to a graph signal <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{N}"/>, the operation <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\boldsymbol{x}"/> takes a local linear combination of the signal values at one-hop neighbors. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}^2\boldsymbol{x}"/> takes a local linear combination of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\boldsymbol{x}"/>, referring to two-hop neighbors. Consequently, a graph filter <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;p(\boldsymbol{A})"/> of order <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N-1"/> represents the mixing values that are at most <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N-1"/> hops away.

**References**

- Geert Leus, Antonio G. Marques, José M. F. Moura, Antonio Ortega, David I Shuman (2023). [Graph Signal Processing: History, Development, Impact, and Outlook](https://arxiv.org/pdf/2303.12211). arXiv:2303.12211.
- A Sandryhaila, JMF Moura (2013). [Discrete signal processing on graphs: Graph filters](https://users.ece.cmu.edu/~asandryh/papers/icassp13.pdf). Section 3: Graph Filters.
- Henry Kenlay, Dorina Thanou, [Xiaowen Dong](https://web.media.mit.edu/~xdong/) (2020). [On The Stability of Polynomial Spectral Graph Filters](https://web.media.mit.edu/~xdong/paper/icassp20.pdf). ICASSP 2020.
- Xiaowen Dong, Dorina Thanou, Michael Rabbat, and Pascal Frossard (2019). [Learning Graphs From Data: A signal representation perspective](https://web.media.mit.edu/~xdong/paper/spm19.pdf). IEEE Signal Processing Magazine.
- Eylem Tugçe Güneyi, Berkay Yaldız, Abdullah Canbolat, and Elif Vural (2024). [Learning Graph ARMA Processes From Time-Vertex Spectra](https://doi.org/10.1109/TSP.2023.3329948). IEEE Transactions on Signal Processing, 72: 47 - 56.

<br>


### 7th Commit
#### Graph Signals

For any graph <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{G}=\{\mathcal{V},\mathcal{E}\}"/> where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{V}=\{1,2,\ldots,N\}"/> is a finite set of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> vertices, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{E} \subseteq \mathcal{V}\times\mathcal{V}"/> is the set of edges. Graph signals can be formally represented as vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{N}"/> where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{n}"/> (or say <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}(n)"/> in the following) stores the signal value at the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/>th vertex in <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{V}"/>. The graph Fourier transform of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> is element-wise defined as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\hat{\boldsymbol{x}}(k)=\langle\boldsymbol{x},\boldsymbol{\psi}_k\rangle=\sum_{n=1}^{N}\boldsymbol{x}(n)\boldsymbol{\psi}_{k}^{*}(n)"/></p>

or another form such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\hat{\boldsymbol{x}}=\boldsymbol{\Psi}^{H}\boldsymbol{x}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\Psi}"/> consists of the eigenvectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\psi}_k,\,k=1,2,\ldots,N"/>. The notation <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\cdot^{*}"/> is the conjugate of complex values, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\cdot^{H}"/> is the conjugate transpose.

The above graph Fourier transform can also be generalized to the graph signals in the form of multivariate time series. For instance, on the data <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{N\times T}"/>, we have

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\hat{\boldsymbol{X}}=\boldsymbol{\Psi}^{H}\boldsymbol{X}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\Psi}"/> consists of the eigenvectors of the graph Laplacian matrix.

**References**

- Santiago Segarra, Weiyu Huang, and Alejandro Ribeiro (2020). [Signal Processing on Graphs](https://www.seas.upenn.edu/~ese2240/labs/1200_ch_9_signal_processing_on_graphs.pdf).
- Matthew Begue. [Fourier analysis on graphs](https://www.norbertwiener.umd.edu/Research/lectures/2014/MBegue_Prelim.pdf). Slides.

<br>


### 6th Commit
#### Graph Signal Processing

Graph signal processing not only focuses on the graph typology (e.g., connection between nodes), but also covers the quantity of nodes (i.e., graph signals) with weighted adjacency information.

**References**

- Antonio Ortega, Pascal Frossard, Jelena Kovacevic, Jose M. F. Moura, Pierre Vandergheynst (2017). [Graph Signal Processing: Overview, Challenges and Applications](https://arxiv.org/pdf/1712.00468). arXiv:1712.00468.
- Gonzalo Mateos, Santiago Segarra, Antonio G. Marques, Alejandro Ribeiro (2019). [Connecting the Dots: Identifying Network Structure via Graph Signal Processing](https://doi.org/10.1109/MSP.2018.2890143). IEEE Signal Processing Magazine. 36 (3): 16-43.
- Xiaowen Dong, Dorina Thanou, Laura Toni, Michael Bronstein, and Pascal Frossard (2020). [Graph signal processing for machine learning: A review and new perspectives](https://arxiv.org/pdf/2007.16061). arXiv:2007.16061. [[Slides](https://web.media.mit.edu/~xdong/talk/BDI_GSP.pdf)]
- Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković (2021). [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/pdf/2104.13478). arXiv:2104.13478.
- Wei Hu, Jiahao Pang, Xianming Liu, Dong Tian, Chia-Wen Lin, Anthony Vetro (2022). [Graph Signal Processing for Geometric Data and Beyond: Theory and Applications](https://doi.org/10.1109/TMM.2021.3111440). IEEE Transactions on Multimedia, 24: 3961-3977.
- Geert Leus, Antonio G. Marques, José M. F. Moura, Antonio Ortega, David I Shuman (2023). [Graph Signal Processing: History, Development, Impact, and Outlook](https://arxiv.org/pdf/2303.12211). arXiv:2303.12211.
- [Spectral graph theory for dummies](https://youtu.be/uTUVhsxdGS8?si=3UzuXpUXVSiu3pDo). YouTube.

<br>


### 5th Commit
#### Clifford Product

In [Grassmann algebra](https://en.wikipedia.org/wiki/Exterior_algebra), the inner product between two vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}=x_1\vec{e}_1+x_2\vec{e}_2"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{y}=y_1\vec{e}_1+y_2\vec{e}_2"/> (w/ basis vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{e}_1"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{e}_2"/>) is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\langle\vec{x},\vec{y}\rangle=\|\vec{x}\|_2 \|\vec{y}\|_2 \cos\theta"/></p>
implies to be the multiplication between the magnitude of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}"/> and the projection of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{y}"/> on <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}"/>. Here, the notation <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\cdot\|_2"/> refers to the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_2"/> norm, or say the magnitude. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\theta"/> is the angle between <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{y}"/> in the plane containing them.

In contrast, the outer product (usually called Wedge product) is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}\wedge\vec{y}=\underbrace{(\vec{e}_1\wedge\vec{e}_2)}_{\text{\color{red}orientation}}\underbrace{\|\vec{x}\|_2 \|\vec{y}\|_2 \sin\theta}_{\text{\color{red}area/determinant}}"/></p>
implies to be the multiplication between <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}"/> and the projection of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{y}"/> on the orthogonal direction of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}"/>. Here, the unit bivector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{e}_1\wedge\vec{e}_2"/> represents the orientation (<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;+1"/> or <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;-1"/>) of the hyperplane of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}\wedge\vec{y}"/> (see Section II in [geometric-algebra adaptive filters](https://doi.org/10.1109/TSP.2019.2916028)).

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/inner_wedge_prods.png" width="600" />
</p>

<br>

As a result, they consist of Clifford product (or called geometric product, denoted by the symbol <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\cdot"/>) such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \vec{x}\cdot\vec{y}=&\langle\vec{x},\vec{y}\rangle+\vec{x}\wedge\vec{y} \\ =&\|\vec{x}\|_2\|\vec{y}\|_2(\cos\theta +(\vec{e}_1\wedge\vec{e}_2)\sin\theta) \\ =&\|\vec{x}\|_2\|\vec{y}\|_2e^{(\vec{e}_1\wedge\vec{e}_2)\theta} \end{aligned}"/></p>

In particular, [Clifford algebra](https://en.wikipedia.org/wiki/Clifford_algebra) is important for modeling vector fields, thus demonstrating valuable applications to wind velocity and fluid dynamics (e.g., Navier-Stokes equation).

**References**

- [Spinors for Beginners 11: What is a Clifford Algebra? (and Geometric, Grassmann, Exterior Algebras)](https://www.youtube.com/watch?v=nktgFWLy32U&t=989s). YouTube.
- [A Swift Introduction to Geometric Algebra](https://www.youtube.com/watch?v=60z_hpEAtD8&t=768s). YouTube.
- [Learning on Graphs & Geometry](https://portal.valencelabs.com/logg). Weekly reading groups every Monday at 11 am ET.
- [What's the Clifford algebra?](https://math.stackexchange.com/questions/261509/whats-the-clifford-algebra) Mathematics stackexchange.
- [Introducing CliffordLayers: Neural Network layers inspired by Clifford / Geometric Algebras](https://www.microsoft.com/en-us/research/lab/microsoft-research-ai4science/articles/introducing-cliffordlayers-neural-network-layers-inspired-by-clifford-geometric-algebras/). Microsoft Research AI4Science.
- David Ruhe, Jayesh K. Gupta, Steven de Keninck, Max Welling, Johannes Brandstetter (2023). [Geometric Clifford Algebra Networks](https://arxiv.org/pdf/2302.06594). arXiv:2302.06594.
- Maksim Zhdanov, David Ruhe, Maurice Weiler, Ana Lucic, Johannes Brandstetter, Patrick Forre (2024). [Clifford-Steerable Convolutional Neural Networks](https://arxiv.org/pdf/2402.14730). arXiv:2402.14730.

<br>


### 4th Commit
#### Bayesian Variable Selection

In genetic fine mapping, one critical problem is the variable selection in linear regression. There is a Bayesian variable selection based on the sum of single effects, i.e., the vector with one non-zero element. Given any data <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{m\times n}"/> (of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> explanatory variables) and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}\in\mathbb{R}^{m}"/>, one can build an optimization problem as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\{\boldsymbol{b}_{\ell}\}_{\ell\in[L]}}\,&\frac{1}{2}\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{b}\|_2^2 \\ \text{s.t.}\,&\begin{cases} \displaystyle\boldsymbol{b}=\sum_{\ell\in[L]}\boldsymbol{b}_{\ell} \\ \|\boldsymbol{b}_{\ell}\|_{0}=1,\,\forall \ell\in[L] \\ \boldsymbol{b}_{k}^\top\boldsymbol{b}_{\ell}=0,\,k\neq \ell \end{cases} \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;L\in\mathbb{Z}^{+}"/> (<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;L< n"/>) is predefined by the number of correlated variables. The vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{b}_{\ell},\,\ell\in[L]"/> are the coefficients in this linear regression. This optimization problem can also be written as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{w}}\,&\frac{1}{2}\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\|_2^2 \\ \text{s.t.}\,& \displaystyle\|\boldsymbol{w}\|_0=L,\,L\in\mathbb{Z}^{+}  \end{aligned}"/></p>

Or see Figure 1.1 in [Section 1.1 Non-negative sparse reconstruction](https://hal.science/tel-02376895/document) (Page 2) for an illustration.

**References**
- Gao Wang, Abhishek Sarkar, Peter Carbonetto and Matthew Stephens (2020). [A simple new approach to variable selection in regression, with application to genetic fine mapping](https://doi.org/10.1111/rssb.12388). Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(5), 1273-1300.
- Emil Uffelmann, Qin Qin Huang, Nchangwi Syntia Munung, Jantina de Vries, Yukinori Okada, Alicia R. Martin, Hilary C. Martin, Tuuli Lappalainen & Danielle Posthuma (2021). [Genome-wide association studies](https://www.nature.com/articles/s43586-021-00056-9). Nature Reviews Methods Primers. 1: 59.

<br>


### 3rd Commit
#### Causal Effect Estimation/Imputation

The causal effect estimation problem is usually defined as a matrix completion on the partially observed matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}\in\mathbb{R}^{N\times T}"/> in which <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> units and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> periods are involved. The observed index set is denoted by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega"/>. The optimization is from the [classical matrix factorization techniques for recommender systems (see Koren et al.'09)](https://doi.org/10.1109/MC.2009.263):

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{W},\boldsymbol{X},\boldsymbol{u},\boldsymbol{p}}\,\frac{1}{2}\left\|\mathcal{P}_{\Omega}(\boldsymbol{Y}-\boldsymbol{W}^\top\boldsymbol{X}-\boldsymbol{u}\mathbf{1}_{T}^\top-\mathbf{1}_{N}\boldsymbol{p}^\top)\right\|_F^2"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}\in\mathbb{R}^{R\times N}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{R\times T}"/> are factor matrices, referring to units and periods, respectively. Here, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{u}\in\mathbb{R}^{N}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{p}\in\mathbb{R}^{T}"/> are bias vectors, corresponding to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> units and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> periods, respectively. This idea has also been examined on the tensor factorization (to be honest, performance gains are marginal), see e.g., Bayesian augmented tensor factorization by [Chen et al.'19](https://doi.org/10.1016/j.trc.2019.03.003). In the causal effect imputation, one great challenge is how to handle the structural patterns of missing data as mentioned by [Athey et al.'21](https://doi.org/10.1080/01621459.2021.1891924). The structural missing patterns have been discussed on spatiotemporal data with [autoregressive tensor factorization (for spatiotemporal predictions)](https://doi.org/10.1109/ICDM.2017.146).

<br>


### 2nd Commit
#### Data Standardization in Healthcare

The motivation for discussing the value of standards for health datasets is the risk of algorithmic bias, consequently leading to the possible healthcare inequity. The problem arises from the systemic inequalities in the dataset curation and the unequal opportunities to access the data and research. The aim is to expolore the standards, frameworks, and best practices in health datasets. Some discrete insights throughout the whole paper are summarized as follows,

- AI as a medical device (AIaMD). One concern is the risk of systemic algorithmic bias (well-recognized in the literature) if models are trained on biased training datasets.
- Less accurate performance in certain patient groups when using the biased algorithms.
- Data diversity (Mainly discuss "how to improve"):
  - Challenges: lack of standardization across attribute categories, difficulty in harmonizing several methods of data capture and data-governance restrictions.
  - Inclusiveness is a core tenet of ethical AI in healthcare.
  - Guidance on how to apply the principles in the curation (e.g., developing the data collection strategy), aggregation and use of health data.
- The use of metrics (measuring diversity). How to promote diversity and transparency?
- Future actions: Guidelines for data collection, handling missing data and labeling data.

**References**

- Anmol Arora, Joseph E. Alderman, Joanne Palmer, Shaswath Ganapathi, Elinor Laws, Melissa D. McCradden, Lauren Oakden-Rayner, Stephen R. Pfohl, Marzyeh Ghassemi, Francis McKay, Darren Treanor, Negar Rostamzadeh, Bilal Mateen, Jacqui Gath, Adewole O. Adebajo, Stephanie Kuku, Rubeta Matin, Katherine Heller, Elizabeth Sapey, Neil J. Sebire, Heather Cole-Lewis, Melanie Calvert, Alastair Denniston, Xiaoxuan Liu (2023). [The value of standards for health datasets in artificial intelligence-based applications](https://doi.org/10.1038/s41591-023-02608-w). Nature Medicine, 29: 2929–2938.

<br>


### 1st Commit
#### Large Time Series Forecasting Models

As we know, the training data in the large time series model is from different areas, this means that the model training process highly depends on the selected datasets across various areas, so one question is how to reduce the model biases if we consider the forecasting scenario as traffic flow or human mobility? Because I guess time series data in different areas should demonstrate different data behaviors. Hopefully, it is interesting to develop domain-specific time series datasets (e.g., [Largest multi-city traffic dataset](https://utd19.ethz.ch/)) and large models (e.g., [TimeGPT](https://docs.nixtla.io/)).

**References**

- Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, Doyen Sahoo (2024). [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/pdf/2402.02592). arXiv:2402.02592.

<br>





Motivation & Principle: "不积硅步，无以至千里。" (Small Steps to Accuracy)

<br>