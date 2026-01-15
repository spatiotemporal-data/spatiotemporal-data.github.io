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

# Huber Optimization

### Time Series Forecasting with Huber Autoregression

One can use the `cvxpy` package to solve the optimization of Huber autoregression, which is defined as follows,

$$
\underbrace{\begin{bmatrix} y_{d+1} \\ y_{d+2} \\ \vdots \\ y_{T} \end{bmatrix}}_{\color{red}\texttt{vec}}=\underbrace{\begin{bmatrix} y_{d} & y_{d-1} & \cdots & y_{1} \\ y_{d+1} & y_{d} & \cdots & y_{2} \\ \vdots & \vdots & \ddots & \vdots \\ y_{T-1} & y_{T-2} & \cdots & y_{T-d} \end{bmatrix}}_{\color{red}\texttt{mat}}\underbrace{\begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_d \end{bmatrix}}_{\color{red}\texttt{coef}}+\underbrace{\begin{bmatrix} \varepsilon_{d+1} \\ \varepsilon_{d+2} \\ \vdots \\ \varepsilon_{T} \end{bmatrix}}_{\color{red}\texttt{residual}} \tag{1}
$$

<br>

```python
import cvxpy as cp
import numpy as np

def huber_ar(y, d, delta = 1, solver = cp.OSQP):
    t = y.shape[0]
    vec = y[d :] # T-d
    mat = np.vstack([y[d-i-1 : t-i-1] for i in range(d)]).T # (T-d) x d

    coef = cp.Variable(d)
    residual = vec - mat @ coef
    obj = cp.sum(cp.huber(residual, delta))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver = solver)

    return coef.value
```

<br>

Then, we define the function for one-step time series prediction with univariate autoregression:

$$
\begin{aligned}
\hat{y}_{T-h}&=\sum_{k=1}^{d}a_ky_{T-h-k} \\
&\quad\vdots \\
\hat{y}_{T}&=\sum_{k=1}^{d}a_ky_{T-k}
\end{aligned}
$$

or 

$$
\begin{bmatrix} \hat{y}_{T-h} \\ \hat{y}_{T-h+1} \\ \vdots \\ \hat{y}_{T} \end{bmatrix}=
\begin{bmatrix} y_{T-h-1} & y_{T-h-2} & \cdots & y_{T-h-d} \\ 
y_{T-h} & y_{T-h-1} & \cdots & y_{T-h-d+1} \\ 
\vdots & \vdots & \ddots & \vdots \\
y_{T-1} & y_{T-2} & \cdots & y_{T-d} \\ 
\end{bmatrix}
\begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_d \end{bmatrix}
$$

with the time prediction horizon being $$h\in\mathbb{Z}^{+}$$.


<br>

```python
def time_series_pred(data, coef, horizon):
    t = data.shape[0]
    d = coef.shape[0]
    # pred = np.zeros(horizon)
    mat = np.vstack([y[d-i-1 : t-i-1] for i in range(d)]).T # h x d
    # for i in range(horizon):
    #     pred[i] = np.inner(np.flip(data[t - d - i : ]), coef)

```

<br>


<br>
<br>


<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on January 14, 2026)</p>
