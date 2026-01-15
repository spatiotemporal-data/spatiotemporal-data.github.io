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

# Time Series Forecasting with Huber Autoregression

One can use the `cvxpy` package to solve the optimization of Huber autoregression, which is defined as follows,

```python
import cvxpy as cp
import numpy as np

def huber_ar(y, d, delta = 1, solver = cp.OSQP):
    t = y.shape[0]
    
```



<br>
<br>


<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on January 14, 2026)</p>
