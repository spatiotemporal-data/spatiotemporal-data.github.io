---
layout: default
---

# Time Series Convolution (Supplementary Materials)

<p align="center"><span style="color:gray">A convolutional kernel approach for reinforcing the modeling of time series trends and interpreting temporal patterns, allowing one to leverage Fourier transforms and learn sparse representations.</span></p>

<br>

In this post, we intend to explain the essential ideas of our latent research work:

- Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). [Laplacian convolutional representation for traffic time series imputation](https://doi.org/10.1109/TKDE.2024.3419698). IEEE Transactions on Knowledge and Data Engineering. Early Access.


<br>

## I. Appendix for Imputing Time Series

### A. Circulant Matrix Nuclear Norm Minimization

Below is the Python code for reproducing the circulant matrix nuclear norm minimization on partially observed traffic volume data in Portland, USA.

<br>

```python
import numpy as np

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def circ_opt(y_true, y, lmbda, gamma, maxiter = 50, show_iter = 10):
    T = y.shape
    pos_train = np.where(y != 0)
    y_train = y[pos_train]
    pos_test = np.where((y_true != 0) & (y == 0))
    y_test = y_true[pos_test]
    z = y.copy()
    w = y.copy()
    def y_true, y
    for it in range(maxiter):
        x = update_x(z, w, lmbda)
        z = update_z(y_train, pos_train, x, w, lmbda, gamma)
        w = w + lmbda * (x - z)
        if (it + 1) % show_iter == 0:
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(compute_mape(y_test, x[pos_test])))
            print('RMSE: {:.6}'.format(compute_rmse(y_test, x[pos_test])))
            print()
    return x
```



<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on August 29, 2024.)</p>
