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

### B. Hankel Matrix Factorization & Discrete Fourier Transform

Main function:

```python
import numpy as np

def inverse_hankel(w, q, fft = False):
    dim1 = w.shape[0]
    dim2 = q.shape[0]
    dim = min(dim1, dim2)
    T = dim1 + dim2 - 1
    x_tilde = np.zeros(T)
    for t in range(T):
        w_new = np.zeros(t + 1)
        if t < dim1:
            w_new[: t + 1] = w[: t + 1]
        elif t >= dim1:
            w_new[: dim1] = w
        q_new = np.zeros(t + 1)
        if t < dim2:
            q_new[: t + 1] = q[: t + 1]
        elif t >= dim2:
            q_new[: dim2] = q
        if t < dim:
            rho = t + 1
        else:
            rho = T - (t + 1) + 1
        if fft == True:
            vec = np.fft.ifft(np.fft.fft(w_new) * np.fft.fft(q_new)).real
            x_tilde[t] = vec[t] / rho
        elif fft == False:
            x_tilde[t] = np.inner(w_new, np.flip(q_new)) / rho
    return x_tilde
```

Experiments on generated data:

```python
import numpy as np
np.random.seed(1)
import time

repeat = 100
timing_tensor = np.zeros((9, repeat, 2))
t = 0
for power in range(5, 14, 1):
    print('Power = {}'.format(power))
    w = np.random.rand(2 ** power)
    q = np.random.rand(2 ** power)
    for i in range(repeat):
        start = time.time() * 1000
        x1 = inverse_hankel(w, q, fft = False)
        end = time.time() * 1000
        timing_tensor[t, i, 0] = end - start
    for i in range(repeat):
        start = time.time() * 1000
        x2 = inverse_hankel(w, q, fft = True)
        end = time.time() * 1000
        timing_tensor[t, i, 1] = end - start
        print('Hx running time: %d milliseconds.'%(timing_tensor[t, i, 0]))
        print('Hx-FFT running time: %d milliseconds.'%(timing_tensor[t, i, 1]))
        print()
    t += 1
```

Drawing figures to compare empirical time complexity:

```python
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 11
plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize = (7.5, 2.5))
for i in [1, 2]:
    ax = fig.add_subplot(1, 2, i)
    if i == 1:
        plt.plot(np.array([2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13]),
                np.mean(timing_tensor[: 9, :, 0], axis = 1) / 1000, '-o', color = 'red', alpha = 0.8)
        plt.title('Element-wise multiplication')
    elif i == 2:
        plt.plot(np.array([2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13]),
                np.mean(timing_tensor[: 9, :, 1], axis = 1) / 1000, '-o', color = 'blue', alpha = 0.8)
        plt.title('Circular convolution with FFT')
    plt.xlabel(r'Vector length $n$')
    plt.ylabel('Time (s)')
    labels = [r'$2^{5}$', r'$2^{9}$',
              r'$2^{11}$', r'$2^{12}$', r'$2^{13}$']
    plt.xticks(np.array([2**5, 2**9, 2**11, 2**12, 2**13]), labels)
    ax.tick_params(which = 'minor', direction = 'in')
    ax.grid(color = 'gray', linestyle = 'dashed', linewidth = 0.5, alpha = 0.2)

plt.show()
fig.savefig('empirical_time_complexity_hankel.png', bbox_inches = 'tight')
```

<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on August 29, 2024.)</p>
