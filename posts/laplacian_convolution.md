---
layout: default
---

# Laplacian Convolutional Representation

<br>

Content:
- Motivation
- Prior art on global trend modeling
- Local trend modeling with circular convolution
- LCR: A unified optimization with an efficient implementation
- Python implementation with `numpy`
- Univariate traffic volume/speed imputation
- Multivariate speed field reconstruction

<br>

## Python Implementation

We implement the LCR model with `numpy`...

<br>

```python
import numpy as np

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def laplacian(n, tau):
    ell = np.zeros(n)
    ell[0] = 2 * tau
    for k in range(tau):
        ell[k + 1] = -1
        ell[-k - 1] = -1
    return ell

def prox(z, w, lmbda, denominator):
    T = z.shape[0]
    temp1 = np.fft.fft(lmbda * z - w) / denominator
    temp2 = 1 - T / (denominator * np.abs(temp1))
    temp2[temp2 <= 0] = 0
    return np.fft.ifft(temp1 * temp2).real

def update_z(y_train, pos_train, x, w, lmbda, eta):
    z = x + w / lmbda
    z[pos_train] = (lmbda / (lmbda + eta) * z[pos_train]
                    + eta / (lmbda + eta) * y_train)
    return z

def update_w(x, z, w, lmbda):
    return w + lmbda * (x - z)

def flip_vec(x):
    return np.append(x, np.flip(x))

def inv_flip_vec(vec):
    dim = vec.shape[0]
    T = int(dim / 2)
    return (vec[: T] + np.flip(vec[T :])) / 2

def LCR(y_true, y, lmbda, gamma, tau, maxiter = 50):
    eta = 100 * lmbda
    data_true = flip_vec(y_true)
    data = flip_vec(y)
    T = data.shape
    pos_train = np.where(data != 0)
    data_train = data[pos_train]
    pos_test = np.where((y_true != 0) & (y == 0))
    y_test = y_true[pos_test]
    z = data.copy()
    w = data.copy()
    ell = np.fft.fft(laplacian(T, tau))
    denominator = lmbda + gamma * np.abs(ell) ** 2
    del y_true, y
    show_iter = 100
    for it in range(maxiter):
        x = prox(z, w, lmbda, denominator)
        z = update_z(data_train, pos_train, x, w, lmbda, eta)
        w = update_w(x, z, w, lmbda)
        if (it + 1) % show_iter == 0:
            print(it + 1)
            print(compute_mape(y_test, inv_flip_vec(x)[pos_test]))
            print(compute_rmse(y_test, inv_flip_vec(x)[pos_test]))
            print()
    return inv_flip_vec(x)
```

Some simple examples on [this dataset](https://github.com/xinychen/transdim/tree/master/datasets/Portland-data-set).


```python
import numpy as np
np.random.seed(1)
import time

missing_rate = 0.1
print('Missing rate = {}'.format(missing_rate))

dense_mat = np.load('speed.npy')
d = 3
vec = dense_mat[0, : 96 * 4]
dense_vec = dense_mat[0, : 96 * d]
T = dense_vec.shape[0]
sparse_vec = dense_vec * np.round(np.random.rand(T) + 0.5 - missing_rate)

dense_vec1 = np.append(dense_vec, np.zeros(96))
sparse_vec1 = np.append(sparse_vec, np.zeros(96))
T = dense_vec1.shape[0]

import time
start = time.time()
lmbda = 1e-2 * T
gamma = 5 * lmbda
tau = 2
maxiter = 100
x = LCR(dense_vec1, sparse_vec1, lmbda, gamma, tau, maxiter)
end = time.time()
print('Running time: %d seconds.'%(end - start))
d = 4

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

fig = plt.figure(figsize = (7.5, 2.2))
ax = fig.add_subplot(111)
plt.plot(vec[: 96 * d], 'dodgerblue', linewidth = 1)
plt.plot(np.arange(0, 96 * d), sparse_vec1[: 96 * d], 'o',
         markeredgecolor = 'darkblue', alpha = missing_rate,
         markerfacecolor = 'deepskyblue', markersize = 10)
plt.plot(x[: 96 * d], 'red', linewidth = 4)
plt.xlabel('Time')
plt.ylabel('Speed (mph)')
plt.xlim([0, 96 * d])
plt.ylim([54, 65])
plt.xticks(np.arange(0, 96 * d + 1, 24))
plt.yticks(np.arange(54, 66, 2))
plt.grid(linestyle = '-.', linewidth = 0.5)
ax.tick_params(direction = 'in')

plt.savefig('speeds_{}.pdf'.format(round(missing_rate * 100)),
            bbox_inches = 'tight')
plt.show()

```



<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 17, 2024.)</p>
