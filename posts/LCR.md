---
layout: default
---

# Laplacian Convolutional Representation

<br>

The latest version of LCR model is available at [here](https://xinychen.github.io/papers/Laplacian_convolution.pdf).

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

<br>

Some simple examples on [this dataset](https://github.com/xinychen/transdim/tree/master/datasets/Portland-data-set).

<br>

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

Example: Next-day passenger flow prediction

<br>

```python
import numpy as np
np.random.seed(1)
import time

dense_mat = np.load('sept_15min_occupancy_dense_mat.npy')
d = 27
vec = dense_mat[0, : 96 * (d + 1)]
dense_vec = dense_mat[0, : 96 * d]

dense_vec1 = np.append(dense_vec, np.zeros(96))
sparse_vec1 = np.append(dense_vec, np.zeros(96))
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

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

fig = plt.figure(figsize = (7.5, 2.2))
ax = fig.add_subplot(111)
plt.plot(vec[: 96 * (d + 1)], 'dodgerblue', linewidth = 1)
plt.plot(x[: 96 * (d + 1)], 'red', linewidth = 2)
plt.xlabel('Time')
plt.ylabel('Speed (mph)')
plt.xlim([96 * (d - 3), 96 * (d + 1)])
plt.grid(linestyle = '-.', linewidth = 0.5)
ax.tick_params(direction = 'in')

plt.show()
```

<br>

## LCR-2D without Flipping Operation

```python
import numpy as np

def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])

def laplacian(T, tau):
    ell = np.zeros(T)
    ell[0] = 2 * tau
    for k in range(tau):
        ell[k + 1] = -1
        ell[-k - 1] = -1
    return ell

def prox_2d(z, w, lmbda, denominator):
    N, T = z.shape
    temp1 = np.fft.fft2(lmbda * z - w) / denominator
    temp2 = 1 - N * T / (denominator * np.abs(temp1))
    temp2[temp2 <= 0] = 0
    return np.fft.ifft2(temp1 * temp2).real

def update_z(y_train, pos_train, x, w, lmbda, eta):
    z = x + w / lmbda
    z[pos_train] = (lmbda / (lmbda + eta) * z[pos_train] 
                    + eta / (lmbda + eta) * y_train)
    return z

def update_w(x, z, w, lmbda):
    return w + lmbda * (x - z)

def LCR_2d(y_true, y, lmbda, gamma, tau_s, tau_t, maxiter = 50):
    eta = 100 * lmbda
    if np.isnan(y).any() == False:
        pos_test = np.where((y_true != 0) & (y == 0))
    elif np.isnan(y).any() == True:
        pos_test = np.where((y_true > 0) & (np.isnan(y)))
        y[np.isnan(y)] = 0
    y_test = y_true[pos_test]
    pos_train = np.where(y != 0)
    y_train = y[pos_train]
    z = y.copy()
    w = y.copy()
    ell_s = np.zeros(N)
    ell_s[0] = 1
    # ell_s = laplacian(N, tau_s)
    ell_t = laplacian(T, tau_t)
    ell = np.fft.fft2(np.outer(ell_s, ell_t))
    denominator = lmbda + gamma * np.abs(ell) ** 2
    del y_true, y
    show_iter = 10
    for it in range(maxiter):
        x = prox_2d(z, w, lmbda, denominator)
        z = update_z(y_train, pos_train, x, w, lmbda, eta)
        w = update_w(x, z, w, lmbda)
        if (it + 1) % show_iter == 0:
            print(it + 1)
            print(compute_mape(y_test, x[pos_test]))
            print(compute_rmse(y_test, x[pos_test]))
            print()
    return x
```

<br>

### Large Time Series Imputation

PeMS dataset is available at [our GitHub repository](https://github.com/xinychen/transdim/tree/master/datasets/California-data-set).

Hyperparameters:

- On 30%/50% missing data, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\lambda=10^{-5}NT"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\gamma=10\lambda"/>, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\tau=1"/>;
- On 70% missing data, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\lambda=10^{-5}NT"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\gamma=10\lambda"/>, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\tau=2"/>;
- On 90% missing data, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\lambda=10^{-5}NT"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\gamma=10\lambda"/>, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\tau=3"/>.

<br>

```python
import numpy as np
np.random.seed(1000)

dense_mat = np.load('pems-w1.npz')['arr_0']
for t in range(2, 5):
    dense_mat = np.append(dense_mat, np.load('pems-w{}.npz'.format(t))['arr_0'],
                          axis = 1)
dim1, dim2 = dense_mat.shape

missing_rate = 0.9
sparse_mat = dense_mat * np.round(np.random.rand(dim1, dim2) + 0.5 - missing_rate)
# np.savez_compressed('dense_mat.npz', dense_mat)
# np.savez_compressed('sparse_mat.npz', sparse_mat)

# import cupy as np

# dense_mat = np.load('dense_mat.npz')['arr_0']
# sparse_mat = np.load('sparse_mat.npz')['arr_0']

import time
start = time.time()
N, T = sparse_mat.shape
lmbda = 1e-5 * N * T
gamma = 10 * lmbda
tau_s = 1
tau_t = 3
maxiter = 100
mat_hat = LCR_2d(dense_mat, sparse_mat, lmbda, gamma, tau_s, tau_t, maxiter)
end = time.time()
print('Running time: %d seconds.'%(end - start))
```


<br>
<br>
<br>

```python
import numpy as np


# def sparse_reg(y, A, tau, maxiter = 50):
#     m, n = A.shape
#     x = np.zeros(n)
#     S = np.array([])
#     r = y.copy()
#     for it in range(maxiter):
#         A_tilde = A.copy()
#         ar = A_tilde.T @ r
#         ell = np.argpartition(ar, - 1 * tau)[- 1 * tau :].flatten().tolist()
#         S = np.append(S, ell, axis = 0).astype(int)
#         xS = np.linalg.pinv(A[:, S]) @ y
#         x[S] = xS
#         S = np.argpartition(x, - tau)[- tau :].flatten().tolist()
#         xS = np.linalg.pinv(A[:, S]) @ y
#         x = np.zeros(n)
#         x[S] = xS
#         r = y - A[:, S] @ x[S]
#     return x
```

```python
data = np.load('volume.npy')
y = data[1, :]
mat = circ(y)
A = mat[:, 1 :]
tau = 5
x = sparse_reg(y, A, tau)
print('Non-zero entries of kernel x:')
print(x[x > 0])
print('Index set of the non-zero entries of kernel x:')
print(np.where(x > 0)[0])
print()
```


<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 17, 2024.)</p>
