---
layout: default
---

# Time-Varying Autoregression

#### Discovering Dynamic Patterns from Spatiotemporal Data with Time-Varying Low-Rank Autoregression


Dynamic mechanisms that drive nonlinear systems are universally complex. Straightforwardly, one can investigate the behavior of a system by discovering the interpretable dynamic patterns from real-world data. In practice, when we take observations from a real-world complex system, spatiotemporal data are one of the most widely encountered from relating to space and time and showing the characteristics of time series. Without loss of generality, leveraging time series models not only allows one to analyze spatiotemporal data but also makes it possible to discover inherent spatial and temporal patterns from the data over space and time. The scientific question in our study (see [Chen et al., 2023](https://doi.org/10.1109/TKDE.2023.3294440)) is how to discover interpretable dynamic patterns from spatiotemporal data. We utilize the vector autoregression as a basic tool to explore the spatiotemporal data in real-world applications.

<br>

```bibtex
@article{chen2023discovering,
  title={Discovering dynamic patterns from spatiotemporal data with time-varying low-rank autoregression},
  author={Chen, Xinyu and Zhang, Chengyuan and Chen, Xiaoxu and Saunier, Nicolas and Sun, Lijun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023}
}
```

<br>

## Revisit Vector Autoregression

As a simple yet efficient and classical method for time series modeling, vector autoregression allows one to explicitly find the linear relationship among a sequence of time series (i.e., multivariate time series) changing over time, which can also successfully describe the dynamic behaviors of time series.

Given any stationary time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{s}_1,\boldsymbol{s}_2,\ldots,\boldsymbol{s}_{T}\in\mathbb{R}^{N}"/>, the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;d"/>th-order vector autoregression takes a linear formula as

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{s}_{t}=\sum_{k=1}^{d}{\color{red}\boldsymbol{A}_{k}}\boldsymbol{s}_{t-k}+\underbrace{\boldsymbol{\epsilon}_{t}}_{\text{error}},\,\forall t"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{A}_1,\boldsymbol{A}_2,\ldots,\boldsymbol{A}_{d}\in\mathbb{R}^{N\times N}"/> are the coefficient matrices, which can capture the temporal correlations of the multivariate time series.

One great challenge of modeling time series with vector autoregression is identifying the time-varying system behaviors in the analysis, which is often associated with the nonstationarity issue. Although the nonstationarity and time-varying system behaviors are pretty clear to verify, the problem of discovering underlying data patterns from time-varying systems is challenging and still demands further exploration.

Typically, time-varying vector autoregression takes a sequence of vector autoregressive processes at different times, and it is capable of handling the time-varying system behaviors. For any observed spatiotemporal data in the form of multivariate time series, i.e., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{s}_1,\boldsymbol{s}_2,\ldots,\boldsymbol{s}_{T}\in\mathbb{R}^{N}"/>, our model considers a time-varying linear system as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\min_{\color{red}\{\boldsymbol{A}_{t}\}}~\frac{1}{2}\underbrace{\sum_{t=d+1}^{T}\left\|\boldsymbol{y}_{t}-{\color{red}\boldsymbol{A}_{t}}\boldsymbol{z}_{t}\right\|_2^2}_{\text{time-varying autoregression}}"/></p>

with the data pair:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\begin{aligned} \boldsymbol{y}_{t}&\triangleq \boldsymbol{s}_{t}\in\mathbb{R}^{N} \\ \boldsymbol{z}_{t}&\triangleq\begin{bmatrix} \boldsymbol{s}_{t-1} \\ \vdots \\ \boldsymbol{s}_{t-d} \\ \end{bmatrix} \end{aligned}\in\mathbb{R}^{dN}"/></p>

As the data pair <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\{\boldsymbol{y}_t,\boldsymbol{z}_t\}"/> is readily available, one can learn the coefficient matrices <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{A}_t\in\mathbb{R}^{N\times (dN)}"/>.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/tensor_Atd.png" width="220" />
</p>

<p align = "center">
<b>Figure 1.</b> A collection of coefficient matrices <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{A}_t,\forall t"/> can be represented as a coefficient tensor <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{\mathcal{A}}"/> of size <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;N\times (dN)\times (T-d)"/>, showing <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;dN^2(T-d)"/> parameters for estimation.
</p>

<br>

As shown in Figure 1, the coefficient matrices can be viewed as a coefficient tensor whose parameters are of number <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\color{blue}dN^2(T-d)"/>. Since there are only <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\color{blue}NT"/> time series observations, one technical challenge for solving the optimization problem would arise as the **over-parameterization** issue in the modeling process.

<br>

## Tensor Factorization on the Coefficient Tensor

To compress the coefficient tensor in the time-varying autoregression and capture spatiotemporal patterns simultaneously, we factorize the coefficient tensors into a sequence of components via the use of Tucker tensor decomposition:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\begin{aligned} {\color{red}\boldsymbol{\mathcal{A}}}&=\boldsymbol{\mathcal{G}}\times_1\underbrace{\color{red}\boldsymbol{W}}_{\text{spatial modes}}\times_2\boldsymbol{V}\times_3\underbrace{\color{red}\boldsymbol{X}}_{\text{temporal modes}} \\ \Rightarrow\quad{\color{red}\boldsymbol{A}_{t}}&=\boldsymbol{\mathcal{G}}\times_1{\boldsymbol{W}}\times_2\boldsymbol{V}\times_3{\boldsymbol{x}_{t}^\top} \\ &=\boldsymbol{W}\boldsymbol{G}(\boldsymbol{x}_{t}^\top\otimes\boldsymbol{V})^\top \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{W}\in\mathbb{R}^{N\times R}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{X}\in\mathbb{R}^{(T-d)\times R}"/> can be interpreted as spatial modes/patterns and temporal modes/patterns, respectively.

Putting this tensor factorization with time-varying autoregression together, we have the following time-varying low-rank vector autoregression problem (also see [the optimization with orthogonal constraints](https://spatiotemporal-data.github.io/probs/orth-var/)):

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\min_{\boldsymbol{W},\boldsymbol{G},\boldsymbol{V},\boldsymbol{X}}~\frac{1}{2}\sum_{t=d+1}^{T}\left\|\boldsymbol{y}_{t}-\boldsymbol{W}\boldsymbol{G}(\boldsymbol{x}_{t}^\top\otimes\boldsymbol{V})^\top\boldsymbol{z}_{t}\right\|_2^2"/></p>

We can use the alternating minimization method to solve this optimization problem. Let <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;f"/> be the objective function of the optimization problem, the scheme can be summarized as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\left\{\begin{aligned} \boldsymbol{W}:&=\{\boldsymbol{W}\mid\frac{\partial f}{\partial\boldsymbol{W}}=\boldsymbol{0}\} \\ \boldsymbol{G}:&=\{\boldsymbol{G}\mid\frac{\partial f}{\partial\boldsymbol{G}}=\boldsymbol{0}\} \\ \boldsymbol{V}:&=\{\boldsymbol{V}\mid\frac{\partial f}{\partial\boldsymbol{V}}=\boldsymbol{0}\} \\ \boldsymbol{x}_{t}:&=\{\boldsymbol{x}_{t}\mid\frac{\partial f}{\partial\boldsymbol{x}_{t}}=\boldsymbol{0}\},\,\forall t \end{aligned} \right."/></p>


The Python implementation with `numpy` is given as follows. We plan to give some spatiotemporal data examples such as fluid flow, sea surface temperature, and human mobility for discovering interpretable patterns.

<br>

```python
import numpy as np

def update_cg(w, r, q, Aq, rold):
    alpha = rold / np.inner(q, Aq)
    w = w + alpha * q
    r = r - alpha * Aq
    rnew = np.inner(r, r)
    q = r + (rnew / rold) * q
    return w, r, q, rnew

def ell_v(Y, Z, W, G, V_transpose, X, temp2, d, T):
    rank, dN = V_transpose.shape
    temp = np.zeros((rank, dN))
    for t in range(d, T):
        temp3 = np.outer(X[t, :], Z[:, t - d])
        Pt = temp2 @ np.kron(X[t, :].reshape([rank, 1]), V_transpose) @ Z[:, t - d]
        temp += np.reshape(Pt, [rank, rank], order = 'F') @ temp3
    return temp

def conj_grad_v(Y, Z, W, G, V_transpose, X, d, T, maxiter = 5):
    rank, dN = V_transpose.shape
    temp1 = W @ G
    temp2 = temp1.T @ temp1
    v = np.reshape(V_transpose, -1, order = 'F')
    temp = np.zeros((rank, dN))
    for t in range(d, T):
        temp3 = np.outer(X[t, :], Z[:, t - d])
        Qt = temp1.T @ Y[:, t - d]
        temp += np.reshape(Qt, [rank, rank], order = 'F') @ temp3
    r = np.reshape(temp - ell_v(Y, Z, W, G, V_transpose, X, temp2, d, T), -1, order = 'F')
    q = r.copy()
    rold = np.inner(r, r)
    for it in range(maxiter):
        Q = np.reshape(q, (rank, dN), order = 'F')
        Aq = np.reshape(ell_v(Y, Z, W, G, Q, X, temp2, d, T), -1, order = 'F')
        v, r, q, rold = update_cg(v, r, q, Aq, rold)
    return np.reshape(v, (rank, dN), order = 'F')

def trvar(mat, d, rank, maxiter = 50):
    N, T = mat.shape
    Y = mat[:, d : T]
    Z = np.zeros((d * N, T - d))
    for k in range(d):
        Z[k * N : (k + 1) * N, :] = mat[:, d - (k + 1) : T - (k + 1)]
    u, _, v = np.linalg.svd(Y, full_matrices = False)
    W = u[:, : rank]
    u, _, _ = np.linalg.svd(Z, full_matrices = False)
    V = u[:, : rank]
    u, _, _ = np.linalg.svd(mat.T, full_matrices = False)
    X = u[:, : rank]
    del u
    loss = np.zeros(maxiter)
    for it in range(maxiter):
        temp1 = np.zeros((N, rank * rank))
        temp2 = np.zeros((rank * rank, rank * rank))
        for t in range(d, T):
            temp = np.kron(X[t, :].reshape([rank, 1]), V.T) @ Z[:, t - d]
            temp1 += np.outer(Y[:, t - d], temp)
            temp2 += np.outer(temp, temp)
        G = np.linalg.pinv(W) @ temp1 @ np.linalg.inv(temp2)
        W = temp1 @ G.T @ np.linalg.inv(G @ temp2 @ G.T)
        V = conj_grad_v(Y, Z, W, G, V.T, X, d, T).T
        temp3 = W @ G
        for t in range(d, T):
            X[t, :] = np.linalg.pinv(temp3 @ np.kron(np.eye(rank), (V.T @ Z[:, t - d]).reshape([rank, 1]))) @ Y[:, t - d]
    return W, G, V, X
```

<br>

## Fluid Flow Data

Investigating fluid dynamic systems is of great interest for uncovering large-scale spatiotemporal coherent structures because dominant patterns exist in the flow field. The data-driven models, such as proper orthogonal decomposition and dynamic mode decomposition, have become an important paradigm. To analyze the underlying spatiotemporal patterns of fluid dynamics, we consider the cylinder wake dataset in which the flow shows a supercritical Hopf bifurcation.

[The cylinder wake dataset](http://dmdbook.com/) is collected from the fluid flow passing a circular cylinder with laminar vortex shedding at Reynolds number Re = 100, which is larger than the critical Reynolds number, using direct numerical simulations of the Navier-Stokes equations. This is a representative three-dimensional flow dataset in fluid dynamics, consisting of matrix-variate time series of vorticity field snapshots for the wake behind a cylinder (see Figure 2). The dataset is of size <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;199\times 449\times 150"/>, representing 199-by-449 vorticity fields with 150 time snapshots.

<br>

```python
import numpy as np
import seaborn as sns
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
color = scipy.io.loadmat('CCcool.mat')
cc = color['CC']
newcmp = LinearSegmentedColormap.from_list('', cc)

tensor = np.load('tensor.npz')['arr_0']
tensor = tensor[:, :, : 150]
M, N, T = tensor.shape

plt.rcParams['font.size'] = 13
plt.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure(figsize = (7, 8))
id = np.array([5, 10, 15, 20, 25, 30, 35, 40])
for t in range(8):
    ax = fig.add_subplot(4, 2, t + 1)
    ax = sns.heatmap(tensor[:, :, id[t] - 1], cmap = newcmp, vmin = -5, vmax = 5, cbar = False)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), tensor[:, :, id[t] - 1],
               levels = np.linspace(0.15, 15, 30), colors = 'k', linewidths = 0.7)
    ax.contour(np.linspace(0, N, N), np.linspace(0, M, M), tensor[:, :, id[t] - 1],
               levels = np.linspace(-15, -0.15, 30), colors = 'k', linestyles = 'dashed', linewidths = 0.7)
    plt.xticks([])
    plt.yticks([])
    plt.title(r'$t = {}$'.format(id[t]))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
plt.show()
fig.savefig('fluid_flow_heatmap.png', bbox_inches = 'tight')
```

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/fluid_flow_heatmap.png" alt="drawing" width="300">
</p>

<p align="center"><b>Figure 2</b>: Heatmaps (snapshots) of the fluid flow at times <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;t=5,10,\ldots,40"/>. It shows that the snapshots at times <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;t=5"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;t=35"/> are even same, and the snapshots at times <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;t=10"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;t=40"/> are also even same, implying the seasonality as 30 for the first 50 snapshots.</p>

<br>


<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on December 13, 2023.)</p>
