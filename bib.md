---
layout: default
---

## Step Forward on the Prior Knowledge

[Xinyu Chen](https://xinychen.github.io/) created this page since early 2024 with the purpose of fostering research knowledge, vision, insight, and style. In the meantime, it aims to connect discrete ideas with mathematics and machine learning.

<br>


### 9th Mile
#### Higher-Order Graph & Hypergraph

Higher-order graph...

**References**

- [Higher-order organization of complex networks](https://snap.stanford.edu/higher-order/). Stanford University.
- Quintino Francesco Lotito, Federico Musciotto, Alberto Montresor, Federico Battiston (2022). [Higher-order motif analysis in hypergraphs](https://www.nature.com/articles/s42005-022-00858-7). Communications Physics, volume 5, Article number: 79.
- Christian Bick, Elizabeth Gross, Heather A. Harrington, and Michael T. Schaub (2023). [What are higher-order networks?](https://doi.org/10.1137/21M1414024) SIAM Review. 65(3).

<br>


### 8th Mile
#### Eigenvalues of Directed Cycles

Graph signal processing mentioned an interesting property of directed cycle (see Figure 2 in the [literature](https://arxiv.org/pdf/2303.12211)). The adjacency matrix of an directed cycle has a set of unit eigenvalues as follows.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/eigenvalues_directed_cycle.png" width="250" />
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


### 7th Mile
#### Graph Filter

Defining graph-aware operator plays an important role for characterizing a signal <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{N}"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> vertices over a graph <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;G"/>. One simple idea is introducing the adjacency matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> so that the operation is <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\boldsymbol{x}"/>. In that case, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> is a simple operator that accounts for the local connectivity of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;G"/>. One example is using the classical unit delay (seems to be time-shift) such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}=\begin{bmatrix} 0 & 0 & 0 & \cdots & 1 \\ 1 & 0 & 0 & \cdots & 0 \\ 0 & 1 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 0 \end{bmatrix}\in\mathbb{R}^{N\times N}"/></p>

The simplest signal operation as multiplication by the adjacency matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}"/> defines graph filters as matrix polynomials of the form

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;p(\boldsymbol{A})=p_0\boldsymbol{I}_N+p_1\boldsymbol{A}+\cdots+p_{N-1}\boldsymbol{A}^{N-1}"/></p>

For instance, we have

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}=\begin{bmatrix} 0 & 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \end{bmatrix}\quad\quad \boldsymbol{A}^2=\begin{bmatrix} 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \end{bmatrix}"/></p>

When applying the polynomial filter to a graph signal <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{N}"/>, the operation <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\boldsymbol{x}"/> takes a local linear combination of the signal values at one-hop neighbors. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}^2\boldsymbol{x}"/> takes a local linear combination of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\boldsymbol{x}"/>, referring to two-hop neighbors. Consequently, a graph filter <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;p(\boldsymbol{A})"/> of order <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N-1"/> represents the mixing values that are at most <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N-1"/> hops away.

**References**

- Geert Leus, Antonio G. Marques, José M. F. Moura, Antonio Ortega, David I Shuman (2023). [Graph Signal Processing: History, Development, Impact, and Outlook](https://arxiv.org/pdf/2303.12211). arXiv:2303.12211.
- A Sandryhaila, JMF Moura (2013). [Discrete signal processing on graphs: Graph filters](https://users.ece.cmu.edu/~asandryh/papers/icassp13.pdf). Section 3: Graph Filters.
- Henry Kenlay, Dorina Thanou, [Xiaowen Dong](https://web.media.mit.edu/~xdong/) (2020). [On The Stability of Polynomial Spectral Graph Filters](https://web.media.mit.edu/~xdong/paper/icassp20.pdf). ICASSP 2020.
- Eylem Tugçe Güneyi, Berkay Yaldız, Abdullah Canbolat, and Elif Vural (2024). [Learning Graph ARMA Processes From Time-Vertex Spectra](https://doi.org/10.1109/TSP.2023.3329948). IEEE Transactions on Signal Processing, 72: 47 - 56.

<br>


### 6th Mile
#### Graph Signals

For any graph <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;G=\{\mathcal{V},\mathcal{E}\}"/> where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{V}=\{1,2,\ldots,N\}"/> is a finite set of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> vertices, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{E} \subseteq \mathcal{V}\times\mathcal{V}"/> is the set of edges. Graph signals can be formally represented as vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{N}"/> where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;x_{n}"/> (or say <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}(n)"/> in the following) stores the signal value at the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/>th vertex in <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{V}"/>. The graph Fourier transform of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> is element-wise defined as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\hat{\boldsymbol{x}}(k)=\langle\boldsymbol{x},\boldsymbol{\psi}_k\rangle=\sum_{n=1}^{N}\boldsymbol{x}(n)\boldsymbol{\psi}_{k}^{*}(n)"/></p>

or another form such that

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\hat{\boldsymbol{x}}=\boldsymbol{\Psi}^{H}\boldsymbol{x}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\Psi}"/> consists of the eigenvectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\psi}_k,\,k=1,2,\ldots,N"/>. The notation <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\cdot^{*}"/> is the conjugate of complex values, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\cdot^{H}"/> is the conjugate transpose.

**References**

- Santiago Segarra, Weiyu Huang, and Alejandro Ribeiro (2020). [Signal Processing on Graphs](https://www.seas.upenn.edu/~ese2240/labs/1200_ch_9_signal_processing_on_graphs.pdf).
- Matthew Begue. [Fourier analysis on graphs](https://www.norbertwiener.umd.edu/Research/lectures/2014/MBegue_Prelim.pdf). Slides.

<br>


### 5th Mile
#### Graph Signal Processing

Graph signal processing not only focuses on the graph typology (e.g., connection between nodes), but also covers the quantity of nodes (i.e., graph signals) with weighted adjacency information.

**References**

- Antonio Ortega, Pascal Frossard, Jelena Kovacevic, Jose M. F. Moura, Pierre Vandergheynst (2017). [Graph Signal Processing: Overview, Challenges and Applications](https://arxiv.org/pdf/1712.00468). arXiv:1712.00468.
- Xiaowen Dong, Dorina Thanou, Laura Toni, Michael Bronstein, and Pascal Frossard (2020). [Graph signal processing for machine learning: A review and new perspectives](https://arxiv.org/pdf/2007.16061). arXiv:2007.16061. [[Slides](https://web.media.mit.edu/~xdong/talk/BDI_GSP.pdf)]
- Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković (2021). [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/pdf/2104.13478). arXiv:2104.13478.
- Wei Hu, Jiahao Pang, Xianming Liu, Dong Tian, Chia-Wen Lin, Anthony Vetro (2022). [Graph Signal Processing for Geometric Data and Beyond: Theory and Applications](https://doi.org/10.1109/TMM.2021.3111440). IEEE Transactions on Multimedia, 24: 3961-3977.
- Geert Leus, Antonio G. Marques, José M. F. Moura, Antonio Ortega, David I Shuman (2023). [Graph Signal Processing: History, Development, Impact, and Outlook](https://arxiv.org/pdf/2303.12211). arXiv:2303.12211.

<br>


### 4th Mile
#### Clifford Product

In [Grassmann algebra](https://en.wikipedia.org/wiki/Exterior_algebra), the inner product between two vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}=x_1\vec{e}_1+x_2\vec{e}_2"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{y}=y_1\vec{e}_1+y_2\vec{e}_2"/> (w/ basis vectors <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{e}_1"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{e}_2"/>) is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\langle\vec{x},\vec{y}\rangle=\|\vec{x}\|_2 \|\vec{y}\|_2 \cos\theta"/></p>
implies to be the multiplication between <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}"/> and the projection of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{y}"/> on <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}"/>. Here, the notation <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\cdot\|_2"/> refers to the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_2"/> norm, or say the magnitude. <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\theta"/> is the angle between <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{x}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\vec{y}"/> in the plane containing them.

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


### 3rd Mile
#### Causal Effect Estimation/Imputation

The causal effect estimation problem is usually defined as a matrix completion on the partially observed matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}\in\mathbb{R}^{N\times T}"/> in which <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> units and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> periods are involved. The observed index set is denoted by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega"/>. The optimization is from the [classical matrix factorization techniques for recommender systems (see Koren et al.'09)](https://doi.org/10.1109/MC.2009.263):

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{W},\boldsymbol{X},\boldsymbol{u},\boldsymbol{p}}~\frac{1}{2}\left\|\mathcal{P}_{\Omega}(\boldsymbol{Y}-\boldsymbol{W}^\top\boldsymbol{X}-\boldsymbol{u}\mathbf{1}_{T}^\top-\mathbf{1}_{N}\boldsymbol{p}^\top)\right\|_F^2"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}\in\mathbb{R}^{R\times N}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{R\times T}"/> are factor matrices, referring to units and periods, respectively. Here, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{u}\in\mathbb{R}^{N}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{p}\in\mathbb{R}^{T}"/> are bias vectors, corresponding to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> units and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> periods, respectively. This idea has also been examined on the tensor factorization (to be honest, performance gains are marginal), see e.g., Bayesian augmented tensor factorization by [Chen et al.'19](https://doi.org/10.1016/j.trc.2019.03.003). In the causal effect imputation, one great challenge is how to handle the structural patterns of missing data as mentioned by [Athey et al.'21](https://doi.org/10.1080/01621459.2021.1891924). The structural missing patterns have been discussed on spatiotemporal data with [autoregressive tensor factorization (for spatiotemporal predictions)](https://doi.org/10.1109/ICDM.2017.146).

<br>


### 2nd Mile
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


### 1st Mile
#### Large Time Series Forecasting Models

As we know, the training data in the large time series model is from different areas, this means that the model training process highly depends on the selected datasets across various areas, so one question is how to reduce the model biases if we consider the forecasting scenario as traffic flow or human mobility? Because I guess time series data in different areas should demonstrate different data behaviors. Hopefully, it is interesting to develop domain-specific time series datasets (e.g., [Largest multi-city traffic dataset](https://utd19.ethz.ch/)) and large models (e.g., [TimeGPT](https://docs.nixtla.io/)).

**References**

- Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, Doyen Sahoo (2024). [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/pdf/2402.02592). arXiv:2402.02592.

<br>





Motivation & Principle: 不积硅步，无以至千里。
