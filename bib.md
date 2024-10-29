---
layout: default
---

## Step Forward on the Prior Knowledge

[Xinyu Chen (陈新宇)](https://xinychen.github.io/) created this page since early 2024 with the purpose of fostering research knowledge, vision, insight, and style. In the meantime, it aims to connect random concepts with mathematics and machine learning.

<br>

### 28th Mile
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

### 27th Mile
#### Importance of Sparsity in Interpretable Machine Learning

Sparsity is an important type of model-based interpretability methods. Typically, the practitioner can impose sparsity on the model by limiting the number of nonzero parameters. When the number of nonzero parameters is sufficiently small, a practitioner can interpret the variables corresponding to those parameters as being meaningfully related to the outcome in question and can also interpret the magnitude and direction of the parameters. Two important methods including <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm penalty (e.g., LASSO) and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-norm constraint (e.g., OMP). Model sparsity is often useful for high-dimensional problems, where the goal is to identify key features for further analysis.

**References**

- W. James Murdoch, Chandan Singh, Karl Kumbier, Reza Abbasi-Asl, and Bin Yu (2019). [Definitions, methods, and applications in interpretable machine learning](https://www.pnas.org/doi/10.1073/pnas.1900654116). PNAS.


<br>

### 26th Mile
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
- Jiachang Liu, Sam Rosen, Chudi Zhong, Cynthia Rudin (2023). [OKRidge: Scalable Optimal k-Sparse Ridge Regression](https://arxiv.org/pdf/2304.06686). NeurIPS 2023.

<br>

### 25th Mile
#### Mobile Service Usage Data

- Orlando E. Martínez-Durive et al. (2023). [The NetMob23 Dataset: A High-resolution Multi-region Service-level Mobile Data Traffic Cartography](https://arxiv.org/abs/2305.06933). arXiv:2305.06933.
- André Zanella (2024). [Characterizing Large-Scale Mobile Traffic Measurements for Urban, Social and Networks Sciences](https://dspace.networks.imdea.org/handle/20.500.12761/1852). PhD thesis.

<br>

### 24th Mile
#### Optimization in Reinforcement Learning

**References**

- Jalaj Bhandari and Daniel Russo (2024). [Global Optimality Guarantees for Policy Gradient Methods](https://doi.org/10.1287/opre.2021.0014). Operations Research, 72(5): 1906 - 1927.
- Lucia Falconi, Andrea Martinelli, and John Lygeros (2024). [Data-driven optimal control via linear programming: boundedness guarantees](https://doi.org/10.1109/TAC.2024.3465536). IEEE Transactions on Automatic Control.

<br>

### 23rd Mile
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

### 22nd Mile
#### Revisiting <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-Norm Minimization




**References**
- Lijun Ding (2023). [One dimensional least absolute deviation problem](https://threesquirrelsdotblog.com/2023/06/08/one-dimensional-least-absolute-deviation-problem/). Blog post.
- Gregory Gundersen (2022). [Weighted Least Squares](https://gregorygundersen.com/blog/2022/08/09/weighted-ols/). Blog post.
- stephentu's blog (2014). [Subdifferential of a norm](https://stephentu.github.io/blog/convex-analysis/2014/10/01/subdifferential-of-a-norm.html). Blog post.

<br>

### 21st Mile
#### Research Seminars

- [Computational Research in Boston and Beyond (CRIBB) seminar series](https://math.mit.edu/crib/): A forum for interactions among scientists and engineers throughout the Boston area working on a range of computational problems. This forum consists of a monthly seminar where individuals present their work.
- [Param-Intelligence (𝝅) seminar series](https://sites.google.com/view/paramintelligencelab/seminar-series): A dynamic platform for researchers, engineers, and students to explore and discuss the latest advancements in integrating machine learning with scientific computing. Key topics include data-driven modeling, physics-informed neural surrogates, neural operators, and hybrid computational methods, with a strong focus on real-world applications across various fields of computational science and engineering.

<br>

### 20th Mile
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

### 19th Mile
#### Causal Inference for Geosciences

Learning causal interactions from time series of complex dynamical systems is of great significance in real-world systems. But the questions arise as: 1) How to formulate causal inference for complex dynamical systems? 2) How to detect causal links? 3) How to quantify causal interactions?

**References**
- Jakob Runge (2017). [Causal inference and complex network methods for the geosciences](https://www.belmontforum.org/wp-content/uploads/2018/04/Causal-Inference-and-Complex-Network-Methods-for-the-Geosciences.pdf). Slides.
- Jakob Runge, Andreas Gerhardus, Gherardo Varando, Veronika Eyring & Gustau Camps-Valls (2023). [Causal inference for time series](https://www.nature.com/articles/s43017-023-00431-y). Nature Reviews Earth & Environment, 4: 487–505.
- Jitkomut Songsiri (2013). [Sparse autoregressive model estimation for learning Granger causality in time series](https://doi.org/10.1109/ICASSP.2013.6638248). 2013 IEEE International Conference on Acoustics, Speech and Signal Processing.

<br>

### 18th Mile
#### Tensor Factorization for Knowledge Graph Completion

Knowledge graph completion is a kind of link prediction problems, inferring missing "facts" based on existing ones. Tucker
decomposition of the binary tensor representation of knowledge graph triples allows one to make data completion.

**References**
- [TuckER: Tensor Factorization for Knowledge Graph Completion](https://github.com/ibalazevic/TuckER). GitHub.
- Ivana Balazevic, Carl Allen, Timothy M. Hospedales (2019). TuckER: Tensor Factorization for Knowledge Graph Completion. arXiv:1901.09590. [[PDF](https://arxiv.org/pdf/1901.09590)]

<br>

### 17th Mile
#### RESCAL: Tensor-Based Relational Learning

Multi-relational data is everywhere in real-world applications such as computational biology, social networks, and semantic web. This type of data is often represented in the form of graphs or networks where nodes represent entities, and edges represent different types of relationships.

Instead of using the classical Tucker and CP tensor decomposition, RESCAL takes the inherent structure of dyadic relational data into account, whose tensor factorization on the tensor variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\mathcal{X}}:=\{\boldsymbol{X}_k\}_k"/> (i.e., frontal tensor slices) is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}_k=\boldsymbol{A}\boldsymbol{S}_{k}\boldsymbol{A}^\top"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{n\times r}"/> is the global entity factor matrix, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{S}\in\mathbb{R}^{r\times r}"/> specifies the interaction of the latent components. Such kind of methods can be used to solve link prediction, collective classification, and link-based clustering.

**References**

- Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel (2011). A Three-Way Model for Collective Learning on Multi-Relational Data. ICML 2011. [[PDF](https://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf)] [[Slides](https://www.cip.ifi.lmu.de/~nickel/data/slides-icml2011.pdf)]
- Maximilian Nickel (2013). [Tensor Factorization for Relational Learning](https://edoc.ub.uni-muenchen.de/16056/1/Nickel_Maximilian.pdf). PhD thesis.
- Denis Krompaß, Maximilian Nickel, Xueyan Jiang, Volker Tresp (2013). [Non-Negative Tensor Factorization with RESCAL](https://www.dbs.ifi.lmu.de/~krompass/papers/NonNegativeTensorFactorizationWithRescal.pdf). ECML Workshop 2013.
- Elynn Y. Chen, Rong Chen (2019). Modeling Dynamic Transport Network with Matrix Factor Models: with an Application to International Trade Flow. arXiv:1901.00769. [[PDF](https://arxiv.org/pdf/1901.00769)]
- Zhanhong Cheng. [factor_matrix_time_series](https://github.com/chengzhanhong/factor_matrix_time_series). GitHub.

<br>

### 16th Mile
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

### 15th Mile
#### Synthetic Sweden Mobility

The Synthetic Sweden Mobility (SySMo) model provides a simplified yet statistically realistic microscopic representation of the real population of Sweden. The agents in this synthetic population contain socioeconomic attributes, household characteristics, and corresponding activity plans for an average weekday. This agent-based modelling approach derives the transportation demand from the agents’ planned activities using various transport modes (e.g., car, public transport, bike, and walking). The dataset is available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10648077).

Going back to the individual mobility trajectory, there would be some opportunities to approach taxi trajectory data such as

- [1 million+ trips collected by 13,000+ taxi cabs during 5 days in Harbin, China](https://github.com/boathit/deepgtt)
- [Daily GPS trajectory data of 664 electric taxis in Shenzhen, China](https://guangwang.me/#/data)
- Takahiro Yabe, Kota Tsubouchi, Toru Shimizu, Yoshihide Sekimoto, Kaoru Sezaki, Esteban Moro & Alex Pentland (2024). [YJMob100K: City-scale and longitudinal dataset of anonymized human mobility trajectories](https://www.nature.com/articles/s41597-024-03237-9). Scientific Data. [[Data](https://zenodo.org/records/13237029)] [[Challenge](https://wp.nyu.edu/humobchallenge2024/)]
- [The dataset comprises trajectory data of traffic participants, along with traffic light data, current local weather data, and air quality data from the Application Platform Intelligent Mobility (AIM) Research Intersection](https://zenodo.org/records/11396372)


<br>

### 14th Mile
#### Prediction on Extreme Floods

AI increases global access to reliable flood forecasts (see [dataset](https://zenodo.org/doi/10.5281/zenodo.8139379)).

Another weather forecasting dataset for consideration: [Rain forecasts world-wide on an expansive data set with over a magnitude more hi-res rain radar data](https://weather4cast.net/neurips2024/challenge/).

**References**
- Nearing, G., Cohen, D., Dube, V. et al. (2024). [Global prediction of extreme floods in ungauged watersheds](https://doi.org/10.1038/s41586-024-07145-1). Nature, 627: 559–563.

<br>

### 13th Mile
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

### 12th Mile
#### Economic Complexity

**References**

- César A. Hidalgo (2021). [Economic complexity theory and applications](https://doi.org/10.1038/s42254-020-00275-1). Nature Reviews Physics. 3: 92-113.

<br>

### 11th Mile
#### Time-Varying Autoregressive Models

Vector autoregression (VAR) has a key assumption that the coeffcients are invariant across time (i.e., time-invariant), but it is not always true when accounting for psychological phenomena such as the phase transition from a healthy to unhealthy state (or vice versa). Consequently, time-varying vector autoregressive models are of great significance for capturing the parameter changes in response to interventions. From the statistical perspective, there are two types of lagged effects between pairs of variables: **autocorrelations** (e.g., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{t}\to\boldsymbol{x}_{t+1}"/>) and **cross-lagged effects** (e.g., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{t}\to\boldsymbol{y}_{t+1}"/>). The time-varying autoregressive models can be solved by using generalized additive model and kernel smoothing estimation.

**References**

- Haslbeck, J. M., Bringmann, L. F., & Waldorp, L. J. (2021). [A tutorial on estimating time-varying vector autoregressive models](https://doi.org/10.1080/00273171.2020.1743630). Multivariate Behavioral Research, 56(1), 120-149.

<br>

### 10th Mile
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


### 9th Mile
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


### 8th Mile
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


### 7th Mile
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


### 6th Mile
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


### 5th Mile
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


### 4th Mile
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


### 3rd Mile
#### Causal Effect Estimation/Imputation

The causal effect estimation problem is usually defined as a matrix completion on the partially observed matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}\in\mathbb{R}^{N\times T}"/> in which <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> units and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> periods are involved. The observed index set is denoted by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega"/>. The optimization is from the [classical matrix factorization techniques for recommender systems (see Koren et al.'09)](https://doi.org/10.1109/MC.2009.263):

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{W},\boldsymbol{X},\boldsymbol{u},\boldsymbol{p}}\,\frac{1}{2}\left\|\mathcal{P}_{\Omega}(\boldsymbol{Y}-\boldsymbol{W}^\top\boldsymbol{X}-\boldsymbol{u}\mathbf{1}_{T}^\top-\mathbf{1}_{N}\boldsymbol{p}^\top)\right\|_F^2"/></p>

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

<br>