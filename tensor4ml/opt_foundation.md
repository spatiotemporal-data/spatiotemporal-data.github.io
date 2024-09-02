---
layout: default
---

### Foundation of Optimization

<br>

#### I. Gradient Descent Methods

##### A. Gradient Descent

##### B. Steepest Gradient Descent

##### C. Conjugate Gradient Descent

##### D. Proximal Gradient Descent

The optimization problem of LASSO is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \min_{\boldsymbol{x}}\, \frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2+\gamma\|\boldsymbol{x}\|_1 "/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\gamma\geq 0"/> is the regularization parameter that controls the sparsity of variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>. The objective function has two terms:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \begin{cases} f(\boldsymbol{x})=\frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2 \\ g(\boldsymbol{x})=\gamma\|\boldsymbol{x}\|_1 \end{cases}"/></p>

The derivative of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f(\boldsymbol{x})"/>, namely, gradient, is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \nabla f(\boldsymbol{x})=\frac{\operatorname{d}f(\boldsymbol{x})}{\operatorname{d}\boldsymbol{x}}=\boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}) "/></p>

To elaborate on the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm, one great challenge is that the function <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;g(\boldsymbol{x})"/> is not necessarily differentiable. Thus, the derivative is expressed in terms of a subgradient. Suppose a standard <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm minimization problem with a penalty term:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \min_{\boldsymbol{x}}\, \frac{1}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_2^2+\gamma\|\boldsymbol{x}\|_1 "/></p>


<br>

#### II. Alternating Minimization

##### A. Alternating Least Squares

Alternating minimization is classical method for solving matrix and tensor factorization problems. In what follows, we use a matrix factorization example on the partially observed matrix to elaborate on the essential idea of alternating least squares. For any partially observed matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}\in\mathbb{R}^{N\times T}"/> with the observed index set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega"/>, the rank-<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;R"/> matrix factorization can be formulated on the partially observed entries of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}"/>, and it takes the following form:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{W},\boldsymbol{X}}\,\frac{1}{2}\|\mathcal{P}_{\Omega}(\boldsymbol{Y}-\boldsymbol{W}^\top\boldsymbol{X})\|_F^2+\frac{\rho}{2}(\|\boldsymbol{W}\|_F^2+\|\boldsymbol{X}\|_F^2)"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;R<\min\{N,T\}"/> is a low rank for the approximation via the use of matrix factorization. In particular, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}\in\mathbb{R}^{R\times N}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{R\times T}"/> are factor matrices, while <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\rho"/> is the weight parameter of the regularization terms for preventing overfitting. To represent the matrix factorization on the observed index set <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega"/> of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{P}_{\Omega}:\mathbb{R}^{N\times T}\to\mathbb{R}^{N\times T}"/> denotes the orthogonal projection supported on <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega"/>. For any <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;(n,t)"/>-th entry of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}"/>, the operator <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{P}_{\Omega}(\cdot)"/> can be described as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;[\mathcal{P}_{\Omega}(\boldsymbol{Y})]_{n,t}=\begin{cases} y_{n,t}, & \text{if}\,(n,t)\in\Omega, \\ 0, & \text{otherwise.} \end{cases}"/></p>


<br>

---

<span style="color:gray">
**Example 1.** Given matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}=\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ \end{bmatrix}"/>, if the observed index set is <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\Omega=\{(1,1), (2,2)\}"/>, then we have
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{P}_{\Omega}(\boldsymbol{Y})=\begin{bmatrix} 1 & 0 \\ 0 & 4 \\ \end{bmatrix}"/></p>


---

<br>

Let the objective function of the aforementioned matrix factorization be <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f"/> (or <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f(\boldsymbol{W},\boldsymbol{X})"/> when mentioning all variables), then the partial derivatives of <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f"/> with respect to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}"/> are given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\frac{\partial f}{\partial\boldsymbol{W}}=-\boldsymbol{X}\mathcal{P}_{\Omega}^\top(\boldsymbol{Y}-\boldsymbol{W}^\top\boldsymbol{X})+\rho\boldsymbol{W} "/></p>

and

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\frac{\partial f}{\partial\boldsymbol{X}}=-\boldsymbol{W}\mathcal{P}_{\Omega}(\boldsymbol{Y}-\boldsymbol{W}^\top\boldsymbol{X})+\rho\boldsymbol{X}"/></p>

respectively. One can find the optimal solution to each subproblem by letting the partial derivative be zeros. In this case, the alternating minimization takes an iterative process by solving two subproblems:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{cases} \displaystyle\boldsymbol{W}:=\arg\min_{\boldsymbol{W}}\,f(\boldsymbol{W},\boldsymbol{X}) \\ \displaystyle\boldsymbol{X}:=\arg\min_{\boldsymbol{X}}\,f(\boldsymbol{W},\boldsymbol{X}) \end{cases} "/></p>

The closed-form solution to each subproblem is given in the column vector of the factor matrix. If <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}_{i}\in\mathbb{R}^{R},\,\forall i\in[N]"/> is the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space; i"/>-th column vector of the factor matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{W}\in\mathbb{R}^{R\times N}"/>, then the least squares solution is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}_{i}:=\Bigl(\sum_{t:(i,t)\in\Omega}\boldsymbol{x}_{t}\boldsymbol{x}_t^\top+\rho\boldsymbol{I}_{R}\Bigr)^{-1}\sum_{t:(i,t)\in\Omega}\boldsymbol{x}_{t}y_{i,t}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{t}\in\mathbb{R}^{R},\,t\in[T]"/> is the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>-th column vector of the factor matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{X}\in\mathbb{R}^{R\times T}"/>. The least squares solution to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{t}"/> is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}_{t}:=\Bigl(\sum_{i:(i,t)\in\Omega}\boldsymbol{w}_{i}\boldsymbol{w}_i^\top+\rho\boldsymbol{I}_{R}\Bigr)^{-1}\sum_{i:(i,t)\in\Omega}\boldsymbol{w}_{i}y_{i,t}"/></p>

Since each subproblem in the alternating minimization scheme has least squares solutions, the scheme is also referred to as the alternating least squares.

<br>

---

<span style="color:gray">
Matrix factorization with Panda
<span>

---

<br>

##### B. Alternating Minimization with Conjugate Gradient Descent

Smoothing matrix factorization on Panda

<br>

#### III. Alternating Direction Method of Multipliers

The Alternating Direction Method of Multipliers (ADMM) is an optimization algorithm designed to solve complex problems by decomposing them into a sequence of easy-to-solve subproblems ([Boyd et al., 2011](http://dx.doi.org/10.1561/2200000016)). ADMM efficiently handles both the primal and dual aspects of the optimization problem, ensuring convergence to a solution that satisfies the original problem's constraints. The method synthesizes concepts from dual decomposition and augmented Lagrangian methods: dual decomposition divides the original problem into two or more subproblems, while the augmented Lagrangian methods use an augmented Lagrangian function to effectively handle constraints within the optimization process. This combination makes ADMM particularly powerful for solving large-scale and constrained optimization problems.

<br>

##### A. Problem Formulation

Typically, ADMM solves a class of optimization problems of the form

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x},\boldsymbol{z}}\,& f(\boldsymbol{x})+g(\boldsymbol{z}) \\ \text{s.t.}\,& \boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}=\boldsymbol{c} \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{n}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}\in\mathbb{R}^{m}"/> are the optimization variables. The objective function has two convex functions <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f(\boldsymbol{x})"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;g(\boldsymbol{z})"/>. In the constraint, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{p\times n}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{B}\in\mathbb{R}^{p\times m}"/> are matrices, while <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{c}\in\mathbb{R}^{p}"/> is a vector.

<br>

##### B. Augmented Lagrangian Method

Augmented Lagrangian methods are usually used to solve constrained optimization problems, in which the algorithms replace a constrained optimization problem by a series of unconstrained problems and add penalty terms to the objective function. For the aforementioned constrained problem, the augmented Lagrangian method has the following unconstrained objective function:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})= f(\boldsymbol{x})+g(\boldsymbol{z})+\frac{\lambda}{2}\|\boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}-\boldsymbol{c}\|_2^2+\langle\boldsymbol{w},\boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}-\boldsymbol{c}\rangle"/></p>

where the dual variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}\in\mathbb{R}^{p}"/> is an estimate of the Lagrange multiplier, and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\lambda"/> is a penalty parameter that controls the convergence rate. Notably, it is not necessary to take <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\lambda"/> as great as possible in order to solve the original constrained problem. The presence of dual variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> allows <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\lambda"/> to stay much smaller.

Thus, ADMM solves the original constrained problem by iteratively updating the variables <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>, and the dual variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{cases} \displaystyle\boldsymbol{x}:=\arg\min_{\boldsymbol{x}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ \displaystyle\boldsymbol{z}:=\arg\min_{\boldsymbol{z}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ \boldsymbol{w}:=\boldsymbol{w}+\lambda(\boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}-\boldsymbol{c}) \end{cases}"/></p>

In this case, ADMM performs between these updates util convergence. In terms of the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>, it takes the following subproblem:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \boldsymbol{x}:=&\arg\min_{\boldsymbol{x}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ =&\arg\min_{\boldsymbol{x}}\,f(\boldsymbol{x})+\frac{\lambda}{2}\|\boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}-\boldsymbol{c}\|_2^2+\langle\boldsymbol{w},\boldsymbol{A}\boldsymbol{x}\rangle \\ =&\arg\min_{\boldsymbol{x}}\,f(\boldsymbol{x})+\frac{\lambda}{2}\langle\boldsymbol{A}\boldsymbol{x},\boldsymbol{A}\boldsymbol{x}\rangle+\lambda\langle\boldsymbol{A}\boldsymbol{x},\boldsymbol{B}\boldsymbol{z}-\boldsymbol{c}\rangle+\lambda\langle\boldsymbol{A}\boldsymbol{x},\boldsymbol{w}/\lambda\rangle \\ =&\arg\min_{\boldsymbol{x}}\,f(\boldsymbol{x})+\frac{\lambda}{2}\|\boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}-\boldsymbol{c}+\boldsymbol{w}/\lambda\|_2^2 \end{aligned}"/></p>

In terms of the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>, we can write the subproblem as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \boldsymbol{z}:=&\arg\min_{\boldsymbol{z}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ =&\arg\min_{\boldsymbol{z}}\,g(\boldsymbol{z})+\frac{\lambda}{2}\|\boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}-\boldsymbol{c}\|_2^2+\langle\boldsymbol{w},\boldsymbol{B}\boldsymbol{z}\rangle \\ =&\arg\min_{\boldsymbol{z}}\,g(\boldsymbol{z})+\frac{\lambda}{2}\langle\boldsymbol{B}\boldsymbol{z},\boldsymbol{B}\boldsymbol{z}\rangle+\lambda\langle\boldsymbol{B}\boldsymbol{z},\boldsymbol{A}\boldsymbol{x}-\boldsymbol{c}\rangle+\lambda\langle\boldsymbol{B}\boldsymbol{z},\boldsymbol{w}/\lambda\rangle \\ =&\arg\min_{\boldsymbol{z}}\,g(\boldsymbol{z})+\frac{\lambda}{2}\|\boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}-\boldsymbol{c}+\boldsymbol{w}/\lambda\|_2^2 \end{aligned}"/></p>


<br>

##### C. LASSO

Least Absolute Shrinkage and Selection Operator (LASSO) has wide applications to machine learning. The optimization problem of LASSO is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \min_{\boldsymbol{x}}\, \frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2+\gamma\|\boldsymbol{x}\|_1 "/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\gamma"/> is the regularization parameter that controls the sparsity of variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>. Using the variable splitting, this original problem can be rewritten to an optimization with a constraint:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space; \begin{aligned} \min_{\boldsymbol{x},\boldsymbol{z}}\,&\frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2+\gamma\|\boldsymbol{z}\|_1 \\ \text{s.t.}\,&\boldsymbol{x}=\boldsymbol{z} \end{aligned}"/></p>

First of all, the augmented Lagrangian function can be written as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})= \frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2+\gamma\|\boldsymbol{z}\|_1+\frac{\lambda}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_2^2+\langle\boldsymbol{w},\boldsymbol{x}-\boldsymbol{z}\rangle"/></p>

Thus, the variables <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>, and the dual variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{w}"/> can be updated iteratively as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{cases} \displaystyle\boldsymbol{x}:=\arg\min_{\boldsymbol{x}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ \displaystyle\boldsymbol{z}:=\arg\min_{\boldsymbol{z}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ \boldsymbol{w}:=\boldsymbol{w}+\lambda(\boldsymbol{x}-\boldsymbol{z}) \end{cases}"/></p>

In terms of the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/>, the subproblem has a least squares solution:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \boldsymbol{x}:=&\arg\min_{\boldsymbol{x}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ =&\arg\min_{\boldsymbol{x}}\,\frac{1}{2}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2+\frac{\lambda}{2}\|\boldsymbol{x}-\boldsymbol{z}\|_2^2+\langle\boldsymbol{w},\boldsymbol{x}\rangle \\ =&(\boldsymbol{A}^\top\boldsymbol{A}+\lambda\boldsymbol{I})^{-1}(\boldsymbol{A}^\top\boldsymbol{b}+\lambda\boldsymbol{z}-\boldsymbol{w}) \end{aligned}"/></p>

where the partial derivative of the augmented Lagrangian function with respect to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> is given by

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \frac{\partial\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})}{\partial\boldsymbol{x}}=&\boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b})+\lambda(\boldsymbol{x}-\boldsymbol{z})+\boldsymbol{w} \\ =&(\boldsymbol{A}^\top\boldsymbol{A}+\lambda\boldsymbol{I})\boldsymbol{x}-\boldsymbol{A}^\top\boldsymbol{b}-\lambda\boldsymbol{z}+\boldsymbol{w} \end{aligned}"/></p>

Let this partial derivative be a zero vector, then the closed-form solution is the least squares.

In terms of the variable <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}"/>, the subproblem is indeed an <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm minimization:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \boldsymbol{z}:=&\arg\min_{\boldsymbol{z}}\,\mathcal{L}_{\lambda}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w}) \\ =&\arg\min_{\boldsymbol{z}}\,\gamma\|\boldsymbol{z}\|_1+\frac{\lambda}{2}\|\boldsymbol{z}-\boldsymbol{x}\|_2^2-\langle\boldsymbol{w},\boldsymbol{z}\rangle \\ =&\arg\min_{\boldsymbol{z}}\,\gamma\|\boldsymbol{z}\|_1+\frac{\lambda}{2}\|\boldsymbol{z}-\boldsymbol{x}-\boldsymbol{w}/\lambda\|_2^2 \end{aligned}"/></p>

<br>

<span style="color:gray">
**Example 1.**
</span>


<br>


#### IV. Greedy Methods for <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-Norm Optimization

##### A. Orthogonal Matching Pursuit


##### B. Subspace Pursuit


<br>

#### V. Bayesian Optimization

##### A. Conjugate Priors

##### B. Bayesian Inference

##### C. Bayesian Linear Regression

<br>

#### VI. Power Iteration

##### A. Eigenvalue Decomposition

##### B. Randomized Singular Value Decomposition

<br>

