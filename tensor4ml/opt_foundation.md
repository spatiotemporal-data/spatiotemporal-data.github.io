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

Discuss LASSO regression problems.

<br>

#### II. Power Iteration

##### A. Eigenvalue Decomposition

##### B. Randomized Singular Value Decomposition

<br>

#### III. Alternating Minimization

##### A. Alternating Least Squares

Example. Matrix factorization with Panda

##### B. Alternating Minimization with Conjugate Gradient Descent

Smoothing matrix factorization on Panda

<br>

#### IV. Alternating Direction Method of Multipliers

The Alternating Direction Method of Multipliers (ADMM) is an optimization algorithm designed to solve complex problems by decomposing them into a sequence of easy-to-solve subproblems. ADMM efficiently handles both the primal and dual aspects of the optimization problem, ensuring convergence to a solution that satisfies the original problem's constraints. The method synthesizes concepts from dual decomposition and augmented Lagrangian methods: dual decomposition divides the original problem into two or more subproblems, while the augmented Lagrangian methods use an augmented Lagrangian function to effectively handle constraints within the optimization process. This combination makes ADMM particularly powerful for solving large-scale and constrained optimization problems.

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

The optimization problem of LASSO is

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

#### V. Greedy Method for <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-Norm Optimization

##### A. Orthogonal Matching Pursuit


##### B. Subspace Pursuit


<br>
