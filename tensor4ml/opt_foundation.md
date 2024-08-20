---
layout: default
---

# Foundation of Optimization

## I. Alternating Direction Method of Multipliers

The Alternating Direction Method of Multipliers (ADMM) is an optimization algorithm designed to solve complex problems by decomposing them into a sequence of easy-to-solve subproblems. ADMM efficiently handles both the primal and dual aspects of the optimization problem, ensuring convergence to a solution that satisfies the original problem's constraints. The method synthesizes concepts from dual decomposition and augmented Lagrangian methods: dual decomposition divides the original problem into two or more subproblems, while the augmented Lagrangian methods use an augmented Lagrangian function to effectively handle constraints within the optimization process. This combination makes ADMM particularly powerful for solving large-scale and constrained optimization problems.

### A. Problem Formulation

Typically, ADMM solves a class of optimization problems of the form

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\begin{aligned} \min_{\boldsymbol{x},\boldsymbol{z}}\,& f(\boldsymbol{x})+g(\boldsymbol{z}) \\ \text{s.t.}\,& \boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}\boldsymbol{z}=\boldsymbol{c} \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{n}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{z}\in\mathbb{R}^{m}"/> are the optimization variables. The objective function has two convex functions <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;f(\boldsymbol{x})"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;g(\boldsymbol{z})"/>. In the constraint, <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{A}\in\mathbb{R}^{p\times n}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{B}\in\mathbb{R}^{p\times m}"/> are matrices, while <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{c}\in\mathbb{R}^{p}"/> is a vector.

#### Augmented Lagrangian Function

The augmented Lagrangian function is

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{L}(\boldsymbol{x},\boldsymbol{z},\boldsymbol{w})\triangleq f(\boldsymbol{x})+g(\boldsymbol{z})+"/></p>


<br>
