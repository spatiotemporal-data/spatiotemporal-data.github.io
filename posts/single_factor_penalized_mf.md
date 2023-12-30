---
layout: default
---

# Single-Factor Penalized Matrix Decomposition

For any positive semidefinite matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{Y}\in\mathbb{R}^{n\times n}"/>, the following optimization problem:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\large&space;\min_{\boldsymbol{x}}~\frac{1}{2}\|\boldsymbol{Y}-\boldsymbol{x}\boldsymbol{x}^{\top}\|_F^2+\frac{\lambda}{2}\|\boldsymbol{x}\|_1"/></p>

can be solved by the following algorithm:

- Initialize <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{x}"/> as the unit vector with equal entries
- Repeat
  - Compute

  <p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{x}:=\mathcal{S}_{\lambda}(\boldsymbol{Y}\boldsymbol{x})/\|\mathcal{S}_{\lambda}(\boldsymbol{Y}\boldsymbol{x})\|_2"/></p>

- Until convergence
- Compute <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;d=\boldsymbol{x}^{\top}\boldsymbol{Y}\boldsymbol{x}"/>
- Compute <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{x}:=\sqrt{d}\boldsymbol{x}"/>

In the algorithm, the soft-thresholding operator is defined as

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\large&space;[\mathcal{S}_{\lambda}(\boldsymbol{x})]_{i}=\begin{cases} x_{i}-\lambda, & \text{if}~~x_{i}>t \\ x_{i}+\lambda, & \text{if}~~x_{i}<-t \\ 0, & \text{otherwise} \end{cases}"/></p>

for <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;i\in\{1,2,\ldots,n\}"/>.

<br>

**References**

- (**Deng et al., 2021**) Correlation tensor decomposition
and its application in spatial imaging data. Journal of the American Statistical Association. [[**DOI**](https://doi.org/10.1080/01621459.2021.1938083)]

<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on December 30, 2023.)</p>