---
layout: default
---

# Single-Factor Penalized Matrix Decomposition

For any positive semidefinite matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{Y}\in\mathbb{R}^{n\times n}"/>, the following optimization problem:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\large&space;\min_{\boldsymbol{x}}~\frac{1}{2}\|\boldsymbol{Y}-\boldsymbol{x}\boldsymbol{x}^{\top}\|_F^2+\frac{\lambda}{2}\|\boldsymbol{x}\|_1"/></p>

can be solved by the following algorithm:

- Initialize <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{x}"/> as the unit vector with equal entries
- Repeat
  - Compute <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{x}:=\mathcal{S}_{\lambda}(\boldsymbol{Y}\boldsymbol{x})/\|\mathcal{S}_{\lambda}(\boldsymbol{Y}\boldsymbol{x})\|_2"/> (Denote <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\mathcal{S}_{\lambda}(\cdot)"/> as the soft-thresholding operator)
- Until convergence
- Compute <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;d=\boldsymbol{x}^{\top}\boldsymbol{Y}\boldsymbol{x}"/>
- Compute <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{x}:=\sqrt{d}\boldsymbol{x}"/>


<br>

**References**

-

Given any stationary time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{s}_1,\boldsymbol{s}_2,\ldots,\boldsymbol{s}_{T}\in\mathbb{R}^{N}"/>, the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;d"/>th-order vector autoregression takes a linear formula as

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{s}_{t}=\sum_{k=1}^{d}{\color{red}\boldsymbol{A}_{k}}\boldsymbol{s}_{t-k}+\underbrace{\boldsymbol{\epsilon}_{t}}_{\text{error}},\,\forall t"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{A}_1,\boldsymbol{A}_2,\ldots,\boldsymbol{A}_{d}\in\mathbb{R}^{N\times N}"/> are the coefficient matrices, which can capture the temporal correlations of the multivariate time series.


<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on December 30, 2023.)</p>
