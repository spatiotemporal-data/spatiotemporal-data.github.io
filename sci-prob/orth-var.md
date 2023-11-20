---
layout: default
---

# Optimizing Interpretable Time-Varying Autoregression with Orthogonal Constraints

<br>

Generally speaking, any spatiotemporal data in the form of a matrix can be written as <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{Y}\in\mathbb{R}^{N\times T}"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;N"/> spatial areas/locations and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;T"/> time steps. 

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\begin{aligned} \min_{\boldsymbol{W},\boldsymbol{G},\boldsymbol{V},\boldsymbol{X}}~&\frac{1}{2}\sum_{t=2}^{T}\|\boldsymbol{y}_t-\boldsymbol{W}\boldsymbol{G}(\boldsymbol{x}_t^\top\otimes\boldsymbol{V})^\top\boldsymbol{y}_{t-1}\|_2^2 \\ \text{s.t.}~~&\begin{cases} \boldsymbol{W}^\top\boldsymbol{W}=\boldsymbol{I}_R \\ \boldsymbol{V}^\top\boldsymbol{V}=\boldsymbol{I}_R \\ \boldsymbol{X}^\top\boldsymbol{X}=\boldsymbol{I}_R \\ \end{cases} \end{aligned}"/></p>
