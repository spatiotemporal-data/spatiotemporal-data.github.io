---
layout: default
---

# Time-Varying Autoregression

#### Discovering Dynamic Patterns from Spatiotemporal Data with Time-Varying Low-Rank Autoregression

<br>

Dynamic mechanisms that drive nonlinear systems are universally complex. Straightforwardly, one can investigate the behavior of a system by discovering the interpretable dynamic patterns from real-world data. In practice, when we take observations from a real-world complex system, spatiotemporal data are one of the most widely encountered from relating to space and time and showing the characteristics of time series. Without loss of generality, leveraging time series models not only allows one to analyze spatiotemporal data but also makes it possible to discover inherent spatial and temporal patterns from the data over space and time.

The scientific question in our study (see [Chen et al., 2023](https://doi.org/10.1109/TKDE.2023.3294440)) is how to discover interpretable dynamic patterns from spatiotemporal data. We utilize the vector autoregression as a basic tool to explore the spatiotemporal data in real-world applications.

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

Typically, time-varying vector autoregression takes a sequence of vector autoregressive processes at different times, and it is capable of handling the time-varying system behaviors. For any observed spatiotemporal data in the form of multivariate time series, i.e., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{s}_1,\boldsymbol{s}_2,\ldots,\boldsymbol{s}_{T}\in\mathbb{R}^{N}"/>, our model considers a time-varying vector autoregression as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\min_{\{\color{red}\boldsymbol{A}_{t}\}}~\frac{1}{2}\left\|\boldsymbol{y}_{t}-{\color{red}\boldsymbol{A}_{t}}\boldsymbol{z}_{t}\right\|_2^2"/></p>

with the data pair:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\begin{aligned} \boldsymbol{y}_{t}&\triangleq \boldsymbol{s}_{t}\in\mathbb{R}^{N} \\ \boldsymbol{z}_{t}&\triangleq\begin{bmatrix} \boldsymbol{s}_{t-1} \\ \vdots \\ \boldsymbol{s}_{t-d} \\ \end{bmatrix} \end{aligned}\in\mathbb{R}^{dN}"/></p>

As the data pair <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\{\boldsymbol{y}_t,\boldsymbol{z}_t\}"/> is readily available, one can learn the coefficient matrices <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{A}_t\in\mathbb{R}^{N\times (dN)}"/>.


<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on December 13, 2023.)</p>