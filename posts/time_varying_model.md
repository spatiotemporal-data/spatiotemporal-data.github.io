---
layout: default
---

# Time-Varying Autoregression

#### Discovering Dynamic Patterns from Spatiotemporal Data with Time-Varying Low-Rank Autoregression

<br>

Dynamic mechanisms that drive nonlinear systems are universally complex. Straightforwardly, one can investigate the behavior of a system by discovering the interpretable dynamic patterns from real-world data. In practice, when we take observations from a real-world complex system, spatiotemporal data are one of the most widely encountered from relating to space and time and showing the characteristics of time series. Without loss of generality, leveraging time series models not only allows one to analyze spatiotemporal data but also makes it possible to discover inherent spatial and temporal patterns from the data over space and time.

The scientific question in our study (see [our paper](https://doi.org/10.1109/TKDE.2023.3294440)) is how to discover interpretable dynamic patterns from spatiotemporal data. We utilize the vector autoregression as a basic tool to explore the spatiotemporal data in real-world applications.

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

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{s}_{t}=\sum_{k=1}^{d}\boldsymbol{A}_{k}\boldsymbol{s}_{t-k}+\underbrace{\boldsymbol{\epsilon}_{t}}_{\text{error}},\,\forall t"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{A}_1,\boldsymbol{A}_2,\ldots,\boldsymbol{A}_{d}\in\mathbb{R}^{N\times N}"/> are the coefficient matrices, which can capture the temporal correlations of the multivariate time series.


<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on December 13, 2023.)</p>
