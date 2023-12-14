---
layout: default
---

# Time-Varying Autoregression

#### Discovering Dynamic Patterns from Spatiotemporal Data with Time-Varying Low-Rank Autoregression

<br>

Dynamic mechanisms that drive nonlinear systems are universally complex. Straightforwardly, one can investigate the behavior of a system by discovering the interpretable dynamic patterns from real-world data. In practice, when we take observations from a real-world complex system, spatiotemporal data are one of the most widely encountered from relating to space and time and showing the characteristics of time series. Without loss of generality, leveraging time series models not only allows one to analyze spatiotemporal data but also makes it possible to discover inherent spatial and temporal patterns from the data over space and time.

The scientific question in our study is how to discover interpretable dynamic patterns from spatiotemporal data. We utilize the vector autoregression as a basic tool to explore the spatiotemporal data in real-world applications.


- Chen, X., Zhang, C., Chen, X., Saunier, N., & Sun, L. (2023). Discovering Dynamic Patterns from Spatiotemporal Data with Time-Varying Low-Rank Autoregression. IEEE Transactions on Knowledge and Data Engineering. Early access.

or

```bibtex
@article{chen2023discovering,
  title={Discovering dynamic patterns from spatiotemporal data with time-varying low-rank autoregression},
  author={Chen, Xinyu and Zhang, Chengyuan and Chen, Xiaoxu and Saunier, Nicolas and Sun, Lijun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023}
}
```

## Revisit Vector Autoregression

As a simple yet efficient and classical method for time series modeling, vector autoregression allows one to explicitly find the linear relationship among a sequence of time series (i.e., multivariate time series) changing over time, which can also successfully describe the dynamic behavior of time series.

Given any stationary time series <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;\boldsymbol{s}_1,\boldsymbol{s}_2,\ldots,\boldsymbol{s}_{T}\in\mathbb{R}^{N}"/>, the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;d"/>th-order vector autoregression takes a linear formula as



<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on December 13, 2023.)</p>
