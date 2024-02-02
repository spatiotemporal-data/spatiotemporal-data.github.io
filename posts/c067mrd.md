---
layout: default
---

# Structured State Space Models

<br>

- (**Stock & Watson**) Dynamic factor models, factor-augmented vector autoregressions, and structural vector autoregressions in macroeconomics. 2016.

<br>

One question confused me a lot: What is the difference among state space model (SSM), dynamic factor model (DFM), and temporal matrix factorization (TMF).

- SSM & DFM:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\large&space;\begin{cases} \boldsymbol{y}_{t}=\boldsymbol{W}\boldsymbol{x}_{t}+\boldsymbol{\eta}_{t}\quad\text{(Observation equation)} \\ \boldsymbol{x}_{t+1}=\boldsymbol{A}\boldsymbol{x}_{t}+\boldsymbol{e}_{t}\quad\text{(State transition equation)} \end{cases}"/></p>

- TMF:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\large&space;\begin{cases} \mathcal{P}_{\Omega_t}(\boldsymbol{y}_{t})=\mathcal{P}_{\Omega_t}(\boldsymbol{W}\boldsymbol{x}_{t}}+\boldsymbol{\eta}_{t}\quad\text{(Matrix factorization)} \\ \boldsymbol{x}_{t+1}=\boldsymbol{A}\boldsymbol{x}_{t}+\boldsymbol{e}_{t}\quad\text{(Latent vector autoregression)} \end{cases}"/></p>


<br>

- (**Gu & Dao 2023**) Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752. [[PDF](https://arxiv.org/pdf/2312.00752.pdf)] [[The Annotated S4](https://srush.github.io/annotated-s4/)]

- (**Smith et al., 2022**) Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933. (ICLR'23) [[PDF](https://arxiv.org/pdf/2208.04933.pdf)]

- (**Smith et al., 2023**) Convolutional state space models for long-range spatiotemporal modeling. arXiv preprint arXiv:2310.19694. [[PDF](https://arxiv.org/pdf/2310.19694.pdf)]

<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on January 5, 2024.)</p>
