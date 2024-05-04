---
layout: default
---

## Step Forward on the "Prior Knowledge"

Research Knowledge & Vision & Insight & Style

<br>

### 4th Mile

Graph siginal processing

**References**

- Antonio Ortega, Pascal Frossard, Jelena Kovacevic, Jose M. F. Moura, Pierre Vandergheynst (2017). [Graph Signal Processing: Overview, Challenges and Applications](https://arxiv.org/pdf/1712.00468). arXiv:1712.00468.
- Xiaowen Dong, Dorina Thanou, Laura Toni, Michael Bronstein, and Pascal Frossard (2020). [Graph signal processing for machine learning: A review and new perspectives](https://arxiv.org/pdf/2007.16061). arXiv:2007.16061. [[Slides](https://web.media.mit.edu/~xdong/talk/BDI_GSP.pdf)]
- Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković (2021). [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/pdf/2104.13478). arXiv:2104.13478.
- Wei Hu, Jiahao Pang, Xianming Liu, Dong Tian, Chia-Wen Lin, Anthony Vetro (2022). [Graph Signal Processing for Geometric Data and Beyond: Theory and Applications](https://doi.org/10.1109/TMM.2021.3111440). IEEE Transactions on Multimedia, 24: 3961-3977.
- Geert Leus, Antonio G. Marques, José M. F. Moura, Antonio Ortega, David I Shuman (2023). [Graph Signal Processing: History, Development, Impact, and Outlook](https://arxiv.org/pdf/2303.12211). arXiv:2303.12211.

<br>

### 3rd Mile

The causal effect estimation problem is usually defined as a matrix completion on the partially observed matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}\in\mathbb{R}^{N\times T}"/> in which <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> units and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> periods are involved. The optimization is from the classical matrix factorization in recommender systems (see [Koren et al.'09](https://doi.org/10.1109/MC.2009.263)):

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\min_{\boldsymbol{W},\boldsymbol{X},\boldsymbol{u},\boldsymbol{p}}~\frac{1}{2}\|\mathcal{P}_{\Omega}(\boldsymbol{Y}-\boldsymbol{W}^\top\boldsymbol{X}-\boldsymbol{u}\mathbf{1}_{T}^\top-\mathbf{1}_{N}\boldsymbol{p}^\top)\|_F^2"/></p>

where

<br>

### 2nd Mile

The motivation for discussing the value of standards for health datasets is the risk of algorithmic bias, consequently leading to the possible healthcare inequity. The problem arises from the systemic inequalities in the dataset curation and the unequal opportunities to access the data and research. The aim is to expolore the standards, frameworks, and best practices in health datasets. Some discrete insights throughout the whole paper are summarized as follows,

- AI as a medical device (AIaMD). One concern is the risk of systemic algorithmic bias (well-recognized in the literature) if models are trained on biased training datasets.
- Less accurate performance in certain patient groups when using the biased algorithms.
- Data diversity (Mainly discuss "how to improve"):
  - Challenges: lack of standardization across attribute categories, difficulty in harmonizing several methods of data capture and data-governance restrictions.
  - Inclusiveness is a core tenet of ethical AI in healthcare.
  - Guidance on how to apply the principles in the curation (e.g., developing the data collection strategy), aggregation and use of health data.
- The use of metrics (measuring diversity). How to promote diversity and transparency?
- Future actions: Guidelines for data collection, handling missing data and labeling data.

**References**

- Anmol Arora, Joseph E. Alderman, Joanne Palmer, Shaswath Ganapathi, Elinor Laws, Melissa D. McCradden, Lauren Oakden-Rayner, Stephen R. Pfohl, Marzyeh Ghassemi, Francis McKay, Darren Treanor, Negar Rostamzadeh, Bilal Mateen, Jacqui Gath, Adewole O. Adebajo, Stephanie Kuku, Rubeta Matin, Katherine Heller, Elizabeth Sapey, Neil J. Sebire, Heather Cole-Lewis, Melanie Calvert, Alastair Denniston, Xiaoxuan Liu (2023). [The value of standards for health datasets in artificial intelligence-based applications](https://doi.org/10.1038/s41591-023-02608-w). Nature Medicine, 29: 2929–2938.

<br>

### 1st Mile

As we know, the training data in the large time series model is from different areas, this means that the model training process highly depends on the selected datasets across various areas, so one question is how to reduce the model biases if we consider the forecasting scenario as traffic flow or human mobility? Because I guess time series data in different areas should demonstrate different data behaviors. Hopefully, it is interesting to develop domain-specific time series datasets (e.g., [Largest multi-city traffic dataset](https://utd19.ethz.ch/)) and large models.

**References**

- Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, Doyen Sahoo (2024). [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/pdf/2402.02592). arXiv:2402.02592.

<br>



**AI for Science**

- (**Liu et al., 2023**) Topological structure of complex predictions. Nature Machine Intelligence. 2023, 5: 1382-1389. [[DOI](https://doi.org/10.1038/s42256-023-00749-8)]

- (**Savcisens et al., 2024**) Using sequences of life-events to predict human lives. Nature Computational Science. 2024, 4: 43-56. [[DOI](https://doi.org/10.1038/s43588-023-00573-5)]

- (**Chen et al., 2024**) Constructing custom thermodynamics using deep learning. Nature Computational Science. 2024, 4, 66-85. [[DOI](https://doi.org/10.1038/s43588-023-00581-5)]

- (**Liu et al., 2024**) Quantifying spatial under-reporting disparities in resident crowdsourcing. Nature Computational Science. 2024, 4: 57-65. [[DOI](https://doi.org/10.1038/s43588-023-00572-6)]

- (**Greenhill et al., 2024**) Machine learning predicts which rivers, streams, and wetlands the Clean Water Act regulates. Science. 2024, 383: 406-412. [[DOI](https://doi.org/10.1126/science.adi3794)]

- (**Ravuri et al., 2021**) Skilful precipitation nowcasting using deep generative models of radar. Nature. 2021, 597: 672–677. [[DOI](https://doi.org/10.1038/s41586-021-03854-z)]

- (**Bi et al., 2023**) Accurate medium-range global weather forecasting with 3D neural networks. Nature. 2023, 619: 533–538. [[DOI](https://doi.org/10.1038/s41586-023-06185-3)]

- (**Zhang et al., 2023**) Skilful nowcasting of extreme precipitation with NowcastNet. Nature. 2023, 619: 526–532. [[DOI](https://doi.org/10.1038/s41586-023-06184-4)]

- (**Andersson et al., 2021**) Seasonal Arctic sea ice forecasting with probabilistic deep learning. Nature Communications. 2021, 12: 5124. [[DOI](https://doi.org/10.1038/s41467-021-25257-4)]

- (**Mondini et al., 2023**) Deep learning forecast of rainfall-induced shallow landslides. Nature Communications. 2023, 14: 2466. [[DOI](https://doi.org/10.1038/s41467-023-38135-y)]


<br>

**Time Series**

- (**Chen et al., 2023**) ContiFormer: Continuous-time transformer for irregular time series modeling. NeurIPS 2023. [[PDF](https://openreview.net/pdf?id=YJDz4F2AZu)]

- (**Zeng et al., 2023**) Are Transformers Effective for Time Series Forecasting? AAAI 2023. [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/26317)]

- Miles Stoudenmire. Tensor networks for machine learning and applications. IPAM. [[YouTube](https://www.youtube.com/watch?v=q8UTwdjS95k)]

- (**Zhang et al., 2019**) Greedy orthogonal pivoting algorithm for non-negative matrix factorization. ICML 2019. [[PDF](https://proceedings.mlr.press/v97/zhang19r/zhang19r.pdf)]

- (**Mirzal, 2014**) A convergent algorithm for orthogonal nonnegative matrix factorization. Journal of Computational and Applied Mathematics. 2014, 260: 149-166. [[DOI](https://doi.org/10.1016/j.cam.2013.09.022)]


**Reinforcement Learning**

- [Maximum Entropy Inverse Reinforcement Learning](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf). AAAI 2008.

- [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248). arXiv: 1710.11248.

**Machine Learning Material**

- [Understanding Deep Learning (book)](http://udlbook.com/?trk=public_post-text) [[Linkedin](https://www.linkedin.com/posts/simon-prince-615bb9165_phew-finally-finished-all-68-python-notebook-activity-7130190400266330112-5n_X?utm_source=share&utm_medium=member_ios)]

- [Blog post (Gregory Gundersen)](https://gregorygundersen.com/blog/)

<br>

Motivation & Principle: 不积硅步，无以至千里。
