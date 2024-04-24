---
layout: default
---

# Commentator

### April 23, 2024 (Dingyi Zhuang)

In the past decades, machine learning has been an important data-driven computing paradigm for learning from data. However, real-world data usually demonstrate complicated data behaviors (e.g., uncertainty) or underlying patterns. In this study, we hope to connect machine learning with real-world data problems in transport demand modeling and in the meanwhile highlight the importance of uncertainty quantification in the prediction task. There are three critical challenges as proposed:

- Robustness: Solving the unexpected data variation that produced by special events or extreme cases.
- Reliability: Handling the the model's generalization ability to different conditions.
- Fairness: Reducing data/model biases over different groups of the collected dataset.

Due to the randomness of transport demand data, uncertainty quantification is of great significance in the deep learning based prediction methods. Introducing probabilistic assumptions when modeling the transport demand allows one to produce both point estimates and interval estimates. One recent demand prediction model is the probabilistic spatiotemporal graph neural network (Prob STGNN, see [Zhuang et al.'22](https://dl.acm.org/doi/pdf/10.1145/3534678.3539093)).

In terms of fairness, one meaningful task is how to reduce data and model biases in our machine learning algorithms. As shown in Figure 1,

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/fairness_explained.png" alt="drawing" width="500">
</p>

<p align="center"><b>Figure 1</b>: Illustration of fairness modeling in machine learning with the dataset <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/>. The dataset is splitted into <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\normalsize&space;n"/> subsets.</p>


<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 24, 2024.)</p>
