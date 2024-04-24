---
layout: default
---

# Commentator

### April 23, 2024 (Dingyi Zhuang)

In the past decades, machine learning acted as an important data-driven computing paradigm for learning from data. However, real-world data usually demonstrate complicated data behaviors (e.g., uncertainty) and underlying patterns (e.g., biases in the data collection process). In this study, we hope to connect machine learning with real-world data problems in transport demand modeling and in the meanwhile highlight the importance of uncertainty quantification in the prediction task. There are some critical challenges that stem from the following perspectives:

- **Robustness**: Solving the unexpected data variations that produced by special events or extreme cases.
- **Reliability**: Handling the the model's generalization ability to different conditions.
- **Fairness**: Reducing data and model biases over different groups of the collected dataset.

Due to the randomness of transport demand data, uncertainty quantification is of great significance in the deep learning based prediction methods, even not mentioning the possible improvement of prediction accuracy. Introducing probabilistic assumptions when modeling the transport demand allows one to produce both point estimates and interval estimates. One recent demand prediction model is the probabilistic spatiotemporal graph neural network (Prob STGNN, see [Zhuang et al.'22](https://dl.acm.org/doi/pdf/10.1145/3534678.3539093)).

In terms of fairness, one meaningful task is how to reduce data and model biases in the machine learning algorithms. As shown in Figure 1, characterizing the fairness on the data is simply implemented by minimizing the differences of model variables on the splitted subsets. The shift of learning mechanisms in such case could benefit a lot of transport modeling applications.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/fairness_explained.png" alt="drawing" width="600">
</p>

<p align="center"><b>Figure 1</b>: Illustration of fairness modeling in machine learning with the dataset <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/>. The dataset is splitted into <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> subsets.</p>


<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 24, 2024.)</p>