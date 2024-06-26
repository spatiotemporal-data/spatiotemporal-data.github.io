---
layout: default
---

# Optimizing Interpretable Time-Varying Autoregression with Orthogonal Constraints

<br>

Generally speaking, any spatiotemporal data in the form of a matrix can be written as <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{Y}\in\mathbb{R}^{N\times T}"/> with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;N"/> spatial areas/locations and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;T"/> time steps. To discover interpretable spatial/temporal patterns, one can build a time-varying autoregression on the time snapshots <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{y}_1,\boldsymbol{y}_2,\ldots,\boldsymbol{y}_{T}\in\mathbb{R}^{N}"/> ([Chen et al., 2023](https://doi.org/10.1109/TKDE.2023.3294440)). The time-varying coefficients in the autoregression allow one to characterize the time-varying system behavior, but the challenges still remain.

To capture interpretable modes/patterns, one can use tensor factorization formulas to parameterize the coefficients and the optimization problem can be easily built. However, a great challenge would be how to make the modes "more interpretable", specifically, e.g., how to learn orthogonal modes in the modeling process. In this post, we present an optimization problem of the time-varying autoregression with orthogonal constraints as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?\large&space;\begin{aligned} \min_{\boldsymbol{W},\boldsymbol{G},\boldsymbol{V},\boldsymbol{X}}~&\frac{1}{2}\sum_{t=1}^{T-1}\left\|\boldsymbol{y}_{t+1}-\boldsymbol{W}\boldsymbol{G}(\boldsymbol{x}_t\otimes\boldsymbol{V}^\top)\boldsymbol{y}_{t}\right\|_2^2 \\ \text{s.t.}~~&\begin{cases} \boldsymbol{W}^\top\boldsymbol{W}=\boldsymbol{I}_R \\ \boldsymbol{V}^\top\boldsymbol{V}=\boldsymbol{I}_R \\ \boldsymbol{X}^\top\boldsymbol{X}=\boldsymbol{I}_R \\ \end{cases} \end{aligned}"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{W}\in\mathbb{R}^{N\times R}"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{X}\in\mathbb{R}^{(T-1)\times R}"/> refer to as the spatial modes and the temporal modes, respectively. Note that the temporal factors are written in the form of vectors, i.e., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?\large&space;\boldsymbol{x}_1,\boldsymbol{x}_2,\ldots,\boldsymbol{x}_{T-1}\in\mathbb{R}^{R}"/>. This model can discover urban mobility transition patterns.

<br>

**References**

- (**Chen et al., 2024**) Discovering dynamic patterns from spatiotemporal data with time-varying low-rank autoregression. IEEE Transactions on Knowledge and Data Engineering. 2024, 36 (2): 504-517. [[DOI](https://doi.org/10.1109/TKDE.2023.3294440)] [[PDF](https://xinychen.github.io/papers/time_varying_model.pdf)] [[Blog post](https://spatiotemporal-data.github.io/posts/time_varying_model/)]

<br>

**Material:**
- [What is the difference between projected gradient descent and ordinary gradient descent?](https://math.stackexchange.com/q/571068)

<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on November 20, 2023.)</p>
