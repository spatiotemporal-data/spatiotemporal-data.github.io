---
layout: default
---

# Awesome LaTeX Drawing

<p align="center"><span style="color:gray"> Drawing and generating academic graphics in LaTeX for visualization purposes in research. LaTeX always demonstrates several merits such as mathematical accuracy and code-driven design for reproducibility.</span></p>

<p align="center"><span style="color:gray">(Updated on April 20, 2025)</span></p>

<br>

[awesome-latex-drawing](https://github.com/xinychen/awesome-latex-drawing) is a GitHub repository we created in 2019 for visualization purposes, which has gained 1,600 stars until now. We have maintained it with regular updates since its inception. Initially, our goal was to share the LaTeX code used to generate academic graphics in our publications. Personally, the two most notable advantages of LaTeX visualization that we highly value are **mathematical accuracy** and **code-driven design for reproducibility**.

<br>

## LaTeX Visualization Examples

<br>


<b>Example 1 (Normalizing a two-dimensional vector).</b> For any vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}=(x_1,x_2)^\top\in\mathbb{R}^{2}"/>, one can normalize it with <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_2"/>-norm (denoted by <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\cdot\|_2"/>). Formally, the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_2"/>-norm of the vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> is defined as <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{x}\|_2=\sqrt{x_1^2+x_2^2}"/>, which is the square root of the sum of squared entries. Thus, we have the normalized vector as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\frac{\boldsymbol{x}}{\|\boldsymbol{x}\|_2}=\Bigl(\frac{x_1}{\sqrt{x_1^2+x_2^2}},\frac{x_2}{\sqrt{x_1^2+x_2^2}}\Bigr)^\top\in\mathbb{R}^{2}"/></p>

<br>

---

<span style="color:gray">
Figure 1 visualizes a certain vector and the corresponding normalized vector in a two-dimensional Cartesian coordinate system.
</span>

<br>

<div style="text-align: center;">
  <img style="display: inline-block; margin: 0 10px;" src="https://spatiotemporal-data.github.io/images/unit_vector_black.png" width="280" />
  <img style="display: inline-block; margin: 0 20px;" src="https://spatiotemporal-data.github.io/images/unit_vector_white.png" width="280" />
</div>

<p style="font-size: 14px; color: gray" align = "center">
<b>Figure 1.</b> Illustration of normalized vectors in a certain direction, ensuring <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{x}\|_2=1"/>. (<b>Left panel</b>) Dark theme, see <a href="https://github.com/xinychen/awesome-latex-drawing/blob/master/norms/unit_vector_black.tex">LaTeX code</a> on GitHub. (<b>Right panel</b>) White theme, see <a href="https://github.com/xinychen/awesome-latex-drawing/blob/master/norms/unit_vector_white.tex">LaTeX code</a> on GitHub.
</p>

<br>

---

<br>

<b>Example 2 (Manhattan distance).</b> Manhattan distance (also known as taxicab distance) is a method for measuring the distance between two points in a grid-like space, similar to how a taxi would navigate through city streets. It is calculated by summing the absolute differences of the coordinates between the two points.

For any vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}\in\mathbb{R}^{n}"/>, the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm of vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{x}"/> is defined as <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{x}\|_1=\sum_{i=1}^{n}|x_i|"/>, which is the sum of the absolute values of entries.

<br>

---

<span style="color:gray">
Figure 2 visualizes distance between the the point <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;A"/> to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;B"/>. This is equivalent to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_1"/>-norm of vector such that
</span>

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\|\boldsymbol{x}\|_1=x_1+x_2+\cdots+x_7=\sum_{i=1}^{7}x_i"/></p>


<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/manhattan_distance.png" width="380" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>Figure 2.</b> Illustration of Manhattan distance from the point <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;A"/> to <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;B"/> with the distance vector <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;(x_1,x_2,\cdots,x_7)^\top"/>. Please see <a href="https://github.com/xinychen/awesome-latex-drawing/blob/master/norms/manhattan_distance.tex">LaTeX code</a> on GitHub.
</p>

<br>

---

<br>

<b>Example 3 (Sine and cosine functions).</b> Figure 3 illustrates four periodic waveforms consisting of sine and cosine functions with vertical offsets:

- Two sine functions <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;y=\sin(2\pi t)"/> shifted upward by 0.75 and 0.25 units (cyan and blue curves, respectively).
- Two sine functions <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;y=\cos(2\pi t)"/> shifted downward by 0.25 and 0.75 units (red and orange curves, respectively).

All functions share the same angular frequency <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;2\pi"/>, resulting in identical periods of oscillation. The vertical offsets demonstrate how constant terms affect the baseline of trigonometric waveforms while preserving their amplitude and phase relationships. The horizontal axis <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> ranges from 0 to 2, capturing two full cycles of each function.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/sin_cos_functions.png" width="520" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>Figure 3.</b> Illustration of four sequences consisting of sine and cosine functions.
</p>

<br>

---

<br>

<b>Example 4 (Two-step rolling time series prediction).</b> 

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/forecasting.png" width="400" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>Figure 4.</b> Illustration of rolling time series prediction.
</p>

<br>

---





<br>

**Relevant Publications**

<ul style="padding-left: 20px;">
  <li style="font-size: 14px; color: gray; margin-bottom: 15px;">
    Xinyu Chen, HanQin Cai, Fuqiang Liu, Jinhua Zhao (2025). Correlating time series with interpretable convolutional kernels. <em>IEEE Transactions on Knowledge and Data Engineering</em>. [<a href="https://doi.org/10.1109/TKDE.2025.3550877">DOI</a>]
  </li>
  
  <li style="font-size: 14px; color: gray; margin-bottom: 15px;">
    Xinyu Chen, Chengyuan Zhang, Xi-Le Zhao, Nicolas Saunier, Lijun Sun (2025). Forecasting sparse movement speed of urban road networks with nonstationary temporal matrix factorization. <em>Transportation Science</em>. [<a href="https://pubsonline.informs.org/doi/abs/10.1287/trsc.2024.0629">DOI</a>]
  </li>

  <li style="font-size: 14px; color: gray; margin-bottom: 15px;">
    Xinyu Chen, Xi-Le Zhao, Chun Cheng (2024). Forecasting urban traffic states with sparse data using Hankel temporal matrix factorization. <em>INFORMS Journal on Computing</em>. [<a href="https://doi.org/10.1287/ijoc.2022.0197">DOI</a>]
  </li>

  <li style="font-size: 14px; color: gray; margin-bottom: 15px;">
    Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). Laplacian convolutional representation for traffic time series imputation. <em>IEEE Transactions on Knowledge and Data Engineering</em>. 36 (11): 6490-6502. [<a href="https://doi.org/10.1109/TKDE.2024.3419698">DOI</a>]
  </li>

  <li style="font-size: 14px; color: gray; margin-bottom: 15px;">
    Xinyu Chen, Chengyuan Zhang, Xiaoxu Chen, Nicolas Saunier, Lijun Sun (2024). Discovering dynamic patterns from spatiotemporal data with time-varying low-rank autoregression. <em>IEEE Transactions on Knowledge and Data Engineering</em>. 36 (2): 504-517. [<a href="https://doi.org/10.1109/TKDE.2023.3294440">DOI</a>]
  </li>

  <li style="font-size: 14px; color: gray; margin-bottom: 15px;">
    Xinyu Chen, Lijun Sun (2022). Bayesian temporal factorization for multidimensional time series prediction. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence</em>. 44 (9): 4659-4673. [<a href="https://doi.org/10.1109/TPAMI.2021.3066551">DOI</a>]
  </li>
</ul>

<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 17, 2025)</p>
