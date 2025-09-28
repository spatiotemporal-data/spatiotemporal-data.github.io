---
layout: default
---

<script>
window.MathJax = {
  chtml: {
    scale: 0.95,
    minScale: 0.9
  },
  svg: {
    scale: 0.95,
    minScale: 0.9
  },
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>


This project aims at supporting research for all aspects of spatiotemporal data computing with machine learning and addressing many scientific, mathematical, industrial, and engineering problems in:

- Human mobility modeling
- Intelligent & sustainable urban systems
- Optimization & decision making
- Data standardization & valorization & monetization
- Signal processing
- Dynamical systems & control
- Computational engineering & social science


### Machine Learning Explained Series

In the past decade, the development of artificial intelligence and machine learning has been truly remarkable. There are several easy-to-follow posts that are created to explain the essential ideas of this project and inspire innovations:


<div style="
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  width: 100vw;
  background: #FCEDBB;
  color: #626a73;
  padding: 10px;
">

<div style="
  max-width: 740px; /* Optional: Limits how wide it can get */
  margin: 0 auto; /* Centers the text */
  padding: 1px 10px; /* Small side padding for better readability */
">

<p style="font-size: 17px;"><a href="https://spatiotemporal-data.github.io/posts/ts_conv/"><b>Time Series Convolution</b></a> 🌱🌱 </p>

A convolutional kernel approach for reinforcing the modeling of time series trends and interpreting temporal patterns, allowing one to leverage Fourier transforms and learn sparse representations. The interpretable machine learning models such as sparse regression unlock opportunities to better capture the long-term changes and temporal patterns of real-world time series. The content is mainly from our research work below.

<ul style="padding-left: 20px;">
  <li style="font-size: 14.5px; color: #626a73; margin-bottom: 15px;">
    Xinyu Chen, Zhanhong Cheng, HanQin Cai, Nicolas Saunier, Lijun Sun (2024). <a href="https://doi.org/10.1109/TKDE.2024.3419698"><b>Laplacian convolutional representation for traffic time series imputation</b></a>. <em>IEEE Transactions on Knowledge and Data Engineering</em>. 36 (11): 6490-6502.
  </li>

  <li style="font-size: 14.5px; color: #626a73; margin-bottom: 15px;">
    Xinyu Chen, Xi-Le Zhao, Chun Cheng (2024). <a href="https://doi.org/10.1287/ijoc.2022.0197"><b>Forecasting urban traffic states with sparse data using Hankel temporal matrix factorization</b></a>. <em>INFORMS Journal on Computing</em>. Early access.
  </li>

  <li style="font-size: 14.5px; color: #626a73; margin-bottom: 15px;">
    Xinyu Chen, HanQin Cai, Fuqiang Liu, Jinhua Zhao (2025). <a href="https://doi.org/10.1109/TKDE.2025.3550877"><b>Correlating time series with interpretable convolutional kernels</b></a>. <em>IEEE Transactions on Knowledge and Data Engineering</em>. 37 (6): 3272-3283.
  </li>
</ul>

</div>

</div>


<br>

<p align="center">
<video style="max-width: 90%; height: auto;" controls>
  <source src="https://spatiotemporal-data.github.io/video/chicago_ridesharing_ts_example.mov" type="video/mp4">
</video>
</p>

<p style="font-size: 14px; color: gray" align = "center"> 🔨 Annotating the weekly periodicity of hourly ridesharing trip time series in Chicago since April 1, 2024.</p>


<br>


<div style="
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  width: 100vw;
  background: #FCEDBB;
  color: #626a73;
  padding: 10px;
">

<div style="
  max-width: 740px; /* Optional: Limits how wide it can get */
  margin: 0 auto; /* Centers the text */
  padding: 1px 10px; /* Small side padding for better readability */
">

<p style="font-size: 17px;"><a href="https://spatiotemporal-data.github.io/posts/sparse_ar/"><b>Sparse Autoregression</b></a> 🌱🌱 </p>

Interpretable time series autoregression for periodicity quantification (ints ⮕ <b>integers</b>).

<div align="center">
<a href="https://github.com/xinychen/integers">
<img src="https://spatiotemporal-data.github.io/images/integers.png" alt="logo" width="250px"/>
</a>
</div>

The content is mainly from our research work below.

<ul style="padding-left: 20px;">
  <li style="font-size: 14.5px; color: #626a73; margin-bottom: 15px;">
    Xinyu Chen, Vassilis Digalakis Jr, Lijun Ding, Dingyi Zhuang, Jinhua Zhao (2025). <a href="https://xinychen.github.io/papers/sparse_ar.pdf"><b>Interpretable time series autoregression for periodicity quantification</b></a>. arXiv preprint arXiv:2506.22895.
  </li>

  <li style="font-size: 14.5px; color: #626a73; margin-bottom: 15px;">
    Xinyu Chen, Qi Wang, Yunhan Zheng, Nina Cao, HanQin Cai, Jinhua Zhao(2025). <a href="https://xinychen.github.io/papers/mobility_periodicity.pdf"><b>Data-driven discovery of mobility periodicity for understanding urban transportation systems</b></a>. arXiv preprint arXiv:2508.03747.
  </li>
</ul>

</div>

</div>

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/autoregression.png" width="650" />
</p>

<p style="font-size: 14px; color: gray" align = "center"> 
🔨 Identification of the dominant auto-correlations from time series through sparse autoregression. The sparsity constraint allows one to find the dominant auto-correlated time lags (e.g., $k=1,24,167,168$).
</p>

<br>


<div style="
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  width: 100vw;
  background: #FCEDBB;
  color: #626a73;
  padding: 10px;
">

<div style="
  max-width: 740px; /* Optional: Limits how wide it can get */
  margin: 0 auto; /* Centers the text */
  padding: 1px 10px; /* Small side padding for better readability */
">

<p style="font-size: 17px;"><a href="https://spatiotemporal-data.github.io/posts/time_varying_model/"><b>Time-Varying Autoregression</b></a> 🌱🌱 </p>

Time-varying autoregression for discovering spatial and temporal patterns of dynamical systems. The system equation includes both time-varying autoregression on the data space and tensor decomposition on the latent space. The content is mainly from our research work below.

<ul style="padding-left: 20px;">
  <li style="font-size: 14.5px; color: #626a73; margin-bottom: 15px;">
    Xinyu Chen, Chengyuan Zhang, Xiaoxu Chen, Nicolas Saunier, Lijun Sun (2024). <a href="https://doi.org/10.1109/TKDE.2023.3294440"><b>Discovering dynamic patterns from spatiotemporal data with time-varying low-rank autoregression</b></a>. <em>IEEE Transactions on Knowledge and Data Engineering</em>. 36 (2): 504-517.
  </li>

  <li style="font-size: 14.5px; color: #626a73; margin-bottom: 15px;">
    Xinyu Chen, Dingyi Zhuang, HanQin Cai, Shenhao Wang, Jinhua Zhao (2025). <a href="https://doi.org/10.1109/TPAMI.2025.3576719"><b>Dynamic autoregressive tensor factorization for pattern discovery of spatiotemporal systems</b></a>. <em>IEEE Transactions on Pattern Analysis and Machine Intelligence</em>. Early access.
  </li>
</ul>

</div>

</div>

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/opp.png" width="650" />
</p>

<p style="font-size: 14px; color: gray" align = "center"> 🔨 Explaining a simple solution to the optimization of Orthogonal Procrustes Problem (OPP).</p>


<br>

### Research Visualization

🌱 [Awesome LaTeX Drawing](https://spatiotemporal-data.github.io/awesome-latex-drawing/). Drawing and generating academic graphics in LaTeX for visualization purposes in research.


<br>

<div style="
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  width: 100vw;
  background: #626a73;
  color: white;
  padding: 10px;
">
<p style="font-size: 14px; color: white" align = "center"> <b>Interactive Visualization Tool for Analyzing Time Series Periodicity</b> </p>

<p align="center">
<a href="https://xinychen.github.io/ts_periodicity">
<img align="middle" src="https://spatiotemporal-data.github.io/images/ts_periodicity_tool.png" width="600" />
</a>
</p>

<p style="font-size: 14px; color: white" align = "center"> 🔨 Anotating the time series periodicity of hourly ridesharing trip time series in Chicago since April 1, 2024.</p>

</div>

<br>

### Scientific Data Tutorial Series

To advance the development of spatiotemporal data computing in the research community, this project handles various spatiotemporal data:

- [Analyzing millions of taxi trips in the City of Chicago](https://spatiotemporal-data.github.io/Chicago-mobility/taxi-data/)
- [Constructing human mobility tensor on NYC rideshare trip data](https://spatiotemporal-data.github.io/NYC-mobility/rideshare/)
<!-- - [Utilizing international import and export trade data from WTO Stats](https://spatiotemporal-data.github.io/trade/import-export/) -->

To contribute to the open science, this project provides a series of tutorials for beginners to understand machine learning and data science, including

- Tensor decomposition for machine learning (see [the detailed page](https://sites.mit.edu/tensor4ml/) at MIT Sites).

- Spatiotemporal data visualization in Python.
  - [High-resolution sea surface temperature data](https://spatiotemporal-data.github.io/climate/sst/)
  - [Global water vapor patterns](https://spatiotemporal-data.github.io/climate/water-vapor/)
  - [Germany energy consumption](https://spatiotemporal-data.github.io/energy/E-usage-data/)
  - [Station-level USA temperature data](https://spatiotemporal-data.github.io/climate/daymet/)

📂 For those who are interested in broad areas within the scope, we would like to recommend a series of [well-documented reading notes](https://spatiotemporal-data.github.io/bib/).

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/ai4data_w_mob.png" width="720" />
</p>

<br>
<br>