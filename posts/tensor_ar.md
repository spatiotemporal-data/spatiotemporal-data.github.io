---
layout: default
---

# Tensor Autoregression

<p align="center"><span style="color:gray">What is the multidimensional time series? How to characterize these time series with tensor autoregression?</span></p>

<p align="center"><span style="color:gray">(Updated on May 9, 2025)</span></p>

<br>

In the past decades, we witnessed great development of time series models, especially on the univariate time series and multivariate time series. One most classical time series model would be the time series autoregression, including univariate autoregression and vector autoregression. However, almost all of these models are not well-suited to the multidimensional time series. In this post, we introduce a tensor autoregression for modeling multidimensional time series. Let’s get started!

## I. Time Series Models

Time series analysis has great significance in many scientific fields and industrial applications. In the real world, a large variety of time series is univariate or multivariate. Both univariate and multivariate time series are well studied in the past decades and there are many mature methods that have been developed on such kind of data, supporting both analysis and forecasting. However, multidimensional time series are not well explored, but of great significance in the real-world applications.


In practice, multidimensional time series have a more complicated representation than multivariate time series. For instance, multivariate time series usually consist of vector yt of length <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> (i.e., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> variables), at time t, while multidimensional time series, consisting of Yt of size <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;M"/>-by-<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> (i.e., <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;MN"/> variables) at time <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/>, also known as matrix-variate time series, are in the form of third-order tensor. Consequently, multidimensional time series pose some methodological challenges for building a certain framework with the complicated algebraic representation.

## II. Tensor Autoregression

### II-A. Basic Formula

For the matrix-variate time series:

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}_{1},\ldots,\boldsymbol{Y}_{T}\in\mathbb{R}^{M\times N}"/></p>

over <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;T"/> time steps, the formula of the first-order tensor autoregression [1] can be written as follows,

<p align = "center"><img align="middle" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{Y}_{t}=\boldsymbol{\mathcal{A}}\times_{\mathcal{L}}\boldsymbol{Y}_{t-1}+\boldsymbol{E}_{t},t=2,\ldots,T"/></p>

where <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{\mathcal{A}}"/> is the coefficient tensor of size <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;M"/>-by-<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;M"/>-by-<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/>, and the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;M"/>-by-<img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;N"/> matrix <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{E}_{t}"/> is error term. Herein, we use the <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\mathcal{L}"/>-product between the coefficient tensor and matrix-variate time series.

Figure 1 presents the graphical illustration of the above tensor autoregression on the matrix-variate time series. The orange boxes show the matrices at time <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t"/> and <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;t-1"/>, while the red cube refers to the coefficient tensor. Notably, the red cube has three dimensions.



<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/tensor_ar_illustration.webp" width="450" />
</p>

<p style="font-size: 14px; color: gray" align = "center">
<b>Figure 1.</b> Illustration of tensor autoregression on the matrix-variate time series.
</p>

<br>



<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on May 9, 2025.)</p>
