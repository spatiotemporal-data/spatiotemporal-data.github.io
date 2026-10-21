---
layout: default
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

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

# Optimal Spectral Gap for Urban Transportation Networks

<p align="center"><span style="color:gray">Directed Laplacian gap with a stationary-distribution-aware symmetrization: connectivity and bottlenecks.</span></p>

<p align="center"><span style="color:gray">(Updated on October 21, 2026)</span></p>

<br>

**Content:**




<br>

## I. Motivation

We are motivated to use the spectral gap to answer the following questions:

- **Global connectivity**: How easily does mobility connect different parts of a city? We expect to quantify the overall connectivity of a city's mobility network.
- **Community structure and bottlenecks**: Are there groups of regions with strong internal flows but weak flows between groups? We expect to identify mobility communities or bottlenecks.
- **Directional asymmetry**: Does mobility flow disproportionately in particular directions, and are some regions much more central than others?

<br>


## II. Definition

Origin-Destination (OD) mobility network is generally weighted, directed, and not regular. Let $$\boldsymbol{X}\in\mathbb{R}^{n\times n}_{\geq0}$$ be the OD matrix for a time window, where $$x_{ij}$$ is the number of trips from origin zone $i$ to destination zone $j$.

First, we use count the basic statistics as follows,

$$r=\boldsymbol{X}\boldsymbol{1},\quad c=\boldsymbol{X}^\top\boldsymbol{1},\quad N=\boldsymbol{1}^\top\boldsymbol{X}\boldsymbol{1}$$

referring to marginals and total.


<br>
<br>


<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on October 21, 2026)</p>
