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

Origin-Destination (OD) mobility network is generally weighted, directed, and not regular.


<br>
<br>


<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on October 21, 2026)</p>
