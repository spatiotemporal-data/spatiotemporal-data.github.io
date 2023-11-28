---
layout: default
---

# Visualizing Global Water Vapor Patterns in Python

### Exploring Water Vapor Patterns Across Time with High-Resolution Monthly Aggregated Datasets

In the realm of Earth observation, the power of data is truly remarkable. One dataset that has been making waves in the scientific community is the Monthly Aggregated Water Vapor MODIS MCD19A2 (1 km) dataset. The part of this dataset is available on Zenodo (check out this link: https://doi.org/10.5281/zenodo.8226283). In this blog post, we will dive into this dataset’s significance, applications, and how it’s shaping our understanding of water vapor dynamics across the globe. For an intuitive demonstration, we will tell how to visualize the dataset by Python as a sequence of heatmps.

## About the Dataset

Imagine having the ability to observe water vapor behavior on a global scale, month by month, over a span of more than two decades. Thanks to the dedicated work of Leandro Parente, Rolf Simoes, and Tomislav Hengl, such a dataset exists. The Monthly Aggregated Water Vapor MODIS MCD19A2 (1 km) dataset takes us on an illuminating journey into Earth’s water vapor patterns from 2000 to 2022.

### The Power of Water Vapor Data

Derived from MCD19A2 v061 and utilizing MODIS near-IR bands at 0.94μm, this dataset measures the column of water vapor above the ground. Its applications are as vast as the oceans it helps us understand. From water cycle modeling to vegetation and soil mapping, the insights this dataset provides are invaluable for numerous fields.

### Dataset Highlights

1. **Monthly Time-Series**: The heart of the dataset lies in its monthly time-series. These provide a monthly aggregated mean and standard deviation of daily water vapor time-series data. Only clear-sky pixels are considered, ensuring the accuracy of observations. The Whittaker method helps smooth mean and standard deviation values, while quality assessment layers add reliability to the mix.

2. **Yearly Time-Series**: Zooming out, the yearly time-series present aggregated statistics of the monthly data. This zoomed-out perspective offers a bird’s-eye view of water vapor trends over the course of a year.

3. **Long-Term Insights**: The real gem lies in the long-term data. Derived from monthly time-series, it offers aggregated statistics that encapsulate the entire spectrum of observations from 2000 to 2022. It’s like looking at the evolution of Earth’s breath over two decades.

### Behind the Scenes

The data’s journey from raw observations to insightful statistics is a testament to modern technology. Utilizing the power of Google Earth Engine, the data is derived from MCD19A2 v061. Cloudy pixels are removed, and only positive water vapor values are considered. The magic of Scikit-map Python package handles time-series gap-filling and smoothing, while four essential statistics — standard deviation and percentiles 25, 50, and 75 — provide depth to the dataset.

### Limitations and Spatial Details

No dataset is without its boundaries. This dataset doesn’t cover Antarctica, and its geographic scope is defined by the bounding box (-180.00000, -62.00081, 179.99994, 87.37000). With a spatial resolution of 1km, the image size is an impressive 43,200 x 17,924 pixels. To keep up with the latest technological trends, the dataset is available in Cloud Optimized Geotiff (COG) format.

## Step-By-Step Data Visualization

This post gives some Python codes for generating the water vapor heatmaps. For an example, we only visualize some areas in North America because visualizing the global water vapor heatmaps consumes a lot of computer memory. You can adjust the rows and columns of the data matrix if you want to explore different areas in the world.

### Requirements

This Python package and extensions are a number of tools for programming and manipulating the GDAL Geospatial Data Abstraction Library. We need to first install this package.

<br>

```python
pip install GDAL
```

<br>

### Download Dataset

Check out the data at https://doi.org/10.5281/zenodo.8226283 and view/download some data files. Below show three subsets that corresponding to the months of May, June, and July.
