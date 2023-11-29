---
layout: default
---

# Visualizing Germany Energy Consumption Data in Python

The analysis of energy consumption for future energy systems requires appropriate data. In the latest [JERICHO-E-usage dataset](https://www.nature.com/articles/s41597-021-00907-w), comprehensive data on useful energy consumption patterns for heat, cold, mechanical energy, information and communication, and light in high spatial and temporal resolution are available. The data are aggregated to the NUTS2 level, consisting of 38 regions in Germany.

<br>

## Download `.shp` File in the NUTS2 Level

Germany energy consumption data is a publicly available dataset. To visualize geographical distribution of these energy consumption data, we need to find appropriate .shp file and project data on it. We download the .shp from [GISCO statistical unit dataset](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts). In the download page, we set the following options:

- Year: `NUTS 2016`
- File format: `SHP`
- Geometry type: `Polygons (RG)`
- Scale: `03M`
- Coordinate reference system: `EPSG: 4326`

Then we can download the `.shp` file. In Python, `geopandas` is a powerful tool for visualizing geospatial data. It is not hard to use `geopandas` to visualize our data file.


<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/Germany_shape.png" width="300" />
</p>

<p align = "center">
<b>Figure 1.</b> Geographical distribution of 38 regions in the NUTS2 level of Germany.
</p>

<br>

To draw Figure 1, please try the following Python codes. Note that two necessary packages include `geopandas` and `matplotlib`.

<br>

```python
import geopandas as gpd
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (14, 8))
ax = fig.subplots(1)
shape = gpd.read_file("NUTS_RG_03M_2016_4326.shp")
shape_de = shape[(shape['CNTR_CODE'] == 'DE') & (shape['LEVL_CODE'] == 2)]
shape_de.plot(cmap = 'YlOrRd_r', ax = ax)
plt.xticks([])
plt.yticks([])
for _, spine in ax.spines.items():
    spine.set_visible(False)
plt.show()
```

## About the Dataset

The JERICHO-E-usage dataset provides the energy consumption of Germany in the whole year of 2019 ([Priesmann et al., 2021](https://doi.org/10.1038/s41597-021-00907-w)). The dataset is spatially resolved at the NUTS2 level comprising 38 regions and temporally resolved in hours. The dataset distinguished among four sectors, namely, the residential, the industrial, the commerce, and the mobility sectors. Useful energy types comprise space heating, warm water, process heating, space cooling, process cooling, mechanical energy, information and communication technology, and lightling.


<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on November 28, 2023.)</p>
