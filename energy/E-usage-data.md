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
<img align="middle" src="https://spatiotemporal-data.github.io/images/boundaries_community_areas_chicago.png" width="300" />
</p>

<p align = "center">
<b>Figure 1.</b> Boundaries of community areas in the City of Chicago, USA.
</p>

<br>


<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on November 28, 2023.)</p>
