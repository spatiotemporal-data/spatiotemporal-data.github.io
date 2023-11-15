---
layout: default
---

# Analyzing 200 Million Taxi Trips (2013-2022) in the City of Chicago


> [The City of Chicago's open data portal](https://data.cityofchicago.org/) provides a large amount of human mobility data, including [taxi trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew), [TNP rideshare trips](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2018-2022-/m6dm-c72p), [Divvy bikeshare trips](https://divvybikes.com/system-data), and [E-scooter trips](https://catalog.data.gov/dataset/e-scooter-trips).

<br>

| Year    | Taxi trips | Rideshare trips |
| ------- | :--------: | :-------------: |
| 2013    | 27.2M      |                 |
| 2014    | 37.4M      |                 |
| 2015    | 32.4M      |                 |
| 2016    | 31.8M      |                 |
| 2017    | 25M        |                 |
| 2018    | 20.7M      |                 |
| 2019    | 16.5M      | 112M            |
| 2020    | 3.89M      | 50M             |
| 2021    | 3.95M      | 51.2M           |
| 2022    | 6.38M      | 69.1M           |

<br>

## Visualizing Boundaries of Community Areas in Chicago

The data can be viewed on the Chicago Data Portal with a web browser, see [77 community areas in Chicago](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6). To use these data, one can export and download the data in the Shapefile format. In this post, we rename four files of the Shapefile data as follows,

- `areas.dbf`
- `areas.prj`
- `areas.shp`
- `areas.shx`

and place these files at the folder `Boundaries_Community_Areas`.

Then it is not hard to use the `geopandas` and `matplotlib` packages in Python to visualize the boundaries of community areas.

```python
import geopandas as gpd
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (14, 8))
ax = fig.subplots(1)
shape = gpd.read_file("Boundaries _Community_Areas/areas.shp")
shape.plot(cmap = 'RdYlBu_r', ax = ax)
plt.xticks([])
plt.yticks([])
for _, spine in ax.spines.items():
    spine.set_visible(False)
plt.show()
fig.savefig("boundaries_community_areas_chicago.png", bbox_inches = "tight")
```

<br>

Figure 1 shows the boundaries of 77 community areas in the City of Chicago. Note that we can set the `cmap` as `RdYlBu_r` or `YlOrRd_r`.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/boundaries_community_areas_chicago.png" width="300" />
</p>

<p align = "center">
<b>Figure 1.</b> Boundaries of community areas in the City of Chicago, USA.
</p>

## Matching Taxi Trips with Community Areas

- Download [taxi trips in 2022](https://data.cityofchicago.org/Transportation/Taxi-Trips-2022/npd7-ywjz) in the `.csv` format, e.g., `Taxi_Trips_-_2022.csv`.
- Use the `pandas` package in Python to process the raw trip data.
- Match trip orgin/destination with boundaries of the community area.




<br>
<br>
<p align="left">(By <a href="https://xinychen.github.io/">Xinyu Chen</a>, published on November 12, 2023.)</p>
