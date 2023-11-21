---
layout: default
---

# Analyzing Millions of Taxi Trips in the City of Chicago


[The City of Chicago's open data portal](https://data.cityofchicago.org/) provides a large amount of human mobility data, including [taxi trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew), [TNP rideshare trips](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2018-2022-/m6dm-c72p), [Divvy bikeshare trips](https://divvybikes.com/system-data), and [E-scooter trips](https://catalog.data.gov/dataset/e-scooter-trips). There is a brief summary of annual trips of these travel modes in the City of Chicago, depending on the data availability.

<br>

| Year    | Taxi trips | Rideshare trips | Divvy trips | E-Scooter trips |
| ------- | :--------: | :-------------: | :---------: | :-------------: |
| 2013    | 27.2M      |                 |  760K       |                 |
| 2014    | 37.4M      |                 |  2.45M      |                 |
| 2015    | 32.4M      |                 |  3.18M      |                 |
| 2016    | 31.8M      |                 |  3.60M      |                 |
| 2017    | 25M        |                 |  3.83M      |                 |
| 2018    | 20.7M      |                 |  3.60M      |                 |
| 2019    | 16.5M      | 112M            |  3.82M      |  711K           |
| 2020    | 3.89M      | 50M             |  3.54M      |  631K           |
| 2021    | 3.95M      | 51.2M           |  5.60M      |                 |
| 2022    | 6.38M      | 69.1M           |  5.67M      | 1.49M           |

<br>

## Visualizing Boundaries of Community Areas in Chicago

The data can be viewed on the Chicago Data Portal with a web browser, see [77 community areas in Chicago](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6). To use these data, one can export and download the data in the Shapefile format. In this post, we rename four files of the Shapefile data as follows,

- `areas.dbf`
- `areas.prj`
- `areas.shp`
- `areas.shx`

and place these files at the folder `Boundaries_Community_Areas`.

Then it is not hard to use the `geopandas` and `matplotlib` packages in Python to visualize the boundaries of community areas.

<br>

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

<br>

## Matching Taxi Trips with Community Areas

There are three basic steps to follow for processing taxi trip data:
- Download [taxi trips in 2022](https://data.cityofchicago.org/Transportation/Taxi-Trips-2022/npd7-ywjz) in the `.csv` format, e.g., `Taxi_Trips_-_2022.csv`.
- Use the `pandas` package in Python to process the raw trip data.
- Match trip pickup/dropoff locations with boundaries of the community area.

<br>

```python
import pandas as pd

data = pd.read_csv('Taxi_Trips_-_2022.csv')
data.head()
```

<br>

For each taxi trip, one can select some important information:
- `Trip Start Timestamp`: When the trip started, rounded to the nearest 15 minutes.
- `Trip Seconds`: Time of the trip in seconds.
- `Trip Miles`: Distance of the trip in miles.
- `Pickup Community Area`: The Community Area where the trip began. This column will be blank for locations ourside Chicago.
- `Dropoff Community Area`: The Community Area where the trip ended. This column will be blank for locations outside Chicago.

<br>

```python
df = pd.DataFrame()
df['Trip Start Timestamp'] = data['Trip Start Timestamp']
df['Trip Seconds'] = data['Trip Seconds']
df['Trip Miles'] = data['Trip Miles']
df['Pickup Community Area'] = data['Pickup Community Area']
df['Dropoff Community Area'] = data['Dropoff Community Area']
del data
df
```

<br>

By doing so, there are 6,382,425 rows in this new dataframe. For the following analysis, one should remove the trips whose pickup/dropoff locations outside Chicago. In addition, one should clean the outliers that are with `0` (trip) seconds or `0` (trip) miles.

<br>

```python
df = df.dropna() # Remove rows with NaN
df = df.drop(df[df['Trip Seconds'] == 0].index)
df = df.drop(df[df['Trip Miles'] == 0].index)
df = df.reset_index()
df = df.drop(['index'], axis = 1)
df.to_csv('taxi_trip_2022.csv', index = False)

import numpy as np

print(np.mean(df['Trip Seconds'].values))
print(np.mean(df['Trip Miles'].values))
```

<br>

By doing so, there are 4,763,961 remaining taxi trips in the dataframe. If you want to aggregate the trip counts of each pickup/dropoff community area, the simplest way to get row counts per pickup/dropoff community area is by calling `.groupby().size()`.

<br>

```python
pickup_df = df.groupby(['Pickup Community Area']).size().reset_index(name = 'pickup_counts')
dropoff_df = df.groupby(['Dropoff Community Area']).size().reset_index(name = 'dropoff_counts')
```

<br>

## Visualizing Pickup and Dropoff Trips in 2022

It is not hard to first use the `geopandas` package to merge pickup/dropoff trip counts into the `.shp` data and then visualize the trip data with the `matplotlib` package.

<br>

```python
import geopandas as gpd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

chicago = gpd.read_file("Boundaries _Community_Areas/areas.shp")
pickup_df = df.groupby(['Pickup Community Area']).size().reset_index(name = 'pickup_counts')
dropoff_df = df.groupby(['Dropoff Community Area']).size().reset_index(name = 'dropoff_counts')
pickup_df['area_numbe'] = pickup_df['Pickup Community Area']
dropoff_df['area_numbe'] = dropoff_df['Dropoff Community Area']
chicago['area_numbe'] = chicago.area_numbe.astype(float)

pickup = chicago.set_index('area_numbe').join(pickup_df.set_index('area_numbe')).reset_index()
dropoff = chicago.set_index('area_numbe').join(dropoff_df.set_index('area_numbe')).reset_index()

fig = plt.figure(figsize = (14, 8))
for i in [1, 2]:
    ax = fig.add_subplot(1, 2, i)
    if i == 1:
        pickup.plot('pickup_counts', cmap = 'YlOrRd', legend = True,
                    legend_kwds = {'shrink': 0.618, 'label': 'Pickup trip count'},
                    vmin = 0, vmax = 1.4e+6, ax = ax)
    elif i == 2:
        dropoff.plot('dropoff_counts', cmap = 'YlOrRd', legend = True, 
                     legend_kwds = {'shrink': 0.618, 'label': 'Dropoff trip count'},
                     vmin = 0, vmax = 1.4e+6, ax = ax)
    plt.xticks([])
    plt.yticks([])
plt.show()
fig.savefig("pickup_dropoff_trips_chicago_2022.png", bbox_inches = "tight")
```

<br>

Figure 2 shows taxi pickup and dropoff trips (2022) on 77 community areas in the City of Chicago. Note that the average trip duration is **1207.75 seconds** and the average trip distance is **6.16 miles**.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/pickup_dropoff_trips_chicago_2022.png" width="600" />
</p>

<p align = "center">
<b>Figure 2.</b> Taxi pickup and dropoff trips (2022) in the City of Chicago, USA. There are 4,763,961 remaining trips after the data processing.
</p>

<br>

For comparison, Figure 3 shows taxi pickup and dropoff trips (2019) on 77 community areas in the City of Chicago. Note that the average trip duration is **915.62 seconds** and the average trip distance is **3.93 miles**.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/pickup_dropoff_trips_chicago_2019.png" width="600" />
</p>

<p align = "center">
<b>Figure 3.</b> Taxi pickup and dropoff trips (2019) in the City of Chicago, USA. There are 12,484,572 remaining trips after the data processing. See <a href = "https://spatiotemporal-data.github.io/Chicago-mobility/taxi-data-appendix/">the data processing codes</a>.
</p>


<br>

In addition, one can analyze the trips of other travel modes in the City of Chicago. Figure 4 shows E-scooter pickup and dropoff trips on 77 community areas in the City of Chicago, see [how to process and visualize E-scooter trips](https://spatiotemporal-data.github.io/Chicago-mobility/e-scooter/). Note that the average trip duration is **913.18 seconds** and the average trip distance is **2448.60 meters**.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/E_scooter_pickup_dropoff_trips_chicago_2022.png" width="600" />
</p>

<p align = "center">
<b>Figure 4.</b> E-scooter pickup and dropoff trips (2022) in the City of Chicago, USA. There are 1,476,028 remaining trips after the data processing.
</p>

<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on November 12, 2023.)</p>
