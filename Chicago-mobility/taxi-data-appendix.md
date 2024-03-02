---
layout: default
---

## Taxi Data

- Download Chicago taxi dataset in 2019.

<br>

```python
import pandas as pd

data = pd.read_csv('Taxi_Trips_-_2019.csv')
data.head()
```

<br>

- Process the raw data.

<br>

```python
df = pd.DataFrame()
df['Trip Start Timestamp'] = data['Trip Start Timestamp']
df['Trip Seconds'] = data['Trip Seconds']
df['Trip Miles'] = data['Trip Miles']
df['Pickup Community Area'] = data['Pickup Community Area']
df['Dropoff Community Area'] = data['Dropoff Community Area']
del data

df = df.dropna() # Remove rows with NaN
df = df.drop(df[df['Trip Seconds'] == 0].index)
df = df.drop(df[df['Trip Miles'] == 0].index)
df = df.reset_index()
df = df.drop(['index'], axis = 1)
df.to_csv('taxi_trip_2019.csv', index = False)

import numpy as np

print(np.mean(df['Trip Seconds'].values))
print(np.mean(df['Trip Miles'].values))
```

<br>

- Visualize pickup and dropoff trips.

<br>

```python
import geopandas as gpd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

chicago = gpd.read_file("Boundaries_Community_Areas/areas.shp")
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
                    vmin = 0, vmax = 4e+6, ax = ax)
    elif i == 2:
        dropoff.plot('dropoff_counts', cmap = 'YlOrRd', legend = True,
                     legend_kwds = {'shrink': 0.618, 'label': 'Dropoff trip count'},
                     vmin = 0, vmax = 4e+6, ax = ax)
    plt.xticks([])
    plt.yticks([])
plt.show()
fig.savefig("pickup_dropoff_trips_chicago_2019.png", bbox_inches = "tight")
```

<br>

## Ridehailing Data

- Download Chicago rideshare data in 2022, see [the download page](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2022/2tdj-ffvb/data).

Since the original dataset is of size 18.07 GB and has 69,109,780 rows, we preprocess the dataset and make it smaller (i.e., `rideshare_trip_2022.csv` of size 2.93 GB), at least for the future usage.

<br>

```python
import pandas as pd

data = pd.DataFrame()
chunksize = 10 ** 7
for chunk in pd.read_csv('Transportation_Network_Providers_-_Trips_-_2022.csv', chunksize = chunksize):
    df = pd.DataFrame()
    df['Trip Start Timestamp'] = chunk['Trip Start Timestamp']
    df['Trip Seconds'] = chunk['Trip Seconds']
    df['Trip Miles'] = chunk['Trip Miles']
    df['Pickup Community Area'] = chunk['Pickup Community Area']
    df['Dropoff Community Area'] = chunk['Dropoff Community Area']
    data = data.append(df)
    del df
data.to_csv('rideshare_trip_2022.csv', index = False)
```

<br>

- Process the raw data.

<br>

```python
import pandas as pd

df = pd.read_csv('rideshare_trip_2022.csv')
df = df.dropna() # Remove rows with NaN
df = df.drop(df[df['Trip Seconds'] == 0].index)
df = df.drop(df[df['Trip Miles'] == 0].index)
df = df.reset_index()
df = df.drop(['index'], axis = 1)

import numpy as np

print(np.mean(df['Trip Seconds'].values))
print(np.mean(df['Trip Miles'].values))
pickup_df = df.groupby(['Pickup Community Area']).size().reset_index(name = 'pickup_counts')
dropoff_df = df.groupby(['Dropoff Community Area']).size().reset_index(name = 'dropoff_counts')
```

<br>

- Visualize pickup and dropoff trips.

<br>

```python
import geopandas as gpd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

chicago = gpd.read_file("Boundaries_Community_Areas/areas.shp")
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
                    vmin = 0, vmax = 1e+7,
                    ax = ax)
    elif i == 2:
        dropoff.plot('dropoff_counts', cmap = 'YlOrRd', legend = True,
                     legend_kwds = {'shrink': 0.618, 'label': 'Dropoff trip count'},
                     vmin = 0, vmax = 1e+7,
                     ax = ax)
    plt.xticks([])
    plt.yticks([])
plt.show()
fig.savefig("tnp_pickup_dropoff_trips_chicago_2022.png", bbox_inches = "tight")
```
