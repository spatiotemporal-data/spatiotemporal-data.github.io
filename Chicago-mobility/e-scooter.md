---
layout: default
---

# Processing and Visualizing E-Scooter Trips in the City of Chicago

- Download data

```python
import pandas as pd

data = pd.read_csv('E-Scooter_Trips.csv')
data.head()
```

- Extract trip information

```python
df = pd.DataFrame()
df['Start Time'] = data['Start Time']
df['Trip Distance'] = data['Trip Distance']
df['Trip Duration'] = data['Trip Duration']
df['Start Community Area Number'] = data['Start Community Area Number']
df['End Community Area Number'] = data['End Community Area Number']
df['Year'] = pd.to_datetime(df['Start Time'], errors='coerce').dt.year
df
```

- Drop trips

```python
df = df.dropna() # Remove rows with NaN
df = df.drop(df[df['Year'] == 2023].index)
df = df.drop(df[df['Trip Distance'] == 0].index)
df = df.drop(df[df['Trip Duration'] == 0].index)
df = df.reset_index()

df = df.drop(['index'], axis = 1)
df.to_csv('E_scooter_trip_2022.csv', index = False)

import numpy as np

print(np.mean(df['Trip Duration'].values))
print(np.mean(df['Trip Distance'].values))
```

- Visualize trips

```python
import geopandas as gpd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

chicago = gpd.read_file("Boundaries_Community_Areas/areas.shp")
pickup_df = df.groupby(['Start Community Area Number']).size().reset_index(name = 'pickup_counts')
dropoff_df = df.groupby(['End Community Area Number']).size().reset_index(name = 'dropoff_counts')
pickup_df['area_numbe'] = pickup_df['Start Community Area Number']
dropoff_df['area_numbe'] = dropoff_df['End Community Area Number']
chicago['area_numbe'] = chicago.area_numbe.astype(float)

pickup = chicago.set_index('area_numbe').join(pickup_df.set_index('area_numbe')).reset_index()
dropoff = chicago.set_index('area_numbe').join(dropoff_df.set_index('area_numbe')).reset_index()

fig = plt.figure(figsize = (14, 8))
for i in [1, 2]:
    ax = fig.add_subplot(1, 2, i)
    if i == 1:
        pickup.plot('pickup_counts', cmap = 'YlOrRd', legend = True,
                    legend_kwds = {'shrink': 0.618, 'label': 'Pickup trip count'},
                    vmin = 0, vmax = 2.5e+5, ax = ax)
    elif i == 2:
        dropoff.plot('dropoff_counts', cmap = 'YlOrRd', legend = True, 
                     legend_kwds = {'shrink': 0.618, 'label': 'Dropoff trip count'},
                     vmin = 0, vmax = 2.5e+5, ax = ax)
    plt.xticks([])
    plt.yticks([])
plt.show()
fig.savefig("E_scooter_pickup_dropoff_trips_chicago_2022.png", bbox_inches = "tight")
```
