---
layout: default
---

Couple of issues confused me:

- Data description is unclear
- Should rename data files
- Trajectory data itself

<br>

**Data description**

The data description is not readily available.

<br>

**Data file**

Too much data files for each day, e.g., 1024 files on October 1st. The naming system shows differences by the `part-xxxxx-tid...` (e.g., `xxxxx` as the number `00000`) and the `-xxxx-1-c000.snappy.parquet` (e.g., `xxxx` as the number `908` or `1931`). What is the difference among these data files?

<br>

**Trajectory data**

The data format `.snappy.parquet` can be processed with the data processing package `pandas`. In what follows, we have tried to analyze some data files.

<br>

```python
import pandas as pd

df = pd.read_parquet('part-00000-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-908-1-c000.snappy.parquet')
df.head()
```

<br>

corresponding to the first data file with `21161 rows × 10 columns`. The columns include `utc_timestamp`, `latitude`, `longitude` as critical information of the spatiotemporal data. But what is the meaning of the column `caid`, corresponding to unique persons? Using the visualization tools, there seem to many scatters on the map (see Figure 1 and Figure 2). What is the difference between two figures?

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/trajectory/oct1_00000_scatters.png" width="600" />
</p>

<p align = "center">
<b>Figure 1.</b> The scatters of the first data file (i.e., on the file `part-00000-tid....snappy.parquet`) on the map.
</p>

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/trajectory/oct1_00001_scatters.png" width="600" />
</p>

<p align = "center">
<b>Figure 2.</b> The scatters of the second data file (i.e., on the file `part-00001-tid....snappy.parquet`) on the map.
</p>

<br>

For reproducing Figure 1, please use the following Python codes.

<br>

```python
import numpy as np
import matplotlib.pyplot as plt

lng = np.array(df['longitude'], dtype = float)
lat = np.array(df['latitude'], dtype = float)

fig = plt.figure(figsize = (6, 3))
plt.plot(lng, lat, 'r.', markersize = 1)
plt.xlim([103.6, 104.1])
plt.ylim([1.22, 1.47])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('oct1_00001_scatters.png', bbox_inches = 'tight')
plt.show()
```

<br>

For reproducing Figure 2, please use the following Python codes.

<br>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet('part-00001-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-909-1-c000.snappy.parquet')
lng = np.array(df['longitude'], dtype = float)
lat = np.array(df['latitude'], dtype = float)

fig = plt.figure(figsize = (6, 3))
plt.plot(lng, lat, 'r.', markersize = 1)
plt.xlim([103.6, 104.1])
plt.ylim([1.22, 1.47])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('oct1_00001_scatters.png', bbox_inches = 'tight')
plt.show()
```

<br>
