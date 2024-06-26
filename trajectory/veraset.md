---
layout: default
---

**Content:**

- Data files
- Trajectory data
- Data samples
- Time resolution
<!-- - Geographical coverage -->
- Challenges

<br>

**Data Files**

**1024 data files** (~1MB for each file) for each day. The naming system shows differences by the `part-xxxxx-tid...` (e.g., `xxxxx` as the number `00000`) and the `-xxxx-1-c000.snappy.parquet` (e.g., `xxxx` as the number `908` or `1931`).

<!-- What is the difference among these data files? Because these data files are not too large, it is not necessary to have so many data files. -->

<br>

**Trajectory Data**

The data format `.snappy.parquet` can be processed with the data processing package `pandas`. In what follows, we have tried to analyze some data files.

<br>

```python
import pandas as pd

df = pd.read_parquet('part-00000-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-908-1-c000.snappy.parquet')
df.head()
```

<br>

corresponding to the first data file with `21161 rows × 10 columns`. The columns include `utc_timestamp`, `latitude`, `longitude` as critical information of the spatiotemporal data. But what is the meaning of the column `caid`, corresponding to unique persons? Using the visualization tools, there seem to be many scatters on the map (see Figure 1 and Figure 2).

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

**Pre-Processing Data Samples**

We tried to read these data files on October 1st with the `for` loop as follows.

<br>

```python
import pandas as pd

df = pd.read_parquet('part-00000-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-908-1-c000.snappy.parquet')
for i in range(1, 1024):
    ticker = list([i, 908+i])
    if i < 10:
        df = df.append(pd.read_parquet('part-0000{}-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 10 and i < 100:
        df = df.append(pd.read_parquet('part-000{}-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 100 and i < 1000:
        df = df.append(pd.read_parquet('part-00{}-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 1000:
        df = df.append(pd.read_parquet('part-0{}-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
a = df.shape[0]
b = df['caid'].unique().size
print('The number of data samples: {}'.format(a))
print('The number of unique id: {}'.format(b))
print('The average data sample number of each unique id: {}'.format(a / b))
```

<br>

<!-- By running these codes, there are **18,050,633** data samples and **193,215** unique IDs in total on October 1st. Therefore, we have **93.42** data samples for each unique ID. -->

<!-- Are these data only subsamples of the whole data? Will we have more samples (> 193,215 unique IDs)? How many IDs will we have on the whole data? -->

<!-- How about the time resolution (e.g., 15 minutes) of the movement of each person? -->

<br>

<p align = "center">
<b>Table 1.</b> The number of data samples and unique IDs in the first two weeks. The last row shows the total data samples and the total unique IDs (already removed the overlap).
</p>

| Day        | Total data samples | Unique ID | Average samples of each ID |
| ---------- | :----------------: | :-------: | :------------------------: |
| Oct. 1st   | 18,050,633         | 193,215   | 93                         |
| Oct. 2nd   | 15,255,154         | 242,068   | 63                         |
| Oct. 3rd   | 17,679,260         | 183,776   | 96                         |
| Oct. 4th   | 31,815,354         | 267,392   | 119                        |
| Oct. 5th   | 21,137,118         | 241,984   | 87                         |
| Oct. 6th   | 18,479,719         | 245,583   | 75                         |
| Oct. 7th   | 21,299,472         | 235,557   | 90                         |
| Oct. 8th   | 21,496,798         | 264,891   | 81                         |
| Oct. 9th   | 16,698,300         | 151,961   | 110                        |
| Oct. 10th  | 32,036,224         | 266,661   | 120                        |
| Oct. 11th  | 17,375,801         | 218,536   | 80                         |
| Oct. 12th  | 11,428,466         | 136,568   | 84                         |
| Oct. 13th  | 11,380,990         | 161,129   | 71                         |
| Oct. 14th  | 14,277,287         | 177,437   | 80                         |
| Total      | 268,410,576        | 1,002,571 | 268                        |

<br>

During the first two weeks, there are **268 million data samples** and **1 million unique IDs**, see Table 1.

<br>

**Time Resolution**

The time resolution for collecting instant position information is a little random. We compute the time differences between two data points and filter the time differences that are greater than 15 minutes. In Figure 3, we visualize the distribution of individual time resolutions of the first 4,299 IDs.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/trajectory/veraset_time_resolution_dist.png" width="500" />
</p>

<p align = "center">
<b>Figure 3.</b> The average time resolution is <b>3.50 minutes</b>.
</p>

<br>

As shown in Figure 4 and 5, the time resolution curves show some "outliers" which refers to the change point of different trajectories.

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/trajectory/time_resolution_0.png" width="500" />
</p>

<p align = "center">
<b>Figure 4.</b> The time resolution curve changed over time on the first example ID (`i = 0`).
</p>

<br>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/trajectory/time_resolution_5.png" width="500" />
</p>

<p align = "center">
<b>Figure 5.</b> The time resolution curve changed over time on the second example ID (`i = 5`).
</p>

<br>

**Challenges**

1. Identify activity modes (e.g., home, work, and commuting) from pings of each ID.

2. Identify mobility from trajectories of each ID.

<br>

**Other Stuff to Connect**

1. Collect POI data from e.g. Google maps or other platforms.

<br>

## Appendix

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

Searching unique IDs and saving the `DataFrame` as the `.csv` file (e.g., `caid_01.csv`).

<br>

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import glob

## Oct. 1st
df = pd.read_parquet('part-00000-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-908-1-c000.snappy.parquet')
df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
for i in range(1, 1024):
    ticker = list([i, 908+i])
    if i < 10:
        df = df.append(pd.read_parquet('part-0000{}-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 10 and i < 100:
        df = df.append(pd.read_parquet('part-000{}-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 100 and i < 1000:
        df = df.append(pd.read_parquet('part-00{}-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 1000:
        df = df.append(pd.read_parquet('part-0{}-tid-9141560157573588789-586e403e-f6f8-4385-8507-8a0d7c4c91d7-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids1 = df['caid'].unique()
pd.DataFrame({'caid': list(ids1)}).to_csv('caid_01.csv', index = False)

## Oct. 2nd
df = pd.read_parquet('part-00000-tid-8628706099891964741-56053e11-e575-4138-bfd8-237ddcb6e634-3980-1-c000.snappy.parquet')
df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
for i in range(1, 1024):
    ticker = list([i, 3980+i])
    if i < 10:
        df = df.append(pd.read_parquet('part-0000{}-tid-8628706099891964741-56053e11-e575-4138-bfd8-237ddcb6e634-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 10 and i < 100:
        df = df.append(pd.read_parquet('part-000{}-tid-8628706099891964741-56053e11-e575-4138-bfd8-237ddcb6e634-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 100 and i < 1000:
        df = df.append(pd.read_parquet('part-00{}-tid-8628706099891964741-56053e11-e575-4138-bfd8-237ddcb6e634-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 1000:
        df = df.append(pd.read_parquet('part-0{}-tid-8628706099891964741-56053e11-e575-4138-bfd8-237ddcb6e634-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids2 = df['caid'].unique()
pd.DataFrame({'caid': list(ids2)}).to_csv('caid_02.csv', index = False)

## Oct. 3rd
df = pd.read_parquet('part-00000-tid-8345537549904831215-54a7c61d-1b8c-4ce5-b6b3-1cd2b70178b2-1932-1-c000.snappy.parquet')
df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
for i in range(1, 1024):
    ticker = list([i, 1932+i])
    if i < 10:
        df = df.append(pd.read_parquet('part-0000{}-tid-8345537549904831215-54a7c61d-1b8c-4ce5-b6b3-1cd2b70178b2-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 10 and i < 100:
        df = df.append(pd.read_parquet('part-000{}-tid-8345537549904831215-54a7c61d-1b8c-4ce5-b6b3-1cd2b70178b2-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 100 and i < 1000:
        df = df.append(pd.read_parquet('part-00{}-tid-8345537549904831215-54a7c61d-1b8c-4ce5-b6b3-1cd2b70178b2-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 1000:
        df = df.append(pd.read_parquet('part-0{}-tid-8345537549904831215-54a7c61d-1b8c-4ce5-b6b3-1cd2b70178b2-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids3 = df['caid'].unique()
pd.DataFrame({'caid': list(ids3)}).to_csv('caid_03.csv', index = False)

## Oct. 4th
df = pd.read_parquet('part-00000-tid-7676566896644138887-7d3e3480-b84c-44c6-a30a-78d0e1cc5709-2956-1-c000.snappy.parquet')
df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
for i in range(1, 1024):
    ticker = list([i, 2956+i])
    if i < 10:
        df = df.append(pd.read_parquet('part-0000{}-tid-7676566896644138887-7d3e3480-b84c-44c6-a30a-78d0e1cc5709-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 10 and i < 100:
        df = df.append(pd.read_parquet('part-000{}-tid-7676566896644138887-7d3e3480-b84c-44c6-a30a-78d0e1cc5709-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 100 and i < 1000:
        df = df.append(pd.read_parquet('part-00{}-tid-7676566896644138887-7d3e3480-b84c-44c6-a30a-78d0e1cc5709-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    elif i >= 1000:
        df = df.append(pd.read_parquet('part-0{}-tid-7676566896644138887-7d3e3480-b84c-44c6-a30a-78d0e1cc5709-{}-1-c000.snappy.parquet'.format(*ticker)), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids4 = df['caid'].unique()
pd.DataFrame({'caid': list(ids4)}).to_csv('caid_04.csv', index = False)

## Oct. 5th
df = pd.DataFrame(['utc_timestamp', 'caid', 'latitude', 'longitude', 'horizontal_accuracy'])
for file in glob.glob("part*.snappy.parquet"):
    df = df.append(pd.read_parquet(file), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids5 = df['caid'].unique()
pd.DataFrame({'caid': list(ids5)}).to_csv('caid_05.csv', index = False)

## Oct. 6th
df = pd.DataFrame(['utc_timestamp', 'caid', 'latitude', 'longitude', 'horizontal_accuracy'])
for file in glob.glob("part*.snappy.parquet"):
    df = df.append(pd.read_parquet(file), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids6 = df['caid'].unique()
pd.DataFrame({'caid': list(ids6)}).to_csv('caid_06.csv', index = False)

## Oct. 7th
df = pd.DataFrame(['utc_timestamp', 'caid', 'latitude', 'longitude', 'horizontal_accuracy'])
for file in glob.glob("part*.snappy.parquet"):
    df = df.append(pd.read_parquet(file), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids7 = df['caid'].unique()
pd.DataFrame({'caid': list(ids7)}).to_csv('caid_07.csv', index = False)

## Oct. 8th
df = pd.DataFrame(['utc_timestamp', 'caid', 'latitude', 'longitude', 'horizontal_accuracy'])
for file in glob.glob("part*.snappy.parquet"):
    df = df.append(pd.read_parquet(file), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids8 = df['caid'].unique()
pd.DataFrame({'caid': list(ids8)}).to_csv('caid_08.csv', index = False)

## Oct. 9th
df = pd.DataFrame(['utc_timestamp', 'caid', 'latitude', 'longitude', 'horizontal_accuracy'])
for file in glob.glob("part*.snappy.parquet"):
    df = df.append(pd.read_parquet(file), ignore_index = True)
    df = df.drop(['id_type', 'iso_country_code', 'quality_fields', 'geo_fields', 'ip_address'], axis = 1)
ids9 = df['caid'].unique()
pd.DataFrame({'caid': list(ids9)}).to_csv('caid_09.csv', index = False)

```

<br>

Processing unique IDs of each day (only cover the first two weeks in this case) and returning the number of unique IDs.

<br>

```python
import pandas as pd

unique_id = pd.read_csv('caid_01.csv')['caid']
for i in range(2, 10):
    unique_id = set(list(unique_id) + list(pd.read_csv('caid_0{}.csv'.format(i))['caid']))
for i in range(10, 15):
    unique_id = set(list(unique_id) + list(pd.read_csv('caid_{}.csv'.format(i))['caid']))
len(unique_id)
```

<br>

Computing the average time resolution (in minute).

<br>

```python
import numpy as np

ids1 = df['caid'].unique()
df['utc_timestamp'] = pd.to_datetime(df.utc_timestamp)
time_resolution = np.zeros(len(ids1))
t = 0
for i in list(ids1):
    x = (df[df['caid'] == i]).sort_values(by = 'utc_timestamp')['utc_timestamp'].diff().values[1 :] / (1e+9 * 60)
    x = x.astype(float)
    time_resolution[t] = x[x < 15].mean()
    t += 1
```

<br>

For reproducing Figure 3, please use the following Python codes.

<br>

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 13

fig, axes = plt.subplots()
sns.histplot(time_resolution[: 4299], color = 'red', alpha = 0.35, ax = axes)
plt.xlabel('Average time resolution (min)')
plt.ylabel('Count')
plt.xlim([0, 15])
fig.savefig("veraset_time_resolution_dist.png", bbox_inches = "tight")
plt.show()
```

<br>

For reproducing Figure 4 and 5, please use the following Python codes.

<br>

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (6, 3))
i = 5
x = (df[df['caid'] == ids1[i]]).sort_values(by = 'utc_timestamp')['utc_timestamp'].diff().values[1 :] / (1e+9 * 60)
x = x.astype(float)
plt.plot(x)
plt.ylim([-2, 60])
plt.xlabel('Data point (#)')
plt.ylabel('Time resolution (min)')
fig.savefig("atime_resolution_{}.png".format(i), bbox_inches = "tight")
plt.show()
```

<br>

**References**

- [How to read a CSV file that starts with a specific substring in Python pandas?](https://stackoverflow.com/a/44522400/11731946)
- [Reading CSV files in a loop using pandas, then concatenating them](https://stackoverflow.com/a/46502968)
- [How can unrar a file with python](https://stackoverflow.com/questions/17614467/how-can-unrar-a-file-with-python)

<br>
