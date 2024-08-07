---
layout: default
---

# Constructing Human Mobility Tensor on NYC Rideshare Trip Data

Among a large number of open human mobility datasets, [TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) might be one of the most classical sources for doing mobility research and data analysis. This open data includes yellow and green taxi trip records, For-Hire Vehicle (FHV) trip records, and High Volume For-Hire Vehicle (HVFHV) trip records stored in the `.parquet` format, ranging from 2009 to the latest date. The HVFHV trip records have TLC license numbers for Juno (`HV0002`), Uber (`HV0003`), Via (`HV0004`), and Lyft (`HV0005`), see [data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf) for details. In what follows, we use rideshare trip records to mention the HVFHV trip records instead.

## Rideshare Trip Records

The first procedure is downloading the rideshare trip records from [TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) through selecting certain data files, e.g., `High Volume For-Hire Vehicle Trip Records` in April and May in 2024:

- `fhvhv_tripdata_2024-04.parquet` (April 2024, with 476.1 MB)
- `fhvhv_tripdata_2024-05.parquet` (May 2024, with 498.6 MB)

These data files in the format of `.parquet` can be easily processed by `pandas` in Python.


```python
import pandas as pd
import glob

df = pd.concat([pd.read_parquet(file) for file in glob.glob('NYC-taxi/fhvhv_tripdata_2024-*.parquet')])
df.head()
```

<br>

As a result, the output is


```python
hvfhs_license_num	dispatching_base_num	originating_base_num	request_datetime	on_scene_datetime	pickup_datetime	dropoff_datetime	PULocationID	DOLocationID	trip_miles	...	sales_tax	congestion_surcharge	airport_fee	tips	driver_pay	shared_request_flag	shared_match_flag	access_a_ride_flag	wav_request_flag	wav_match_flag
0	HV0003	B03404	B03404	2024-04-30 23:55:50	2024-04-30 23:59:00	2024-05-01 00:00:36	2024-05-01 00:38:21	138	21	18.680	...	4.48	0.00	2.5	0.00	47.41	N	N	N	N	N
1	HV0003	B03404	B03404	2024-05-01 00:41:32	2024-05-01 00:47:40	2024-05-01 00:49:40	2024-05-01 00:57:08	21	22	1.710	...	1.03	0.00	0.0	0.00	6.69	N	N	N	N	N
2	HV0003	B03404	B03404	2024-05-01 00:09:01	2024-05-01 00:10:51	2024-05-01 00:11:31	2024-05-01 00:33:51	140	129	5.000	...	2.65	0.75	0.0	0.00	19.82	Y	N	N	N	N
3	HV0003	B03404	B03404	2024-05-01 00:25:43	2024-05-01 00:48:00	2024-05-01 00:48:30	2024-05-01 01:10:17	138	48	9.250	...	3.47	2.75	2.5	0.00	25.28	N	N	N	N	N
4	HV0005	B03406	None	2024-05-01 00:00:08	NaT	2024-05-01 00:09:29	2024-05-01 00:28:51	4	25	4.996	...	2.38	2.75	0.0	6.99	22.24	N	N	N	N	Y
5 rows × 24 columns
```

<br>

To construct human mobility tensors, the columns we are interested are `pickup_datetime` (starting time), `PULocationID` (pickup location ID), and `DOLocationID` (dropoff location ID). Therefore, we generate a new dataframe as follows.


```python
data = pd.DataFrame()
data['pickup_datetime'] = df['pickup_datetime']
data['PULocationID'] = df['PULocationID']
data['DOLocationID'] = df['DOLocationID']
data['month'] =  data['pickup_datetime'].dt.month
data['day'] = data['pickup_datetime'].dt.day
data['hour'] = data['pickup_datetime'].dt.hour
data
```

<br>

As a result, the output is


```python
pickup_datetime	PULocationID	DOLocationID	month	day	hour
0	2024-05-01 00:00:36	138	21	5	1	0
1	2024-05-01 00:49:40	21	22	5	1	0
2	2024-05-01 00:11:31	140	129	5	1	0
3	2024-05-01 00:48:30	138	48	5	1	0
4	2024-05-01 00:09:29	4	25	5	1	0
...	...	...	...	...	...	...
19733033	2024-04-30 23:34:35	76	38	4	30	23
19733034	2024-04-30 23:14:38	126	126	4	30	23
19733035	2024-04-30 23:33:36	126	32	4	30	23
19733036	2024-04-30 23:53:58	32	31	4	30	23
19733037	2024-04-30 23:08:31	50	7	4	30	23
40437576 rows × 6 columns
```

<br>

The number of rideshare trip records in April and May 2024 is 40,437,576 in total. The maximum pickup and dropoff location ID is 265, but in fact, there are 262 unique pickup/dropoff location IDs. As mentioned above, one can extract month, day, and hour information from the pickup datetime.

## Constructing Mobility Tensor

According to the pickup location ID, dropoff location ID, and time step with an hourly time resolution, one can construct a mobility tensor with (origin, destination, time) dimensions, while the entries of this tensor are trip counts.

```python
data = data.groupby(['PULocationID', 'DOLocationID', 'month', 'day', 
                     'hour']).size().reset_index(name = 'count')
data.head()
```

<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on August 7, 2024.)</p>
