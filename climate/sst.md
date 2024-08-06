---
layout: default
---

# Sea Surface Temperature Dataset

The oceans play an important role in the global climate system. Exploiting sea surface temperature allows one to sense the climate and understand the dynamical processes of energy exchange at the sea surface. [NOAA (National Oceanic and Atmospheric Administration)](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/) provides a large amount of sea surface temperature data from September 1, 1981 to the present date.

<br>

## Open Dataset

With the advent of satellite retrievals of sea surface temperature beginning in the early of 1980s, there are a large amount of high-resolution sea surface temperature data available for climate analysis, attracting a lot of attention. [The NOAA 1/4 degree daily Optimum Interpolation Sea Surface Temperature (OISST)](https://www.ncei.noaa.gov/products/climate-data-records/sea-surface-temperature-optimum-interpolation)...

To build an intuitive understanding of sea surface temperature distribution, we consider the latest [sea surface temperature dataset in July, 2024](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/202407/) as an example.

```python
import numpy as np
import netCDF4 as nc

dataset = nc.Dataset('2024-07/oisst-avhrr-v02r01.20240701.nc', 'r').variables

lat = dataset['lat'][:].data
lon = dataset['lon'][:].data
sst = dataset['sst'][0, 0, :, :].data
sst[sst == -999] = np.nan
```

<br>

## Draw SST

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rc('font', family = 'Helvetica')

levels = np.linspace(0, 34, 18)
jet = ['blue', '#007FFF', 'cyan','#7FFF7F', 'yellow', '#FF7F00', 'red', '#7F0000']
cm = LinearSegmentedColormap.from_list('my_jet', jet, N = len(levels))
fig = plt.figure(figsize = (7, 4))
plt.contourf(dataset['lon'][:].data, dataset['lat'][:].data, sst,
             levels = levels, cmap = cm, extend = 'both')
plt.xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
plt.yticks(np.arange(-60, 90, 30), ['60S', '30S', 'EQ', '30N', '60N'])
cbar = plt.colorbar(fraction = 0.022)
plt.show()
fig.savefig('sst_2407_1.png', bbox_inches = 'tight')
```

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/sst_2407_1.png" width="550" />
</p>

<p align = "center">
<b>(a)</b> On July 1, 2024.
</p>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/sst_2407_2.png" width="550" />
</p>

<p align = "center">
<b>(b)</b> On July 2, 2024.
</p>

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/sst_2407_3.png" width="550" />
</p>

<p align = "center">
<b>(c)</b> On July 3, 2024.
</p>

<p align = "center">
<b>Figure 1.</b> Sea surface temperature data.
</p>

<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> and Fuqiang Liu on July 29, 2024.)</p>
