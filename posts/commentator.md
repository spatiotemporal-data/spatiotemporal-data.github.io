---
layout: default
---


## JTL - Transit Lab Seminar

### April 26, 2024

Today, **Eric Plosky** discussed what they have done about "Panel on data standards" at [MobilityData](https://mobilitydata.org/). The team tried to develop and manage open data specifications around the world and advance the data standardization. Overall, this is an open mobility data standard/format as used in [the mobility database](https://mobilitydatabase.org/):

- **Transit**: General Transit Feed Specification (GTFS, see [gtfs-realtime-bindings](https://github.com/MobilityData/gtfs-realtime-bindings)). For example, the transit operations system data include:
  - Vehicle location data
  - Passenger activity data
  - Fare transactions data
  - Bus schedule data
- **Bikeshare**: General Bikeshare Feed Specification (GBFS, see [gbfs](https://github.com/MobilityData/gbfs)). The currently available datasets include:
  - [New York City Citi Bike data](https://citibikenyc.com/system-data)
  - [Chicago Divvy data](https://divvybikes.com/system-data)
  - [Washington D.C. Capital Bikeshare data](https://capitalbikeshare.com/system-data)
  - [Boston Bluebike data](https://bluebikes.com/system-data)
  - [Montreal BIXI data](https://bixi.com/en/open-data/)
  - [Bike Share Toronto ridership data](https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/)
  - [Vancouver Mobi Bike Share data](https://www.mobibikes.ca/en/system-data)
  - [Austin MetroBike Trip data](https://data.austintexas.gov/Transportation-and-Mobility/Austin-MetroBike-Trips/tyfh-5r8s/about_data) (or see [shared micromobility vehicle trips](https://data.austintexas.gov/Transportation-and-Mobility/Shared-Micromobility-Vehicle-Trips/7d8e-dm7r/about_data) & [data visualization platform](https://public.ridereport.com/austin))
  - [Los Angeles Metro Bike Share data](https://bikeshare.metro.net/about/data/) (or see [the data portal](https://data.lacity.org/dataset/Metro-Bike-Share-Trip-Data/sii9-rjps/data))
  - [San Francisco Bay Area Bike Share data](https://www.lyft.com/bikes/bay-wheels/system-data)


<br>

## SRL Meeting

### April 23, 2024 (Dingyi Zhuang)

In the past decades, machine learning acted as an important data-driven computing paradigm for learning from data. However, real-world data usually demonstrate complicated data behaviors (e.g., uncertainty) and underlying patterns (e.g., biases in the data collection process). In this study, we hope to connect machine learning with real-world data problems in transport demand modeling and in the meanwhile highlight the importance of uncertainty quantification in the prediction task. There are some critical challenges that stem from the following perspectives:

- **Robustness**: Solving the unexpected data variations that produced by special events or extreme cases.
- **Reliability**: Handling the the model's generalization ability to different conditions.
- **Fairness**: Reducing data and model biases over different groups of the collected dataset.

Due to the randomness of transport demand data, uncertainty quantification is of great significance in the deep learning based prediction methods, even not mentioning the possible improvement of prediction accuracy. Introducing probabilistic assumptions when modeling the transport demand allows one to produce both point estimates and interval estimates. One recent demand prediction model is the probabilistic spatiotemporal graph neural network (Prob STGNN, see [Zhuang et al.'22](https://dl.acm.org/doi/pdf/10.1145/3534678.3539093)).

In terms of fairness, one meaningful task is how to reduce data and model biases in the machine learning algorithms. As shown in Figure 1, characterizing the fairness on the data is simply implemented by minimizing the differences of model variables on the splitting subsets. The shift of learning mechanisms in such case could benefit a lot of transport modeling applications.

<p align="center">
<img align="middle" src="https://spatiotemporal-data.github.io/images/fairness_explained.png" alt="drawing" width="600">
</p>

<p align="center"><b>Figure 1</b>: Illustration of fairness modeling in machine learning with the dataset <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\boldsymbol{y}"/>. The dataset is grouped into <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;n"/> subsets.</p>

<br>

**References**

- Curtis Northcutt, Lu Jiang, Isaac Chuang (2021). [Confident Learning: Estimating Uncertainty in Dataset Labels](https://doi.org/10.1613/jair.1.12125). Journal of Artificial Intelligence Research, 70: 1373-1411.

<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 24, 2024.)</p>
