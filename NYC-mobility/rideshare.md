---
layout: default
---

# Constructing Human Mobility Tensor on NYC Rideshare Trip Data

Among a large number of open human mobility datasets, [TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) might be one of the most classical sources for doing mobility research and data analysis. This open data includes yellow and green taxi trip records, For-Hire Vehicle (FHV) trip records, and High Volume For-Hire Vehicle (HVFHV) trip records stored in the `.parquet` format, ranging from 2009 to the latest date. The HVFHV trip records have TLC license numbers for Juno (`HV0002`), Uber (`HV0003`), Via (`HV0004`), and Lyft (`HV0005`), see [data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf) for details. In what follows, we use rideshare trip records to mention the HVFHV trip records instead.

## Rideshare Trip Records

The first procedure is downloading the rideshare trip records from [TLC trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) through selecting certain data files, e.g., `High Volume For-Hire Vehicle Trip Records` in April and May in 2024.



<br>

<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on August 7, 2024.)</p>
