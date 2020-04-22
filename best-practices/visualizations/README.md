# Best Practices - Visualizing Large Data

This section shows best practices for visualizing large data.

**Goal:** show an example of visualizing a large dataset.

## The Data
The data is the public airlines data: <https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/airlines_all.05p.csv>

## Workflow
1. Use H2O-3 to plot histograms.
2. Use H2O-3 to calculate custom aggregations and plot with matplotlib.
3. Use H2O-3 to reduce the size of the data using the [H2O Aggregator](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/aggregator.html) function and graph the aggregated data.