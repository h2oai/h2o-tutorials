# Custom Loss Metric Demo

This demo shows how to use H2O's custom loss metric.

**Goal:** Train a model with a custom loss metric.

## The Data
The data is found at: <https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv>

## Workflow
1. Train a GBM model in H2O
2. Write a script to calculate custom metric
3. Train a GBM model in H2O using custom metric as a [`custom_metric_func`](https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/dev/custom_functions.md)
4. Train a Grid of GBMs and choose model based on custom metric
