# Best Practices - Forecasting

This section shows best practices in how to perform forecasting with H2O-3 and Sparkling Water.

Note: We recommend using Sparkling Water for forecasting use cases since you can utilize Spark's windowing and lag functionality.  With this you can add features like Demand from the previous day, Demand from the previous week, etc.

**Goal:** learn about recommended practices for forecasting.

## The Data
The forecasting demo uses teh public dataset from Kaggle: [Product Demand Data](https://www.kaggle.com/felixzhao/productdemandforecasting). 

## Workflow
1. Import and clean data
2. Format data for forecasting
3. Train baseline model
4. Use Spark and H2O-3 to add features for time-series use cases.
5. Train new model with added features.
6. Evaluate improvement to model
7. Analyze final model and residuals

## Running the Notebook

This notebook uses Sparkling Water.  To start the Sparkling Water notebook, use the command: `PYSPARK_DRIVER_PYTHON="ipython" PYSPARK_DRIVER_PYTHON_OPTS="notebook" bin/pysparkling`