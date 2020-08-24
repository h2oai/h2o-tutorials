# Best Practices - Forecasting

This section shows best practices in how to perform forecasting with H2O-3 and Sparkling Water.

Note: We recommend using Sparkling Water for forecasting use cases since you can utilize Spark's windowing and lag functionality.  With this you can add features like Demand from the previous day, Demand from the previous week, etc.

## Contents

* [Forecasting Tutorial](Forecasting-Tutorial.ipynb): walks through an example of forecasting product demand
* [Spend Analysis](spend_analysis.ipynb): walks through an example of forecasting customer spend and finding anomalies

## Running the Notebooks

This notebook uses Sparkling Water.  To start the Sparkling Water notebook, use the command: `PYSPARK_DRIVER_PYTHON="ipython" PYSPARK_DRIVER_PYTHON_OPTS="notebook" bin/pysparkling`