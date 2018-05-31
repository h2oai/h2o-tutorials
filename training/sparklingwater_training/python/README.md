# Sparkling Water Python Users - Launch Instructions

This document provides launch instructions for the [Chicago Crime Demo](ChicagoCrimeDemo.ipynb)

You should have already successfully [installed Sparkling Water](../README.md) before following the launch instructions.


## Launch Instructions

1. Save the ChicagoCrimeDemo.ipynb in your sparkling water folder here: "~/Downloads/sparkling-water-2.3.5"

2. In terminal, export the following environment variables:

  ```bash
  export SPARK_HOME="/path/to/spark/installation" 
  # To launch a local Spark cluster with 3 worker nodes with 2 cores and 1g per node.
  export MASTER="local[*]" 
  ```
3. Go to your sparkling-water-2.3.5 folder and from your terminal, run:

  ```bash
  PYSPARK_DRIVER_PYTHON="ipython" PYSPARK_DRIVER_PYTHON_OPTS="notebook" bin/pysparkling
  ```
  A notebook should open on your browser.

4. Click on the [Chicago Crime Demo](ChicagoCrimeDemo.ipynb) file to open the notebook.

  