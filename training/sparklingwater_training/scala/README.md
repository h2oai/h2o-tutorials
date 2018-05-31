# Sparkling Water Scala Users - Launch Instructions

This document provides launch instructions for the [Chicago Crime Demo](ChicagoCrimeDemo.scala)

You should have already successfully [installed Sparkling Water](../README.md) before following the launch instructions.


## Launch Instructions

1. In terminal, export the following environment variables:

  ```bash
  export SPARK_HOME="/path/to/spark/installation" 
  # To launch a local Spark cluster with 3 worker nodes with 2 cores and 1g per node.
  export MASTER="local[*]" 
  ```
2. Go to your sparkling-water-2.3.5 folder and from your terminal, run:

  ```bash
  bin/sparkling-shell --conf "spark.executor.memory=1g"
  ```
  Now you should see the Spark shell.

3. Enter the [Chicago Crime Demo](ChicagoCrimeDemo.scala) scala code into the Spark shell.

  