# H2O Rains with Databricks Cloud for Spark


## Requirements

  * Databricks account (14 day free trial at http://www.databricks.com/)
  * Your AWS account
  * Sparkling Water jar (Maven coordinates `ai.h2o:sparkling-water-examples_2.10:1.5.6`
  
## Provided artifacts

  * [Databricks package](H2OWorld-Demo-Example.dbc)
  * [Source code of example](H2OWorld-Demo-Example.scala)
  * [Presentation slides](H2OWorld-Demo-Example.pdf)

## Import and Run

  * Login to your Databricks account
  * Select `Workspace > Import` and load provided [Databricks package](H2OWorld-Demo-Example.dbc)
  * Create a cluster
  * Create a new library via importing Maven coordinates `ai.h2o:sparkling-water-examples_2.10:1.5.6` and attach it to cluster 
  * Execute cell step-by-step or run them all
