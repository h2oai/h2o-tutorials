# Best Practices - Missing Value Imputation

This section shows best practices in how to impute missing values.

**Goal:** learn about recommended practices for missing value imputation.

## The Data
This exploration of H2O will use a version of the Lending Club Loan Data that can be found on [Kaggle](https://www.kaggle.com/wendykan/lending-club-loan-data).  The data used can be found here: <https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/lending-club/LoanStats3a.csv>

## Workflow
1. Import and clean data
2. Train baseline model
3. Convert text features to word-embeddings
4. Perform target encoding on categorical features
5. Add anomaly score
6. Evaluate improvement to model
7. Analyze final model