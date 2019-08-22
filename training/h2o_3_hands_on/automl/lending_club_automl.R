# Set up the H2O cluster

library(h2o)
h2o.init(max_mem_size = "6G")


# Import data and Manage Data Types

# This exploration of H2O will use a version of the Lending Club Loan Data that can be found on Kaggle. This data consists of 15 variables:
  
# |   | Column Name | Description | | --- | ----------- | ----------- | 
# | 1 | loan_amnt | Requested loan amount (US dollars) | 
# | 2 | term | Loan term length (months) | 
# | 3 | int_rate | Recommended interest rate | 
# | 4 | emp_length | Employment length (years) | 
# | 5 | home_ownership| Housing status | 
# | 6 | annual_inc | Annual income (US dollars) | 
# | 7 | purpose | Purpose for the loan | 
# | 8 | addr_state | State of residence | 
# | 9 | dti | Debt to income ratio | 
# | 10 | delinq_2yrs | Number of delinquencies in the past 2 years | 
# | 11 | revol_util | Percent of revolving credit line utilized | 
# | 12 | total_acc | Number of active accounts | 
# | 13 | bad_loan | Bad loan indicator | 
# | 14 | longest_credit_length | Age of oldest active account | 
# | 15 | verification_status | Income verification status |


# Use local data file or download from S3
data_path <- "/home/h2o/data/topics/automl/loan.csv"
if (!file.exists(data_path)) {
  data_path <- "https://s3-us-west-2.amazonaws.com/h2o-tutorials/data/topics/automl/loan.csv"
}
# Load data into H2O
train <- h2o.importFile(data_path)


# For classification, the response should be encoded as categorical (aka. "factor" or "enum"). 
# Let's take a look.
h2o.describe(train)

# Next, let's identify the response & predictor columns by saving them as `x` and `y`.  
# The `"int_rate"` column is correlated with the outcome (hence causing label leakage) 
# so we want to remove that from the set of our predictors.
y <- "bad_loan"
x <- setdiff(names(train), c(y, "int_rate"))


# Let's convert the response column to a categorical so that H2O will perform classification
train[,y] <- as.factor(train[,y])


# Train Models Using H2O's AutoML

# Run AutoML, stopping after 6 models.  
# The `max_models` argument specifies the number of individual (or "base") models, 
# and does not include the two ensemble models that are trained at the end.
aml <- h2o.automl(y = y, x = x,
                  training_frame = train,
                  max_models = 6,
                  seed = 1)



## Leaderboard

# Next, we will view the AutoML Leaderboard.  Since we did not specify a `leaderboard_frame` in the `h2o.automl()` 
# function for scoring and ranking the models, the AutoML leaderboard uses cross-validation metrics to rank the 
# models.  A default performance metric for each machine learning task (binary classification, multiclass 
# classification, regression) is specified internally and the leaderboard will be sorted by that metric unless 
# you change the default value of the `sort_metric` parameter.  In the case of binary classification, the default 
# leaderboard ranking metric is Area Under the ROC Curve (AUC).  

# The leader model is stored at `aml@leader` and the leaderboard is stored at `aml@leaderboard`.
lb <- aml@leaderboard

# Now we will view a snapshot of the top models.  Here we should see the two Stacked Ensembles 
# at or near the top of the leaderboard.  Stacked Ensembles can almost always outperform a single model.
print(lb)

# To view the entire leaderboard, specify the `n` argument of the `print.H2OFrame()` 
# function as the total number of rows:
print(lb, n = nrow(lb))