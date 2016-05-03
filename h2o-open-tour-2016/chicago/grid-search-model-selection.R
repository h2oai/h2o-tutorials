# H2O Machine Learning Tutorial: Grid Search and Model Selection
# Prepared for H2O Open Chicago 2016: http://open.h2o.ai/chicago.html


# First step is to download & install the h2o R library
# The latest version is always here: http://www.h2o.ai/download/h2o/r

# Load the H2O library and start up the H2O cluter locally on your machine
library(h2o)
h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O



# Next we will import a cleaned up version of the Lending Club "Bad Loans" dataset
# The purpose here is to predict whether a loan will be bad (not repaid to the lender)
# The response column, bad_loan, is 1 if the loan was bad, and 0 otherwise

# Import the data
loan_csv <- "/Users/me/h2oai/code/demos/lending_club/loan.csv"  # modify this for your machine
# Alternatively, you can import the data directly from a URL
#loan_csv <- "https://s3.amazonaws.com/h2o-datasets/loan.csv"
data <- h2o.importFile(loan_csv)  # 163,994 rows x 15 columns
dim(data)
# [1] 163994     15

# Since we want to train a binary classification model, 
# we must ensure that the response is coded as a factor
# If the response is 0/1, H2O will assume it's numeric,
# which means that H2O will train a regression model instead
data$bad_loan <- as.factor(data$bad_loan)  #encode the binary repsonse as a factor
h2o.levels(data$bad_loan)  #optoional: after encoding, this shows the two factor levels, '0' and '1'
# [1] "0" "1"

# Partition the data into training, validation and test sets
splits <- h2o.splitFrame(data = data, 
                         ratios = c(0.7, 0.15),  #partition data into 70%, 15%, 15% chunks
                         seed = 1)  #setting a seed will guarantee reproducibility
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

# Take a look at the size of each partition
# Notice that h2o.splitFrame uses approximate splitting not exact splitting (for efficiency)
# so these are not exactly 70%, 15% and 15% of the total rows
nrow(train)  # 114914
nrow(valid) # 24499
nrow(test)  # 24581

# Identify response and predictor variables
y <- "bad_loan"
x <- setdiff(names(data), c(y, "int_rate"))  #remove the interest rate column because it's correlated with the outcome
print(x)
# [1] "loan_amnt"             "term"                 
# [3] "emp_length"            "home_ownership"       
# [5] "annual_inc"            "verification_status"  
# [7] "purpose"               "addr_state"           
# [9] "dti"                   "delinq_2yrs"          
# [11] "revol_util"            "total_acc"            
# [13] "longest_credit_length"


# Now that we have prepared the data, we can train some models
# Rather than training models manually one-by-one, we will make
# use of the h2o.grid function to train a bunch of models at once


# Cartesian Grid Search
# By default, h2o.grid will train a Cartesian
# grid search -- all models in the specified grid 


# GBM hyperparamters
gbm_params1 <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 5, 9),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2, 0.5, 1.0))

# Train and validate a grid of GBMs
gbm_grid1 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid1",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params1)

# Get the grid results, sorted by AUC
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf1)
# H2O Grid Details
# ================
#   
# Grid ID: gbm_grid1 
# Used hyper parameters: 
# - sample_rate 
# -  max_depth 
# -  learn_rate 
# -  col_sample_rate 
# Number of models: 36 
# Number of failed models: 0 
# 
# Hyper-Parameter Search Summary: ordered by decreasing auc
# sample_rate max_depth learn_rate col_sample_rate          model_ids               auc
# 1           1         3        0.1             0.5 gbm_grid1_model_19 0.685421713191415
# 2           1         5        0.1             0.2  gbm_grid1_model_9 0.684999851240248
# 3         0.8         3        0.1             0.5 gbm_grid1_model_18 0.684996004193172
# 4           1         5        0.1             0.5 gbm_grid1_model_21  0.68475480088044
# 5           1         3        0.1               1 gbm_grid1_model_31 0.684695040872968
# 
# ---
#   sample_rate max_depth learn_rate col_sample_rate          model_ids               auc
# 31           1         5       0.01               1 gbm_grid1_model_27 0.668421770188742
# 32           1         3       0.01             0.2  gbm_grid1_model_1 0.668216786878714
# 33           1         3       0.01             0.5 gbm_grid1_model_13 0.665042449930355
# 34         0.8         3       0.01             0.5 gbm_grid1_model_12 0.664959640334883
# 35         0.8         3       0.01               1 gbm_grid1_model_24 0.662830457801871
# 36           1         3       0.01               1 gbm_grid1_model_25 0.661749203263473



# Random Grid Search
# This is set to run fairly quickly, increase max_runtime_secs 
# or max_models to cover more of the hyperparameter space.
# Also, you can expand the hyperparameter space of each of the 
# algorithms by modifying the hyper param code below.


# GBM hyperparamters
gbm_params2 <- list(learn_rate = seq(0.01, 0.1, 0.01),
                    max_depth = seq(2, 10, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria2 <- list(strategy = "RandomDiscrete", 
                         max_models = 36)

# Train and validate a grid of GBMs
gbm_grid2 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid2",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params2,
                      search_criteria = search_criteria2)

gbm_gridperf2 <- h2o.getGrid(grid_id = "gbm_grid2", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf2)
# H2O Grid Details
# ================
#   
# Grid ID: gbm_grid2 
# Used hyper parameters: 
# -  sample_rate 
# -  max_depth 
# -  learn_rate 
# -  col_sample_rate 
# Number of models: 36 
# Number of failed models: 0 
# 
# Hyper-Parameter Search Summary: ordered by decreasing auc
# sample_rate max_depth learn_rate col_sample_rate          model_ids               auc
# 1         0.9         3       0.09             0.7 gbm_grid2_model_12 0.686216400866229
# 2           1         4       0.07             0.9  gbm_grid2_model_5 0.685706972276958
# 3         0.7         5       0.08             0.6  gbm_grid2_model_2 0.684295182287367
# 4         0.5         7       0.06             0.4 gbm_grid2_model_23 0.683939570192207
# 5         0.8         7       0.04             0.4 gbm_grid2_model_16 0.683870143525371
# 
# ---
#   sample_rate max_depth learn_rate col_sample_rate          model_ids               auc
# 31         0.8        10       0.06             0.5 gbm_grid2_model_19 0.674279253550808
# 32         0.8         9       0.09             0.9 gbm_grid2_model_26 0.673252315393779
# 33         0.5         3       0.02             0.3 gbm_grid2_model_10 0.672512930382942
# 34         0.5         9       0.09             0.3 gbm_grid2_model_11 0.672470062508521
# 35         0.7         5       0.01             0.8 gbm_grid2_model_31  0.67178319023517
# 36           1        10        0.1             0.8  gbm_grid2_model_9  0.66928075695065



# Looks like learn_rate = 0.1 does well here, which was the biggest 
# learn_rate in our previous search, so maybe we want to 
# add some models to our grid search with a higher learn_rate.
# We can add models to the same grid, by re-using the same model_id.
# Let's add as many new models as we can train in 60 seconds by setting
# max_runtime_secs = 60 in search_criteria.

gbm_params <- list(learn_rate = seq(0.1, 0.3, 0.01),  #updated
                   max_depth = seq(2, 10, 1),
                   sample_rate = seq(0.9, 1.0, 0.05),  #updated
                   col_sample_rate = seq(0.1, 1.0, 0.1))
search_criteria <- list(strategy = "RandomDiscrete", 
                         max_runtime_secs = 60)  #updated


gbm_grid <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid2",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params,
                      search_criteria = search_criteria2)

gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid2", 
                             sort_by = "auc", 
                             decreasing = TRUE)
print(gbm_gridperf)
# H2O Grid Details
# ================
#   
# Grid ID: gbm_grid2 
# Used hyper parameters: 
# -  sample_rate 
# -  max_depth 
# -  learn_rate 
# -  col_sample_rate 
# Number of models: 72 
# Number of failed models: 0 
# 
# Hyper-Parameter Search Summary: ordered by decreasing auc
# sample_rate max_depth learn_rate col_sample_rate          model_ids               auc
# 1           1         4        0.1             0.4 gbm_grid2_model_47   0.6866268688012
# 2         0.9         3       0.09             0.7 gbm_grid2_model_12 0.686216400866229
# 3         0.9         3       0.23             0.2 gbm_grid2_model_49 0.685769102632132
# 4           1         4       0.07             0.9  gbm_grid2_model_5 0.685706972276958
# 5        0.95         4       0.12             0.3 gbm_grid2_model_69 0.685149079613059
# 
# ---
#   sample_rate max_depth learn_rate col_sample_rate          model_ids               auc
# 67           1         9       0.22             0.8 gbm_grid2_model_55 0.651326010648147
# 68           1        10       0.23             0.2 gbm_grid2_model_68 0.651131266150378
# 69           1         9       0.23             0.4 gbm_grid2_model_64 0.649976105805347
# 70         0.9        10        0.2             0.4 gbm_grid2_model_59 0.649033295919983
# 71        0.95        10       0.21             0.5 gbm_grid2_model_65 0.648789760403075
# 72        0.95         9       0.24             0.3 gbm_grid2_model_63 0.648310176398551


# Grab the model_id for the top GBM model, chosen by validation AUC
best_gbm_model_id <- gbm_gridperf@model_ids[[1]]
best_gbm <- h2o.getModel(best_gbm_model_id)

# Now let's evaluate the model performance on a test set
# so we get an honest estimate of top model performance
best_gbm_perf <- h2o.performance(model = best_gbm, 
                                 newdata = test)
h2o.auc(best_gbm_perf)  # 0.6840662

# As we can see, this is slighly less than the AUC on the validation set
# of the top model, but this is a more honest estimate of performance.  
# The validation set was used to select the best model, but should not 
# be used to also evaluate the best model's performance.




# Next we will explore some of the deep learning
# hyperparameters in a random grid search


# Deeplearning hyperparamters
activation_opt <- c("Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
hyper_params <- list(activation = activation_opt,
                     l1 = l1_opt,
                     l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_runtime_secs = 120)


dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = train,
                    validation_frame = valid,
                    seed = 1,
                    hidden = c(10,10),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)

dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "auc", 
                           decreasing = TRUE)
print(dl_gridperf)
# H2O Grid Details
# ================
#   
# Grid ID: dl_grid 
# Used hyper parameters: 
# -  l2 
# -  l1 
# -  activation 
# Number of models: 53 
# Number of failed models: 0 
# 
# Hyper-Parameter Search Summary: ordered by decreasing auc
# l2    l1 activation        model_ids               auc
# 1 1e-04     0     Maxout dl_grid_model_21 0.682435320295004
# 2 0.001 1e-05  Rectifier dl_grid_model_48 0.681218073187401
# 3     0 1e-04     Maxout dl_grid_model_17 0.679177558005678
# 4 1e-05 1e-04  Rectifier dl_grid_model_31 0.677314159563525
# 5 0.001 0.001  Rectifier dl_grid_model_43 0.676308690762695
# 
# ---
#   l2    l1           activation        model_ids               auc
# 48     0   0.1 RectifierWithDropout dl_grid_model_22 0.499974988744935
# 49  0.01   0.1    MaxoutWithDropout dl_grid_model_23  0.49994997748987
# 50     0   0.1            Rectifier  dl_grid_model_3 0.499835546911143
# 51     0   0.1    MaxoutWithDropout  dl_grid_model_2 0.499433764802004
# 52 1e-05   0.1               Maxout dl_grid_model_38 0.499116982816559
# 53   0.1 0.001    MaxoutWithDropout dl_grid_model_34 0.497202019776002

# Note that that these results are not reproducible since we are not using a single core H2O cluster
# H2O's DL requires a single core to be used in order to get reproducible results

# Grab the model_id for the top DL model, chosen by validation AUC
best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)

# Now let's evaluate the model performance on a test set
# so we get an honest estimate of top model performance
best_dl_perf <- h2o.performance(model = best_dl, 
                                newdata = test)
h2o.auc(best_dl_perf)  # 0.6778661 



