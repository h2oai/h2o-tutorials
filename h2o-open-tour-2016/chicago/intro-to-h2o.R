# Introductory H2O Machine Learning Tutorial
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
# We will start by training a single model from each of the H2O supervised algos:
# 1. Generalized Linear Model (GLM)
# 2. Random Forest (RF)
# 3. Gradient Boosting Machine (RF)
# 4. Deep Learning (DL)
# 5. Naive Bayes (NB)


# 1. Let's start with a basic binomial Generalized Linear Model
# By default, h2o.glm uses a regularized, elastic net model
glm_fit1 <- h2o.glm(x = x, 
                    y = y, 
                    training_frame = train,
                    model_id = "glm_fit1",
                    family = "binomial")  #similar to R's glm, h2o.glm has the family argument

# Next we will do some automatic tuning by passing in a validation frame and setting 
# `lambda_search = True`.  Since we are training a GLM with regularization, we should 
# try to find the right amount of regularization (to avoid overfitting).  The model 
# parameter, `lambda`, controls the amount of regularization in a GLM model and we can 
# find the optimal value for `lambda` automatically by setting `lambda_search = TRUE` 
# and passing in a validation frame (which is used to evaluate model performance using a 
# particular value of lambda).
glm_fit2 <- h2o.glm(x = x, 
                    y = y, 
                    training_frame = train,
                    model_id = "glm_fit2",
                    validation_frame = valid,
                    family = "binomial",
                    lambda_search = TRUE)

# Let's compare the performance of the two GLMs
glm_perf1 <- h2o.performance(model = glm_fit1,
                             newdata = test)
glm_perf2 <- h2o.performance(model = glm_fit2,
                             newdata = test)

# Print model performance
glm_perf1
glm_perf2

# Instead of printing the entire model performance metrics object, 
# it is probably easier to print just the metric that you are interested in comparing.
# Retreive test set AUC
h2o.auc(glm_perf1)  #0.673463297871
h2o.auc(glm_perf2)  #0.673426207356



# Compare test set AUC to validation set AUC
glm_fit2@model$validation_metrics  #0.6734262073556496
h2o.auc(glm_fit2, valid = TRUE)  #you can also use h2o.auc directly on a model to retreive AUC

# This shows that the AUC evaluated on the validation set (0.673) is slightly higher 
# than the AUC evaluated on the held-out test set (0.671).





# 2. Random Forest
# H2O's Random Forest (RF) is implements a distributed version of the standard 
# Random Forest algorithm and variable importance measures.
# First we will train a basic Random Forest model with default parameters. 
# Random Forest will infer the response distribution from the response encoding. 
# A seed is required for reproducibility.
rf_fit1 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit1",
                            seed = 1)

# Next we will increase the number of trees used in the forest by setting `ntrees = 100`.  
# The default number of trees in an H2O Random Forest is 50, so this RF will be twice as 
# big as the default.  Usually increasing the number of trees in an RF will increase 
# performance as well.  Unlike Gradient Boosting Machines (GBMs), Random Forests are fairly 
# resistant (although not free from) overfitting by increasing the number of trees.  
# See the GBM example below for additional guidance on preventing overfitting using H2O's 
# early stopping functionality.
rf_fit2 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit2",
                            #validation_frame = valid,  #only used if stopping_rounds > 0
                            ntrees = 100,
                            seed = 1)

# Let's compare the performance of the two RFs
rf_perf1 <- h2o.performance(model = rf_fit1,
                            newdata = test)
rf_perf2 <- h2o.performance(model = rf_fit2,
                            newdata = test)

# Print model performance
rf_perf1
rf_perf2

# Retreive test set AUC
h2o.auc(rf_perf1)  # 0.665035
h2o.auc(rf_perf2)  # 0.6718425


# Cross-validate performance
# Rather than using held-out test set to evaluate model performance, a user may wish 
# to estimate model performance using cross-validation. Using the RF algorithm 
# (with default model parameters) as an example, we demonstrate how to perform k-fold 
# cross-validation using H2O. No custom code or loops are required, you simply specify 
# the number of desired folds in the nfolds argument.
# Since we are not going to use a test set here, we can use the original (full) dataset, 
# which we called data rather than the subsampled train dataset. Note that this will 
# take approximately k (nfolds) times longer than training a single RF model, since it 
# will train k models in the cross-validation process (trained on n(k-1)/k rows), in 
# addition to the final model trained on the full training_frame dataset with n rows.

rf_fit3 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit3",
                            seed = 1,
                            nfolds = 5)

# To evaluate the cross-validated AUC, do the following:
h2o.auc(rf_fit3, xval = TRUE)  # 0.6614636




# 3. Gradient Boosting Machine
# H2O's Gradient Boosting Machine (GBM) offers a Stochastic GBM, which can 
# increase performance quite a bit compared to the original GBM implementation.

# Now we will train a basic GBM model
# GBM will infer the response distribution from the response encoding if not specified 
# explicitly through the `distribution` argument. A seed is required for reproducibility.
gbm_fit1 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit1",
                    seed = 1)

# Next we will increase the number of trees used in the GBM by setting `ntrees=500`.  
# The default number of trees in an H2O GBM is 50, so this GBM will trained using ten times 
# the default.  Increasing the number of trees in a GBM is one way to increase performance 
# of the model, however, you have to be careful not to overfit your model to the training data 
# by using too many trees.  To automatically find the optimal number of trees, you must use 
# H2O's early stopping functionality.  This example will not do that, however, the following 
# example will.
gbm_fit2 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit2",
                    #validation_frame = valid,  #only used if stopping_rounds > 0
                    ntrees = 500,
                    seed = 1)

# We will again set `ntrees = 500`, however, this time we will use early stopping in order to 
# prevent overfitting (from too many trees).  All of H2O's algorithms have early stopping available, 
# however, with the exception of Deep Learning, it is not enabled by default.  
# There are several parameters that should be used to control early stopping.  The three that are 
# generic to all the algorithms are: `stopping_rounds`, `stopping_metric` and `stopping_tolerance`.  
# The stopping metric is the metric by which you'd like to measure performance, and so we will choose 
# AUC here.  The `score_tree_interval` is a parameter specific to Random Forest and GBM.  
# Setting `score_tree_interval = 5` will score the model after every five trees.  The parameters we 
# have set below specify that the model will stop training after there have been three scoring intervals 
# where the AUC has not increased more than 0.0005.  Since we have specified a validation frame, 
# the stopping tolerance will be computed on validation AUC rather than training AUC. 
gbm_fit3 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit3",
                    validation_frame = valid,  #only used if stopping_rounds > 0
                    ntrees = 500,
                    score_tree_interval = 5,      #used for early stopping
                    stopping_rounds = 3,          #used for early stopping
                    stopping_metric = "AUC",      #used for early stopping
                    stopping_tolerance = 0.0005,  #used for early stopping
                    seed = 1)


# Let's compare the performance of the two GBMs
gbm_perf1 <- h2o.performance(model = gbm_fit1,
                             newdata = test)
gbm_perf2 <- h2o.performance(model = gbm_fit2,
                             newdata = test)
gbm_perf3 <- h2o.performance(model = gbm_fit3,
                             newdata = test)

# Print model performance
gbm_perf1
gbm_perf2
gbm_perf3

# Retreive test set AUC
h2o.auc(gbm_perf1)  # 0.6822778
h2o.auc(gbm_perf2)  # 0.6711076
h2o.auc(gbm_perf3)  # 0.6830188

# To examine the scoring history, use the `scoring_history` method on a trained model.  
# If `score_tree_interval` is not specified, it will score at various intervals, as we can 
# see for `h2o.scoreHistory()` below.  However, regular 5-tree intervals are used 
# for `h2o.scoreHistory()`.  
# The `gbm_fit2` was trained only using a training set (no validation set), so the scoring 
# history is calculated for training set performance metrics only.

h2o.scoreHistory(gbm_fit2)
# Scoring History: 
#   timestamp   duration number_of_trees training_MSE training_logloss
# 1 2016-05-03 05:40:39  0.002 sec               0      0.14864          0.47385
# 2 2016-05-03 05:40:39  0.058 sec               1      0.14713          0.46889
# 3 2016-05-03 05:40:39  0.118 sec               2      0.14589          0.46490
# 4 2016-05-03 05:40:39  0.185 sec               3      0.14482          0.46156
# 5 2016-05-03 05:40:40  0.268 sec               4      0.14394          0.45880
# training_AUC training_lift training_classification_error
# 1      0.50000       1.00000                       0.81838
# 2      0.65925       2.38109                       0.33866
# 3      0.66289       3.00711                       0.34291
# 4      0.66845       3.04423                       0.34750
# 5      0.67210       3.01847                       0.34745
# 
# ---
#   timestamp   duration number_of_trees training_MSE training_logloss
# 18 2016-05-03 05:40:42  3.077 sec              17      0.13848          0.44197
# 19 2016-05-03 05:40:43  3.409 sec              18      0.13827          0.44137
# 20 2016-05-03 05:40:43  3.722 sec              19      0.13809          0.44077
# 21 2016-05-03 05:40:47  7.743 sec             127      0.13085          0.41852
# 22 2016-05-03 05:40:57 18.068 sec             376      0.12221          0.39360
# 23 2016-05-03 05:41:05 25.612 sec             500      0.11856          0.38323
# training_AUC training_lift training_classification_error
# 18      0.69287       3.34373                       0.31390
# 19      0.69397       3.36579                       0.30929
# 20      0.69471       3.32749                       0.30696
# 21      0.73800       4.24674                       0.25145
# 22      0.78650       5.00320                       0.21881
# 23      0.80459       5.23780                       0.19928


# When early stopping is used, we see that training stopped at 105 trees instead of the full 500.  
# Since we used a validation set in `gbm_fit3`, both training and validation performance metrics 
# are stored in the scoring history object.  Take a look at the validation AUC to observe that the 
# correct stopping tolerance was enforced.

h2o.scoreHistory(gbm_fit3)
# Scoring History: 
#   timestamp   duration number_of_trees training_MSE training_logloss
# 1 2016-05-03 05:41:43  0.002 sec               0      0.14864          0.47385
# 2 2016-05-03 05:41:44  0.215 sec               5      0.14318          0.45646
# 3 2016-05-03 05:41:44  0.517 sec              10      0.14052          0.44831
# 4 2016-05-03 05:41:44  0.933 sec              15      0.13898          0.44357
# 5 2016-05-03 05:41:45  1.430 sec              20      0.13790          0.44016
# training_AUC training_lift training_classification_error validation_MSE
# 1      0.50000       1.00000                       0.81838        0.15205
# 2      0.67348       3.03205                       0.37514        0.14748
# 3      0.68220       3.28925                       0.34638        0.14551
# 4      0.68936       3.33504                       0.33957        0.14454
# 5      0.69609       3.37537                       0.31112        0.14394
# validation_logloss validation_AUC validation_lift
# 1            0.48192        0.50000         1.00000
# 2            0.46723        0.65481         2.22307
# 3            0.46108        0.66127         2.46829
# 4            0.45805        0.66488         2.46661
# 5            0.45613        0.66825         2.59758
# validation_classification_error
# 1                         0.81301
# 2                         0.35957
# 3                         0.37687
# 4                         0.36006
# 5                         0.34454
# 
# ---
#   timestamp   duration number_of_trees training_MSE training_logloss
# 17 2016-05-03 05:41:55 11.300 sec              80      0.13286          0.42446
# 18 2016-05-03 05:41:56 12.534 sec              85      0.13257          0.42359
# 19 2016-05-03 05:41:57 13.757 sec              90      0.13238          0.42302
# 20 2016-05-03 05:41:58 15.093 sec              95      0.13211          0.42223
# 21 2016-05-03 05:42:00 16.444 sec             100      0.13191          0.42163
# 22 2016-05-03 05:42:01 17.815 sec             105      0.13176          0.42120
# training_AUC training_lift training_classification_error validation_MSE
# 17      0.72591       3.98341                       0.28458        0.14235
# 18      0.72766       4.04565                       0.28077        0.14233
# 19      0.72888       4.08395                       0.27202        0.14234
# 20      0.73056       4.11268                       0.27464        0.14232
# 21      0.73179       4.11747                       0.25854        0.14235
# 22      0.73260       4.14620                       0.26655        0.14232
# validation_logloss validation_AUC validation_lift
# 17            0.45083        0.67953         2.61941
# 18            0.45072        0.67994         2.66307
# 19            0.45074        0.68002         2.72855
# 20            0.45069        0.68009         2.61941
# 21            0.45074        0.67998         2.59758
# 22            0.45064        0.68025         2.57575
# validation_classification_error
# 17                         0.36875
# 18                         0.37140
# 19                         0.36377
# 20                         0.37014
# 21                         0.36132
# 22                         0.36614




# Look at scoring history for third GBM model
plot(gbm_fit3, 
     timestep = "number_of_trees", 
     metric = "AUC")
plot(gbm_fit3, 
     timestep = "number_of_trees", 
     metric = "logloss")




# 4. Deep Learning
# H2O's Deep Learning algorithm is a multilayer feed-forward artificial neural network.  
# It can also be used to train an autoencoder, however, in the example below we will train 
# a standard supervised prediction model.

# Train a default DL
# First we will train a basic DL model with default parameters. DL will infer the response 
# distribution from the response encoding if not specified explicitly through the `distribution` 
# argument.  H2O's DL will not be reproducbible if run on more than a single core, so in this example, 
# the performance metrics below may vary slightly from what you see on your machine.
# In H2O's DL, early stopping is enabled by default, so below, it will use the training set and 
# default stopping parameters to perform early stopping.
dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit1",
                            seed = 1)

# Train a DL with new architecture and more epochs.
# Next we will increase the number of epochs used in the GBM by setting `epochs=20` (the default is 10).  
# Increasing the number of epochs in a deep neural net may increase performance of the model, however, 
# you have to be careful not to overfit your model.  To automatically find the optimal number of epochs, 
# you must use H2O's early stopping functionality.  Unlike the rest of the H2O algorithms, H2O's DL will 
# use early by default, so we will first turn it off in the next example by setting `stopping_rounds=0`, 
#for comparison.
dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            #validation_frame = valid,  #only used if stopping_rounds > 0
                            stopping_rounds = 0,  # disable early stopping
                            hidden= c(10,10),
                            epochs = 20,
                            seed = 1)

# Train a DL with early stopping
# This example will use the same model parameters as `dl_fit2`, however, we will turn on early 
# stopping and specify the stopping criterion.  We will also pass a validation set, as is recommended 
# for early stopping.
dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            validation_frame = valid,  #in DL, early stopping is on by default
                            hidden = c(10,10),
                            epochs = 20,
                            stopping_rounds = 3,          #used for early stopping
                            stopping_metric = "AUC",      #used for early stopping
                            stopping_tolerance = 0.0005,  #used for early stopping
                            seed = 1)


# Let's compare the performance of the three DL models
dl_perf1 <- h2o.performance(model = dl_fit1,
                            newdata = test)
dl_perf2 <- h2o.performance(model = dl_fit2,
                            newdata = test)
dl_perf3 <- h2o.performance(model = dl_fit3,
                            newdata = test)

# Print model performance
dl_perf1
dl_perf2
dl_perf3

# Retreive test set AUC
h2o.auc(dl_perf1)  # 0.6813354
h2o.auc(dl_perf2)  # 0.6778485
h2o.auc(dl_perf3)  # 0.6790762

# Scoring history
h2o.scoreHistory(dl_fit3)
# Scoring History: 
#   timestamp   duration  training_speed   epochs iterations
# 1 2016-05-03 05:49:14  0.000 sec                  0.00000          0
# 2 2016-05-03 05:49:15  0.273 sec 504060 rows/sec  0.86851          1
# 3 2016-05-03 05:49:17  3.034 sec 792820 rows/sec 20.00783         23
# samples training_MSE training_r2 training_logloss training_AUC
# 1       0.000000                                                       
# 2   99804.000000      0.14209     0.04981          0.45144      0.66472
# 3 2299180.000000      0.14048     0.06058          0.44870      0.68433
# training_lift training_classification_error validation_MSE validation_r2
# 1                                                                         
# 2       2.42799                       0.32696        0.14427       0.05103
# 3       2.70390                       0.34153        0.14419       0.05154
# validation_logloss validation_AUC validation_lift
# 1                                                  
# 2            0.45766        0.66295         2.46661
# 3            0.45972        0.67482         2.51027
# validation_classification_error
# 1                                
# 2                         0.35471
# 3                         0.35801


# Look at scoring history for third DL model
plot(dl_fit3, 
     timestep = "epochs", 
     metric = "AUC")





# 5. Naive Bayes model
# The Naive Bayes (NB) algorithm does not usually beat an algorithm like a Random Forest 
# or GBM, however it is still a popular algorithm, especially in the text domain (when your 
# input is text encoded as "Bag of Words", for example).  The Naive Bayes algorithm is for 
# binary or multiclass classification problems only, not regression.  Therefore, your response 
# must be a factor instead of numeric.

# First we will train a basic NB model with default parameters. 
nb_fit1 <- h2o.naiveBayes(x = x,
                          y = y,
                          training_frame = train,
                          model_id = "nb_fit1")

# Train a NB model with Laplace Smoothing
# One of the few tunable model parameters for the Naive Bayes algorithm is the amount of Laplace 
# smoothing. The H2O Naive Bayes model will not use any Laplace smoothing by default.
nb_fit2 <- h2o.naiveBayes(x = x,
                          y = y,
                          training_frame = train,
                          model_id = "nb_fit2",
                          laplace = 6)

# Let's compare the performance of the two NB models
nb_perf1 <- h2o.performance(model = nb_fit1,
                            newdata = test)
nb_perf2 <- h2o.performance(model = nb_fit2,
                            newdata = test)

# Print model performance
nb_perf1
nb_perf2

# Retreive test set AUC
h2o.auc(nb_perf1)  # 0.6488014
h2o.auc(nb_perf2)  # 0.6490678

