# Introductory H2O Machine Learning Tutorial
# Prepared for H2O Open Chicago 2016: http://open.h2o.ai/chicago.html


# First step is to download & install the h2o R library
# The latest version is always here: http://www.h2o.ai/download/h2o/r

# Load the H2O library and start up the H2O cluster locally on your machine
library(h2o)
h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O



# Next we will import a cleaned up version of the Lending Club "Bad Loans" dataset
# The purpose here is to predict whether a loan will be bad (not repaid to the lender)
# The response column, bad_loan, is 1 if the loan was bad, and 0 otherwise

# Import the data
loan_csv <- "/Volumes/H2OTOUR/loan.csv"  # modify this for your machine
# Alternatively, you can import the data directly from a URL
#loan_csv <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
data <- h2o.importFile(loan_csv)  # 163,987 rows x 15 columns
dim(data)
# [1] 163987     15

# Since we want to train a binary classification model, 
# we must ensure that the response is coded as a factor
# If the response is 0/1, H2O will assume it's numeric,
# which means that H2O will train a regression model instead
data$bad_loan <- as.factor(data$bad_loan)  #encode the binary repsonse as a factor
h2o.levels(data$bad_loan)  #optional: after encoding, this shows the two factor levels, '0' and '1'
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
nrow(train)  # 114908
nrow(valid) # 24498
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
# 3. Gradient Boosting Machine (GBM)
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
h2o.auc(glm_perf1)  #0.677449084114
h2o.auc(glm_perf2)  #0.677675858276



# Compare test AUC to the training AUC and validation AUC
h2o.auc(glm_fit2, train = TRUE)  #0.674306164325 
h2o.auc(glm_fit2, valid = TRUE)  #0.675512216705
glm_fit2@model$validation_metrics  #0.675512216705





# 2. Random Forest
# H2O's Random Forest (RF) implements a distributed version of the standard 
# Random Forest algorithm and variable importance measures.
# First we will train a basic Random Forest model with default parameters. 
# The Random Forest model will infer the response distribution from the response encoding. 
# A seed is required for reproducibility.
rf_fit1 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit1",
                            seed = 1)

# Next we will increase the number of trees used in the forest by setting `ntrees = 100`.  
# The default number of trees in an H2O Random Forest is 50, so this RF will be twice as 
# big as the default.  Usually increasing the number of trees in a RF will increase 
# performance as well.  Unlike Gradient Boosting Machines (GBMs), Random Forests are fairly 
# resistant (although not free from) overfitting.
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
h2o.auc(rf_perf1)  # 0.662266990734
h2o.auc(rf_perf2)  # 0.66525468051


# Cross-validate performance
# Rather than using held-out test set to evaluate model performance, a user may wish 
# to estimate model performance using cross-validation. Using the RF algorithm 
# (with default model parameters) as an example, we demonstrate how to perform k-fold 
# cross-validation using H2O. No custom code or loops are required, you simply specify 
# the number of desired folds in the nfolds argument.
# Since we are not going to use a test set here, we can use the original (full) dataset, 
# which we called data rather than the subsampled `train` dataset. Note that this will 
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
h2o.auc(rf_fit3, xval = TRUE)  # 0.661201482614




# 3. Gradient Boosting Machine
# H2O's Gradient Boosting Machine (GBM) offers a Stochastic GBM, which can 
# increase performance quite a bit compared to the original GBM implementation.

# Now we will train a basic GBM model
# The GBM model will infer the response distribution from the response encoding if not specified 
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
# however early stopping is not enabled by default (with the exception of Deep Learning).  
# There are several parameters that should be used to control early stopping.  The three that are 
# common to all the algorithms are: `stopping_rounds`, `stopping_metric` and `stopping_tolerance`.  
# The stopping metric is the metric by which you'd like to measure performance, and so we will choose 
# AUC here.  The `score_tree_interval` is a parameter specific to the Random Forest model and the GBM.  
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
h2o.auc(gbm_perf1)  # 0.682765594191
h2o.auc(gbm_perf2)  # 0.671854616713
h2o.auc(gbm_perf3)  # 0.68309902855


# To examine the scoring history, use the `scoring_history` method on a trained model.  
# If `score_tree_interval` is not specified, it will score at various intervals, as we can 
# see for `h2o.scoreHistory()` below.  However, regular 5-tree intervals are used 
# for `h2o.scoreHistory()`.  
# The `gbm_fit2` was trained only using a training set (no validation set), so the scoring 
# history is calculated for training set performance metrics only.

h2o.scoreHistory(gbm_fit2)


# When early stopping is used, we see that training stopped at 105 trees instead of the full 500.  
# Since we used a validation set in `gbm_fit3`, both training and validation performance metrics 
# are stored in the scoring history object.  Take a look at the validation AUC to observe that the 
# correct stopping tolerance was enforced.

h2o.scoreHistory(gbm_fit3)




# Look at scoring history for third GBM model
plot(gbm_fit3, 
     timestep = "number_of_trees", 
     metric = "AUC")
plot(gbm_fit3, 
     timestep = "number_of_trees", 
     metric = "logloss")




# 4. Deep Learning
# H2O's Deep Learning algorithm is a multilayer feed-forward artificial neural network.  
# It can also be used to train an autoencoder. In this example we will train 
# a standard supervised prediction model.

# Train a default DL
# First we will train a basic DL model with default parameters. The DL model will infer the response 
# distribution from the response encoding if it is not specified explicitly through the `distribution` 
# argument.  H2O's DL will not be reproducible if it is run on more than a single core, so in this example, 
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
# you have to be careful not to overfit your model to your training data.  To automatically find the optimal number of epochs, 
# you must use H2O's early stopping functionality.  Unlike the rest of the H2O algorithms, H2O's DL will 
# use early stopping by default, so for comparison we will first turn off early stopping.  We do this in the next example 
# by setting `stopping_rounds=0`.
dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            #validation_frame = valid,  #only used if stopping_rounds > 0
                            epochs = 20,
                            hidden= c(10,10),
                            stopping_rounds = 0,  # disable early stopping
                            seed = 1)

# Train a DL with early stopping
# This example will use the same model parameters as `dl_fit2`. This time, we will turn on 
# early stopping and specify the stopping criterion.  We will also pass a validation set, as is
# recommended for early stopping.
dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            validation_frame = valid,  #in DL, early stopping is on by default
                            epochs = 20,
                            hidden = c(10,10),
                            score_interval = 1,           #used for early stopping
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
h2o.auc(dl_perf1)  # 0.6774335
h2o.auc(dl_perf2)  # 0.678446
h2o.auc(dl_perf3)  # 0.6770498

# Scoring history
h2o.scoreHistory(dl_fit3)
# Scoring History: 
#   timestamp   duration  training_speed   epochs
# 1 2016-05-03 10:33:29  0.000 sec                  0.00000
# 2 2016-05-03 10:33:29  0.347 sec 424697 rows/sec  0.86851
# 3 2016-05-03 10:33:30  1.356 sec 601925 rows/sec  6.09185
# 4 2016-05-03 10:33:31  2.348 sec 717617 rows/sec 13.05168
# 5 2016-05-03 10:33:32  3.281 sec 777538 rows/sec 20.00783
# 6 2016-05-03 10:33:32  3.345 sec 777275 rows/sec 20.00783
# iterations        samples training_MSE training_r2
# 1          0       0.000000                         
# 2          1   99804.000000      0.14402     0.03691
# 3          7  700039.000000      0.14157     0.05333
# 4         15 1499821.000000      0.14033     0.06159
# 5         23 2299180.000000      0.14079     0.05853
# 6         23 2299180.000000      0.14157     0.05333
# training_logloss training_AUC training_lift
# 1                                            
# 2          0.45930      0.66685       2.20727
# 3          0.45220      0.68133       2.59354
# 4          0.44710      0.67993       2.70390
# 5          0.45100      0.68192       2.81426
# 6          0.45220      0.68133       2.59354
# training_classification_error validation_MSE validation_r2
# 1                                                           
# 2                       0.36145        0.14682       0.03426
# 3                       0.33647        0.14500       0.04619
# 4                       0.37126        0.14411       0.05204
# 5                       0.32868        0.14474       0.04793
# 6                       0.33647        0.14500       0.04619
# validation_logloss validation_AUC validation_lift
# 1                                                  
# 2            0.46692        0.66582         2.53209
# 3            0.46256        0.67354         2.64124
# 4            0.45789        0.66986         2.44478
# 5            0.46292        0.67117         2.70672
# 6            0.46256        0.67354         2.64124
# validation_classification_error
# 1                                
# 2                         0.37197
# 3                         0.34716
# 4                         0.34385
# 5                         0.36544
# 6                         0.34716


# Look at scoring history for third DL model
plot(dl_fit3, 
     timestep = "epochs", 
     metric = "AUC")





# 5. Naive Bayes model
# The Naive Bayes (NB) algorithm does not usually beat an algorithm like a Random Forest 
# or GBM, however it is still a popular algorithm, especially in the text domain (when your 
# input is text encoded as "Bag of Words", for example).  The Naive Bayes algorithm is for 
# binary or multiclass classification problems only, not regression.  Therefore, your response 
# must be a factor instead of a numeric.

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

