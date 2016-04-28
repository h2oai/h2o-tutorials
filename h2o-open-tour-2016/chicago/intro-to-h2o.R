# Introductory H2O Machine Learning Tutorial
# Prepared by for H2O Open Chicago 2016: http://open.h2o.ai/chicago.html


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
loan_csv <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
data <- h2o.importFile(loan_csv)  # 163,994 rows x 15 columns

# Since we want to train a binary classification model, 
# we must ensure that the response is coded as a factor
# If the repsonse is 0/1, H2O will assume it's numeric,
# which means that H2O will train a regression model instead
data$bad_loan <- as.factor(data$bad_loan)  #encode the binary repsonse as a factor


# Partition the data into training, validation and test sets
splits <- h2o.splitFrame(data, 
                         ratios = c(0.7, 0.15),  #partition data into 70%, 15%, 15% chunks
                         seed = 1)  #setting a seed will guarantee reproducibility
train <- splits[[1]]
valid <- splits[[3]]
test <- splits[[2]]

# Identify response and predictor variables
y <- "bad_loan"
x <- setdiff(names(data), c(y, "int_rate"))  #remove the interest rate column because it's correlated with the outcome



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

# Next we will do some automatic tuning by passing in 
# a validation frame and set lambda_search = TRUE
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

# Retreive test set AUC
h2o.auc(glm_perf1)
h2o.auc(glm_perf2)

# Compare test set AUC to validation set AUC
glm_fit2@model$validation_metrics





# 2. Now we will train a basic Random Forest model
rf_fit1 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit1",
                            seed = 1)

# Next we will increase the number of trees
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
h2o.auc(rf_perf1)  # 0.6569221
h2o.auc(rf_perf2)  # 0.6630125





# 3. Now we will train a basic GBM model
gbm_fit1 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit1",
                    seed = 1)

# Next we will increase the number of trees
gbm_fit2 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit2",
                    #validation_frame = valid,  #only used if stopping_rounds > 0
                    ntrees = 500,
                    seed = 1)

# Now let's use early stopping to find optimal ntrees
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
h2o.auc(gbm_perf1)  # 0.6828525
h2o.auc(gbm_perf2)  # 0.6839284
h2o.auc(gbm_perf3)  # 0.6841286

# Look at scoring history for third GBM model
plot(gbm_fit3, 
     timestep = "number_of_trees", 
     metric = "AUC")
plot(gbm_fit3, 
     timestep = "number_of_trees", 
     metric = "logloss")





# 4. Now we will train a Deep Learning model
dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit1",
                            seed = 1)

# Next we will increase the number of trees
dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            #validation_frame = valid,  #only used if stopping_rounds > 0
                            epochs = 20,
                            seed = 1)

# Now let's use early stopping to find optimal ntrees
dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            validation_frame = valid,  #in DL, early stopping is on by default
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
h2o.auc(dl_perf1)  # 0.6760118
h2o.auc(dl_perf2)  # 0.6662219
h2o.auc(dl_perf3)  # 0.6793781

# Look at scoring history for third DL model
plot(dl_fit3, 
     timestep = "epochs", 
     metric = "AUC")





# 5. Lastly, let's take a look at a Naive Bayes model
nb_fit1 <- h2o.naiveBayes(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "nb_fit1")

# Next we add Laplace smoothing
nb_fit2 <- h2o.naiveBayes(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "nb_fit2",
                            laplace = 6)

# Let's compare the performance of the two GBMs
nb_perf1 <- h2o.performance(model = nb_fit1,
                            newdata = test)
nb_perf2 <- h2o.performance(model = nb_fit2,
                            newdata = test)

# Print model performance
nb_perf1
nb_perf2

# Retreive test set AUC
h2o.auc(nb_perf1)  # 0.6501662
h2o.auc(nb_perf2)  # 0.6092679

