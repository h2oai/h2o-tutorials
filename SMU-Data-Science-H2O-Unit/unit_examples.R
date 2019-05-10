#
# SMU Data Science H2O Unit - R
#

# Cleanup R environment 
rm(list=ls())

# Load the H2O library 
library(h2o)

# Connect to already running instance 
h2o.connect(ip = "localhost", port = 54321)
h2o.shutdown(prompt = FALSE)

# or Start up the H2O cluster locally on your machine
# (like in we did in the segment)
h2o.init(nthreads = -1,        #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O
h2o.clusterStatus()

#
# Classification (binomial) with H2O and R
#

# Import Chicago crime data
chicago_crime_hex = h2o.importFile(path="https://s3.amazonaws.com/h2o-public-test-data/smalldata/chicago/chicagoCrimes10k.csv.zip", 
                                   destination_frame = "chicago.crime.hex")  # 9,999 rows x 22 columns
# chicago_crime_hex = h2o.getFrame("chicago.crime.hex")
dim(chicago_crime_hex)
names(chicago_crime_hex)

y = "Arrest"
x = setdiff(names(chicago_crime_hex), 
            c(y,"Case Number","Block","Description",
              "Location Description","Location",
              "Date","Updated On"))

# Build GLM classification model
glm_crime_model = h2o.glm(x = x, y = y, 
                    training_frame = chicago_crime_hex,
                    model_id = "glm_crime_r_model",
                    family = "binomial", nfolds = 3,
                    seed = 75205)

h2o.coef_norm(glm_crime_model)
h2o.varimp(glm_crime_model)

glm_crime_model@model$model_summary
glm_crime_model@model$coefficients_table
glm_crime_model@model$cross_validation_metrics_summary

h2o.auc(h2o.performance(glm_crime_model, xval = TRUE)) 
h2o.r2(h2o.performance(glm_crime_model, xval = TRUE))

# Build Random Forest model
rf_crime_model = h2o.randomForest(x = x, y = y,
                            training_frame = chicago_crime_hex,
                            model_id = "rf_crime_r_model",
                            nfolds = 3, seed = 75205)

h2o.auc(h2o.performance(rf_crime_model, xval = TRUE)) 
h2o.r2(h2o.performance(rf_crime_model, xval = TRUE))

# Build AutoML model
splits = h2o.splitFrame(chicago_crime_hex, ratios = 0.8, seed = 75205,
               destination_frames = c("chicago.crime.train", "chicago.crime.test"))
chicago_crime_train = splits[[1]]
chicago_crime_test = splits[[2]]

automl_crime_model = h2o.automl(y = y, x = x,
                            training_frame = chicago_crime_train,
                            leaderboard_frame = chicago_crime_test,
                            nfolds = 3, seed = 75205, 
                            max_runtime_secs = 60,
                            include_algos = c("GLM","DRF","GBM"),
                            project_name = "automl_crime_r",
                            keep_cross_validation_predictions = FALSE,
                            keep_cross_validation_models = FALSE,
                            keep_cross_validation_fold_assignment = FALSE)

automl_crime_model@leaderboard

# 
# Regression with H2O and R
#
wine_quality_hex = h2o.importFile(path="https://s3.amazonaws.com/h2o-public-test-data/smalldata/wine/winequality-redwhite.csv",
                                  header = TRUE, destination_frame = "wine.hex")  # 6,497 rows x 13 columns
# wine_quality_hex = h2o.getFrame("wine.hex")

dim(wine_quality_hex)
names(wine_quality_hex)

y = "quality"
x = setdiff(names(wine_quality_hex), y)

# Build GLM regression model
glm_wine_model = h2o.glm(x = x, y = y, 
                         training_frame = wine_quality_hex,
                         model_id = "glm_wine_r_model",
                         family = "gaussian", nfolds = 3,
                         seed = 75205)

glm_wine_model@model$model_summary
glm_wine_model@model$coefficients_table
glm_wine_model@model$cross_validation_metrics_summary

h2o.performance(glm_wine_model, xval = TRUE)

h2o.rmse(h2o.performance(glm_wine_model, xval = TRUE)) 
h2o.r2(h2o.performance(glm_wine_model, xval = TRUE))

# Build GBM regression model
gbm_wine_model = h2o.gbm(x = x, y = y,
                         training_frame = wine_quality_hex,
                         model_id = "gbm_crime_r_model",
                         nfolds = 3, seed = 75205)

rf_crime_model@model$cross_validation_metrics_summary

h2o.rmse(h2o.performance(rf_crime_model, xval = TRUE)) 
h2o.r2(h2o.performance(rf_crime_model, xval = TRUE))

# Build AutoML model
splits = h2o.splitFrame(wine_quality_hex, ratios = 0.8, seed = 75205,
                        destination_frames = c("wine.train", "wine.test"))
wine_quality_train = splits[[1]]
wine_quality_test = splits[[2]]

automl_wine_model = h2o.automl(y = y, x = x,
                               training_frame = wine_quality_train,
                               leaderboard_frame = wine_quality_test,
                               nfolds = 3, seed = 75205, 
                               max_runtime_secs = 60,
                               include_algos = c("GLM","DRF","GBM"),
                               project_name = "automl_wine_quality_r",
                               keep_cross_validation_predictions = FALSE,
                               keep_cross_validation_models = FALSE,
                               keep_cross_validation_fold_assignment = FALSE)

automl_wine_model@leaderboard
