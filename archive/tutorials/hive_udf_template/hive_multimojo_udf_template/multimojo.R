library(h2o)

# Download the data into the pums2013 directory if necessary.
pumsdir <- "pums2013"
if (! file.exists(pumsdir)) {
  dir.create(pumsdir)
}

trainfile <- file.path(pumsdir, "adult_2013_train.csv.gz")
if (! file.exists(trainfile)) {
  download.file("http://h2o-training.s3.amazonaws.com/pums2013/adult_2013_train.csv.gz", trainfile)
}

testfile  <- file.path(pumsdir, "adult_2013_test.csv.gz")
if (! file.exists(testfile)) {
  download.file("http://h2o-training.s3.amazonaws.com/pums2013/adult_2013_test.csv.gz", testfile)
}

# Run the example.
h2o.init(nthreads = -1)

adult_2013_train <- h2o.importFile(trainfile, destination_frame = "adult_2013_train")
adult_2013_test <- h2o.importFile(testfile, destination_frame = "adult_2013_test")

dim(adult_2013_train)
dim(adult_2013_test)

actual_log_wagp <- h2o.assign(adult_2013_test[, "LOG_WAGP"], key = "actual_log_wagp")

for (j in c("COW", "SCHL", "MAR", "INDP", "RELP", "RAC1P", "SEX", "POBP")) {
  adult_2013_train[[j]] <- as.factor(adult_2013_train[[j]])
  adult_2013_test[[j]]  <- as.factor(adult_2013_test[[j]])
}

predset <- c("RELP", "SCHL", "COW", "MAR", "INDP", "RAC1P", "SEX", "POBP", "AGEP",
             "WKHP", "LOG_CAPGAIN", "LOG_CAPLOSS")

# Train the model.
log_wagp_gbm1 <- h2o.gbm(x = predset,
                        y = "LOG_WAGP",
                        training_frame = adult_2013_train,
                        model_id = "GBMModel1",
                        distribution = "gaussian",
                        max_depth = 5,
                        ntrees = 110,
                        learn_rate = 0.25,
                        validation_frame = adult_2013_test)

log_wagp_gbm2 <- h2o.gbm(x = predset,
                         y = "LOG_WAGP",
                         training_frame = adult_2013_train,
                         model_id = "GBMModel2",
                         distribution = "gaussian",
                         max_depth = 10,
                         ntrees = 120,
                         learn_rate = 0.25,
                         validation_frame = adult_2013_test)

log_wagp_gbm3 <- h2o.gbm(x = predset,
                         y = "LOG_WAGP",
                         training_frame = adult_2013_train,
                         model_id = "GBMModel3",
                         distribution = "gaussian",
                         max_depth = 8,
                         ntrees = 130,
                         learn_rate = 0.25,
                         validation_frame = adult_2013_test)

log_wagp_gbm4 <- h2o.gbm(x = predset,
                         y = "LOG_WAGP",
                         training_frame = adult_2013_train,
                         model_id = "GBMModel4",
                         distribution = "gaussian",
                         max_depth =12,
                         ntrees = 105,
                         learn_rate = 0.25,
                         validation_frame = adult_2013_test)

log_wagp_gbm5 <- h2o.gbm(x = predset,
                         y = "LOG_WAGP",
                         training_frame = adult_2013_train,
                         model_id = "GBMModel5",
                         distribution = "gaussian",
                         max_depth = 7,
                         ntrees = 115,
                         learn_rate = 0.25,
                         validation_frame = adult_2013_test)

log_wagp_rf1 <- h2o.randomForest(x = predset,
                                 y = "LOG_WAGP",
                                 training_frame = adult_2013_train,
                                 model_id = "RFModel1",
                                 max_depth = 5,
                                 ntrees = 110, 
                                 validation_frame = adult_2013_test)

log_wagp_rf2 <- h2o.randomForest(x = predset,
                                 y = "LOG_WAGP",
                                 training_frame = adult_2013_train,
                                 model_id = "RFModel2",
                                 max_depth = 10,
                                 ntrees = 120, 
                                 validation_frame = adult_2013_test)

log_wagp_rf3 <- h2o.randomForest(x = predset,
                                 y = "LOG_WAGP",
                                 training_frame = adult_2013_train,
                                 model_id = "RFModel3",
                                 max_depth = 8,
                                 ntrees = 130, 
                                 validation_frame = adult_2013_test)

log_wagp_rf4 <- h2o.randomForest(x = predset,
                                 y = "LOG_WAGP",
                                 training_frame = adult_2013_train,
                                 model_id = "RFModel4",
                                 max_depth = 12,
                                 ntrees = 105, 
                                 validation_frame = adult_2013_test)

log_wagp_rf5 <- h2o.randomForest(x = predset,
                                 y = "LOG_WAGP",
                                 training_frame = adult_2013_train,
                                 model_id = "RFModel5",
                                 max_depth = 7,
                                 ntrees = 115, 
                                 validation_frame = adult_2013_test)

model_list <- c(log_wagp_gbm1, log_wagp_gbm2, log_wagp_gbm3, log_wagp_gbm4, log_wagp_gbm5, log_wagp_rf1, log_wagp_rf2, log_wagp_rf3, log_wagp_rf4, log_wagp_rf5)

# Do in-H2O predictions for reference.  This is not using the MOJO.
for (model in model_list) {print(h2o.predict(model, adult_2013_test))}
# Export the MOJOs.
tmpdir_name <- "generated_models"
dir.create(tmpdir_name)
for (model in model_list) {print (h2o.download_mojo(model, tmpdir_name))}
