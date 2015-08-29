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
log_wagp_gbm <- h2o.gbm(x = predset,
                        y = "LOG_WAGP",
                        training_frame = adult_2013_train,
                        model_id = "GBMModel",
                        distribution = "gaussian",
                        max_depth = 5,
                        ntrees = 110,
                        learn_rate = 0.25,
                        validation_frame = adult_2013_test,
                        importance = TRUE)
log_wagp_gbm

# Do in-H2O predictions for reference.  This is not using the POJO.
h2o.predict(log_wagp_gbm, adult_2013_test)
h2o.mse(h2o.performance(log_wagp_gbm, adult_2013_test))

# Export the POJO.
tmpdir_name <- "generated_model"
dir.create(tmpdir_name)
h2o.download_pojo(log_wagp_gbm, tmpdir_name)
