rmLastValues <- function(pattern = "Last.value.")
{
  keys <- h2o.ls(h2oServer)$Key
  if (!is.null(keys))
    h2o.rm(h2oServer, keys)
  invisible(keys)
}

myIP <- "localhost"
myPort <- 54321

library(h2o)
h2oServer <- h2o.init(ip = myIP, port = myPort, startH2O = TRUE)


pumsdir <- file.path("/Users/myhomedir/data/pums2013")
trainfile <- "adult_2013_train.csv.gz"
testfile  <- "adult_2013_test.csv.gz"

adult_2013_train <- h2o.importFile(h2oServer,
                                   path = file.path(pumsdir, trainfile),
                                   destination_frame = "adult_2013_train", sep = ",")

adult_2013_test <- h2o.importFile(h2oServer,
                                  path = file.path(pumsdir, testfile),
                                  destination_frame = "adult_2013_test", sep = ",")

dim(adult_2013_train)
dim(adult_2013_test)

actual_log_wagp <- h2o.assign(adult_2013_test[, "LOG_WAGP"],
                              key = "actual_log_wagp")
rmLastValues()

for (j in c("COW", "SCHL", "MAR", "INDP", "RELP", "RAC1P", "SEX", "POBP")) {
  adult_2013_train[[j]] <- as.factor(adult_2013_train[[j]])
  adult_2013_test[[j]]  <- as.factor(adult_2013_test[[j]])
}
rmLastValues()

predset <- c("RELP", "SCHL", "COW", "MAR", "INDP", "RAC1P", "SEX", "POBP", "AGEP",
                "WKHP", "LOG_CAPGAIN", "LOG_CAPLOSS")

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

tmpdir_name <- "generated_model"
dir.create(tmpdir_name)
h2o.download_pojo(log_wagp_gbm, tmpdir_name)

h2o.predict(log_wagp_gbm, adult_2013_test)

h2o.mse(h2o.performance(log_wagp_gbm, adult_2013_test))
