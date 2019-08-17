# Load package and connect to cluster
library(h2o)
h2o.init(max_mem_size = "6g")

# Import data and manage data types
train_path <- "http://s3.amazonaws.com/h2o-public-test-data/smalldata/diabetes/diabetes_train.csv"
train <- h2o.importFile(train_path)
for (j in c("admission_type_id", "discharge_disposition_id", "admission_source_id",
            "diag_1", "diag_2", "diag_3")) {
  train[[j]] <- h2o.asfactor(train[[j]])
}
train <- h2o.assign(train, "diabetes_train")

# Set target and predictor variables
y <- "readmitted"
x <- h2o.colnames(train)
x <- setdiff(x, c(y, "id", "patient_nbr"))

# Use Auto ML to train models
aml <- h2o.automl(x = x, y = y, training_frame = train, max_runtime_secs = 300)
