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

# Set target and predictor variables
y <- "readmitted"
x <- h2o.colnames(train)
x <- setdiff(x, c(y, "id", "patient_nbr"))

# Create mean target encodings for categorical variables
train[["fold"]] <- h2o.kfold_column(train, nfolds = 5, seed = 2018)
categoricals <- as.list(setdiff(h2o.colnames(train)[h2o.columns_by_type(train, "categorical")], y))
categoricals <- c(categoricals,
                  list(c("diag_1", "diag_2", "diag_3"),
                       c("admission_type_id", "discharge_disposition_id")))
train_encode <- h2o.target_encode_create(train, x = categoricals, y = y, fold_column = "fold")
train <- h2o.target_encode_apply(train, x = categoricals, y = y, target_encode_map = train_encode,
                                 holdout_type = "KFold", fold_column = "fold", seed = 2018)
x <- setdiff(x, unlist(categoricals))
x <- c(x, h2o.colnames(train)[grep("TargetEncode_", h2o.colnames(train))])

# Checkpoint data set
train <- h2o.assign(train, "diabetes_train")

# Use Auto ML to train models
aml <- h2o.automl(x = x, y = y, training_frame = train, max_runtime_secs = 300)
