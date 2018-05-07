# Load package and connect to cluster
library(h2o)
h2o.init(max_mem_size = "6g")

# Import data and manage data types
train_path <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
train <- h2o.importFile(train_path, destination_frame = "loan_train")
train["bad_loan"] <- h2o.asfactor(train["bad_loan"])

# Set target and predictor variables
y <- "bad_loan"
x <- h2o.colnames(train)
x <- setdiff(x, c(y, "int_rate"))

# Use Auto ML to train models
aml <- h2o.automl(x = x, y = y, training_frame = train, max_runtime_secs = 300)
