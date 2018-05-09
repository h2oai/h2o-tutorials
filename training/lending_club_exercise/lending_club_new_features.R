# Load package and connect to cluster
library(h2o)
h2o.init(max_mem_size = "6g")

# Import data and manage data types
train_path <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
train <- h2o.importFile(train_path, destination_frame = "loan_init")
train["bad_loan"] <- h2o.asfactor(train["bad_loan"])

# Set target and predictor variables
y <- "bad_loan"
x <- h2o.colnames(train)
x <- setdiff(x, c(y, "int_rate"))

# Create Cross-Validation Fold Column
cv_nfolds <- 5
cv_seed <- 545
train["fold"] <- h2o.kfold_column(train, nfolds = cv_nfolds, seed = cv_seed)

# Create Target Encoding for Home Ownership, Loan Purpose, and State of Residence
mapping <- h2o.target_encode_create(data = train,
                                    x = list("home_ownership", "purpose", "addr_state"),
                                    y = "bad_loan",
                                    fold_column = "fold")

train <- h2o.target_encode_apply(data = train,
                                 x = list("home_ownership", "purpose", "addr_state"),
                                 y = "bad_loan", mapping,
                                 holdout_type = "KFold", fold_column = "fold")

x <- setdiff(x, c("home_ownership", "purpose", "addr_state"))
x <- c(x, "TargetEncode_home_ownership", "TargetEncode_purpose", "TargetEncode_addr_state")

# Create interaction effects amongst Annual Income, Debt to Income Ratio, and
# Revolving Credit Utiliized using k-Means Clustering
log_train <- log1p(train[c("annual_inc", "dti", "revol_util")])
log_train <- h2o.scale(log_train)
for (j in colnames(log_train)) {
  log_train[[j]] <- h2o.ifelse(log_train[[j]] == NA, 0, log_train[[j]])
}

train <- h2o.cbind(train, log_train)

clusters <- h2o.kmeans(training_frame = train,
                       x = colnames(log_train),
                       fold_column = "fold",
                       k = 2,
                       standardize = FALSE,
                       seed = 234)

cdist <- h2o.distance(x = log_train,
                      y = as.h2o(h2o.centers(clusters)),
                      measure = "l2")
colnames(cdist) <- sprintf("ClusterDist%d", seq_len(ncol(cdist)))

train <- h2o.cbind(train, cdist)

x <- c(x, colnames(cdist))

# Checkpoint data in H2O K/V store
train <- h2o.assign(train, "loan_trans")

# Use Auto ML to train models
aml <- h2o.automl(x = x, y = y, training_frame = train, nfolds = cv_nfolds,
                  fold_column = "fold", max_runtime_secs = 300)
