# Load package and connect to cluster
library(h2o)
h2o.init(max_mem_size = "6g")


# Import data and manage data types
loan_path <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
loan_path <- "/Users/patrickaboyoun/Downloads/loan.csv"
train <- h2o.importFile(loan_path, destination_frame = "loan")
train["bad_loan"] <- h2o.asfactor(train["bad_loan"])


# Set target and predictor variables
y <- "bad_loan"
x_orig <- h2o.colnames(train)
x_orig <- setdiff(x_orig, c(y, "int_rate"))
x_trans <- x_orig


# Create Cross-Validation Fold Column
cv_nfolds <- 5
cv_seed <- 987

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

x_trans <- setdiff(x_trans, c("home_ownership", "purpose", "addr_state"))
x_trans <- c(x_trans, "TargetEncode_home_ownership", "TargetEncode_purpose", "TargetEncode_addr_state")


# Helper Functions
h2o.pmin <- function(x, y) {
  h2o.ifelse(x < y, x, y)
}

h2o.pmax <- function(x, y) {
  h2o.ifelse(x > y, x, y)
}

h2o.winsorize <- function(x, minval, maxval) {
  h2o.pmax(minval, h2o.pmin(x, maxval))
}


# Separate Typical from Extreme Loan Amount
breaks <- unique(h2o.quantile(train["loan_amnt"], seq(0, 1, by = 0.01)))
breaks[1] <- 0
train["loan_amnt_cat"] <- h2o.cut(train["loan_amnt"], breaks = breaks)

df <- as.data.frame(h2o.group_by(train, "loan_amnt_cat",
                                 nrow("bad_loan"),
                                 min("loan_amnt"),
                                 mean("bad_loan")))
print(df)

library(splines)
symbols(df$min_loan_amnt, qlogis(df$mean_bad_loan), circles = sqrt(df$nrow / pi),
        inches = 0.1, xlab = "loan_amnt", ylab = "logit(bad_loan)")

lines(df$min_loan_amnt,
     predict(lm(qlogis(mean_bad_loan) ~
                  bs(min_loan_amnt, knots = seq(5000, 30000, by = 5000),
                     degree = 1), data = df, weights = df$nrow)),
     col = "blue")
abline(v = c(5000, 30000), col = "orange", lty = 2)

train["loan_amnt_winz"] <- h2o.winsorize(train["loan_amnt"], 5000, 30000)
train["loan_amnt_tail"] <- train["loan_amnt"] - train["loan_amnt_winz"]

x_trans <- setdiff(x_trans, "loan_amnt")
x_trans <- c(x_trans, "loan_amnt_winz", "loan_amnt_tail")


# Convert Term to a 0/1 Indicator
h2o.table(train["term"])

train["term_60months"] <- train["term"] == "60 months"

x_trans <- setdiff(x_trans, "term")
x_trans <- c(x_trans, "term_60months")


# Create Missing Value Indicator for Employment Length
train["emp_length_missing"] <- train["emp_length"] == NA

h2o.table(train["emp_length_missing"])

x_trans <- c(x_trans, "emp_length_missing")


# Separate Typical from Extreme Annual Income
breaks <- unique(h2o.quantile(train["annual_inc"], seq(0, 1, by = 0.01)))
breaks[1] <- 0
train["annual_inc_cat"] <- h2o.cut(train["annual_inc"], breaks = breaks)

df <- as.data.frame(h2o.group_by(train, "annual_inc_cat",
                                 nrow("bad_loan"),
                                 min("annual_inc"),
                                 mean("bad_loan")))
df <- na.omit(df)
print(df)

library(splines)
symbols(df$min_annual_inc, qlogis(df$mean_bad_loan), circles = sqrt(df$nrow / pi),
        inches = 0.1, xlab = "annual_inc", ylab = "logit(bad_loan)")

lines(df$min_annual_inc,
      predict(lm(qlogis(mean_bad_loan) ~
                   bs(min_annual_inc, knots = seq(2500, 140000, by = 20000),
                      degree = 1), data = df, weights = df$nrow)),
      col = "blue")
abline(v = c(10000, 105000), col = "orange", lty = 2)

train["annual_inc_winz"] <- h2o.winsorize(train["annual_inc"], 10000, 105000)
train["annual_inc_tail"] <- train["annual_inc"] - train["annual_inc_winz"]

x_trans <- setdiff(x_trans, "annual_inc")
x_trans <- c(x_trans, "annual_inc_winz", "annual_inc_tail")


# Separate Typical from Extreme Debt to Income Ratio
breaks <- unique(h2o.quantile(train["dti"], seq(0, 1, by = 0.01)))
breaks[1] <- -0.5
train["dti_cat"] <- h2o.cut(train["dti"], breaks = breaks)

df <- as.data.frame(h2o.group_by(train, "dti_cat",
                                 nrow("bad_loan"),
                                 min("dti"),
                                 mean("bad_loan")))
df <- na.omit(df)
print(df)

library(splines)
symbols(df$min_dti, qlogis(df$mean_bad_loan), circles = sqrt(df$nrow / pi),
        inches = 0.1, xlab = "dti", ylab = "logit(bad_loan)")

lines(df$min_dti,
      predict(lm(qlogis(mean_bad_loan) ~
                   bs(min_dti, knots = seq(2, 38, by = 2),
                      degree = 1), data = df, weights = df$nrow)),
      col = "blue", lwd = 2)
abline(v = c(5, 30), col = "orange", lty = 2)

train["dti_winz"] <- h2o.winsorize(train["dti"], 5, 30)
train["dti_tail"] <- train["dti"] - train["dti_winz"]

x_trans <- setdiff(x_trans, "dti")
x_trans <- c(x_trans, "dti_winz", "dti_tail")


# Separate Typical from Extreme Number of Delinquencies in the Past 2 Years
breaks <- unique(h2o.quantile(train["delinq_2yrs"], seq(0, 1, by = 0.01)))
breaks <- c(-0.5, breaks)
train["delinq_2yrs_cat"] <- h2o.cut(train["delinq_2yrs"], breaks = breaks)

df <- as.data.frame(h2o.group_by(train, "delinq_2yrs_cat",
                                 nrow("bad_loan"),
                                 min("delinq_2yrs"),
                                 mean("bad_loan")))
df <- na.omit(df)
print(df)

symbols(df$min_delinq_2yrs, qlogis(df$mean_bad_loan), circles = sqrt(df$nrow / pi),
        inches = 0.1, xlab = "delinq_2yrs", ylab = "logit(bad_loan)")

lines(df$min_delinq_2yrs, qlogis(df$mean_bad_loan), col = "blue")
abline(v = 3, col = "orange", lty = 2)

train["delinq_2yrs_winz"] <- h2o.ifelse(train["delinq_2yrs"] <= 3, train["delinq_2yrs"], 3)
train["delinq_2yrs_tail"] <- train["delinq_2yrs"] - train["delinq_2yrs_winz"]

x_trans <- setdiff(x_trans, "delinq_2yrs")
x_trans <- c(x_trans, "delinq_2yrs_winz", "delinq_2yrs_tail")


# Separate Typical from Extreme Revolving Credit Line Utilized
breaks <- unique(h2o.quantile(train["revol_util"], seq(0, 1, by = 0.01)))
breaks[1] <- -0.1
train["revol_util_cat"] <- h2o.cut(train["revol_util"], breaks = breaks)

df <- as.data.frame(h2o.group_by(train, "revol_util_cat",
                                 nrow("bad_loan"),
                                 min("revol_util"),
                                 mean("bad_loan")))
df <- na.omit(df)
print(df)

library(splines)
symbols(df$min_revol_util, qlogis(df$mean_bad_loan), circles = sqrt(df$nrow / pi),
        inches = 0.1, xlab = "revol_util", ylab = "logit(bad_loan)")

lines(df$min_revol_util,
      predict(lm(qlogis(mean_bad_loan) ~
                   bs(min_revol_util, knots = seq(0, 100, by = 10),
                      degree = 1), data = df, weights = df$nrow)),
      col = "blue", lwd = 2)
abline(v = c(0, 100), col = "orange", lty = 2)

train["revol_util_0"] <- train["revol_util"] == 0
train["revol_util_winz"] <- h2o.ifelse(train["revol_util"] <= 100, train["revol_util"], 100)
train["revol_util_tail"] <- train["revol_util"] - train["revol_util_winz"]

x_trans <- setdiff(x_trans, "revol_util")
x_trans <- c(x_trans, "revol_util_0", "revol_util_winz", "revol_util_tail")


# Separate Typical from Extreme Number of Credit Lines
breaks <- unique(h2o.quantile(train["total_acc"], seq(0, 1, by = 0.01)))
breaks[1] <- 0
train["total_acc_cat"] <- h2o.cut(train["total_acc"], breaks = breaks)

df <- as.data.frame(h2o.group_by(train, "total_acc_cat",
                                 nrow("bad_loan"),
                                 min("total_acc"),
                                 mean("bad_loan")))
df <- na.omit(df)
print(df)

library(splines)
symbols(df$min_total_acc, qlogis(df$mean_bad_loan), circles = sqrt(df$nrow / pi),
        inches = 0.1, xlab = "total_acc", ylab = "logit(bad_loan)")

lines(df$min_total_acc,
      predict(lm(qlogis(mean_bad_loan) ~
                   bs(min_total_acc, knots = seq(5, 55, by = 10),
                      degree = 1), data = df, weights = df$nrow)),
      col = "blue", lwd = 2)
abline(v = 50, col = "orange", lty = 2)

train["total_acc_winz"] <- h2o.ifelse(train["total_acc"] <= 50, train["total_acc"], 50)
train["total_acc_tail"] <- train["total_acc"] - train["total_acc_winz"]

x_trans <- setdiff(x_trans, "total_acc")
x_trans <- c(x_trans, "total_acc_winz", "total_acc_tail")


# Separate Typical from Extreme Longest Credit Length
breaks <- -1:30
train["longest_credit_length_cat"] <- h2o.cut(train["longest_credit_length"], breaks = breaks)

df <- as.data.frame(h2o.group_by(train, "longest_credit_length_cat",
                                 nrow("bad_loan"),
                                 min("longest_credit_length"),
                                 mean("bad_loan")))
df <- na.omit(df)
print(df)

library(splines)
symbols(df$min_longest_credit_length, qlogis(df$mean_bad_loan), circles = sqrt(df$nrow / pi),
        inches = 0.1, xlab = "longest_credit_length", ylab = "logit(bad_loan)")

lines(df$min_longest_credit_length,
      predict(lm(qlogis(mean_bad_loan) ~
                   bs(min_longest_credit_length, knots = seq(3, 28, by = 5),
                      degree = 1), data = df, weights = df$nrow)),
      col = "blue", lwd = 1)
abline(v = c(3, 20), col = "orange", lty = 2)

train["longest_credit_length_winz"] <- h2o.winsorize(train["longest_credit_length"], 3, 20)
train["longest_credit_length_tail"] <- train["longest_credit_length"] - train["longest_credit_length_winz"]

x_trans <- setdiff(x_trans, "longest_credit_length")
x_trans <- c(x_trans, "longest_credit_length_winz", "longest_credit_length_tail")


# Convert Income Verification Status to a 0/1 Indicator
h2o.table(train["verification_status"])

train["verified"] <- train["verification_status"] == "verified"

x_trans <- setdiff(x_trans, "verification_status")
x_trans <- c(x_trans, "verified")


# Checkpoint data in H2O K/V store
train <- h2o.assign(train, "loan_model1")


# Use Automatic Machine Learning to create an initial set of models
aml1 <- h2o.automl(x = x_trans, y = y, training_frame = train, nfolds = cv_nfolds,
                   fold_column = "fold", max_runtime_secs = 300, seed = 2307,
                   project_name = "loan_aml1")


# Create interaction effects amongst Annual Income, Debt to Income Ratio, and
# Revolving Credit Utiliized using k-Means Clustering
clusters <- h2o.kmeans(training_frame = train,
                       x = c("annual_inc_winz", "dti_winz", "revol_util_winz"),
                       fold_column = "fold",
                       k = 4,
                       seed = 423)

cdist <- h2o.distance(x = h2o.scale(train[c("annual_inc_winz", "dti_winz", "revol_util_winz")]),
                      y = as.h2o(as.data.frame(clusters@model$centers_std[, -1])),
                      measure = "l2")
colnames(cdist) <- sprintf("ClusterDist%d", 1:4)

train <- h2o.cbind(train, cdist)

x_trans <- c(x_trans, colnames(cdist))


# Checkpoint data in H2O K/V store
train <- h2o.assign(train, "loan_model2")


# Use Automatic Machine Learning to create a revised set of models
aml2 <- h2o.automl(x = x_trans, y = y, training_frame = train, nfolds = cv_nfolds,
                   fold_column = "fold", max_runtime_secs = 300, seed = 2307,
                   project_name = "loan_aml2")
