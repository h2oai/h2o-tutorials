library(h2o)
h2o.init(nthreads = -1)

## If possible download from the s3 link and change the path to the dataset.
small_test <- "http://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/lending-club/LoanStats3a.csv"

## Task 1: Import Data
## Parse with user imposed schema which changes the column types of column:
## 'int_rate', 'revol_util', 'emp_length', 'verification_status' to String instead of Enum
col_types <- c('Numeric', 'Numeric', 'Numeric', 'Numeric', 'Numeric', 'Enum', 'String', 'Numeric', 
               'Enum', 'Enum', 'Enum', 'String', 'Enum', 'Numeric', 'String', 'Time', 'Enum', 'Enum', 
               'String', 'Enum', 'Enum', 'Enum', 'Enum', 'Enum', 'Numeric', 'Numeric', 'Time', 'Numeric', 
               'Enum', 'Enum', 'Numeric', 'Numeric', 'Numeric', 'String', 'Numeric', 'Enum', 'Numeric', 
               'Numeric', 'Numeric', 'Numeric', 'Numeric', 'Numeric', 'Numeric', 'Numeric', 'Numeric', 
               'Enum', 'Numeric', 'Enum', 'Time', 'Numeric', 'Enum', 'Numeric')
loanStats <- h2o.importFile(path = small_test, col.types = col_types)

## Task 2: Look at the levels in the response column loan_status
## Hint: Use h2o.table function on the response column, use as.data.frame to return the table to R
as.data.frame(h2o.table(loanStats$loan_status))

## Task 3: Filter out all loans that are completed, aka subset data
## Hint: "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)" are ongoing loans
loanStats <- loanStats[!(loanStats$loan_status %in% c("Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)")), ]

## Task 4: Bin the response variable to good/bad loans only, use your best judgment for what is a good/bad loan
## Create new column called bad_loan which should be a binary variable
## Hint: You can turn the bad_loan column into factor using as.factor
loanStats$bad_loan <- loanStats$loan_status %in% c("Charged Off", "Default", "Does not meet the credit policy.  Status:Charged Off")
loanStats$bad_loan <- as.factor(loanStats$bad_loan)

## Task 5: String munging to clean string columns before converting to numeric
## Hint: Columns that need munging includes "int_rate", "revol_util", "emp_length"

## Example for int_rate using h2o.strsplit, trim, as.numeric
loanStats$int_rate <- h2o.strsplit(loanStats$int_rate, split = "%")
loanStats$int_rate <- h2o.trim(loanStats$int_rate)
loanStats$int_rate <- as.numeric(loanStats$int_rate)

## Now try for revol_util yourself
loanStats$revol_util <- h2o.strsplit(loanStats$revol_util, split = "%")
loanStats$revol_util <- h2o.trim(loanStats$revol_util)
loanStats$revol_util <- as.numeric(loanStats$revol_util)

## Now we're going to clean up emp_length.
## Use h2o.sub to remove " year" and " years", also translate n/a to ""
loanStats$emp_length <- h2o.sub(x = loanStats$emp_length, pattern = "([ ]*+[a-zA-Z].*)|(n/a)", replacement = "")
## Use h2o.trim to remove any trailing spaces
loanStats$emp_length <- h2o.trim(loanStats$emp_length)
## Use h2o.sub to convert < 1 to 0 years and do the same for 10 + to 10 
## Hint: Be mindful of spaces between characters
loanStats$emp_length <- h2o.sub(x = loanStats$emp_length, pattern = "< 1", replacement = "0")
loanStats$emp_length <- h2o.sub(x = loanStats$emp_length, pattern = "10\\\\+", replacement = "10")
loanStats$emp_length <- as.numeric(loanStats$emp_length)

## Task 6: Create new feature called "credit_length_in_years"
## Hint: Use the columns "earliest_cr_line" and "issue_d"
loanStats$credit_length_in_years <- h2o.year(loanStats$issue_d) - h2o.year(loanStats$earliest_cr_line)

## Task 7: Use h2o.sub to create two levels for column "verification_status" ie "verified" and "not verified"
## Hint: Use h2o.table to examine levels within "verification_status"
loanStats$verification_status <- h2o.sub(x = loanStats$verification_status, pattern = "VERIFIED - income source", replacement = "verified")
loanStats$verification_status <- h2o.sub(x = loanStats$verification_status, pattern = "VERIFIED - income", replacement = "verified")
loanStats$verification_status <- as.factor(loanStats$verification_status)
h2o.table(loanStats$verification_status)

## Task 8: Define your response and predictor variables
myY <- "bad_loan"
myX <-  c("loan_amnt", "term", "home_ownership", "annual_inc", "verification_status", "purpose",
          "addr_state", "dti", "delinq_2yrs", "open_acc", "pub_rec", "revol_bal", "total_acc",
          "emp_length", "credit_length_in_years", "inq_last_6mths", "revol_util")

## Task 9: Do a test-train split (80-20)
## Hint: Use h2o.splitFrame ONLY once
split <- h2o.splitFrame(loanStats, ratios = 0.8)
train <- split[[1]]
valid <- split[[2]]
## Hint: Use h2o.table to see if the ratio of the response class is maintained
h2o.table(loanStats[,myY])
h2o.table(train[,myY])
h2o.table(valid[,myY])

## Task 10: Build model predicting good/bad loan 
## Note: Use any of the classification methods available including GLM, GBM, Random Forest, and Deep Learning
glm_model <- h2o.glm(x = myX, y = myY, training_frame = train, validation_frame = valid,
                     family = "binomial", model_id = "GLM_BadLoan")
gbm_model <- h2o.gbm(x = myX, y = myY, training_frame = train, validation_frame = valid,
                     learn_rate = 0.05, score_each_iteration = T, ntrees = 100, model_id = "GBM_BadLoan")

## Task 11: Plot the scoring history to make sure you're not overfitting
## Hint: Use plot function on the model object
plot(gbm_model)

## Task 12: Plot the ROC curve for the binomial models and get auc using h2o.auc
## Hint: Use h2o.performance and plot to grab the modelmetrics and then plotting the modelmetrics
perf <- h2o.performance(model = gbm_model)
plot(perf, train = T)
plot(perf, valid = T)
h2o.auc(gbm_model, train = T)
h2o.auc(gbm_model, valid = T)

## Task 13: Check the variable importance and generate confusion matrix for max F1 threshold
## Hint: Use h2o.varimp for non-GLM model and use h2o.confusionMatrix
h2o.varimp(gbm_model)
h2o.confusionMatrix(gbm_model)

## Task 14: Score the entire data set using the model
## Hint: Use h2o.predict.
pred <- h2o.predict(gbm_model, loanStats)

## Extra: Calculate the money gain/loss if model is implemented
## Calculate the total amount of money earned or lost per loan
loanStats$earned <- loanStats$total_pymnt - loanStats$loan_amnt

## Calculate how much money will be lost to false negative, vs how much will be saved due to true positives
loanStats$pred <- pred[,1]
net <- as.data.frame(h2o.group_by(data = loanStats, by = c("bad_loan", "pred"), gb.control = list(na.methods = "ignore"), sum("earned")))
n1  <- net[ net$bad_loan == 0 & net$pred == 0, 3]
n2  <- net[ net$bad_loan == 0 & net$pred == 1, 3]
n3  <- net[ net$bad_loan == 1 & net$pred == 1, 3]
n4  <- net[ net$bad_loan == 1 & net$pred == 0, 3]


## Function defined to pretty print numerics as dollars
printMoney <- function(x){
  x <- round(abs(x),2)
  format(x, digits=10, nsmall=2, decimal.mark=".", big.mark=",")
}

## Calculate the amount of earned
print(paste0("Total amount of profit still earned using the model : $", printMoney(n1) , ""))
print(paste0("Total amount of profit forfeitted using the model : $", printMoney(n2) , ""))
print(paste0("Total amount of loss that could have been prevented : $", printMoney(n3) , ""))
print(paste0("Total amount of loss that still would've accrued : $", printMoney(n4) , ""))
## Calculate Net
print(paste0("Total profit by implementing model : $", printMoney( n1 - n2 + n3 - n4)))

