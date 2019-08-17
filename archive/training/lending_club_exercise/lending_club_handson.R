# Load package and connect to cluster
library(h2o)
h2o.init()

## If possible download from the s3 link and change the path to the dataset.
small_test <- "http://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/lending-club/LoanStats3a.csv"

## Task 1: Import Data
loan_stats <- h2o.importFile(path = small_test, parse = FALSE)

## Specify some column types to "String" that we want to munge later
parseSetup <- h2o.parseSetup(loan_stats)
col_types <- parseSetup$column_types
col_types[parseSetup$column_names %in% c("int_rate", "revol_util", "emp_length", "verification_status")] <- "String"

loan_stats <- h2o.parseRaw(data = loan_stats, destination_frame = "loanStats", col.types = col_types)

## Task 2: Look at the levels in the response column loan_status
## Hint: Use h2o.table function on the response column, use as.data.frame to return the table to R
as.data.frame(h2o.table(loan_stats$loan_status))

## Task 3: Filter out all loans that are still in progress and therefore cannot be deemed good/bad loans.
## Hint: "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)" are ongoing loans
loan_stats <- loan_stats[!(loan_stats$loan_status %in% c("Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)")), ]

## Task 4: Bin the response variable to good/bad loans only, use your best judgment for what is a good/bad loan
## Create new column called bad_loan which should be a binary variable
## Hint: You can turn the bad_loan column into factor using as.factor



## Task 5: String munging to clean string columns before converting to numeric
## Hint: Columns that need munging includes "int_rate", "revol_util", "emp_length"

## Example for int_rate using h2o.gsub, trim, as.numeric
loan_stats$int_rate <- h2o.gsub(x = loan_stats$int_rate, pattern = "%", replacement = "")
loan_stats$int_rate <- h2o.trim(loan_stats$int_rate)
loan_stats$int_rate <- as.numeric(loan_stats$int_rate)

## Now try for revol_util yourself




## Now we're going to clean up emp_length.
## Use h2o.sub to remove " year" and " years", also translate n/a to ""
loan_stats$emp_length <- h2o.sub(x = loan_stats$emp_length, pattern = "([ ]*+[a-zA-Z].*)|(n/a)", replacement = "")
## Use h2o.trim to remove any trailing spaces
loan_stats$emp_length <- h2o.trim(loan_stats$emp_length)
## Use h2o.sub to convert < 1 to 0 years and do the same for 10 + to 10, then convert to numeric
## Hint: Be mindful of spaces between characters




## Task 6: Create new column called credit_length
## Hint: Do this by subtracting the earliest_cr year from the issue_d year


## Task 7: Use h2o.sub to create two levels for column "verification_status" ie "verified" and "not verified"
## Hint: Use h2o.table to examine levels within "verification_status", warning messages can be ignored




## Task 8: Create Cross-Validation 5-Fold Column Called "fold"
## Hint: Use h2o.kfold_column




## Task 9: Create Cross-Validation Target Encoding for Categorial Variables
##         Home Ownership, Loan Purpose, and State of Residence
## Hint: use h2o.target_encode_create to create a target encoding mapping and
##      target_encode_apply to add the new columns to loan_stats












## Task 10: Define your response and predictor variables
myY <- "bad_loan"
myX <-  c()




## Task 11: Use Auto ML to build predictive models for good/bad loan
## Note: Limit max_runtime_secs to something reasonable, e.g. 300 seconds



## Task 12: Select the best single model


## Task 13: Plot the scoring history to make sure you're not overfitting
## Hint: Use plot function on the model object


## Task 14: Plot the ROC curve for the binomial models and get auc using h2o.auc
## Hint: Use h2o.performance and plot to grab the modelmetrics and then plotting the modelmetrics










## Task 15: Check the variable importance and generate confusion matrix for max F1 threshold
## Hint: Use h2o.varimp_plot, h2o.varimp, and h2o.confusionMatrix





## Task 16: Score the entire data set using the model
## Hint: Use h2o.predict.


## Extra: Calculate the money gain/loss if model is implemented
## Calculate the total amount of money earned or lost per loan
loan_stats$earned <- loan_stats$total_pymnt - loan_stats$loan_amnt

## Calculate how much money will be lost to false negative, vs how much will be saved due to true positives
loan_stats$pred <- pred[, 1]
net <- as.data.frame(h2o.group_by(data = loan_stats,
                                  by = c("bad_loan", "pred"),
                                  gb.control = list(na.methods = "ignore"),
                                  sum("earned")))
n1 <- net[net$bad_loan == 0 & net$pred == 0, 3]
n2 <- net[net$bad_loan == 0 & net$pred == 1, 3]
n3 <- net[net$bad_loan == 1 & net$pred == 1, 3]
n4 <- net[net$bad_loan == 1 & net$pred == 0, 3]

## Function defined to pretty print numerics as dollars
printMoney <- function(x){
  x <- round(abs(x), 2)
  format(x, digits = 10, nsmall = 2, decimal.mark = ".", big.mark = ",")
}

## Calculate the amount earned
writeLines(sprintf("Total amount of profit still earned using the model : $%s", printMoney(n1)))
writeLines(sprintf("Total amount of profit forfeitted using the model : $%s", printMoney(n2)))
writeLines(sprintf("Total amount of loss that could have been prevented : $%s", printMoney(n3)))
writeLines(sprintf("Total amount of loss that still would've accrued : $%s", printMoney(n4)))
## Calculate Net
writeLines(sprintf("Total profit by implementing model : $%s", printMoney( n1 - n2 + abs(n3) - abs(n4))))

# Shutdown h2o instance
