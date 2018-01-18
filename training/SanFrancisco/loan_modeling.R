library(h2o)
h2o.init(nthreads = -1)

print("Import approved and rejected loan requests...")
file_path = "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
loans  <- h2o.importFile(path = file_path)
h2o.colnames(loans)
loans$bad_loan <- as.factor(loans$bad_loan)

loan_split = h2o.splitFrame(loans, ratios = c(0.8,0.1))

loan_train = loan_split[[1]]
loan_valid = loan_split[[2]]
loan_test = loan_split[[3]]

myY = "bad_loan"
myX = c("loan_amnt", "longest_credit_length", "revol_util", "emp_length",
        "home_ownership", "annual_inc", "purpose", "addr_state", "dti",
        "delinq_2yrs", "total_acc", "verification_status", "term")

gbm_model <- h2o.gbm(x = myX, y = myY,
                 training_frame = loan_train, validation_frame = loan_valid,
                 score_each_iteration = T,
                 ntrees = 100, max_depth = 5, learn_rate = 0.05,
                 model_id = "BadLoanModel")

print(gbm_model)

loan_test_result = h2o.predict(gbm_model, loan_test)

myY = "int_rate"
myX = c("loan_amnt", "longest_credit_length", "revol_util", "emp_length",
        "home_ownership", "annual_inc", "purpose", "addr_state", "dti",
        "delinq_2yrs", "total_acc", "verification_status", "term")

gbm_model_intrate <- h2o.gbm(x = myX, y = myY,
                             training_frame = loan_train, validation_frame = loan_valid,
                             score_each_iteration = T, 
                             ntrees = 100, max_depth = 5, learn_rate = 0.05,
                             model_id = "InterestRateModel")

print(gbm_model_intrate)

loan_int_test_result = h2o.predict(gbm_model_intrate, loan_test)

loan_int_test_result


