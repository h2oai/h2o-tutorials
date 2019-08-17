library(h2o)

# Step 1 (of 10). Start H2O-3 cluster -------------------------------------

h2o.init(bind_to_localhost = F)

# Step 2 (of 10). Import data ---------------------------------------------

# View Data
# https://s3-us-west-2.amazonaws.com/h2o-tutorials/data/topics/lending/lending_club/LoanStats3a.csv
loans <- h2o.importFile("/home/h2o/data/topics/lending/lending_club/LoanStats3a.csv", 
                        col.types = c("int_rate" = "string",
                                      "revol_util" = "string",
                                      "emp_length" = "string",
                                      "verification_status" = "string"))

dim(loans)

head(loans)

# Filter Loans
# 
# Now we will filter out loans that are ongoing.  These are loans with loan 
# status like "Current" and "In Grace Period".

num_unfiltered_loans = nrow(loans)
num_unfiltered_loans

head(h2o.table(loans$loan_status), 20)

ongoing_status = c("Current", 
                   "In Grace Period", 
                   "Late (16-30 days)",
                   "Late (31-120 days)")

loans = loans[!loans$loan_status %in% ongoing_status, ]

num_filtered_loans = nrow(loans)
num_filtered_loans

num_loans_filtered_out = num_unfiltered_loans - num_filtered_loans
num_loans_filtered_out

# Create Response Column
# 
# Our response column will be called: bad_loan.  The bad_loan column will be 
# positive if the loan was not completely paid off.

loans$bad_loan = loans$loan_status != "Fully Paid"
loans$bad_loan = as.factor(loans$bad_loan) # convert to enum/factor

bad_loan_dist = h2o.table(loans$bad_loan)
bad_loan_dist$Percentage = h2o.round(100 * bad_loan_dist$Count / nrow(loans))
bad_loan_dist
# About one in five loans eventually become bad.

# Step 3 (of 10). Clean data ----------------------------------------------
#
# We have multiple columns that are numeric but are being treated as string 
# because of their syntax. In this section, we will convert these to numeric. 
# Our machine learning models will have a greater ability to understand numeric 
# features than strings.

head(loans[c("int_rate", "revol_util", "emp_length")], 10)

# Convert int_rate to numeric
loans$int_rate = h2o.gsub(pattern = "%", replacement = "", x = loans$int_rate) # strip %
loans$int_rate = h2o.trim(loans$int_rate) # trim ws
loans$int_rate = as.numeric(loans$int_rate) # change to a numeric

head(loans$int_rate, 10)

# Now that we have converted interest rate to numeric, we can use the hist 
# function to see the distribution of interest rate for good loans and bad loans.

# Bad Loans
h2o.hist(loans[loans$bad_loan == "1", "int_rate"])

# Good Loans
h2o.hist(loans[loans$bad_loan == "0", "int_rate"])
# The distribution of interest rate is very different for good loans. 
# This may be a helpful predictor in our model.

# Convert 'revol_util' to numeric
loans$revol_util = h2o.gsub(pattern = "%", replacement = "", x = loans$revol_util) # strip %
loans$revol_util = h2o.trim(loans$revol_util) # trim ws
loans$revol_util = as.numeric(loans$revol_util) #change to a numeric 

# Convert emp_length to numeric
# 
# Use gsub to remove " year" and " years" also translate n/a to "" 
loans$emp_length = h2o.gsub(pattern = "([ ]*+[a-zA-Z].*)|(n/a)", replacement = "",
                            x = loans$emp_length) 

# Use trim to remove any trailing spaces 
loans$emp_length = h2o.trim(loans$emp_length)

# Convert emp_length to numeric 
# Use sub to convert < 1 to 0 years and do the same for 10 + to 10
# Hint: Be mindful of spaces between characters
loans$emp_length = h2o.gsub(pattern = "< 1", replacement = "0", x = loans$emp_length)
loans$emp_length = h2o.gsub(pattern = "10\\\\+", replacement = "10", x = loans$emp_length)
loans$emp_length = as.numeric(loans$emp_length)

head(loans[c("int_rate", "revol_util", "emp_length")], 10)

# We can also clean up the verification status column. There are multiple values
# that mean verified: 'VERIFIED - income' and 'VERIFIED - income source'. We will 
# replace these values with 'verified'.

head(loans["verification_status"], 10)

loans$verification_status = h2o.sub(pattern = "VERIFIED - income source",
                                    replacement = "verified",
                                    x = loans$verification_status)
loans$verification_status = h2o.sub(pattern="VERIFIED - income",
                                    replacement="verified",
                                    x = loans$verification_status)
loans$verification_status = as.factor(loans$verification_status)

h2o.table(loans$verification_status)

# Step 4 (of 10). Feature engineering -------------------------------------
# 
# Now that we have cleaned our data, we can add some new columns to our dataset 
# that may help improve the performance of our supervised learning models.
# 
# The new columns we will create are:
#   
# * 'credit_length': the time from their earliest credit line to when they were issued the loan
# * 'issue_d_year' and 'issue_d_month': extract year and month from the issue date
# * word embeddings from the loan description
# 
# 
# Credit Length
# 
# We can extract the credit length by subtracting the year they had their 
# earliest credit line from the year when they issued the loan.
loans$credit_length = h2o.year(loans$issue_d) - h2o.year(loans$earliest_cr_line)
head(loans$credit_length, 10)

# Issue Date Expansion
#
# We can extract the year and month from the issue date. We may find that the 
# month or the year when the loan was issued can impact the probability of a bad 
# loan.
loans$issue_d_year = h2o.year(loans$issue_d)
loans$issue_d_month = as.factor(h2o.month(loans$issue_d))  # we will treat month as a enum/factor since its cyclical
head(loans[c("issue_d_year", "issue_d_month")], 10)

# Word Embeddings
# 
# One of the columns in our dataset is a description of why the loan was 
# requested. The first few descriptions in the dataset are shown below.
head(loans$desc)

# This information may be important to the model but supervised learning 
# algorithms have a hard time understanding text. Instead we will convert these
# strings to a numeric vector using the Word2Vec algorithm.

STOP_WORDS = c("ax","i","you","edu","s","t","m","subject","can","lines","re","what",
               "there","all","we","one","the","a","an","of","or","in","for","by","on",
               "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have",
               "from","at","my","be","by","not","that","to","from","com","org","like","likes","so")

tokenize <- function(sentences, stop_word = STOP_WORDS) {
  tokenized = h2o.tokenize(sentences, "\\\\W+")
  tokenized_lower = h2o.tolower(tokenized)
  tokenized_filtered = tokenized_lower[(h2o.nchar(tokenized_lower) >= 2) | is.na(tokenized_lower), ]
  tokenized_words = tokenized_filtered[h2o.grep("[0-9]", tokenized_filtered, invert=T, output.logical=T), ]
  tokenized_words = tokenized_words[is.na(tokenized_words) | !(tokenized_words %in% STOP_WORDS), ]
  return(tokenized_words)
}

# Break loan description into sequence of words
words = tokenize(h2o.ascharacter(loans$desc))

# Train Word2Vec Model
w2v_model = h2o.word2vec(training_frame = words, vec_size = 100, model_id = 'w2v')

# Sanity check - find synonyms for the word 'car'
h2o.findSynonyms(w2v_model, "car", count = 5)

# Calculate a vector for each description
desc_vecs = h2o.transform(w2v_model, words, aggregate_method = "AVERAGE")

head(desc_vecs)

# Add aggregated word embeddings 
loans = h2o.cbind(loans, desc_vecs)


# Step 5 (of 10). Model training ------------------------------------------
#
# Now that we have cleaned our data and added new columns, we will train a model to predict bad loans.

frames = h2o.splitFrame(loans, ratios = .75, seed = 25)
train = frames[[1]]
test = frames[[2]]

cols_to_remove = c("initial_list_status",
                   "out_prncp",
                   "out_prncp_inv",
                   "total_pymnt",
                   "total_pymnt_inv",
                   "total_rec_prncp", 
                   "total_rec_int",
                   "total_rec_late_fee",
                   "recoveries",
                   "collection_recovery_fee",
                   "last_pymnt_d", 
                   "last_pymnt_amnt",
                   "next_pymnt_d",
                   "last_credit_pull_d",
                   "collections_12_mths_ex_med" , 
                   "mths_since_last_major_derog",
                   "policy_code",
                   "loan_status",
                   "funded_amnt",
                   "funded_amnt_inv",
                   "mths_since_last_delinq",
                   "mths_since_last_record",
                   "id",
                   "member_id",
                   "desc",
                   "zip_code")

predictors = setdiff(colnames(loans), cols_to_remove)

gbm_model = h2o.gbm(x = predictors, 
                    y = "bad_loan",
                    training_frame = train,
                    validation_frame = test,
                    stopping_metric = "logloss",
                    stopping_rounds = 5,  # early stopping
                    score_tree_interval = 5,
                    ntrees = 500,
                    model_id = "gbm",
                    nfolds = 5,
                    seed = 25,
                    fold_assignment = 'Stratified')

# Step 6 (of 10). Examine model accuracy ----------------------------------
#
# The plot below shows the performance of the model as more trees are built. 
# This graph can help us see if our model is overfitting. Our early stopping 
# kicked in at 100 trees. This is where the model was no longer improving 
# performance on the test data.

plot(gbm_model)

# The ROC curve of the training and testing data are shown below. The area under 
# the ROC curve is much higher for the training data than the testing data 
# indicating that the model may be beginning to memorize the training data.

print(paste("AUC for training data:", h2o.auc(h2o.performance(gbm_model, train = T))))
plot(h2o.performance(gbm_model, train = T), main = 'Training Data')

print(paste("AUC for x-val data:", h2o.auc(h2o.performance(gbm_model, xval = T))))
plot(h2o.performance(gbm_model, train = T), main = 'Cross-validation')

print(paste("AUC for test data:", h2o.auc(h2o.performance(gbm_model, valid = T))))
plot(h2o.performance(gbm_model, train = T), main = 'Test Data')


# Step 7 (of 10). Interpret model -----------------------------------------

# The variable importance plot shows us which variables are most important to 
# predicting bad_loan. We can use partial dependency plots to learn more about 
# how these variables affect the prediction.

h2o.varimp_plot(gbm_model, 20)
# The partial dependency plot of the 'inq_last_6mths' predictor shows us that, as 
# the number of inquiries in the last 6 months increases, the likelihood of the 
# loan defaulting also increases.

pdp = h2o.partialPlot(gbm_model, cols = "inq_last_6mths", data = train)

head(h2o.table(loans$inq_last_6mths), 100)


# Step 8 (of 10). Save and reuse model ------------------------------------
# 
# The model can either be embedded into a self-contained Java MOJO/POJO package
# or it can be saved and later loaded directly in H2O-3 cluster. For production
# use, we recommend to use MOJO as it is optimised for speed. See 
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html for further
# information.
h2o.download_mojo(gbm_model)


# We can save the model to disk for later batch scoring in H2O cluster.
model_path = h2o.saveModel(gbm_model, path = '.' , force=T)
print(model_path)
loaded_model = h2o.loadModel(path=model_path)

# We can also score new data using the predict function:
bad_loan_hat = predict(gbm_model, test)
head(bad_loan_hat, 15)

# Step 9 (of 10). AutoML (optional) ---------------------------------------
#
# AutoML can be used for automating the machine learning workflow, which 
# includes automatic training and tuning of many models within a user-specified 
# time-limit. Stacked Ensembles will be automatically trained on collections of 
# individual models to produce highly predictive ensemble models.

aml = h2o.automl(x=predictors, 
                 y='bad_loan', 
                 training_frame=train,
                 max_runtime_secs = 300,
                 seed = 25)

# The leaderboard contains the performance metrics of the models generated by AutoML:
aml@leaderboard

# Since we provided only training frame during training, the models are sorted 
# by their cross-validated performance metrics (AUROC by default for 
# classification). We can evaluate the best model ('leader') on the test data:

perf = h2o.performance(aml@leader, newdata = test)
plot(perf)
print(h2o.auc(perf))
print(perf)

# Step 10 (of 10). Shutdown H2O cluster -----------------------------------

h2o.shutdown(prompt = FALSE)

# Bonus -------------------------------------------------------------------
#
# This tutorial: https://github.com/h2oai/h2o-tutorials/tree/master/training/h2o_3_hands_on/lending_club_loan_default
# Other tutorials: https://github.com/h2oai/h2o-tutorials/
# H2O Documentation: http://docs.h2o.ai/
# Awesome H2O: https://github.com/h2oai/awesome-h2o

