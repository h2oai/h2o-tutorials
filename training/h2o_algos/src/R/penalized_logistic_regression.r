
# import h2o lib and allow it to use max. threads
library(h2o)
h2o.init(nthreads = -1)

# location of clean data file
path <- "/Users/phall/Documents/aetna/share/data/loan.csv"

# import file
frame <- h2o.importFile(path)

# strings automatically parsed as enums (categorical)
# numbers automatically parsed as numeric
# bad_loan is numeric, but categorical
frame$bad_loan <- as.factor(frame$bad_loan)

# find missing numeric and impute
for (name in names(frame)) {
  if (any(is.na(frame[name]))) {
      h2o.impute(frame, name, "median")
  }
}

h2o.describe(frame) # summarize table, check for missing

# assign target and inputs
y <- 'bad_loan'
X <- names(frame)[names(frame) != y]
print(y)
print(X)

# split into training and test for cross validation
split <- h2o.splitFrame(frame, ratios = 0.7)
train <- split[[1]]
test <- split[[2]]

# elastic net regularized regression
#   - binomial family for logistic regression
#   - L1 for variable selection
#   - L2 for handling multicollinearity
#   - IRLS for handling outliers
#   - standardization very important for penalized regression variable selection
#   - with lamba parameter tuning for variable selection and regularization

# train
loan_glm <- h2o.glm(x = X, 
                    y = y,
                    training_frame = train,
                    validation_frame = test,
                    family = "binomial",
                    model_id = "loan_glm",
                    solver = "IRLSM",
                    standardize = TRUE, 
                    lambda_search = TRUE)

# print model
loan_glm

# view detailed results at http://ip:port/flow/index.html

# print sorted, non-zero model parameters
coef <- as.data.frame(h2o.coef(loan_glm))
names(coef) <- "coef"
coef <- coef[order(-coef$coef), , drop = FALSE]
coef <- coef[coef$coef != 0, , drop = FALSE] 
coef

h2o.shutdown(prompt = FALSE)
