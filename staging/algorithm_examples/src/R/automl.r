
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

# start automl process
# automl loosely based on: http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
auto <- h2o.automl(x = X, 
                   y = y,
                   training_frame = train, # training automatically split into 70% train, 30% validation
                   leaderboard_frame = test,
                   max_runtime_secs = 300) # will run for 300 seconds, then build a stacked ensemble

# view leaderboard
lb <- auto@leaderboard
lb

# view best model
best <- auto@leader
best # can only be used for predict with .predict(), no MOJO for stacked ensemble yet

h2o.shutdown(prompt = FALSE)
