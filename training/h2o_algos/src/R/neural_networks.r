
# import h2o lib and allow it to use max. threads
library(h2o)
h2o.init(nthreads = -1)

# location of clean data file
path <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"

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

# split into training and valid
split <- h2o.splitFrame(frame, ratios = 0.7)
train <- split[[1]]
valid <- split[[2]]

# neural network

# train
loan_nn <- h2o.deeplearning(
    x = X,
    y = y,
    training_frame = train,
    validation_frame = valid,
    epochs = 50,                     # read over the data 50 times, but in mini-batches
    hidden = c(100),                 # 100 hidden units in 1 hidden layer
    input_dropout_ratio = 0.2,       # randomly drop 20% of inputs for each iteration, helps w/ generalization
    hidden_dropout_ratios = c(0.05), # randomly set 5% of hidden weights to 0 each iteration, helps w/ generalization
    activation = "TanhWithDropout",  # bounded activation function that allows for dropout, tanh, more stable
    l1 = 0.001,                      # L1 penalty can help generalization   
    l2 = 0.01,                       # L2 penalty can increase stability in presence of highly correlated inputs
    adaptive_rate = TRUE,            # adjust magnitude of weight updates automatically (+stability, +accuracy)
    stopping_rounds = 5,             # stop after validation error does not decrease for 5 iterations
    score_each_iteration = TRUE,     # score validation error on every iteration, use with caution
    model_id = "loan_nn")            # for easy lookup in flow

# print model information
loan_nn

# view detailed results at http://ip:port/flow/index.html

# NN with random hyperparameter search
# train many different NN models with random hyperparameters

# define random grid search parameters
hidden_opts = c(c(170, 320), c(80, 190), c(320, 160, 80), c(100), c(50, 50, 50, 50))
l1_opts = seq(0.001, 0.01, 0.001)
l2_opts = seq(0.001, 0.01, 0.001)
input_dropout_ratio_opts = seq(0.01, 0.5, 0.01)


hyper_params = list(hidden = hidden_opts,
                    l1 = l1_opts,
                    l2 = l2_opts,
                    input_dropout_ratio = input_dropout_ratio_opts)

# search a random subset of these hyper-parmameters
# max runtime and max models are enforced
# and the search will stop after not improving much over the best 5 random models
search_criteria = list(strategy = "RandomDiscrete", 
                       max_runtime_secs = 600, 
                       max_models = 20, 
                       stopping_metric = "AUC", 
                       stopping_tolerance = 0.0001, 
                       stopping_rounds = 5, 
                       seed = 123456)

# execute training w/ grid search
loan_nn2 <- h2o.grid("deeplearning", 
                     grid_id = "nn_grid",
                     x = X, 
                     y = y, 
                     training_frame = train,
                     validation_frame = valid,
                     
                     # per model stopping criteria 
                     stopping_rounds = 2,
                     stopping_tolerance = 1e-3,
                     stopping_metric = "AUC",
                     
                     # how often to score (affects early stopping, and training speed)
                     score_interval = 5, 
                     
                     # seed to control sampling
                     seed = 12345,
                     
                     # grid serach options
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# view detailed results at http://ip:port/flow/index.html

# show grid results
sorted_grid <- h2o.getGrid(grid_id = "nn_grid")
print(sorted_grid)

# select best model
best_model <- h2o.getModel(sorted_grid@model_ids[[1]])
summary(best_model)

# use partial dependence plots to get insight into important relationships
h2o.partialPlot(best_model, valid, "int_rate")

h2o.shutdown(prompt = FALSE)
