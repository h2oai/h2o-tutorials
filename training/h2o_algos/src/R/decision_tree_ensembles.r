
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

# summarize table
# decision tree ensembles can run on missing values
h2o.describe(frame)

# assign target and inputs
y <- 'bad_loan'
X <- names(frame)[names(frame) != y]
print(y)
print(X)

# split into training and valid
split <- h2o.splitFrame(frame, ratios = 0.7)
train <- split[[1]]
valid <- split[[2]]

# random forest
# random forest is often the best guess model with little tuning

# train
loan_rf <- h2o.randomForest(
    x = X,
    y = y,
    training_frame = train,
    validation_frame = valid,
    ntrees = 500,                      # Up to 500 decision trees in the forest 
    max_depth = 30,                    # trees can grow to depth of 30
    stopping_rounds = 2,               # stop after validation error does not decrease for 5 iterations/new trees
    score_each_iteration = TRUE,       # score validation error on every iteration/new tree
    model_id = "loan_rf")              # for easy lookup in flow

# print model information
loan_rf

# view detailed results at http://ip:port/flow/index.html

# GBM with random hyperparameter search
# GBM often more accurate than random forest but requires some tuning

# define random grid search parameters
ntrees_opts = seq(100, 500, 50)
max_depth_opts = seq(2, 20, 2)
sample_rate_opts = seq(0.1, 0.9, 0.1)
col_sample_rate_opts = seq(0.1, 0.9, 0.1)

hyper_params = list(ntrees = ntrees_opts,
                    max_depth = max_depth_opts,
                    sample_rate = sample_rate_opts,
                    col_sample_rate = col_sample_rate_opts)

# search a random subset of these hyper-parmameters
# max runtime and max models are enforced
# and the search will stop after not improving much over the best 5 random models
search_criteria = list(strategy = "RandomDiscrete", 
                       max_runtime_secs = 300, 
                       max_models = 10, 
                       stopping_metric = "AUC", 
                       stopping_tolerance = 0.0001, 
                       stopping_rounds = 5, 
                       seed = 12345)

# execute training w/ grid search
loan_gbm <- h2o.grid("xgboost", 
                     grid_id = "gbm_grid",
                     x = X, 
                     y = y, 
                     training_frame = train,
                     validation_frame = valid,
                     
                     # per model stopping criteria 
                     stopping_rounds = 2,
                     stopping_tolerance = 1e-3,
                     stopping_metric = "AUC",
                     
                     # how often to score (affects early stopping, training speed)
                     score_tree_interval = 5, 
                     
                     # seed to control sampling 
                     seed = 12345,
                     # grid serach options
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# view detailed results at http://ip:port/flow/index.html

# show grid results
sorted_grid <- h2o.getGrid(grid_id = "gbm_grid")
print(sorted_grid)

# select best model
best_model <- h2o.getModel(sorted_grid@model_ids[[1]])
summary(best_model)

# use variable importance to get insight into important relationships
h2o.varimp(best_model)

# use partial dependence plots to get insight into important relationships
h2o.partialPlot(best_model, valid, "int_rate")

h2o.shutdown(prompt = FALSE)
