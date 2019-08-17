# This script is meant to show an example of gbm tuning
# It is based off of this blog post: http://blog.h2o.ai/2016/06/h2o-gbm-tuning-tutorial-for-r/

# Launch H2O Cluster
library('h2o')
h2o.init(nthreads = -1)


# Import data
df <- h2o.importFile(path = "../data/titanic.csv")

# Convert some integer columns to factor/categorical
df$survived <- as.factor(df$survived)
df$ticket <- as.factor(df$ticket)

# Set predictors and response variable
response <- "survived"
predictors <- colnames(df)[!(colnames(df) %in% c("survived", "name"))]

# Split the data for machine learning
splits <- h2o.splitFrame(
  data = df, 
  ratios = c(0.6,0.2),   ## only need to specify 2 fractions, the 3rd is implied
  destination_frames = c("train.hex", "valid.hex", "test.hex"), seed = 1234
)
train <- splits[[1]]
valid <- splits[[2]]
test  <- splits[[3]]

# Establish a baseline

glm_model <- h2o.glm(x = predictors, y = response, training_frame = train, validation_frame = valid,
                     family = "binomial", model_id = "glm_default.hex")

drf_model <- h2o.randomForest(x = predictors, y = response, training_frame = train, validation_frame = valid,
                              model_id = "drf_default.hex")

gbm_model <- h2o.gbm(x = predictors, y = response, training_frame = train, validation_frame = valid,
                     model_id = "gbm_default.hex")

dl_model <- h2o.deeplearning(x = predictors, y = response, training_frame = train, validation_frame = valid,
                             model_id = "dl_default.hex")


baseline_results <- data.frame('model' = c("GLM", "DRF", "GBM", "DL"),
                               'training auc' = c(h2o.auc(glm_model, train = T), h2o.auc(drf_model, train = T), h2o.auc(gbm_model, train = T), h2o.auc(dl_model, train = T)),
                               'validation_auc' = c(h2o.auc(glm_model, train = T), h2o.auc(drf_model, valid = T), h2o.auc(gbm_model, valid = T), h2o.auc(dl_model, valid = T)))

print(baseline_results)

# Investigate Model

# Build Partial Dependency Plots to understand the model
h2o.partialPlot(gbm_model, train, cols = c("sex", "age"), plot = TRUE, plot_stddev = FALSE)

# Decrease Learning Rate

gbm_learn_rate <- h2o.gbm(x = predictors, y = response, training_frame = train, validation_frame = valid,
                          learn_rate = 0.05, model_id = "gbm_learnrate.hex")

print(paste0("Learn Rate AUC: ", h2o.auc(gbm_learn_rate, valid = TRUE)))

# Early Stopping
gbm_early_stopping <- h2o.gbm(x = predictors, y = response, training_frame = train, validation_frame = valid,
                              # Early stopping once the moving average (window length = 5) of the validation AUC 
                              # doesn’t improve by at least 0.1% for 5 consecutive scoring events
                              score_tree_interval = 10, stopping_rounds = 5, stopping_metric = "AUC", stopping_tolerance = 0.001,
                              ntrees = 5000,
                              learn_rate = 0.05, model_id = "gbm_early_stopping.hex")

print(paste0("Early Stopping AUC: ", h2o.auc(gbm_early_stopping, valid = TRUE)))

# Use Cartesian Grid Search to find best max depth
# Max depth can have a big impact on training time so we will first narrow down the best max depths
hyper_params = list( max_depth = seq(1, 25, 2))

grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## full Cartesian hyper-parameter search
  search_criteria = list(strategy = "Cartesian"),
  
  ## which algorithm to run
  algorithm="gbm",
  
  ## identifier for the grid, to later retrieve it
  grid_id="depth_grid",
  
  ## standard model parameters
  x = predictors, 
  y = response, 
  training_frame = train, 
  validation_frame = valid,
  
  ## more trees is better if the learning rate is small enough 
  ## here, use "more than enough" trees - we have early stopping
  ntrees = 5000,                                                            
  
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  
  ## sample 80% of rows per tree
  sample_rate = 0.8,                                                       
  
  ## sample 80% of columns per split
  col_sample_rate = 0.8, 
  
  ## fix a random number generator seed for reproducibility
  seed = 1234,                                                             
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5,
  stopping_tolerance = 0.001,
  stopping_metric = "AUC", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10                                                
)

## by default, display the grid search results sorted by increasing logloss (since this is a classification task)
grid                                                                       

## sort the grid models by decreasing AUC
sorted_grid <- h2o.getGrid("depth_grid", sort_by="auc", decreasing = TRUE)    
sorted_grid@summary_table[c(1:5), ]

## find the range of max_depth for the top 5 models
top_depths = sorted_grid@summary_table$max_depth[1:5]                       
new_min = min(as.numeric(top_depths))
new_max = max(as.numeric(top_depths))

# Final Random Discrete Hyper-parameterization
hyper_params_tune = list( 
  ## restrict the search to the range of max_depth established above
  max_depth = seq(new_min, new_max, 1),                                      
  
  ## search a large space of row sampling rates per tree
  sample_rate = seq(0.2, 1, 0.01),                                             
  
  ## search a large space of column sampling rates per split
  col_sample_rate = seq(0.2, 1, 0.01),                                
  
  ## search a large space of the number of min rows in a terminal node
  min_rows = 2^seq(0, log2(nrow(train))-1, 1),                                                    
  
  ## search a large space of the number of bins for split-finding for categorical columns
  nbins_cats = 2^seq(4, 12, 1),                                                 
  
  ## try all histogram types (QuantilesGlobal are good for numeric columns with outliers)
  histogram_type = c("UniformAdaptive", "QuantilesGlobal")       
)

search_criteria_tune = list(
  ## Random grid search
  strategy = "RandomDiscrete",      
  
  ## limit the runtime to 60 minutes
  max_runtime_secs = 3600,         
  
  ## build no more than 100 models
  max_models = 100,                  
  
  ## random number generator seed to make sampling of parameter combinations reproducible
  seed = 1234,                        
  
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 5,                
  stopping_metric = "AUC",
  stopping_tolerance = 0.001
)

final_grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params_tune,
  
  ## hyper-parameter search configuration (see above)
  search_criteria = search_criteria_tune,
  
  ## which algorithm to run
  algorithm = "gbm",
  
  ## identifier for the grid, to later retrieve it
  grid_id = "final_grid", 
  
  ## standard model parameters
  x = predictors, 
  y = response, 
  training_frame = train, 
  validation_frame = valid,
  
  ## more trees is better if the learning rate is small enough
  ## use "more than enough" trees - we have early stopping
  ntrees = 5000,                                                            
  
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  
  ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
  max_runtime_secs = 3600,                                                 
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 0.001, stopping_metric = "AUC", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234                                                             
)

## Sort the grid models by AUC
sorted_final_grid <- h2o.getGrid("final_grid", sort_by = "auc", decreasing = TRUE)    
sorted_final_grid@summary_table[c(1:5), ]

# Final Test Scoring
# How well does our best model do on the final hold out dataset

best_model <- h2o.getModel(sorted_final_grid@model_ids[[1]])

print(paste0("AUC on validation: ", h2o.auc(best_model, valid = TRUE)))
print(paste0("AUC on test: ", h2o.auc(h2o.performance(best_model, test))))

# Shutdown h2o cluster
h2o.shutdown()