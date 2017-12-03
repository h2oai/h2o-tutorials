library(h2o)
h2o.init()

# Import a binary classfication dataset:
# We will use a modified version of of the Product Backorders dataset
# For demonstration purposes, we are using a rebalanced subset of the original dataset
# Source: https://www.kaggle.com/tiredgeek/predict-bo-trial/data

# If using the docker image, use this path
data_path <- "/home/h2o/data/automl/product_backorders.csv"

# Load data into H2O
df <- h2o.importFile(data_path)


# For classification, the response should be encoded as categorical (aka. "factor" or "enum")
# Let's take a look
h2o.describe(df)


# Identify response
y <- "went_on_backorder"

# Predictor columns: All columns except the response and a unique ID column ("sku")
x <- setdiff(names(df), c(y, "sku"))


# Run AutoML (for 10 models)
aml <- h2o.automl(y = y, x = x,
                  training_frame = df,
                  max_models = 10,
                  seed = 1)

# View the AutoML Leaderboard
lb <- aml@leaderboard
lb

# Print the entire AutoML Leaderboard
print(lb, nrow(lb))

# The leader model is stored here
aml@leader


# Look at which models contribute to the "StackedEnsemble_AllModels" ensemble:
# Get model ids for all models in the AutoML Leaderboard
model_ids <- as.data.frame(aml@leaderboard$model_id)[,1]
# Grab SE model ID, then get the model
se_allmodels_id <- grep("StackedEnsemble_AllModels", model_ids, value = TRUE)
se <- h2o.getModel(se_allmodels_id)
# Get the SE metalearner model
meta <- h2o.getModel(se@model$metalearner$name)

# Look at base learner contributions to the ensemble
h2o.varimp(meta)

# Plot the base learner contributions to the ensemble
h2o.varimp_plot(meta)


# Save the leader model in binary format
h2o.saveModel(aml@leader, path = "./product_backorders_automl_leadermodel_bin")

# Export the leader model as a MOJO for production use
h2o.download_mojo(aml@leader, path = "./")
