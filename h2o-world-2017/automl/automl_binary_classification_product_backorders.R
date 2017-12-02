library(h2o)
h2o.init()

# Import a binary classfication dataset
# We will use a subset of the Product Backorders dataset
# Source: https://www.kaggle.com/haimfeld87/predict-product-backorders-with-smote-and-rf/data
df <- h2o.importFile("/Users/me/h2oai/code/h2o-tutorials/h2o-world-2017/automl/product_backorders.csv")
#df <- h2o.importFile("/home/h2o/data/automl/product_backorders.csv")

# For classification, the response should be encoded as categorical (aka. "factor" or "enum")
# Let's take a look
h2o.describe(df)


# Identify response & predictor columns
y <- "went_on_backorder"
x <- setdiff(names(df), c(y, "sku"))


splits <- h2o.splitFrame(df, ratios = 0.8, seed = 1)
train <- splits[[1]]
test <- splits[[2]]


# Run AutoML (for 10 models)
aml <- h2o.automl(y = y, x = x,
                  training_frame = train,
                  max_models = 10,
                  seed = 1)

# View the AutoML Leaderboard
lb <- aml@leaderboard
lb

# The leader model is stored here
aml@leader


# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly
pred <- h2o.predict(aml, test)  # predict(aml, test) also works or:
pred <- h2o.predict(aml@leader, test)


# You can also use the standard h2o.performance() function on a test set (leader model)
perf <- h2o.performance(aml, test)
