# Regression Example based on Boston Housing Data


# ------------------------------------------------------------------------------
# Start H2O
# ------------------------------------------------------------------------------

library(h2o)
h2o.init(nthreads = -1)

# ------------------------------------------------------------------------------
# Import file from internet
# ------------------------------------------------------------------------------

# Import from GitHub
# train = h2o.importFile("https://github.com/woobe/h2o_training_2017_10/raw/master/examples/regression/house_price_train.csv")
# test = h2o.importFile("https://github.com/woobe/h2o_training_2017_10/raw/master/examples/regression/house_price_test.csv")

# or import the files locally if you have them

train_main = h2o.importFile("https://raw.githubusercontent.com/h2oai/h2o-tutorials/master/training/ceder-rapids/prostate.csv")

data_split = h2o.splitFrame(train_main, ratios = 0.75, seed = 1234)
train = data_split[[1]]
test = data_split[[2]]

# ------------------------------------------------------------------------------
# Have a quick look
# ------------------------------------------------------------------------------

h2o.describe(train)
h2o.describe(test)

h2o.summary(train)
h2o.summary(test)


# ------------------------------------------------------------------------------
# Define features and target
# ------------------------------------------------------------------------------

features = setdiff(colnames(train), "CAPSULE")
target = "CAPSULE"


# ------------------------------------------------------------------------------
# Build Out-of-Box GLM Model
# ------------------------------------------------------------------------------

model_glm = h2o.glm(x = features,
                    y = target,
                    training_frame = train,
                    seed = 1234)
summary(model_glm)
h2o.varimp(model_glm)


#
# Changing response variable to categorical for classification
#
train_main$CAPSULE = h2o.asfactor(train_main$CAPSULE)

data_split = h2o.splitFrame(train_main, ratios = 0.75, seed = 1234)
train = data_split[[1]]
test = data_split[[2]]


# ------------------------------------------------------------------------------
# Build Out-of-Box Distributed Random Forest Model
# ------------------------------------------------------------------------------

model_drf = h2o.randomForest(x = features,
                             y = target,
                             training_frame = train,
                             seed = 1234)
summary(model_drf)
h2o.varimp(model_drf)
h2o.varimp_plot(model_drf, num_of_features = 5)


# ------------------------------------------------------------------------------
# Build Out-of-Box Gradient Boosting Machine
# ------------------------------------------------------------------------------

model_gbm = h2o.gbm(x = features,
                    y = target,
                    training_frame = train,
                    seed = 1234)
summary(model_gbm)
h2o.varimp(model_gbm)
h2o.varimp_plot(model_gbm, num_of_features = 10)


# ------------------------------------------------------------------------------
# Build Out-of-Box Deep Neural Network
# ------------------------------------------------------------------------------

model_dnn = h2o.deeplearning(x = features,
                             y = target,
                             training_frame = train,
                             seed = 1234)
summary(model_dnn)
h2o.varimp(model_dnn)
h2o.varimp_plot(model_dnn, num_of_features = 10)


# ------------------------------------------------------------------------------
# Use models for prediction
# ------------------------------------------------------------------------------

yhat_glm = h2o.predict(model_glm, newdata = test)
yhat_drf = h2o.predict(model_drf, newdata = test)
yhat_gbm = h2o.predict(model_gbm, newdata = test)
yhat_dnn = h2o.predict(model_dnn, newdata = test)

head(yhat_glm)
head(yhat_drf)
head(yhat_gbm)
head(yhat_dnn)


# ------------------------------------------------------------------------------
# Convert H2O data frame into normal R data frame
# ------------------------------------------------------------------------------

df_yhat_glm = as.data.frame(yhat_glm)
head(df_yhat_glm)


# ------------------------------------------------------------------------------
# Use a Validation set
# ------------------------------------------------------------------------------

# Split training data into 75:25 train:valid
data_split = h2o.splitFrame(train_main, ratios = 0.75, seed = 1234)
new_train = data_split[[1]]
valid = data_split[[2]]

model_gbm2 = h2o.gbm(x = features,
                     y = target,
                     training_frame = new_train,
                     validation_frame = valid,   # use validation set
                     seed = 1234)
summary(model_gbm2)


# ------------------------------------------------------------------------------
# Use Cross-Validation
# ------------------------------------------------------------------------------

model_gbm_cv = h2o.gbm(x = features,
                       y = target,
                       training_frame = train,
                       nfolds = 3,   # setting for cross-validation
                       fold_assignment = "AUTO",
                       seed = 1234)
print(model_gbm_cv)


# ------------------------------------------------------------------------------
# Use H2O AutoML
# ------------------------------------------------------------------------------

aml = h2o.automl(x = features,
                 y = target,
                 training_frame = train,
                 seed = 1234,
                 max_runtime_secs = 60, # increase this to build more models
                 max_models = 100) # set a hard limit of no. of models

print(aml@leaderboard)

# Best Model (leader) based on AutoML
model_best = aml@leader
yhat_best = h2o.predict(model_best, newdata = test)

# or Choose it yourself
# model_handpick = h2o.getModel(model_id = "copy_id_of_model_you_want")

