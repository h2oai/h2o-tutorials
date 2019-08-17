# Import Modules
import h2o
import pandas

from h2o.estimators.deepwater import H2ODeepWaterEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

project_path="/gtc-2017"

# Connect or Start H2O
h2o.init()

# Import Data
train = h2o.import_file(project_path+"/data/train-odd.csv.gz")
valid = h2o.import_file(project_path+"/data/test-odd.csv.gz")

features = list(range(0,784))
target = 784

train[features] = train[features]/255
train[target] = train[target].asfactor()
valid[features] = valid[features]/255
valid[target] = valid[target].asfactor()

# Explore Data
print(train.head())

# Cross-Validation
nfolds = 5

# Build GBM Model
gbm_model = H2OGradientBoostingEstimator(distribution="bernoulli", ntrees=100, nfolds=nfolds, ignore_const_cols=False, keep_cross_validation_predictions=True, fold_assignment="Modulo")
gbm_model.train(x=features, y=target, training_frame=train, model_id="gbm_model")
gbm_model.show()

# Build GLM Model
glm_model = H2OGeneralizedLinearEstimator(family="binomial", lambda_=0.0001, alpha=0.5, nfolds=nfolds, ignore_const_cols=False, keep_cross_validation_predictions=True, fold_assignment="Modulo")
glm_model.train(x=features, y=target, training_frame=train, model_id="glm_model")
glm_model.show()

# Build Deep Water Model
dw_model = H2ODeepWaterEstimator(epochs=3, network="lenet", ignore_const_cols=False, image_shape=[28,28], channels=1, standardize=False, seed=1234, nfolds=nfolds, keep_cross_validation_predictions=True, fold_assignment="Modulo")
dw_model.train(x=features, y=target, training_frame=train, model_id="dw_model")
dw_model.show()

# Ensemble Models
stack_all = H2OStackedEnsembleEstimator(base_models=[gbm_model.model_id, glm_model.model_id, dw_model.model_id])
stack_all.train(x=features, y=target, training_frame=train, validation_frame=valid, model_id="stack_all")
stack_all.model_performance()
