# Import Modules
import h2o
import pandas

project_path="/gtc-2017"

# Connect or Start H2O
h2o.init()

# Import Data
mnist_training = h2o.import_file(project_path+"/data/mnist-training.csv")
mnist_testing = h2o.import_file(project_path+"/data/mnist-testing.csv")

mnist_training["label"] = mnist_training["label"].asfactor()
mnist_testing["label"] = mnist_testing["label"].asfactor()

# Explore Data
print(mnist_training.head())

# Build Deep Water Model
from h2o.estimators.deepwater import H2ODeepWaterEstimator
model_mnist_lenet_mx = H2ODeepWaterEstimator(epochs=80, network="lenet")
model_mnist_lenet_mx.train(x=["uri"], y="label", training_frame=mnist_training, validation_frame=mnist_testing, model_id="model_mnist_lenet_mx")

model_mnist_lenet_mx.show()

# Extract Deep Features from Model
extracted_features = model_mnist_lenet_mx.deepfeatures(mnist_training, "flatten0_output")

print(extracted_features.dim)
print(extracted_features[0:4,:])

# Build New Training Set with Deep Features
extracted_features["label"] = mnist_training["label"]

deep_train, deep_valid = extracted_features.split_frame(ratios=[0.8], destination_frames=["deep_train", "deep_valid"])

# Build GBM Model Using Deep Features
from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm_mnist_lenet_mx = H2OGradientBoostingEstimator(ntrees=80)
gbm_mnist_lenet_mx.train(x=[x for x in extracted_features.columns if x != "label"], y="label", training_frame=deep_train, validation_frame=deep_valid, model_id="gbm_mnist_lenet_mx")

gbm_mnist_lenet_mx.show()
