# Import Modules
import h2o
import pandas
import random

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
extracted_features = model_mnist_lenet_mx.deepfeatures(mnist_testing, "flatten0_output")

print(extracted_features.dim)
print(extracted_features[0:4,:])

# Run Similarity Analysis
num_ref_images = 3

pd_mnist_testing = mnist_testing.as_data_frame()
pd_mnist_sample = pd_mnist_testing.sample(n=num_ref_images)

print(pd_mnist_sample)

list_mnist_sample_label = pd_mnist_sample["label"].tolist()
list_mnist_sample_index = pd_mnist_sample.index.values.tolist()
list_mnist_sample_index.sort()
list_mnist_all_index = pd_mnist_testing.index.values.tolist()
set_not_sample_index = set(list_mnist_all_index).symmetric_difference(set(list_mnist_sample_index))
list_mnist_not_sample_index = list(set_not_sample_index)

ref_images = mnist_testing[list_mnist_sample_index]

ref_digits_features = extracted_features[list_mnist_sample_index,:]
rest_digits_features = extracted_features[list_mnist_not_sample_index,:]
h2o.assign(ref_digits_features, "ref_digits_features")
h2o.assign(rest_digits_features, "rest_digits_features")

similarities = rest_digits_features.distance(ref_digits_features, "cosine")
print(similarities.head())

pd_similarities = similarities.as_data_frame()
pd_similarities.sort_values(by="C1", ascending=False, inplace=True)
print(pd_similarities.head())
