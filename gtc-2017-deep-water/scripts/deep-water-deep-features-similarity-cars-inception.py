# Import Modules
import h2o
import pandas
import random

project_path="/gtc-2017"

# Connect or Start H2O
h2o.init()

# Import Data
cars = h2o.import_file(project_path+"/data/cars_train.csv", header=1, destination_frame="cars")

cars["label"] = cars["label"].asfactor()

# Explore Data
print(cars.head())

# Build Deep Water Model
from h2o.estimators.deepwater import H2ODeepWaterEstimator
model = H2ODeepWaterEstimator(epochs=0, mini_batch_size=32, network="user", network_definition_file=project_path+"/models/Inception_BN-symbol.json", network_parameters_file=project_path+"/models/Inception_BN-0039.params", mean_image_file=project_path+"/models/mean_224.nd", image_shape=[224,224], channels=3)
model.train(x="uri", y="label", training_frame=cars)

# Extract Deep Features from Model
extracted_features = model.deepfeatures(cars, "global_pool_output")

print(extracted_features.dim)
print(extracted_features[0:4,:])

# Run Similarity Analysis
num_ref_images = 3

pd_cars = cars.as_data_frame()
pd_cars_sample = pd_cars.sample(n=num_ref_images, random_state=1234)

print(pd_cars_sample)

list_cars_sample_label = pd_cars_sample["label"].tolist()
list_cars_sample_index = pd_cars_sample.index.values.tolist()
list_cars_sample_index.sort()
list_cars_all_index = pd_cars.index.values.tolist()
set_not_sample_index = set(list_cars_all_index).symmetric_difference(set(list_cars_sample_index))
list_cars_not_sample_index = list(set_not_sample_index)

ref_cars_features = extracted_features[list_cars_sample_index,:]
rest_cars_features = extracted_features[list_cars_not_sample_index,:]
h2o.assign(ref_cars_features, "ref_cars_features")
h2o.assign(rest_cars_features, "rest_cars_features")

similarities = rest_cars_features.distance(ref_cars_features, "cosine")
print(similarities.head())

pd_similarities = similarities.as_data_frame()
pd_similarities.sort_values(by="C1", ascending=False, inplace=True)
print(pd_similarities.head())
