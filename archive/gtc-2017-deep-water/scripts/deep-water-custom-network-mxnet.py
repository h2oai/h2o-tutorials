# Import Modules
import sys, os
sys.path.insert(1, "/usr/local/lib/python2.7/dist-packages/mxnet-0.7.0-py2.7.egg")
sys.path.insert(1, "/usr/local/lib/python2.7/dist-packages")

import h2o
import pandas
import mxnet as mx

project_path = "/gtc-2017"

# Connect or Start H2O
h2o.init()

# Import Data
mnist_training = h2o.import_file(project_path+"/data/mnist-training.csv")
mnist_testing = h2o.import_file(project_path+"/data/mnist-testing.csv")

mnist_training["label"] = mnist_training["label"].asfactor()
mnist_testing["label"] = mnist_testing["label"].asfactor()

# Explore Data
print(mnist_training.head())

num_classes = mnist_training["label"].nlevels()[0]
print(num_classes)

# Define Custom Network
def lenet(num_classes):
	# Input layer
	data = mx.symbol.Variable("data")

	# Convolution layer 1
	conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
	act1 = mx.symbol.Activation(data=conv1, act_type="tanh")
	pool1 = mx.symbol.Pooling(data=act1, pool_type="max", kernel=(2,2), stride=(2,2))

	# Convolution layer 2
	conv2 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=50)
	act2 = mx.symbol.Activation(data=conv2, act_type="tanh")
	pool2 = mx.symbol.Pooling(data=act2, pool_type="max", kernel=(2,2), stride=(2,2))

	# Fully connected layer 1
	flatten = mx.symbol.Flatten(data=pool2)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
	act3 = mx.symbol.Activation(data=fc1, act_type="tanh")

	# Fully connected layer 2
	fc2 = mx.symbol.FullyConnected(data=act3, num_hidden=500)
	out = mx.symbol.SoftmaxOutput(data=fc2, name="softmax")
	return out

sym_my_lenet = lenet(num_classes)

network_def_path = project_path+"/models/sym_my_lenet.json"
sym_my_lenet.save(network_def_path)

# Build Deep Water MXNet Model
from h2o.estimators.deepwater import H2ODeepWaterEstimator
model_mnist_mylenet_mx = H2ODeepWaterEstimator(epochs=80, network_definition_file=network_def_path, image_shape=[28,28], channels=1, model_id="model_mnist_mylenet_mx")
model_mnist_mylenet_mx.train(x=["uri"], y="label", training_frame=mnist_training, validation_frame=mnist_testing)

model_mnist_mylenet_mx.show()
