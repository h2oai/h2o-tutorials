# Import Modules
import sys, os

import h2o
import pandas

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
def simple_model(w, h, channels, num_classes):
	import json
	import tensorflow as tf
	from tensorflow.python.framework import ops

	graph = tf.Graph()
	with graph.as_default():
		size = w * h * channels
		x = tf.placeholder(tf.float32, [None, size])
		W = tf.Variable(tf.zeros([size, num_classes]))
		b = tf.Variable(tf.zeros([num_classes]))
		y = tf.matmul(x, W) + b
        
		predictions = tf.nn.softmax(y)

		y_ = tf.placeholder(tf.float32, [None, num_classes])
   
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

		tf.add_to_collection(ops.GraphKeys.TRAIN_OP, train_step)
		tf.add_to_collection("predictions", predictions)

		global_step = tf.Variable(0, name="global_step", trainable=False)

		init = tf.global_variables_initializer()
		tf.add_to_collection(ops.GraphKeys.INIT_OP, init.name)
		tf.add_to_collection("logits", y)
		saver = tf.train.Saver()
		meta = json.dumps({
			"inputs": {"batch_image_input": x.name, "categorical_labels": y_.name},
			"outputs": {"categorical_logits": y.name},
			"parameters": {"global_step": global_step.name}
		})
        
		tf.add_to_collection("meta", meta)
		filename = project_path+"/models/mymodel_tensorflow.meta"
		tf.train.export_meta_graph(filename, saver_def=saver.as_saver_def())
	return(filename)

network_def_path = simple_model(28, 28, 1, num_classes)

# Build Deep Water MXNet Model
from h2o.estimators.deepwater import H2ODeepWaterEstimator
model_mnist_mymodel_tf = H2ODeepWaterEstimator(epochs=80, network_definition_file=network_def_path, backend="tensorflow", image_shape=[28,28], channels=1, model_id="model_mnist_mymodel_tf")
model_mnist_mymodel_tf.train(x=["uri"], y="label", training_frame=mnist_training, validation_frame=mnist_testing)

model_mnist_mymodel_tf.show()
