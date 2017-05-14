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
import tensorflow as tf
import json
from keras.layers.core import Dense, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.objectives import categorical_crossentropy
from tensorflow.python.framework import ops

def keras_model(w, h, channels, num_classes):
	graph = tf.Graph()
	with graph.as_default():
		size = w * h * channels

		inp = tf.placeholder(tf.float32, [None, size])

		labels = tf.placeholder(tf.float32, [None, num_classes])

		x = Reshape((w, h, channels))(inp)
		x = Conv2D(20, (5,5), padding="same", activation="relu")(x)
		x = MaxPooling2D((2,2))(x)
 
		x = Conv2D(50, (5,5), padding="same", activation="relu")(x)
		x = MaxPooling2D((2,2))(x)

		x = Flatten()(x)

		x = Dense(num_classes)(x)

		out = Dense(num_classes)(x)

		predictions = tf.nn.softmax(out)

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=out))
		train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

		init_op = tf.global_variables_initializer()

		tf.add_to_collection(ops.GraphKeys.INIT_OP, init_op.name)
		tf.add_to_collection(ops.GraphKeys.TRAIN_OP, train_step)
		tf.add_to_collection("logits", out)
		tf.add_to_collection("predictions", predictions)

		meta = json.dumps({
			"inputs": {"batch_image_input": inp.name, "categorical_labels": labels.name},
			"outputs": {"categorical_logits": out.name, "layers": ",".join([m.name for m in tf.get_default_graph().get_operations()])},
			"parameters": {}
		})

		tf.add_to_collection("meta", meta)

		saver = tf.train.Saver()
		filename= project_path+"/models/keras_tensorflow_lenet.meta"
		tf.train.export_meta_graph(filename, saver_def=saver.as_saver_def())
	return(filename)

network_def_path = keras_model(28, 28, 1, num_classes)

# Build Deep Water MXNet Model
from h2o.estimators.deepwater import H2ODeepWaterEstimator
model_mnist_mylenet_keras = H2ODeepWaterEstimator(epochs=80, network_definition_file=network_def_path, backend="tensorflow", image_shape=[28,28], channels=1, model_id="model_mnist_mylenet_keras")
model_mnist_mylenet_keras.train(x=["uri"], y="label", training_frame=mnist_training, validation_frame=mnist_testing)

model_mnist_mylenet_keras.show()
