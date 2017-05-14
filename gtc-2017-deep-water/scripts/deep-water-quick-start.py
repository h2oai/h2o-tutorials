# Import Modules
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

#	Build Deep Water Models
from h2o.estimators.deepwater import H2ODeepWaterEstimator
model_mnist_lenet_mx = H2ODeepWaterEstimator(epochs=80, network="lenet")
model_mnist_lenet_mx.train(x=["uri"], y="label", training_frame=mnist_training, validation_frame=mnist_testing, model_id="model_mnist_lenet_mx")
model_mnist_lenet_mx.show()

model_mnist_lenet_tf = H2ODeepWaterEstimator(epochs=80, network="lenet", backend="tensorflow")
model_mnist_lenet_tf.train(x=["uri"], y="label", training_frame=mnist_training, validation_frame=mnist_testing, model_id="model_mnist_lenet_tf")
model_mnist_lenet_tf.show()

#model_mnist_lenet_caffe = H2ODeepWaterEstimator(epochs=80, network="lenet", backend="caffe")
#model_mnist_lenet_caffe.train(x=["uri"], y="label", training_frame=mnist_training, validation_frame=mnist_testing, model_id="model_mnist_lenet_caffe")
#model_mnist_lenet_caffe.show()

#	Save H2O Model
h2o.save_model(model=model_mnist_lenet_mx, path=project_path+"/models/", force=True)
h2o.save_model(model=model_mnist_lenet_tf, path=project_path+"/models/", force=True)
#h2o.save_model(model=model_mnist_lenet_caffe, path=project_path+"/models/", force=True)

# Export MOJOs for Deployment
model_mnist_lenet_mx.download_mojo(path=project_path+"/models/", get_genmodel_jar=True)
model_mnist_lenet_tf.download_mojo(path=project_path+"/models/", get_genmodel_jar=True)
#model_mnist_lenet_caffe.download_mojo(path=project_path+"/models/", get_genmodel_jar=True)
