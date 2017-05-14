##	Overview
This repository contains the material from the instructor-led lab _Train Deep Learning Models with H2O Deep Water_ at [GTC](http://www.gputechconf.com/) 2017 led by Wen Phan, Magnus Stensmo, and Arno Candel.  The material is a good introduction to Deep Water.

The contents of the repository and how to use them are described below.  The presentation slides are [here](https://github.com/h2oai/gtc-2017/blob/master/deepwater-lab-gtc2017.pdf) .

During the lab, no previous experience with H2O was necessary.  However, if you are completely new to H2O, here are some useful links and videos.

*	[User Guide](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html)
* 	[Quick Start Videos](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/quick-start-videos.html)
*  [General Installation Instructions and Dependencies](http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/7/index.html)

Additional, Deep Water examples can be found [here](https://github.com/h2oai/h2o-3/tree/master/examples/deeplearning/notebooks).

Deep Water expands H2O deep learning capabilities to leverage GPUs, open source deep learning frameworks, and modern deep learning architectures.  [H2O Deep Learning](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html) can still be used to build world-class multi-layer perception (MLP) deep learning networks using CPUs on a distributed in-memory cluster.

##	Installation
To go run through the examples in this repo, you will need to have Deep Water installed.  There are various ways to install Deep Water, and they are all described in detail in the Deep Water repository: <https://github.com/h2oai/deepwater>

Deep Water is not yet officially released, so all installations still require testing.

The various installation methods include:

*	Building from source
*	Downloading build artifacts
*	Launching a pre-Release Amazon Machine Image (AMI)
* 	Pulling a pre-Release Docker image.

The recommended installation method is using the Docker image since the proper installation and configuration of the various dependencies (e.g. drivers, libraries, backends) are done for the user.  The other methods offer flexibility (such as building for specific hardware), but assume the installer has the requisite knowledge to do so.


##	Docker Image
*	__Starting Docker__: `nvidia-docker run -it -p 54321:54321 -p 8888:8888 -p 55001:55001 -v $PWD:/host opsh2oai/h2o-deepwater`

Each of the arguments after `-p` specify how to map the ports of the container to the running host.  Feel free to add addition ports you wish to expose.  The reason for the ports specified above are:

*	__Ports__
	*	`54321`: H2O Flow
	*	`8888`: Jupyter Notebook
	* 	`55001`: Prediction service.  This can be changed and just arbitrary set.

*	__Artifacts__

All build artifacts are already on the container in the `/opt` directory.  They have been installed for you.

1.	`/opt/h2o.jar`: This is H2O will Deep Water.
2.	`/opt/dist/deepwater-all.jar`: All Deep Water dependencies in a JAR.
3. `/opt/dist/h2o-3.11.0.234-py2.py3-none-any.whl`: Python module
4. `/opt/dist/h2o_3.11.0.234.tar.gz`: R package
5. `/opt/dist/mxnet-0.7.0-py2.7.egg`: MXNet Python module
6. `/opt/dist/tensorflow-1.1.0rc0-cp27-cp27mu-linux_x86_64.whl`: TensorFlow Python module.
7.	`/opt/steam`: [Steam repo](https://github.com/h2oai/steam).  This is needed for the prediction service builder.

*	__Starting H2O__: `java -jar /opt/h2o.jar &`

*	__Starting Jupyter Notebook__:  If you intend to go through the GTC lab material, it is recommended you start the `jupyter` from the directory `/gtc-2017`.  To start Jupyter: `jupyter notebook --allow-root --ip=* &`

## GTC Lab Directory Structure
This repo is in the `/gtc-2017` folder on the container.  The following are the directories in this repo.

###	`/data`
This directory contains the data used for the examples.

####	MNIST
This is the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of 28x28 pixel black-and-white handwritten digit images.  The training set contains 60,000 records and the testing/validation set contains 10,000 records.

*	__Vectorized__: This is this data in vectorized form with 785 columns (784 pixel values and a target column).  Column 785 is the target/label.  The training and testing/validation set are `train.csv.gz` and `test.csv.gz`, respectively.

*	__PNG__: This is the data in PNG format.  This data was created from Myle Ott's repo: <https://github.com/myleott/mnist_png>.  The compressed file is `mnist_png.tar.gz`.  When extracted, training and testing are in separate folders.  Within each of those are folders, all images of the same digit are in a subfolder named after the digit.

*	__Odd / Even Target, Vectorized__: A custom MNIST data set is created where the target is not the digit label.  Instead, the target indicates whether the digit is odd or even.  The training and testing set are `train-odd.csv.gz` and `test-odd.csv.gz`, respectively.

*	__Image Schema Dataset__:  Deep Water accepts a data set with a custom two-column image schema, where the first column indicates the path to an image and the second column is the target value.  For example:

```
uri,label
/gtc-2017/data/mnist_png/training/6/6453.png,6
/gtc-2017/data/mnist_png/training/6/13970.png,6
/gtc-2017/data/mnist_png/training/6/21967.png,6
```

The training and testing set are `mnist-training.csv` and `mnist-testing.csv`, respectively.  Those data sets assumes the `/gtc-2017` directory is at the root level.

####	Cars
Jonathan Krause (Stanford University) created a cars dataset: <http://ai.stanford.edu/~jkrause/cars/car_dataset.html>

The `get-cars-data.sh` script grabs a resized version of this data set that can be used.  The target is the label of the image, either car or truck.  The `cars_train.csv` is a two-column CSV in the Deep Water image schema.  Here are some sample values:

```
"uri","label"
"/gtc-2017/data/cars_train/00001.jpg","truck"
"/gtc-2017/data/cars_train/00002.jpg","truck"
"/gtc-2017/data/cars_train/00004.jpg","car"
"/gtc-2017/data/cars_train/00005.jpg","car"
"/gtc-2017/data/cars_train/00007.jpg","car"
"/gtc-2017/data/cars_train/00008.jpg","truck"
"/gtc-2017/data/cars_train/00009.jpg","truck"
```

The data set assumes the `/gtc-2017` directory is at the root level.

###	`/notebooks`
The notebooks are simple examples introducing and highlighting Deep Water features.

*	[Quick Start](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Quickstart.ipynb): This is a quick start notebook, illustrating the simplicity and power of the Deep Water API.  [LeNet](http://yann.lecun.com/exdb/lenet/) convolutional neural networks (ConvNet) are trained on the MNIST data set using all three backends: MXNet, TensorFlow, and Caffe.  H2O models are persisted to disk (in the `/model` directory).  MOJOs are also saved.

* [MXNet Custom Network](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Custom%20Network%20MXNet.ipynb): This is an example of building the LeNet ConvNet explicitly with the MXNet [Symbol](http://mxnet.io/tutorials/basic/symbol.html) Python API, illustrating how a custom user-defined network can be created with MXNet.

*	[TensorFlow Custom Network](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Custom%20Network%20TensorFlow.ipynb): This is an example of building a custom user-defined network with the TensorFlow Python API.

*	[Keras Custom Network](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Custom%20Network%20Keras.ipynb): A TensorFlow network (e.g. computation graph) can also be created with the [Keras](https://keras.io/) API.

*	[MXNet Pretrained Network](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Pre-Trained%20Networks.ipynb): This notebook illustrates using the Deep Water `network`, `network_definition_file`, and `network_parameters_file` parameters to load a pre-trained network into Deep Water.

*	[Stacked Ensemble](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Stacked%20Ensemble.ipynb): Deep Water models can be ensembled with other H2O models.  This notebook ensembles gradient boosting machine (GBM), generalized linear model (GLM), and Deep Water (deep learning) models.

*	[Deep Features Classifier](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Deep%20Features%20Classifier.ipynb): This notebook illustrates how deep features (e.g. hidden layer feature representations) can be used as features for training a classifier using the MNIST data set.

*	[Deep Features Similarity MNIST](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Deep%20Features%20Similarity%20MNIST.ipynb): This notebook illustrates how deep features (e.g. hidden layer feature representations) can be used for similarity analysis using the MNIST data set.

*	[Deep Features Similarity Cars](https://github.com/h2oai/gtc-2017/blob/master/notebooks/Deep%20Water%20Deep%20Features%20Similarity%20Cars%20Inception.ipynb): This notebook illustrates how deep features (e.g. hidden layer feature representations) can be used for similarity analysis.  The cars data set and a pre-trained Inception network is used.

Keep in mind that Deep Water models are first-class H2O models.  Other features note showcased in notebooks include grid search and checkpointing.

###	`/scripts`
The scripts are equivalent to the notebooks.  Any plotting is removed.

###	`/models`
The models folder is a place to store model artifacts, including: netowrk graphs, network parameters, mean image files, H2O models, and MOJOs.

*	__H2O Models__: There are existing H2O models from previous runs and left here for convenience (e.g. if you want to try to load a model).  These models will be overwritten if you run the "Quick Start" notebook.  The H2O models are: `model_mnist_lenet_mx`, `model_mnist_lenet_tf`,  and `model_mnist_lenet_caffe`.  The `mx`, `tf`, and `caffe` suffixes refer to Deep Water models using the MXNet, TensorFlow, and Caffe backends, respectively.

*	__MOJOs__: There are existing MOJOs from previous runs and left here for convenience (e.g. if you want to try to deploy a model).  These models will be overwritten if you run the "Quick Start" notebook.  The MOJOs are: `model_mnist_lenet_mx.zip`, `model_mnist_lenet_tf.zip`,  and `model_mnist_lenet_caffe.zip`.  To learn more about MOJOs, click [here](http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html).

*	__MXNet Files__: These are graph and parameter files that can be used to load a pre-trained network: `model_mnist_lenet_mx.json`,
`model_mnist_lenet_mx_params`.

*	__TensorFlow Files__: These are graph files that can be used to load a pre-defined network: `mymodel_tensorflow.meta`.  An example of using a pre-trained TensorFlow model can be found [here](https://github.com/h2oai/h2o-3/blob/master/examples/deeplearning/notebooks/deeplearning_tensorflow_cat_dog_mouse_lenet.ipynb).

*	__Inception with Batch Normalization__: These are files of a pre-trained GoogLeNet (Inception) with batch normalization: `Inception_BN-0039.params`, `Inception_BN-symbol.json`, and `mean_224.nd`.  `inception-bn_old.tar` is the unextracted file.  These files were obtain here:  <http://data.dmlc.ml/mxnet/models/imagenet/inception-bn_old.tar.gz>

*	__`h2o-genmodel.jar`__: This is the H2O POJO and MOJO dependency JAR file. (see [here](http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html)).

###	`/deploy`
Trained Deep Water models can be export as a MOJO (Model Object Optimized format) and deployed.  A MOJO contains the deep learning model graph and its parameters.  To learn more about MOJOs: <http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html>

This folder contains artifacts that can be used to test deployment.

*	__Data__: Example MNIST image files are stored here for your convenience: `0.png`, `1.png`, `2.png`, `3.png`, and `5.png`.

*	__Example Deployment__: A MOJO can be "wrapped" as a web prediction service with a REST endpoint.  H2O provides a Prediction Service Builder as part of the Steam solution: <https://github.com/h2oai/steam/tree/master/prediction-service-builder>.

The prediction service builder is included as part of the Docker image.  The following are steps to deploy a MOJO as a prediction service.  Scripts are provided for convenience, but assume certain location of artifacts.  This can be modified accordingly.

1.	__Start Prediction Service Builder__: The Prediction Service Builder is a web service itself.  You send the MOJO and the required dependencies to the web service and then you get a Web Archive (WAR) file back, which is a ready-to-run web service for Java that can be run with [Jetty](http://www.eclipse.org/jetty/) or tomcat or other web service systems.

To start the Prediction Service Builder, you can use the `start-prediction-service-builder.sh` script.  The service builder will be running at `http://localhost:55000`.

2.	__Get Dependencies__: To build a WAR file, the Prediction Service Builder needs the MOJO and the appropriate dependencies.  In the case of Deep Water, this includes the `deepwater-all.jar` and `h2o-genmodel.jar`.  The example `get-all-dependencies-gtc.sh` script copies these dependencies from the container and pulls the example `model_mnist_lenet_mx.zip` MOJO from the `/models` directory.

3. __Build Prediction Service__:  The `build-prediction-service.sh` script will submit the dependencies to the Prediction Service Builder and create a WAR file.

4. __Run Prediction Service__: The `run-prediction-service.sh` script launches the WAR file and starts the prediction service.  The script starts the prediction service on port 55001: `http://localhost:55001`.

5. __Predict__:  You can visit `http://localhost:55001` and use the built-in prediction service web UI.  The `predict.sh` is a command line script to test the prediction service with a curl.  The script will send an image of "1" to the service and get the prediction
result back as JSON.

##	Other Examples
Other examples can be found here: <https://github.com/h2oai/h2o-3/tree/master/examples/deeplearning/notebooks>

