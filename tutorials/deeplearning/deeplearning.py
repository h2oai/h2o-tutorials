
# coding: utf-8

# # Introduction
# This tutorial shows how a H2O [Deep Learning](http://en.wikipedia.org/wiki/Deep_learning) model can be used to do supervised classification and regression. This tutorial covers usage of H2O from R. A python version of this tutorial will be available as well in a separate document. This file is available in plain R, R markdown and regular markdown formats, and the plots are available as PDF files. More examples and explanations can be found in our [H2O Deep Learning booklet](http://h2o.ai/resources/) and on our [H2O Github Repository](http://github.com/h2oai/h2o-3/).
# 

# ### H2O Python Module
# 
# Load the H2O Python module.

# In[1]:

import h2o


# ### Start H2O
# Start up a 1-node H2O server on your local machine, and allow it to use all CPU cores and up to 2GB of memory:

# In[2]:

h2o.init(max_mem_size_GB = 2)            #uses all cores by default
h2o.remove_all()                          #clean slate, in case cluster was already running


# To learn more about the h2o package itself, we can use Python's builtin help() function.

# In[3]:

help(h2o)


# help() can be used on H2O functions and models. Jupyter's builtin shift-tab functionality also works

# In[4]:

from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
help(H2ODeepLearningEstimator)
help(h2o.import_file)


# ##H2O Deep Learning
# While H2O Deep Learning has many parameters, it was designed to be just as easy to use as the other supervised training methods in H2O. Early stopping, automatic data standardization and handling of categorical variables and missing values and adaptive learning rates (per weight) reduce the amount of parameters the user has to specify. Often, it's just the number and sizes of hidden layers, the number of epochs and the activation function and maybe some regularization techniques.
# 

# ### Let's have some fun first: Decision Boundaries
# We start with a small dataset representing red and black dots on a plane, arranged in the shape of two nested spirals. Then we task H2O's machine learning methods to separate the red and black dots, i.e., recognize each spiral as such by assigning each point in the plane to one of the two spirals.

# We visualize the nature of H2O Deep Learning (DL), H2O's tree methods (GBM/DRF) and H2O's generalized linear modeling (GLM) by plotting the decision boundary between the red and black spirals:

# In[5]:

#get_ipython().magic(u'matplotlib inline')
#IMPORT ALL THE THINGS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator


# First, we need to upload our datasets to the the H2O cluster. The data is imported into H2OFrames, which operate similarly in function to pandas DataFrames.  
# 
# In this case, the cluster is running on our laptops. Data files are imported by their relative locations to this notebook.

# In[6]:

spiral = h2o.upload_file("../data/spiral.csv")
grid  = h2o.upload_file("../data/grid.csv")


# Spiral is a simple data set consisting of two spirals of black and red dots.  
# Grid is a 201 by 201 matrix with dimensions [-1.5, 1.5] by [-1.5, 1.5].
# 
# To visualize these datasets, we can pull them from H2OFrames into pandas DataFrames for easier plotting.

# In[7]:

spiral_df = spiral.as_data_frame(use_pandas=True)
grid_df = grid.as_data_frame(use_pandas=True)
grid_x, grid_y = grid_df.x.reshape(201,201), grid_df.y.reshape(201,201)
spiral_r = spiral_df[spiral_df.color == "Red"]
spiral_k = spiral_df[spiral_df.color == "Black"]

spiral_xr, spiral_yr = spiral_r[spiral_r.columns[0]], spiral_r[spiral_r.columns[1]]
spiral_xk, spiral_yk = spiral_k[spiral_k.columns[0]], spiral_k[spiral_k.columns[1]]
    
markersize_ = 7**2
plt.figure(figsize = (5,5))
plt.scatter(spiral_xr, spiral_yr, c = 'r', s=markersize_)
plt.scatter(spiral_xk, spiral_yk, c = 'k', s=markersize_)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.title("Spiral");


# ###Model Construction
# H2O in Python is designed to be very similar in look and feel to to scikit-learn. Models are initialized individually with desired or default parameters and then trained on data.  
# 
# Note that the below examples use model.train(), as opposed the traditional model.fit()
# This is because h2o-py takes the data frame AND column indices for the feature and response columns, while scikit-learn takes in feature frames.
# 
# H2O supports model.fit() so that it can be incorporated into a scikit-learn pipeline, but we advise using train() in all other cases.

# In[8]:

X = spiral.col_names[0:2]
y = spiral.col_names[2]
dl_model = H2ODeepLearningEstimator(epochs=1000)
dl_model.train(X, y, spiral)

gbm_model = H2OGradientBoostingEstimator()
gbm_model.train(X, y, spiral)

drf_model = H2ORandomForestEstimator()
drf_model.train(X, y, spiral)

glm_model = H2OGeneralizedLinearEstimator(family="binomial")
glm_model.fit(spiral[X], spiral[y])                                #model.fit() example

models = [dl_model, gbm_model, drf_model, glm_model]
m_names = ["Deep Learning", "Gradient Boosted Method", "Distributed Random Forest", "Generalized Linear Model"]


# Now that we've trained four models to classify points as red or black based on their (x,y) coordinates.  
# To see how our models have performed, we ask them to predict the colors of the grid.
# 
# Since we'll be doing a lot of spiral plotting, let's write a little helper function to keep things clean.

# In[79]:

def plot_spirals(models, model_names):
    fig, ax = plt.subplots(2,2, figsize=(12,12))
    for k, subplot in enumerate(ax.flatten()):
        subplot.scatter(spiral_xr, spiral_yr, c = 'r', s=markersize_)
        subplot.scatter(spiral_xk, spiral_yk, c = 'k', s=markersize_)
        subplot.axis([-1.5, 1.5, -1.5, 1.5])
        subplot.set_title(model_names[k])
        subplot.set_xlabel('x')
        subplot.set_ylabel('y')
        pred_z = models[k].predict(grid).as_data_frame(True)
        subplot.contour(grid_x, grid_y, (pred_z['predict'] == 'Black').astype(np.int).reshape(201,201), colors='b')


# Below are four graphs of the contour plots of the predictions, so that we can see how exactly the algorithms grouped the points into black and red.

# In[37]:

plot_spirals(models, m_names)


# ###A Deeper Dive into Deep Learning
# 
# Now let's explore the evolution of our deep learning model over training time (number of passes over the data, aka epochs).  
# We will use checkpointing to ensure that we continue training the same model

# In[77]:

dl_1 = H2ODeepLearningEstimator(epochs=1)
dl_1.train(X, y, spiral)

dl_250 = H2ODeepLearningEstimator(checkpoint=dl_1, epochs=250)
dl_250.train(X, y, spiral)

dl_500 = H2ODeepLearningEstimator(checkpoint=dl_250, epochs=500)
dl_500.train(X, y, spiral)

dl_750 = H2ODeepLearningEstimator(checkpoint=dl_500, epochs=750)
dl_750.train(X, y, spiral)


# You can see how the network learns the structure of the spirals with enough training time. 

# In[80]:

models_dl = [dl_1, dl_250, dl_500, dl_750]
m_names_dl = ["DL " + str(int(model.get_params()['epochs']['actual_value'])) +                                      " Epochs" for model in models_dl]

plot_spirals(models_dl, m_names_dl)


# ###Deep Learning Network Architecture
# Of course, there is far more to constructing Deep Learning models than simply having them run longer.  
# Consider the four following setups.
# 
# 1. Single layer, 1000 nodes
# 2. Two layers, 200 nodes each
# 3. Three layers, 42 nodes each
# 4. Four layers, 11 -> 13 -> 17 -> 19
# 
# The H2O Architecture uses the hidden keyword to control model network architecture.  
# Hidden takes a list of integers, representing the number of nodes in each layer.

# In[30]:

dl_1 = H2ODeepLearningEstimator(hidden=[1000], epochs=500)
dl_1.train(X, y, spiral)

dl_2 = H2ODeepLearningEstimator(hidden=[200,200], epochs=500)
dl_2.train(X, y, spiral)

dl_3 = H2ODeepLearningEstimator(hidden=[42,42,42], epochs=500)
dl_3.train(X, y, spiral)

dl_4 = H2ODeepLearningEstimator(hidden=[11,13,17,19], epochs = 1000)
dl_4.train(X, y, spiral)


# It is clear that different configurations can achieve similar performance, and that tuning will be required for optimal performance.

# In[31]:

models_network = [dl_1, dl_2, dl_3, dl_4]
m_names_network = ["1000", "200 x 200", "42 x 42 x 42", "11 x 13 x 17 x 19"]

plot_spirals(models_network, m_names_network)


# ###Activation Functions
# Next, we compare between different activation functions, including one with 50% dropout regularization in the hidden layers:

# In[33]:

models_act = []
m_names_act = []
for i,method in enumerate(["Tanh","Maxout","Rectifier","RectifierWithDropout"]):
    models_act.append(H2ODeepLearningEstimator(activation=method, hidden=[100,100], epochs=1000))
    models_act[i].train(X, y, spiral)
    m_names_act.append("DL "+ method + " Activation")


# In[35]:

plot_spirals(models_act, m_names_act)


# Clearly, the dropout rate was too high or the number of epochs was too low for the last configuration, which often ends up performing the best on larger datasets where generalization is important.  
# 
# More information about the parameters can be found in the [H2O Deep Learning booklet](http://h2o.ai/resources/).

# ## Covertype Dataset
# The following examples use the Covertype dataset from UC Irvine, which concerns predicting forest cover based on cartographical data.  
# We import the full covertype dataset (581k rows, 13 columns, 10 numerical, 3 categorical) and then split the data 3 ways:  
#   
# 60% for training  
# 20% for validation (hyper parameter tuning)  
# 20% for final testing  

# In[85]:

covtype_df = h2o.import_file("../data/covtype.full.csv")

#split the data as described above
train, valid, test = covtype_df.split_frame([0.6, 0.2], seed=1234)

#Prepare predictors and response columns
covtype_X = covtype_df.col_names[:-1]     #last column is cover_type, 
covtype_y = covtype_df.col_names[-1]    


# ####First Impressions
# Let's run our first Deep Learning model on the covtype dataset.   
# We want to predict the `Cover_Type` column, a categorical feature with 7 levels, and the Deep Learning model will be tasked to perform (multi-class) classification. It uses the other 12 predictors of the dataset, of which 10 are numerical, and 2 are categorical with a total of 44 levels.  
# 
# We can expect the Deep Learning model to have 56 input neurons (after automatic one-hot encoding). First run will be only one epoch to get a feel for the model construction.

# In[88]:

#set the model_id for easy lookup in Flow
covtype_model_v1 = H2ODeepLearningEstimator(model_id="covtype_v1", epochs=1, variable_importances=True)
covtype_model_v1.train(covtype_X, covtype_y, training_frame = train, validation_frame = valid)
print covtype_model_v1


# ###Shutdown H2O Cluster
# Shut down the cluster now that we are done using it.

# In[ ]:

h2o.shutdown()

