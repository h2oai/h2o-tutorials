
# coding: utf-8

# # Introduction
# This tutorial shows how H2O [Gradient Boosted Models](https://en.wikipedia.org/wiki/Gradient_boosting) and [Random Forest](https://en.wikipedia.org/wiki/Random_forest) models can be used to do supervised classification and regression. This tutorial covers usage of H2O from Python. An R version of this tutorial will be available as well in a separate document. This file is available in plain R, R markdown, regular markdown, plain Python and iPython Notebook formats. More examples and explanations can be found in our [H2O GBM booklet](http://h2o.ai/resources/) and on our [H2O Github Repository](http://github.com/h2oai/h2o-3/).
# 

# ## Task: Predicting forest cover type from cartographic variables only
# 
# The actual forest cover type for a given observation (30 x 30 meter cell) was determined from the US Forest Service (USFS). We are using the UC Irvine Covertype dataset.

# ### H2O Python Module
# 
# Load the H2O Python module.

# In[ ]:

import h2o
import os


# ### Start H2O
# Start up a 1-node H2O cloud on your local machine, and allow it to use all CPU cores and up to 2GB of memory:

# In[ ]:

h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
h2o.remove_all()                          #clean slate, in case cluster was already running


# To learn more about the h2o package itself, we can use Python's builtin help() function.

# In[ ]:

help(h2o)


# help() can be used on H2O functions and models. Jupyter's builtin shift-tab functionality also works

# In[ ]:

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
help(H2OGradientBoostingEstimator)
help(h2o.import_file)


# ## H2O GBM and RF
# 
# While H2O Gradient Boosting Models and H2O Random Forest have many flexible parameters options, they were designed to be just as easy to use as the other supervised training methods in H2O. Early stopping, automatic data standardization and handling of categorical variables and missing values and adaptive learning rates (per weight) reduce the amount of parameters the user has to specify. Often, it's just the number and sizes of hidden layers, the number of epochs and the activation function and maybe some regularization techniques. 

# ### Getting started
# 
# We begin by importing our data into H2OFrames, which operate similarly in function to pandas DataFrames but exist on the H2O cloud itself.  
# 
# In this case, the H2O cluster is running on our laptops. Data files are imported by their relative locations to this notebook.

# In[ ]:

covtype_df = h2o.import_file(os.path.realpath("../data/covtype.full.csv"))


# We import the full covertype dataset (581k rows, 13 columns, 10 numerical, 3 categorical) and then split the data 3 ways:  
#   
# 60% for training  
# 20% for validation (hyper parameter tuning)  
# 20% for final testing  
# 
#  We will train a data set on one set and use the others to test the validity of the model by ensuring that it can predict accurately on data the model has not been shown.  
#  
#  The second set will be used for validation most of the time.  
#  
#  The third set will be withheld until the end, to ensure that our validation accuracy is consistent with data we have never seen during the iterative process. 

# In[ ]:

#split the data as described above
train, valid, test = covtype_df.split_frame([0.6, 0.2], seed=1234)

#Prepare predictors and response columns
covtype_X = covtype_df.col_names[:-1]     #last column is Cover_Type, our desired response variable 
covtype_y = covtype_df.col_names[-1]    


# ### The First Random Forest
# We build our first model with the following parameters
# 
# **model_id:** Not required, but allows us to easily find our model in the [Flow](http://localhost:54321/) interface  
# **ntrees:** Maximum number of trees used by the random forest. Default value is 50. We can afford to increase this, as our early-stopping criterion will decide when the random forest is sufficiently accurate.  
# **stopping_rounds:** Stopping criterion described above. Stops fitting new trees when 2-tree rolling average is within 0.001 (default) of the two prior rolling averages. Can be thought of as a convergence setting.  
# **score_each_teration:** predict against training and validation for each tree. Default will skip several.  
# **seed:** set the randomization seed so we can reproduce results
# 

# In[ ]:

rf_v1 = H2ORandomForestEstimator(
    model_id="rf_covType_v1",
    ntrees=200,
    stopping_rounds=2,
    score_each_iteration=True,
    seed=1000000)


# ### Model Construction
# H2O in Python is designed to be very similar in look and feel to to scikit-learn. Models are initialized individually with desired or default parameters and then trained on data.  
# 
# **Note that the below example uses model.train() as opposed the traditional model.fit()**  
# This is because h2o-py takes column indices for the feature and response columns AND the whole data frame, while scikit-learn takes in a feature frame and a response frame.
# 
# H2O supports model.fit() so that it can be incorporated into a scikit-learn pipeline, but we advise using train() in all other cases.

# In[ ]:

rf_v1.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)


# Note that the progress bar does not behave linearly. H2O estimates completion time initially based on the number of epochs specified. However, convergence can allow for early stops, in which case the bar jumps to 100%.
# 
# We can view information about the model in [Flow](http://localhost:54321/) or within Python. To find more information in Flow, enter `getModel "rf_covType_v1"` into a cell and run in place pressing Ctrl-Enter. Alternatively, you can click on the Models tab, select List All Models, and click on the model named "rf_covType_v1" as specified in our model construction above.
# 
# In Python, we can call the model itself to get an overview of its stats.

# In[ ]:

rf_v1


# To look at validation statistics, we can use the scoring history function.

# In[ ]:

rf_v1.score_history()


# Here we can see the hit ratio table.

# In[ ]:

rf_v1.hit_ratio_table(valid=True)


# ### Now for GBM
# 
# First we will use all default settings, then make some changes to improve our predictions.

# In[ ]:

gbm_v1 = H2OGradientBoostingEstimator(
    model_id="gbm_covType_v1",
    seed=2000000
)
gbm_v1.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)


# In[ ]:

gbm_v1.score_history()


# In[ ]:

gbm_v1.hit_ratio_table(valid=True)


# This default GBM is much worse than our original random forest.  
# 
# 
# The GBM is far from converging, so there are three primary knobs to adjust to get our performance up if we want to keep a similar run time.  
# 
# 1: Adding trees will help. The default is 50.  
# 2: Increasing the learning rate will also help. The contribution of each tree will be stronger, so the model will move further away from the overall mean.  
# 3: Increasing the depth will help. This is the parameter that is the least straightforward. Tuning trees and learning rate both have direct impact that is easy to understand. Changing the depth means you are adjusting the "weakness" of each learner. Adding depth makes each tree fit the data closer.  
#   
# The first configuration will attack depth the most, since we've seen the random forest focus on a continuous variable (elevation) and 40-class factor (soil type) the most.  
# 
# Also we will take a look at how to review a model while it is running.  

# ### GBM Round 2
# 
# Let's do the following:
# 
# 1. decrease the number of trees to speed up runtime(from default 50 to 20)
# 2. increase the learning rate (from default 0.1 to 0.2)
# 3. increase the depth (from default 5 to 10)

# In[ ]:

gbm_v2 = H2OGradientBoostingEstimator(
    ntrees=20,
    learn_rate=0.2,
    max_depth=10,
    stopping_tolerance=0.01, #10-fold increase in threshold as defined in rf_v1
    stopping_rounds=2,
    score_each_iteration=True,
    model_id="gbm_covType_v2",
    seed=2000000
)
gbm_v2.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)


# ### Live Performance Monitoring
# 
# While this is running, we can actually look at the model. To do this we simply need a new connection to H2O. 
# 
# This Python notebook will run the model, so we need either another notebook or the web browser (or R, etc.). In this demo, we will use [Flow](http://localhost:54321) in our web browser http://localhost:54321 and the focus will be to look at model performance, since we are using Python to control H2O. 

# In[ ]:

gbm_v2.hit_ratio_table(valid=True)


# This has moved us in the right direction, but still lower accuracy than the random forest.  
# 
# It still has yet to converge, so we can make it more aggressive.  
# 
# We can now add the stochastic nature of random forest into the GBM using some of the new H2O settings. This will help generalize and also provide a quicker runtime, so we can add a few more trees.

# ### GBM: Third Time is the Charm
# 
# 1. Add a few trees(from 20 to 30)
# 2. Increase learning rate (to 0.3)
# 3. Use a random 70% of rows to fit each tree
# 4. Use a random 70% of columns to fit each tree

# In[ ]:

gbm_v3 = H2OGradientBoostingEstimator(
    ntrees=30,
    learn_rate=0.3,
    max_depth=10,
    sample_rate=0.7,
    col_sample_rate=0.7,
    stopping_rounds=2,
    stopping_tolerance=0.01, #10-fold increase in threshold as defined in rf_v1
    score_each_iteration=True,
    model_id="gbm_covType_v3",
    seed=2000000
)
gbm_v3.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)


# In[ ]:

gbm_v3.hit_ratio_table(valid=True)


# ### Parity
# 
# Now the GBM is close to the initial random forest.
# 
# However, we used a default random forest. Random forest's primary strength is how well it runs with standard parameters, and while there are only a few parameters to tune, we can experiment with those to see if it will make a difference.  
# 
# The main parameters to tune are the tree depth and the mtries, which is the number of predictors to use.  
# 
# The default depth of trees is 20. It is common to increase this number, to the point that in some implementations, the depth is unlimited. We will increase ours from 20 to 30.  
# 
# Note that the default mtries depends on whether classification or regression is being run. The default for classification is one-third of the columns. The default for regression is the square root of the number of columns.  

# ### Random Forest #2

# In[ ]:

rf_v2 = H2ORandomForestEstimator(
    model_id="rf_covType_v2",
    ntrees=200,
    max_depth=30,
    stopping_rounds=2,
    stopping_tolerance=0.01,
    score_each_iteration=True,
    seed=3000000)
rf_v2.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)


# In[ ]:

rf_v2.hit_ratio_table(valid=True)


# ### Final Predictions
# 
# Now that we have our validation accuracy up beyond 95%, we can start considering our test data.  
# We have withheld an extra test set to ensure that after all the parameter tuning we have repeatedly applied with the validation data, we still have a completely pristine data set upon which to test the predictive capacity of our model.

# In[ ]:

#Excludes the "Cover_Type" column from the features provided
final_rf_predictions = rf_v2.predict(test[:-1])


# Technically, our model won't look at the ["Cover_Type"] column within the test data, as it is trained on a set of features not including "Cover_Type". It is up to the user whether to include it in the test frame provided for predictions, as it has no effect whatsoever.
# 
# Let's take a peek at the first few rows of predictions returned by our model.

# In[ ]:

final_rf_predictions


# Let's compare these predictions to the accuracy we got from our experimentation

# In[ ]:

#validation set accuracy
rf_v2.hit_ratio_table(valid=True)


# In[ ]:

#test set accuracy
(final_rf_predictions['predict']==test['Cover_Type']).as_data_frame(use_pandas=True).mean()


# Our final error rates are very similar between validation and test sets. This suggests that we did not overfit the validation set during our experimentation. This concludes our demo of H2O GBM and H2O Random Forests.
# 
# 
# ### Shut down the cluster
# Shut down the cluster now that we are done using it.

# In[ ]:

h2o.shutdown(prompt=False)


# ### Possible Further Steps
# 
# Model-agnostic gains can be found in improving handling of categorical features. We could experiment with the nbins and nbins_cats settings to control the H2O splitting.The general guidance is to lower the number to increase generalization (avoid overfitting), increase to better fit the distribution.  
#  
# A good example of adjusting this value is for nbins_cats to be increased to match the number of values in a category. Though usually unnecessary, this can improve performance if a problem has a very important categorical predictor.  
# 
# 
# With regards to our Random Forest, we could further experiment with deeper trees or a higher percentage of columns used (mtries).  
# 
# The GBM can be set to converge a slower for optimal accuracy. If we were to relax our runtime requirements a little bit, we could balance the learn rate and number of trees used.  
# 
# In a production setting where fine-grain accuracy is beneficial, it is common to set the learn rate to a very small number, such as 0.01 or smaller, and add trees to match.  
# 
# Use of early stopping is very powerful in allowing the setting of a low learning rate and the building as many trees as needed until the desired convergence is met.
# 
# ### More information can be found in the [H2O Gradient Boosted Models booklet](http://h2o.ai/resources/), in our [H2O SlideShare Presentations](http://www.slideshare.net/0xdata/presentations), our [H2O YouTube channel](https://www.youtube.com/user/0xdata/), as well as on our [H2O Github Repository](https://github.com/h2oai/h2o-3/), especially in our [H2O GBM R tests](https://github.com/h2oai/h2o-3/tree/master/h2o-r/tests/testdir_algos/gbm), and [H2O GBM Python tests](https://github.com/h2oai/h2o-3/tree/master/h2o-py/tests/testdir_algos/gbm).
