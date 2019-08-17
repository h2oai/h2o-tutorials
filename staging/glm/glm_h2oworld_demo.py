
# coding: utf-8

# # Introduction
# This tutorial shows how a H2O [Generalized Linear Model](https://en.wikipedia.org/wiki/Generalized_linear_model) model can be used to do supervised classification. This tutorial covers usage of H2O from Python. An R version of this tutorial will be available as well in a separate document. This file is available in plain R, plain Python and iPython Notebook formats. More examples and explanations can be found in our [H2O Generalized Linear Modeling booklet](http://h2o.ai/resources/) and on our [H2O Github Repository](http://github.com/h2oai/h2o-3/).
# 

# ### H2O Python Module
# 
# Load the H2O Python module.

# In[ ]:

import h2o


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

from h2o.estimators.glm import H2OGeneralizedLinearEstimator
help(H2OGeneralizedLinearEstimator)
help(h2o.import_file)


# Since we use pandas DataFrames to simplify some processes later in this demo, let's import both pandas and numpy.

# In[ ]:

import pandas as pd
import numpy as np


# ##H2O GLM
# 
# Generalized linear models (GLMs) are an extension of traditional linear models. They have gained popularity in statistical data analysis due to:  
# 
# 1. the flexibility of the model structure unifying the typical regression methods (such as linear regression and logistic regression for binary classification)  
# 2. the recent availability of model-fitting software  
# 3. the ability to scale well with large datasets  
# 
# H2O's GLM algorithm fits generalized linear models to the data by maximizing the log-likelihood. The elastic net penalty can be used for parameter regularization. The model fitting computation is distributed, extremely fast, and scales extremely well for models with a limited number of predictors with non-zero coefficients (~ low thousands).  

# ###Getting started
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
train, valid, test = covtype_df.split_frame([0.7, 0.15], seed=1234)

#Prepare predictors and response columns
covtype_X = covtype_df.col_names[:-1]     #last column is Cover_Type, our desired response variable 
covtype_y = covtype_df.col_names[-1]    


# ###The First Multinomial Model
# 
# Our goal is to perform classification on cartographical data into tree cover categories.
# 
# This is a multinomial problem, so let's begin by building a multinomial GLM model with default parameters!
# 
# We will use the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm to ensure that this demo can be run in almost all environments.

# In[ ]:

glm_multi_v1 = H2OGeneralizedLinearEstimator(
                    model_id='glm_v1',            #allows us to easily locate this model in Flow
                    family='multinomial',
                    solver='L_BFGS')


# ###Model Construction
# H2O in Python is designed to be very similar in look and feel to to scikit-learn. Models are initialized individually with desired or default parameters and then trained on data.  
# 
# **Note that the below example uses model.train() as opposed the traditional model.fit()**  
# This is because h2o-py takes column indices for the feature and response columns AND the whole data frame, while scikit-learn takes in a feature frame and a response frame.
# 
# H2O supports model.fit() so that it can be incorporated into a scikit-learn pipeline, but we advise using train() in all other cases.

# In[ ]:

glm_multi_v1.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)


# We can view information about the model in [Flow](http://localhost:54321/) or within Python. To find more information in Flow, enter `getModel "rf_covType_v1"` into a cell and run in place pressing Ctrl-Enter. Alternatively, you can click on the Models tab, select List All Models, and click on the model named "rf_covType_v1" as specified in our model construction above.
# 
# In Python, we can use call the model itself to get an overview of its stats,

# In[ ]:

glm_multi_v1


# To find out a little more about its performance, we can look at its hit ratio table.

# In[ ]:

glm_multi_v1.hit_ratio_table(valid=True)


# ###Updating our GLM Estimator
# As we can see, the k=1 hit ratio indicates that we're very far off of a good estimator. Judging by our training and validation scores, we don't seem to be overfitting. Perhaps we're over-regularizing?
# 
# Let's try again with a lower lambda value.

# In[ ]:

glm_multi_v2 = H2OGeneralizedLinearEstimator(
                    model_id='glm_v2',           
                    family='multinomial',
                    solver='L_BFGS',
                    Lambda=0.0001                 #default value 0.001
)
glm_multi_v2.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)


# In[ ]:

glm_multi_v2


# In[ ]:

glm_multi_v2.hit_ratio_table(valid=True)


# There's a noticeable improvement in the MSE, and our hit ratio has improved from coin-flip to 72%. 
# 
# Let's look at the confusion matrix to see if we can gather any more insight on the errors in our multinomial classification.

# In[ ]:

glm_multi_v2.confusion_matrix(valid)


# ### Class 1 & 2 Struggles
# 
# As we can see in the above confusion matrix, our model is struggling to correctly distinguish between covertype classes 1 and 2. To learn more about this, let's shrink the scope of our problem to a binomial classification.

# ##Binomial Classification
# 
# Let's only look at the rows where coverage is class_1 or class_2

# In[ ]:

c1 = covtype_df[covtype_df['Cover_Type'] == 'class_1']
c2 = covtype_df[covtype_df['Cover_Type'] == 'class_2']
df_b = c1.rbind(c2)


# Once again, let's split this into train, valid and test sets

# In[ ]:

#split the data as described above
train_b, valid_b, test_b = df_b.split_frame([0.7, 0.15], seed=1234)


# Let's build a binomial classifier with the default parameters

# In[ ]:

glm_binom_v1 = H2OGeneralizedLinearEstimator(
                    model_id="glm_v3",
                    solver="L_BFGS",
                    family="binomial")
glm_binom_v1.train(covtype_X, covtype_y, training_frame=train_b, validation_frame=valid_b)


# In[ ]:

glm_binom_v1.accuracy()


# As we can see, the data in its natural state does not classify particularly cleanly into class_1 or class_2.

# ###Featurization
# 
# Let's add some features to this binomial model to see if we can improve its predictive capacity. We'll do a combination of binning (by converting several numeric fields to categorical) and interaction variables. To do this cleanly, we use the two helper functions defined below

# In[ ]:

def cut_column(train_df, train, valid, test, col):
    '''
    Convenience function to change a column from numerical to categorical
    We use train_df only for bucketing with histograms.
    Uses np.histogram to generate a histogram, with the buckets forming the categories of our new categorical.
    Picks buckets based on training data, then applies the same classification to the test and validation sets
    
    Assumes that train, valid, test will have the same histogram behavior.
    '''
    only_col= train_df[col]                            #Isolate the column in question from the training frame
    counts, breaks = np.histogram(only_col, bins=20)   #Generate counts and breaks for our histogram
    min_val = min(only_col)-1                          #Establish min and max values
    max_val = max(only_col)+1
    
    new_b = [min_val]                                  #Redefine breaks such that each bucket has enough support
    for i in xrange(19):
        if counts[i] > 1000 and counts[i+1] > 1000:
            new_b.append(breaks[i+1])
    new_b.append(max_val)
    
    names = [col + '_' + str(x) for x in xrange(len(new_b)-1)]  #Generate names for buckets, these will be categorical names

    train[col+"_cut"] = train[col].cut(breaks=new_b, labels=names)
    valid[col+"_cut"] = valid[col].cut(breaks=new_b, labels=names)
    test[col+"_cut"] = test[col].cut(breaks=new_b, labels=names)


# In[ ]:

def add_features(train, valid, test):
    '''
    Helper function to add a specific set of features to our covertype dataset
    '''
    #pull train dataset into Python
    train_df = train.as_data_frame(True)
    
    #Make categoricals for several columns
    cut_column(train_df, train, valid, test, "Elevation")
    cut_column(train_df, train, valid, test, "Hillshade_Noon")
    cut_column(train_df, train, valid, test, "Hillshade_9am")
    cut_column(train_df, train, valid, test, "Hillshade_3pm")
    cut_column(train_df, train, valid, test, "Horizontal_Distance_To_Hydrology")
    cut_column(train_df, train, valid, test, "Slope")
    cut_column(train_df, train, valid, test, "Horizontal_Distance_To_Roadways")
    cut_column(train_df, train, valid, test, "Aspect")
    
    
    #Add interaction columns for a subset of columns
    interaction_cols1 = ["Elevation_cut",
                         "Wilderness_Area",
                         "Soil_Type",
                         "Hillshade_Noon_cut",
                         "Hillshade_9am_cut",
                         "Hillshade_3pm_cut",
                         "Horizontal_Distance_To_Hydrology_cut",
                         "Slope_cut",
                         "Horizontal_Distance_To_Roadways_cut",
                         "Aspect_cut"]

    train_cols = train.interaction(factors=interaction_cols1,    #Generate pairwise columns
                                   pairwise=True,
                                   max_factors=1000,
                                   min_occurrence=100,
                                   destination_frame="itrain")
    valid_cols = valid.interaction(factors=interaction_cols1,
                                   pairwise=True,
                                   max_factors=1000,
                                   min_occurrence=100,
                                   destination_frame="ivalid")
    test_cols = test.interaction(factors=interaction_cols1,
                                   pairwise=True,
                                   max_factors=1000,
                                   min_occurrence=100,
                                   destination_frame="itest")
    
    train = train.cbind(train_cols)                              #Append pairwise columns to H2OFrames
    valid = valid.cbind(valid_cols)
    test = test.cbind(test_cols)
    
    
    #Add a three-way interaction for Hillshade
    interaction_cols2 = ["Hillshade_Noon_cut","Hillshade_9am_cut","Hillshade_3pm_cut"]
    
    train_cols = train.interaction(factors=interaction_cols2,    #Generate pairwise columns
                                   pairwise=False,
                                   max_factors=1000,
                                   min_occurrence=100,
                                   destination_frame="itrain")
    valid_cols = valid.interaction(factors=interaction_cols2,
                                   pairwise=False,
                                   max_factors=1000,
                                   min_occurrence=100,
                                   destination_frame="ivalid")
    test_cols = test.interaction(factors=interaction_cols2,
                                   pairwise=False,
                                   max_factors=1000,
                                   min_occurrence=100,
                                   destination_frame="itest")
    
    train = train.cbind(train_cols)                              #Append pairwise columns to H2OFrames
    valid = valid.cbind(valid_cols)
    test = test.cbind(test_cols)
    
    return train, valid, test


# ####Add features to our binomial data

# In[ ]:

train_bf, valid_bf, test_bf = add_features(train_b, valid_b, test_b)


# In[ ]:

glm_binom_feat_1 = H2OGeneralizedLinearEstimator(family='binomial', solver='L_BFGS', model_id='glm_v4')
glm_binom_feat_1.train(covtype_X, covtype_y, training_frame=train_bf, validation_frame=valid_bf)


# In[ ]:

glm_binom_feat_1.accuracy(valid=True)


# We appear to have marginal improvement in accuracy! Inspecting in flow, we see that we may be over-regularizing like in our very first model, so we once again decrement lambda.

# In[ ]:

glm_binom_feat_2 = H2OGeneralizedLinearEstimator(family='binomial', solver='L_BFGS', model_id='glm_v5', Lambda=0.001)
glm_binom_feat_2.train(covtype_X, covtype_y, training_frame=train_bf, validation_frame=valid_bf)


# In[ ]:

glm_binom_feat_2.accuracy(valid=True)


# Our Validation accuracy is increasing! Let's try adding in lambda search to see if we can possibly improve any further.

# In[ ]:

glm_binom_feat_3 = H2OGeneralizedLinearEstimator(family='binomial', model_id='glm_v6', lambda_search=True)
glm_binom_feat_3.train(covtype_X, covtype_y, training_frame=train_bf, validation_frame=valid_bf)


# In[ ]:

glm_binom_feat_3.accuracy(valid=True)


# This yields minimal improvements over lambda=0.001. Thus, we can conclude that the optimal lambda value is quite close to 0.001

# ###Revisiting the Multinomial
# 
# We've managed to reduce the error in classification between class_1 and class_2 by adding some features and categorizing others. Let's apply these changes to our original multinomial model to see what sorts of gains we can achieve. First let's featurize our data.

# In[ ]:

train_f, valid_f, test_f = add_features(train, valid, test)


# Let's build a final multinomial classifier with our featurized data and a near-optimal lambda of 0.001

# In[ ]:

glm_multi_v3 = H2OGeneralizedLinearEstimator(
                    model_id='glm_v7',           
                    family='multinomial',
                    solver='L_BFGS',
                    Lambda=0.0001)
glm_multi_v3.train(covtype_X, covtype_y, training_frame=train_f, validation_frame=valid_f)


# In[ ]:

glm_multi_v3.hit_ratio_table(valid=True)


# Our hit ratio has improved dramatically since our first multinomial!

# ### More information can be found in the [H2O Generalized Linear Modeling Booklet](http://h2o.ai/resources/), in our [H2O SlideShare Presentations](http://www.slideshare.net/0xdata/presentations), our [H2O YouTube channel](https://www.youtube.com/user/0xdata/), as well as on our [H2O Github Repository](https://github.com/h2oai/h2o-3/), especially in our [H2O GLM R tests](https://github.com/h2oai/h2o-3/tree/master/h2o-r/tests/testdir_algos), and [H2O GLM tests](https://github.com/h2oai/h2o-3/tree/master/h2o-py/tests/testdir_algos/glm).

# In[ ]:

h2o.shutdown(prompt=False)

