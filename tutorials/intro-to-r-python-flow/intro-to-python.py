
# coding: utf-8

# # Predicting Airline Delays in Python
# 
# The following is a demonstration of predicting potential flight delays using a publicly available airlines dataset. For this example, the dataset used is a small sample of what is more than two decades worth of flight data in order to ensure the download and import process would not take more than a minute or two.
# 
# ## The Data
# 
# The data comes originally from [RITA](http://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) where it is described in detail. To use the entire 26 years worth of flight information to more accurately predict delays and cancellation please download one of the following and change the path to the data in the notebook: 
# 
#   * [2 Thousand Rows - 4.3MB](https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv)
#   * [5.8 Million Rows - 580MB](https://s3.amazonaws.com/h2o-airlines-unpacked/airlines_all.05p.csv)
#   * [152 Million Rows (Years: 1987-2013) - 14.5GB](https://s3.amazonaws.com/h2o-airlines-unpacked/allyears.1987.2013.csv)
# 
# ## Business Benefits
# 
# There are obvious benefits to predicting potential delays and logistic issues for a business. It helps the user make contingency plans and corrections to avoid undesirable outcomes. Recommendation engines can forewarn flyers of possible delays and rank flight options accordingly, other businesses might pay more for a flight to ensure certain shipments arrive on time, and airline carriers can use the information to better their flight plans. The goal is to have the machine take in all the possible factors that will affect a flight and return the probability of a flight being delayed.

# ### Load the H2O module and start an local H2O cluster
# 
# Connection to an H2O cloud is established through the `h2o.init` function from the `h2o` module. To connect to a pre-existing H2O cluster make sure to edit the H2O location with argument `myIP` and `myPort`.
# 

# In[ ]:

import h2o
import os
import tabulate
import operator 
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# In[ ]:

h2o.init()


# ### Import Data into H2O
# 
# We will use the `h2o.importFile` function to do a parallel read of the data into the H2O distributed key-value store. During import of the data, features Year, Month, DayOfWeek, and FlightNum were set to be parsed as enumerator or categorical rather than numeric columns. Once the data is in H2O, get an overview of the airlines dataset quickly by using `describe`.
# 

# In[ ]:

airlines_hex = h2o.import_file(path = os.path.realpath("../data/allyears2k.csv"),
                               destination_frame = "airlines.hex")
airlines_hex.describe()


# ### Building a GLM Model
# 
# Run a logistic regression model using function `h2o.glm` and selecting “binomial” for parameter `Family`. Add some regularization by setting alpha to 0.5 and lambda to 1e-05.

# In[ ]:

# Set predictor and response variables
myY = "IsDepDelayed"
myX = ["Dest", "Origin", "DayofMonth", "Year", "UniqueCarrier", "DayOfWeek", "Month", "Distance"]

# GLM - Predict Delays
glm_model = H2OGeneralizedLinearEstimator(
    family = "binomial",standardize = True, solver = "IRLSM",
    link = "logit", alpha = 0.5, model_id = "glm_model_from_python" )
glm_model.train(x               = myX,
               y               = myY,
               training_frame  = airlines_hex)


# In[ ]:

print "AUC of the training set : " + str(glm_model.auc())
# Variable importances from each algorithm
# Calculate magnitude of normalized GLM coefficients
glm_varimp = glm_model.coef_norm()
for k,v in glm_varimp.iteritems():
    glm_varimp[k] = abs(glm_varimp[k])
    
# Sort in descending order by magnitude
glm_sorted = sorted(glm_varimp.items(), key = operator.itemgetter(1), reverse = True)
table = tabulate.tabulate(glm_sorted, headers = ["Predictor", "Normalized Coefficient"], tablefmt = "orgtbl")
print "Variable Importances:\n\n" + table


# ### Building a Deep Learning Model
# 
# Build a binary classfication model using function `h2o.deeplearning` and selecting “bernoulli” for parameter `Distribution`. Run 100 passes over the data by setting parameter `epoch` to 100.

# In[ ]:

# Deep Learning - Predict Delays
deeplearning_model = H2ODeepLearningEstimator(
    distribution = "bernoulli", model_id = "deeplearning_model_from_python",
    epochs = 100, hidden = [200,200],  
    seed = 6765686131094811000, variable_importances = True)
deeplearning_model.train(x               = myX,
                         y               = myY,
                         training_frame  = airlines_hex)


# In[ ]:

print "AUC of the training set : " + str(deeplearning_model.auc())
deeplearning_model.varimp(table)


# ### Shut down the cluster
# 
# Shut down the cluster now that we are done using it.

# In[ ]:

h2o.shutdown(prompt=False)

