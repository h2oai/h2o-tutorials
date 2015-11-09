
# coding: utf-8

# In[ ]:

import h2o
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import os


# In[ ]:

h2o.init()
h2o.remove_all() # Clean slate - just in case the cluster was already running


# In[ ]:

from h2o.h2o import _locate # private function. used to find files within h2o git project directory.

# Import and parse ACS 2013 5-year DP02 demographic data
acs_orig = h2o.import_file(path=os.path.realpath("../data/ACS_13_5YR_DP02_cleaned.zip"), col_types = (["enum"] + ["numeric"]*149))
acs_orig.describe()

acs_zcta_col = acs_orig["ZCTA5"].asfactor()
acs_full = acs_orig.drop("ZCTA5")


# In[ ]:

# Import and parse WHD 2014-2015 labor violations data
whd_zcta = h2o.import_file(path=os.path.realpath("../data/whd_zcta_cleaned.zip"), col_types = (["enum"]*7 + ["numeric"]*97))
whd_zcta["zcta5_cd"] = whd_zcta["zcta5_cd"].asfactor()
whd_zcta.describe()


# In[ ]:

# Run GLRM to reduce ZCTA demographics to 10 archetypes
acs_model = H2OGeneralizedLowRankEstimator(k = 10,
                                           transform = "STANDARDIZE",
                                           loss = "Quadratic",
                                           regularization_x = "Quadratic",
                                           regularization_y = "L1",
                                           gamma_x = 0.25,
                                           gamma_y = 0.5,
                                           max_iterations = 100)
acs_model.train(x = acs_full.names, training_frame= acs_full)
print acs_model


# In[ ]:

# Plot objective function value each iteration
acs_model_score = acs_model.score_history()
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.title("Objective Function Value per Iteration")
plt.plot(acs_model_score["iteration"], acs_model_score["objective"])
plt.show()


# In[ ]:

# Embedding of ZCTAs into archetypes (X)
zcta_arch_x = h2o.get_frame(acs_model._model_json["output"]["representation_name"])
zcta_arch_x.head()


# In[ ]:

# Plot a few ZCTAs on the first two archetypes
idx = ((acs_zcta_col == "10065") |   # Manhattan, NY (Upper East Side)
       (acs_zcta_col == "11219") |   # Manhattan, NY (East Harlem)
       (acs_zcta_col == "66753") |   # McCune, KS
       (acs_zcta_col == "84104") |   # Salt Lake City, UT
       (acs_zcta_col == "94086") |   # Sunnyvale, CA
       (acs_zcta_col == "95014"))    # Cupertino, CA

city_arch = np.array(h2o.as_list(zcta_arch_x[idx,[0,1]]))
plt.xlabel("First Archetype")
plt.ylabel("Second Archetype")
plt.title("Archetype Representation of Zip Code Tabulation Areas")
plt.plot(city_arch[:,0], city_arch[:,1], "o")

# Label city names corresponding to ZCTAs
city_names = ["Upper East Side", "East Harlem", "McCune", "Salt Lake City", "Sunnyvale", "Cupertino"]
for i, txt in enumerate(city_names):
   plt.annotate(txt, (city_arch[i,0], city_arch[i,1]))
plt.show()


# In[ ]:

# Archetypes to full feature mapping (Y)
arch_feat_y = acs_model._model_json["output"]["archetypes"]
print arch_feat_y


# In[ ]:

# Split WHD data into test/train with 20/80 ratio
split = whd_zcta["flsa_repeat_violator"].runif()
train = whd_zcta[split <= 0.8]
test = whd_zcta[split > 0.8]

# Build a DL model to predict repeat violators and score
s = time.time()
dl_orig = H2ODeepLearningEstimator(epochs = 0.1, hidden = [50,50,50], distribution = "multinomial")
idx_x = train.names
idx_x.remove("flsa_repeat_violator")
idx_x = idx_x[4:]
dl_orig.train(x               =idx_x,
              y               ="flsa_repeat_violator",
              training_frame  =train,
              validation_frame=test)
orig_elapsed = time.time() - s


# In[ ]:

# Replace zcta5_cd column in WHD data with GLRM archetypes
zcta_arch_x["zcta5_cd"] = acs_zcta_col
whd_arch = whd_zcta.merge(zcta_arch_x, allLeft = True, allRite = False)
whd_arch = whd_arch.drop("zcta5_cd")
whd_arch.describe()


# In[ ]:

# Split WHD data into test/train with 20/80 ratio
train_mod = whd_arch[split <= 0.8]
test_mod = whd_arch[split > 0.8]

# Build a DL model to predict repeat violators and score
s = time.time()
dl_mod = H2ODeepLearningEstimator(epochs = 0.1, hidden = [50,50,50], distribution = "multinomial")

dl_mod.train(x               =idx_x,
             y               ="flsa_repeat_violator",
             training_frame  =train,
             validation_frame=test)

mod_elapsed = time.time() - s


# In[ ]:

# Model performance comparison
train_ll_orig = dl_orig.model_performance(train).logloss()
test_ll_orig  = dl_orig.model_performance(test ).logloss()
train_ll_mod  = dl_mod .model_performance(train).logloss()
test_ll_mod   = dl_mod .model_performance(test ).logloss()

# Print results in pretty HTML table
header = ["Metric"   , "Original"    , "Reduced"    ]
table = [
         ["Runtime"  , orig_elapsed  , mod_elapsed  ],
         ["Train LogLoss", train_ll_orig, train_ll_mod],
         ["Test LogLoss" , test_ll_orig , test_ll_mod ],
        ]
h2o.H2ODisplay(table,header)


# ### Shut down the cluster
# 
# Shut down the cluster now that we are done using it.

# In[ ]:

h2o.shutdown(prompt=False)

