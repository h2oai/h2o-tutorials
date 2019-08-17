
# coding: utf-8

# In[ ]:

import h2o
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
import os


# In[ ]:

h2o.init()
h2o.remove_all() # Clean slate - just in case the cluster was already running


# In[ ]:

from h2o.h2o import _locate # private function. used to find files within h2o git project directory.

# Import walking gait data
gait = h2o.import_file(path=os.path.realpath("../data/subject01_walk1.csv"))
gait.describe()


# In[ ]:

# Plot first row of data on x- vs. y-coordinate features
gait_row = gait[1,:].drop("Time")
gait_row_np = np.array(h2o.as_list(gait_row))
x_coords = range(0, gait_row_np.shape[1], 3)
y_coords = range(1, gait_row_np.shape[1], 3)

x_pts = gait_row_np[0,x_coords]
y_pts = gait_row_np[0,y_coords]
plt.plot(x_pts, y_pts, 'bo')

# Add feature labels to each point
feat_names = [nam[:-2] for nam in gait_row.col_names[1::3]]
for i in xrange(len(feat_names)):
    plt.annotate(feat_names[i], xy = [x_pts[i], y_pts[i]])
plt.title("Location of Body Parts at Time 0")
plt.xlabel("X-Coordinate Weight")
plt.ylabel("Y-Coordinate Weight")
plt.show()


# In[ ]:

# Basic GLRM using quadratic loss and no regularization (PCA)
model = H2OGeneralizedLowRankEstimator(k=10, loss="Quadratic", regularization_x="None", regularization_y="None", max_iterations=1000)
model.train(x=range(1,gait.ncol), training_frame=gait)
model.show()


# In[ ]:

# Plot objective function value each iteration
model_score = model.score_history()
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.title("Objective Function Value per Iteration")
print model_score
plt.plot(model_score["iteration"], model_score["objective"])
plt.show()


# In[ ]:

# Archetype to feature mapping (Y)
gait_y = model._model_json["output"]["archetypes"]
print gait_y

gait_y_np = np.array(model.archetypes())
x_coords = range(0, gait_y_np.shape[1], 3)
y_coords = range(1, gait_y_np.shape[1], 3)

# Plot archetypes on x- vs. y-coordinate features
for k in xrange(gait_y_np.shape[0]):
    x_pts = gait_y_np[k, x_coords]
    y_pts = gait_y_np[k, y_coords]
    plt.plot(x_pts, y_pts, 'bo')

    # Add feature labels to each point
    feat_names = [nam[:-1] for nam in gait_y.col_header[1::3]]
    for i in xrange(len(feat_names)):
        plt.annotate(feat_names[i], xy = [x_pts[i], y_pts[i]])
    plt.title("Feature Weights of Archetype " + str(k+1))
    plt.xlabel("X-Coordinate Weight")
    plt.ylabel("Y-Coordinate Weight")
    plt.show()


# In[ ]:

# Projection into archetype space (X)
x_key = model._model_json["output"]["representation_name"]
gait_x = h2o.get_frame(x_key)
gait_x.show()

time_np = np.array(h2o.as_list(gait["Time"]))
gait_x_np = np.array(h2o.as_list(gait_x))

# Plot archetypes over time
lines = []
for i in xrange(gait_x_np.shape[1]):
    lines += plt.plot(time_np, gait_x_np[:,i], '-')
plt.title("Archetypes over Time")
plt.xlabel("Time")
plt.ylabel("Archetypal Projection")
plt.legend(lines, gait_x.col_names)
plt.show()


# In[ ]:

# Reconstruct data from X and Y
pred = model.predict(gait)
pred.head()


# In[ ]:

# Plot original and reconstructed L.Acromium.X over time
lacro_np = np.array(h2o.as_list(gait["L.Acromium.X"]))
lacro_pred_np = np.array(h2o.as_list(pred["reconstr_L.Acromium.X"]))
line_orig = plt.plot(time_np, lacro_np, '-')
line_imp = plt.plot(time_np, lacro_pred_np, '-')

plt.title("Position of Left Acromium over Time")
plt.xlabel("Time")
plt.ylabel("X-Coordinate of Left Acromium")
blue_patch = mpatches.Patch(color = 'blue', label = 'Original')
green_patch = mpatches.Patch(color = 'green', label='Imputed')
plt.legend([blue_patch, green_patch], ["Original", "Imputed"])
plt.show()


# In[ ]:

# Import walking gait data with missing values
gait_miss = h2o.import_file(path = os.path.realpath("../data/subject01_walk1_miss15.csv"))
gait_miss.describe()


# In[ ]:

# Basic GLRM using quadratic loss and no regularization

model2 = H2OGeneralizedLowRankEstimator(k=10, init="SVD", svd_method = "GramSVD", loss="Quadratic", regularization_x="None", regularization_y="None", max_iterations=2000, min_step_size=1e-6)
model2.train(x=range(1,gait_miss.ncol), training_frame=gait_miss, validation_frame=gait)
model2.show()


# In[ ]:

# Plot objective function value each iteration
model2_score = model2.score_history()
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.title("Objective Function Value per Iteration")
plt.plot(model2_score["iteration"], model2_score["objective"])
plt.show()


# In[ ]:

# Impute missing data from X and Y
pred2 = model2.predict(gait_miss)
pred2.head()


# In[ ]:

# Plot original and imputed L.Acromium.X over time
lacro_pred_np2 = np.array(h2o.as_list(pred2["reconstr_L.Acromium.X"]))
plt.plot(time_np, lacro_np, 'b-')
plt.plot(time_np, lacro_pred_np2, 'g-')

# Mark points where training data contains missing values
idx_miss = zip(*gait_miss["L.Acromium.X"].isna().which().as_data_frame(True).values.tolist())
plt.plot(time_np[idx_miss], lacro_np[idx_miss], "o", marker = "x", ms = 8, mew = 1.5, mec = "r")

plt.title("Position of Left Acromium over Time")
plt.xlabel("Time")
plt.ylabel("X-Coordinate of Left Acromium")
blue_patch = mpatches.Patch(color = 'blue', label = 'Original')
green_patch = mpatches.Patch(color = 'green', label = 'Imputed')
red_patch = mpatches.Patch(color = 'red', label = "Missing")
plt.legend([blue_patch, green_patch, red_patch], ["Original", "Imputed", "Missing"])
plt.show()


# ### Shut down the cluster
# 
# Shut down the cluster now that we are done using it.

# In[ ]:

h2o.shutdown(prompt=False)

