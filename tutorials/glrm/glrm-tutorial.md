> # UNDER CONSTRUCTION

# Generalized Low Rank Models
- Overview
- What is a Low Rank Model?
- Why use Low Rank Models?
	- Memory
	- Speed
	- Feature Engineering
	- Missing Data
- Example 1: Visualizing Walking Stances
	- Basic Model Building
	- Plotting Archetypal Features
	- Imputing Missing Values
- Example 2: Compressing Zip Codes
	- Condensing Categorical Data
	- Runtime and Accuracy Comparison
- References

## Overview

This tutorial introduces the Generalized Low Rank Model (GLRM), a convex optimization algorithm for compressing data, imputing missing values and identifying key features. It demonstrates how to build a GLRM in H2O and integrate it into a machine learning pipeline.

## What is a Low Rank Model?

## Why use Low Rank Models?

- **Memory:** By saving only the X and Y matrices, you can significantly reduce the amount of memory required to store a large dataset. A file that is 10 GB can be compressed down to 100 MB. When you need the original data again, you can reconstruct it on the fly from X and Y with minimal loss in accuracy.
- **Speed:** 
- **Feature Engineering:**
- **Missing Data:** Reconstructing a dataset from X and Y will automatically impute missing values. This imputation is accomplished by intelligently leveraging the information contained in the known values of each feature, as well as user-provided parameters such as the loss function.

## Example 1: Visualizing Walking Stances

For our first example, we will use data on [Subject 01's walking stances](https://simtk.org/project/xml/downloads.xml?group_id=603) from an experiment carried out by Hamner and Delp (2013) [2]. Each of the 151 row of the dataset contains the (x, y, z) coordinates of major body parts recorded at a specific time.

###### Initialize the H2O server and import our walking stance data
	library(h2o)
	h2o.init()
	pathToData <- "/data/h2o-training/glrm/subject01_walk1.csv"
	gait.hex <- h2o.importFile(path = pathToData, destination_frame = "gait.hex")

###### Get a summary of the imported dataset
	dim(gait.hex)
	summary(gait.hex)

###### Build a basic GLRM using quadratic loss and no regularization. Since this dataset has no missing values, this is equivalent to principal components analysis (PCA). We skip the first column since it is the time index, set the rank k = 10, and allow the algorithm to run for a maximum of 1,000 iterations.
	gait.glrm <- h2o.glrm(training_frame = gait.hex, cols = 2:ncol(gait.hex), k = 10, loss = "Quadratic", 
	                      regularization_x = "None", regularization_y = "None", max_iterations = 1000)

###### To ensure our algorithm converged, we should always plot the objective function value per iteration after model-building is complete.
	plot(gait.glrm)

###### The rows of the Y matrix represent the principal stances, or archetypes, that Subject 01 took while walking.
	gait.y <- gait.glrm@model$archetypes

###### We can visualize each of the 10 stances by plotting the (x, y) coordinate weights of every body part.
	gait.y.mat <- as.matrix(gait.y)
	x_coords <- seq(1, ncol(gait.y), by = 3)
	y_coords <- seq(2, ncol(gait.y), by = 3)
	feat_nams <- sapply(colnames(gait.y), function(nam) { substr(nam, 1, nchar(nam)-1) })
	feat_nams <- as.character(feat_nams[x_coords])
	for(k in 1:10) {
		plot(gait.y.mat[k,x_coords], gait.y.mat[k,y_coords], xlab = "X-Coordinate Weight", ylab = "Y-Coordinate Weight", main = paste("Feature Weights of Archetype", k), col = "blue", pch = 19, lty = "solid")
		text(gait.y.mat[k,x_coords], gait.y.mat[k,y_coords], labels = feat_nams, cex = 0.7, pos = 3)
		cat("Press [Enter] to continue")
		line <- readline()
	}

###### The rows of the X matrix decompose each bodily position Subject 01 took at a specific time into a combination of the principal stances. Let's plot each principal stance over time to see how they alternate.
	gait.x <- h2o.getFrame(gait.glrm@model$representation_name)
	time.df <- as.data.frame(gait.hex$Time[1:150])[,1]
	gait.x.df <- as.data.frame(gait.x[1:150,])
	matplot(time.df, gait.x.df, xlab = "Time", ylab = "Archetypal Projection", main = "Archetypes over Time", type = "l", lty = 1, col = 1:5)
	legend("topright", legend = colnames(gait.x.df), col = 1:5, pch = 1)

###### We can reconstruct our original training data from X and Y.
	gait.pred <- predict(gait.glrm, gait.hex)
	head(gait.pred)

###### For comparison, let's plot the original and reconstructed data of a specific feature over time: the x-coordinate of the left acromium.
	lacro.df <- as.data.frame(gait.hex$L.Acromium.X[1:150])
	lacro.pred.df <- as.data.frame(gait.pred$reconstr_L.Acromium.X[1:150])
	matplot(time.df, cbind(lacro.df, lacro.pred.df), xlab = "Time", ylab = "X-Coordinate of Left Acromium", main = "Position of Left Acromium over Time", type = "l", lty = 1, col = c(1,4))
	legend("topright", legend = c("Original", "Reconstructed"), col = c(1,4), pch = 1)

## Example 2: Compressing Zip Codes

## References

[1] M. Udell, C. Horn, R. Zadeh, S. Boyd (2014). [Generalized Low Rank Models](http://arxiv.org/abs/1410.0342). Unpublished manuscript, Stanford Electrical Engineering Department.
[2] Hamner, S.R., Delp, S.L. [Muscle contributions to fore-aft and vertical body mass center accelerations over a range of running speeds](http://nmbl.stanford.edu/publications/pdf/Hamner2012.pdf). Journal of Biomechanics, vol 46, pp 780-787. (2013)