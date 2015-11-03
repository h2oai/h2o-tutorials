> # UNDER CONSTRUCTION

# Generalized Low Rank Models
- Overview
- What is a Low Rank Model?
- Why use Low Rank Models?
	- Memory
	- Speed
	- Feature Engineering
	- Missing Data Imputation
- Example 1: Visualizing Walking Stances
	- Basic Model Building
	- Plotting Archetypal Features
	- Imputing Missing Values
- Example 2: Compressing Zip Codes
	- Condensing Categorical Data
	- Runtime and Accuracy Comparison
- References

## Overview

This tutorial introduces the Generalized Low Rank Model (GLRM), a machine learning method for compressing data, imputing missing values and identifying key features. It demonstrates how to build a GLRM in H2O and integrate it into a data science pipeline.

## What is a Low Rank Model?

## Why use Low Rank Models?

- **Memory:** By saving only the X and Y matrices, we can significantly reduce the amount of memory required to store a large dataset. A file that is 10 GB can be compressed down to 100 MB. When we need the original data again, we can reconstruct it on the fly from X and Y with minimal loss in accuracy.
- **Speed:** We can use GLRM to compress data with high-dimensional, mixed-type features into a few numeric columns. This leads to a huge speed-up in model-building and prediction, especially by machine learning algorithms that scale poorly with the size of the feature space. Below, we will see an example with 10x speed-up and no accuracy loss in deep learning.
- **Feature Engineering:** The Y matrix represents the most important combinations of features from the training data. These condensed features, called "archetypes", can be analyzed, visualized and used in fitting other machine learning models. 
- **Missing Data Imputation:** Reconstructing a dataset from X and Y will automatically impute missing values. This imputation is accomplished by intelligently leveraging the information contained in the known values of each feature, as well as user-provided parameters such as the loss function.

## Example 1: Visualizing Walking Stances

For our first example, we will use data on [Subject 01's walking stances](https://simtk.org/project/xml/downloads.xml?group_id=603) from an experiment carried out by Hamner and Delp (2013) [2]. Each of the 151 row of the dataset contains the (x, y, z) coordinates of major body parts recorded at a specific point in time.

#### Basic Model Building

###### Initialize the H2O server and import our walking stance data.
	library(h2o)
	h2o.init()
	pathToData <- "/data/h2o-training/glrm/subject01_walk1.csv"
	gait.hex <- h2o.importFile(path = pathToData, destination_frame = "gait.hex")

###### Get a summary of the imported dataset.
	dim(gait.hex)
	summary(gait.hex)

###### Build a basic GLRM using quadratic loss and no regularization. Since this dataset has no missing values, this is equivalent to principal components analysis (PCA). We skip the first column since it is the time index, set the rank k = 10, and allow the algorithm to run for a maximum of 1,000 iterations.
	gait.glrm <- h2o.glrm(training_frame = gait.hex, cols = 2:ncol(gait.hex), k = 10, loss = "Quadratic", 
	                      regularization_x = "None", regularization_y = "None", max_iterations = 1000)

###### To ensure our algorithm converged, we should always plot the objective function value per iteration after model-building is complete.
	plot(gait.glrm)

#### Plotting Archetypal Features

###### The rows of the Y matrix represent the principal stances, or archetypes, that Subject 01 took while walking. We can visualize each of the 10 stances by plotting the (x, y) coordinate weights of every body part.
	gait.y <- gait.glrm@model$archetypes
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

#### Imputing Missing Values

Suppose that due to a sensor malfunction, our walking stance data has missing values randomly interspersed. We can use GLRM to reconstruct these missing values from the existing data.

###### Import walking stance data containing 15% missing values.
	pathToMissingData <- "/data/h2o-training/glrm/subject01_walk1_miss15.csv"
	gait.miss <- h2o.importFile(path = pathToMissingData, destination_Frame = "gait.miss")

###### Get a summary of the imported dataset.
	dim(gait.miss)
	summary(gait.miss)
	sum(is.na(gait.miss))

###### Build a basic GLRM with quadratic loss and no regularization, validating on our original dataset with no missing values. We change the algorithm initialization method, increase the maximum number of iterations to 2,000, and reduce the minimum step size to 1e-6 to ensure it converges.
	gait.glrm2 <- h2o.glrm(training_frame = gait.miss, validation_frame = gait.hex, cols = 2:ncol(gait.miss), k = 10, init = "SVD", svd_method = "GramSVD",
	                      loss = "Quadratic", regularization_x = "None", regularization_y = "None", max_iterations = 2000, min_step_size = 1e-6)
	plot(gait.glrm2)

###### Impute missing values in our training data from X and Y.
	gait.pred2 <- predict(gait.glrm2, gait.miss)
	head(gait.pred2)
	sum(is.na(gait.pred2))

###### Plot original and reconstructed data of the x-coordinate of the left acromium. Red x's mark the points where the training data contains a missing value, so we can see how accurate our imputation is.
	lacro.pred.df2 <- as.data.frame(gait.pred2$reconstr_L.Acromium.X[1:150])
	matplot(time.df, cbind(lacro.df, lacro.pred.df2), xlab = "Time", ylab = "X-Coordinate of Left Acromium", main = "Position of Left Acromium over Time", type = "l", lty = 1, col = c(1,4))
	legend("topright", legend = c("Original", "Imputed"), col = c(1,4), pch = 1)
	lacro.miss.df <- as.data.frame(gait.miss$L.Acromium.X[1:150])
	idx_miss <- which(is.na(lacro.miss.df))
	points(time.df[idx_miss], lacro.df[idx_miss,1], col = 2, pch = 4, lty = 2)

## Example 2: Compressing Zip Codes

For our second example, we will be using two datasets. The first is compliance actions carried out by the U.S. Labor Department's [Wage and Hour Division (WHD)](http://ogesdw.dol.gov/views/data_summary.php) from 2014-2015. This includes information on each investigation, including the zip code tabulation area (ZCTA) at which the firm is located, number of violations found, and civil penalties assessed. We want to predict whether a firm is a repeat and/or willful violator. In order to do this, we need to encode the categorical ZCTA column in a meaningful way. One common approach is to replace ZCTA with indicator variables for every unique level, but due to its high cardinality (there are over 32,000 ZCTAs!), this is slow and leads to overfitting.

Instead, we will use GLRM to condense ZCTAs into a few numeric columns representing the demographics of that area. Our second dataset is the 2009-2013 [American Community Survey (ACS)](http://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk) 5-year estimates of household characteristics. Each row contains information for a unique ZCTA, such as average household size, number of children, education level and ethnicity. By transforming the WHD data with GLRM, we not only address the speed and overfitting issue, but also transfer knowledge between similar ZCTAs in our model.

#### Condensing Categorical Data

###### Initialize the H2O server and import the ACS dataset.
	library(h2o)
	h2o.init()
	pathToACSData <- "/data/h2o-training/glrm/ACS_13_5YR_DP02_cleaned.zip"
	acs_orig <- h2o.uploadFile(path = pathToACSData, col.types = c("enum", rep("numeric", 149)))

###### Save and drop the zip code tabulation area column.
	acs_zcta_col <- acs_orig$ZCTA5
	acs_full <- acs_orig[,-which(colnames(acs_orig) == "ZCTA5")]

###### Get a summary of the ACS dataset.
	dim(acs_full)
	summary(acs_full)

###### Build a GLRM to reduce ZCTA demographics to k = 10 archetypes. We standardize the data before performing the fit to ensure differences in scale between columns don't unduly affect the algorithm. For the loss function, we select quadratic again, but this time, we apply regularization to X and Y in order to sparsify the resulting features.
	acs_model <- h2o.glrm(training_frame = acs_full, k = 10, transform = "STANDARDIZE", 
	                      loss = "Quadratic", regularization_x = "Quadratic", 
	                      regularization_y = "L1", max_iterations = 100, gamma_x = 0.25, gamma_y = 0.5)
	plot(acs_model)

###### Plot a few interesting ZCTAs on the first two archetypes. We should see cities with similar demographics, such as Sunnyvale and Cupertino, grouped close together, while very different cities, such as the rural town McCune and the upper east side of Manhattan, fall far apart on the graph.
	idx <- ((acs_zcta_col == "10065") |   # Manhattan, NY (Upper East Side)
        	(acs_zcta_col == "11219") |   # Manhattan, NY (East Harlem)
        	(acs_zcta_col == "66753") |   # McCune, KS
        	(acs_zcta_col == "84104") |   # Salt Lake City, UT
        	(acs_zcta_col == "94086") |   # Sunnyvale, CA
        	(acs_zcta_col == "95014"))    # Cupertino, CA
	city_arch <- as.data.frame(zcta_arch_x[idx,1:2])
	xeps <- (max(city_arch[,1]) - min(city_arch[,1])) / 10
	yeps <- (max(city_arch[,2]) - min(city_arch[,2])) / 10
	xlims <- c(min(city_arch[,1]) - xeps, max(city_arch[,1]) + xeps)
	ylims <- c(min(city_arch[,2]) - yeps, max(city_arch[,2]) + yeps)
	plot(city_arch[,1], city_arch[,2], xlim = xlims, ylim = ylims, xlab = "First Archetype", ylab = "Second Archetype", main = "Archetype Representation of Zip Code Tabulation Areas")
	text(city_arch[,1], city_arch[,2], labels = c("Upper East Side", "East Harlem", "McCune", "Salt Lake City", "Sunnyvale", "Cupertino"), pos = 1)

#### Runtime and Accuracy Comparison

###### Import WHD dataset and get a summary.
	pathToWHDData <- "/data/h2o-training/glrm/whd_zcta_cleaned.zip"
	whd_zcta <- h2o.uploadFile(path = pathToWHDData, col.types = c(rep("enum", 7), rep("numeric", 97)))
	dim(whd_zcta)
	summary(whd_zcta)

## References

[1] M. Udell, C. Horn, R. Zadeh, S. Boyd (2014). [Generalized Low Rank Models](http://arxiv.org/abs/1410.0342). Unpublished manuscript, Stanford Electrical Engineering Department.

[2] Hamner, S.R., Delp, S.L. [Muscle contributions to fore-aft and vertical body mass center accelerations over a range of running speeds](http://nmbl.stanford.edu/publications/pdf/Hamner2012.pdf). Journal of Biomechanics, vol 46, pp 780-787. (2013)