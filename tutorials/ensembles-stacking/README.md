> # !!! UNDER CONSTRUCTION !!!
> (But open the book anyway, we're adding content...)

# Ensembles: Stacking, Super Learner
- Overview
- What is Ensemble Learning?
  - Bagging
  - Boosting
  - Stacking / Super Learning
- H2O Ensemble: Super Learning in H2O

# Overview

In this tutorial, we will discuss ensemble learning with a focus on a type of ensemble learning called stacking or Super Learning.  We present the H2O implementation of the Super Learner algorithm, called "H2O Ensemble."  

Following the introduction to ensemble learning, we will dive into a hands-on code demo of the **h2oEnsemble** R package.


# What is Ensemble Learning?

Ensemble machine learning methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms.

Many of the popular modern machine learning algorithms are actually ensembles.  For example, [Random Forest](https://en.wikipedia.org/wiki/Random_forest) and [Gradient Boosting Machine](https://en.wikipedia.org/wiki/Gradient_boosting) are both ensemble learners.

Common types of ensembles:
- [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
- [Boosting](https://en.wikipedia.org/wiki/Boosting_%28machine_learning%29)
- [Stacking](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)

## Bagging
Bootstrap aggregating, or bagging, is an ensemble method designed to improve the stability and accuracy of machine learning algorithms.  It reduces variance and helps to avoid overfitting.  Bagging is a special case of the model averaging approach and is relatively robust against noisy data and outliers.

One of the most well known bagging ensembles is the Random Forest algorithm, which applies bagging to decision trees.

## Boosting
Boosting is an ensemble method designed to reduce bias and variance.  A boosting algorithm iteratively learns weak classifiers and adds them to a final strong classifier.   

After a weak learner is added, the data is reweighted: examples that are misclassified gain weight and examples that are classified correctly lose weight. Thus, future weak learners focus more on the examples that previous weak learners misclassified.  This causes boosting methods to be not very robust to noisy data and outliers.

Both bagging and boosting are ensembles that take a collection of weak learners and forms a single, strong learner.


## Stacking / Super Learning

Stacking is a broad class of algorithms that involves training a second-level "metalearner" to ensemble a group of base learners. 

The type of ensemble learning implemented in H2O is called "super learning", "stacked regression" or "stacking."  The Super Learner algorithm learns the optimal combination of the base learner fits. In a 2007 article titled, "[Super Learner](http://dx.doi.org/10.2202/1544-6115.1309)," it was shown that the super learner ensemble represents an asymptotically optimal system for learning.

### Super Learner Algorithm
Set up the ensemble:
- Specify a list of L base algorithms (with a specific set of model parameters).
- Specify a metalearning algorithm

Train the ensemble:
- Train each of the L base algorithms on the training set.
- Perform k-fold cross-validation on each of these learners and collect the cross-validated predicted values from each of the L algorithms.
- The N cross-validated predicted values from each of the L algorithms can be combined to form a new N x L matrix.  This matrix, along wtih the original response vector, is called the "level-one" data.
- Train the metalearning algorithm on the level-one data.

The "ensemble model" consists of the L base learning models and the metalearning model.

Predict on new data:
- To generate ensemble predictions, first generate predictions from the base learners.
- Feed those predictions into the metalearner to generate the ensemble prediction.


# H2O Ensemble: Super Learning in H2O

The H2O Super Learner ensemble has been implemented as a stand-alone R package called [h2oEnsemble](https://github.com/h2oai/h2o-3/tree/master/h2o-r/ensemble).  The package is an extension to the [h2o](https://cran.r-project.org/web/packages/h2o/index.html) R package that allows the user to train an ensemble containing H2O algorithms.  As in the **h2o** R package, all of the actual computation in **h2oEnsemble** is performed inside the H2O cluster, rather than in R memory.  

The main computational tasks in the Super Learner ensemble algorithm is the training and cross-validation of the base learners and metalearner.  Therefore, implementing the "plumbing" of the ensemble in R (rather than in Java) does not incur a loss of performance.


## Install H2O Ensemble

To install the **h2oEnsemble** package, you just need to follow the installation instructions on the [README](https://github.com/h2oai/h2o-3/blob/master/h2o-r/ensemble/README.md#install) file.


## Demo

This is an example of binary classification using `h2o.ensemble`.  This example is also included in the R package documentation for h2o.ensemble


### Start H2O Cluster
```r
library(h2oEnsemble)  # Requires version >=0.0.4 of h2oEnsemble
localH2O <-  h2o.init(nthreads = -1)  # Start an H2O cluster with nthreads = num cores on your machine
```


### Load Data into H2O Cluster
```r
# Import a sample binary outcome train/test set into R
train <- h2o.importFile("http://www.stat.berkeley.edu/~ledell/data/higgs_10k.csv")
test <- h2o.importFile("http://www.stat.berkeley.edu/~ledell/data/higgs_test_5k.csv")
y <- "C1"
x <- setdiff(names(train), y)

#For binary classification, response should be a factor
train[,y] <- as.factor(train[,y])  
test[,y] <- as.factor(test[,y])
```


### Specify Base Learners & Metalearner
For this example, we will use the default base learner library, which includes the H2O GLM, Random Forest, GBM and Deep Learner (all using default model parameter values).

```r
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "h2o.deeplearning.wrapper"
```


### Train an Ensemble
Train the ensemble using 5-fold CV to generate level-one data.  Note that more CV folds will take longer to train, but should increase performance.
```r
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = train, 
                    family = "binomial", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5, shuffle = TRUE))
```


### Predict 
Generate predictions on the test set.
```r
pp <- predict(fit, test)
predictions <- as.data.frame(pp$pred)[,3]  #third column, p1 is P(Y==1)
labels <- as.data.frame(test[,y])[,1]
```

### Model Evaluation

Since the response is binomial, we can use Area Under the ROC Curve (AUC) to evaluate the model performance.  We first generate predictions on the test set and then calculate test set AUC using the [cvAUC](https://cran.r-project.org/web/packages/cvAUC/) R package.

```r
# Ensemble test AUC 
library(cvAUC)  # Used to calculate test set AUC (cvAUC version >=1.0.1)
cvAUC::AUC(predictions = predictions, labels = labels)
# 0.7888723

# Base learner test AUC (for comparison)
L <- length(learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pp$basepred)[,l], labels = labels)) 
data.frame(learner, auc)
#                   learner       auc
#1          h2o.glm.wrapper 0.6871288
#2 h2o.randomForest.wrapper 0.7711654
#3          h2o.gbm.wrapper 0.7817075
#4 h2o.deeplearning.wrapper 0.7425813

# Note that the ensemble results above are not reproducible since 
# h2o.deeplearning is not reproducible when using multiple cores,
# and we did not set a seed for h2o.randomForest.wrapper or h2o.gbm.wrapper.
```
Additional note: In a future version, performance metrics such as AUC will be computed automatically, as in the other H2O algos.


### Specifying New Learners

Here is an example of how to generate a base learner library using custom base learners:
```r
h2o.randomForest.1 <- function(..., ntrees = 1000, nbins = 100, seed = 1) {
  h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
}
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", seed = 1) {
  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
}
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", seed = 1) {
  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
}
learner <- c("h2o.randomForest.1", "h2o.deeplearning.1", "h2o.deeplearning.2")
```




TO FINISH...

## Roadmap for H2O Ensemble
H2O Ensemble is currently only available using the R API, however, it will be accessible via all our APIs in a future release.  You can follow the progress of H2O Ensemble development on the [H2O JIRA](https://0xdata.atlassian.net/secure/IssueNavigator.jspa?reset=true&jqlQuery=project+%3D+PUBDEV+AND+component+%3D+Ensemble)   



