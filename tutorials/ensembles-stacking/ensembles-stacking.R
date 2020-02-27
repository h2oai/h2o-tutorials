## Ensembles: Stacking, Super Learner
#- Overview
#- What is Ensemble Learning?
#  - Bagging
#  - Boosting
#  - Stacking / Super Learning
#- H2O Ensemble: Super Learning in H2O
#
## Overview
#
#In this tutorial, we will discuss ensemble learning with a focus on a type of ensemble learning called stacking or Super Learning.  We present the H2O implementation of the Super Learner algorithm, called "H2O Ensemble."  
#
#Following the introduction to ensemble learning, we will dive into a hands-on code demo of the [h2oEnsemble](https://github.com/h2oai/h2o-3/tree/master/h2o-r/ensemble) R package.
#
#
## What is Ensemble Learning?
#
#Ensemble machine learning methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms.
#
#Many of the popular modern machine learning algorithms are actually ensembles.  For example, [Random Forest](https://en.wikipedia.org/wiki/Random_forest) and [Gradient Boosting Machine](https://en.wikipedia.org/wiki/Gradient_boosting) are both ensemble learners.
#
#Common types of ensembles:
#- [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
#- [Boosting](https://en.wikipedia.org/wiki/Boosting_%28machine_learning%29)
#- [Stacking](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)
#
### Bagging
#Bootstrap aggregating, or bagging, is an ensemble method designed to improve the stability and accuracy of machine learning algorithms.  It reduces variance and helps to avoid overfitting.  Bagging is a special case of the model averaging approach and is relatively robust against noisy data and outliers.
#
#One of the most well known bagging ensembles is the Random Forest algorithm, which applies bagging to decision trees.
#
### Boosting
#Boosting is an ensemble method designed to reduce bias and variance.  A boosting algorithm iteratively learns weak classifiers and adds them to a final strong classifier.   
#
#After a weak learner is added, the data is reweighted: examples that are misclassified gain weight and examples that are classified correctly lose weight. Thus, future weak learners focus more on the examples that previous weak learners misclassified.  This causes boosting methods to be not very robust to noisy data and outliers.
#
#Both bagging and boosting are ensembles that take a collection of weak learners and forms a single, strong learner.
#
#
### Stacking / Super Learning
#
#Stacking is a broad class of algorithms that involves training a second-level "metalearner" to ensemble a group of base learners. The type of ensemble learning implemented in H2O is called "super learning", "stacked regression" or "stacking."  Unlike bagging and boosting, the goal in stacking is to ensemble strong, diverse sets of learners together.
#
#### Some Background
#[Leo Breiman](https://en.wikipedia.org/wiki/Leo_Breiman), known for his work on classification and regression trees and the creator of the Random Forest algorithm, formalized stacking in his 1996 paper, ["Stacked Regressions"](http://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf).  Although the idea originated with [David Wolpert](https://en.wikipedia.org/wiki/David_Wolpert) in 1992 under the name ["Stacked Generalization"](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.1533), the modern form of stacking that uses internal k-fold cross-validation was Dr. Breiman's contribution.
#
#However, it wasn't until 2007 that the theoretical background for stacking was developed, which is when the algorithm took on the name, "Super Learner".  Until this time, the mathematical reasons for why stacking worked were unknown and stacking was considered a "black art."  The Super Learner algorithm learns the optimal combination of the base learner fits. In an article titled, ["Super Learner"](http://dx.doi.org/10.2202/1544-6115.1309), by [Mark van der Laan](http://www.stat.berkeley.edu/~laan/Laan/laan.html) et al., proved that the Super Learner ensemble represents an asymptotically optimal system for learning.
#
#
#### Super Learner Algorithm
#
#Here is an outline of the tasks involved in training and testing a Super Learner ensemble.
#
##### Set up the ensemble
#- Specify a list of L base algorithms (with a specific set of model parameters).
#- Specify a metalearning algorithm.
#
##### Train the ensemble
#- Train each of the L base algorithms on the training set.
#- Perform k-fold cross-validation on each of these learners and collect the cross-validated predicted values from each of the L algorithms.
#- The N cross-validated predicted values from each of the L algorithms can be combined to form a new N x L matrix.  This matrix, along wtih the original response vector, is called the "level-one" data.
#- Train the metalearning algorithm on the level-one data.
#- The "ensemble model" consists of the L base learning models and the metalearning model, which can then be used to generate predictions on a test set.
#
##### Predict on new data
#- To generate ensemble predictions, first generate predictions from the base learners.
#- Feed those predictions into the metalearner to generate the ensemble prediction.
#
#
## H2O Ensemble: Super Learning in H2O
#
#H2O Ensemble has been implemented as a stand-alone R package called [h2oEnsemble](https://github.com/h2oai/h2o-3/tree/master/h2o-r/ensemble).  The package is an extension to the [h2o](https://cran.r-project.org/web/packages/h2o/index.html) R package that allows the user to train an ensemble in the H2O cluster using any of the supervised machine learning algorithms H2O.  As in the **h2o** R package, all of the actual computation in **h2oEnsemble** is performed inside the H2O cluster, rather than in R memory.  
#
#The main computational tasks in the Super Learner ensemble algorithm are the training and cross-validation of the base learners and metalearner.  Therefore, implementing the "plumbing" of the ensemble in R (rather than in Java) does not incur a loss of performance.  All training and data processing are performed in the high-performance H2O cluster.
#
#H2O Ensemble currently supports regression and binary classification.  Multi-class support will be added in a future release.
#
#
### Install H2O Ensemble
#
#To install the **h2oEnsemble** package, you just need to follow the installation instructions on the [README](https://github.com/h2oai/h2o-3/blob/master/h2o-r/ensemble/README.md#install) file, also documented here for convenience.
#
#### H2O R Package
#
#First you need to install the H2O R package if you don't already have it installed.  The R installation instructions are at: [http://h2o.ai/download](http://h2o.ai/download)
#
#
#### H2O Ensemble R Package
#
#The recommended way of installing the **h2oEnsemble** R package is directly from GitHub using the [devtools](https://cran.r-project.org/web/packages/devtools/index.html) package (however, [H2O World](http://h2oworld.h2o.ai/) tutorial attendees should install the package from the provided USB stick).
#
##### Install from GitHub
library(devtools)
install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
#
#
### Higgs Demo
#
#This is an example of binary classification using the `h2o.ensemble` function, which is available in **h2oEnsemble**.  This demo uses a subset of the [HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS), which has 28 numeric features and a binary response.  The machine learning task in this example is to distinguish between a signal process which produces Higgs bosons (Y = 1) and a background process which does not (Y = 0).  The dataset contains approximately the same number of positive vs negative examples.  In other words, this is a balanced, rather than imbalanced, dataset.
#   
#If run from plain R, execute R in the directory of this script. If run from RStudio, be sure to setwd() to the location of this script. h2o.init() starts H2O in R's current working directory. h2o.importFile() looks for files from the perspective of where H2O was started.
#
#### Start H2O Cluster
library(h2o)
h2o.init()  
h2o.removeAll() # Clean slate - just in case the cluster was already running
#
#
#### Load Data into H2O Cluster
#
#First, import a sample binary outcome train and test set into the H2O cluster.
train <- h2o.importFile(path = normalizePath("../data/higgs_10k.csv"))
test <- h2o.importFile(path = normalizePath("../data/higgs_test_5k.csv"))
#
#Identify predictors and response
y <- "response"
x <- setdiff(names(train), y)
#
#For binary classification, the response should be encoded as factor (also known as the [enum](https://docs.oracle.com/javase/tutorial/java/javaOO/enum.html) type in Java).  The user can specify column types in the `h2o.importFile` command, or you can convert the response column as follows:
train[,y] <- as.factor(train[,y])  
test[,y] <- as.factor(test[,y])
#
#Number of CV folds (to generate level-one data for stacking)
nfolds <- 5
#
#### There are a few ways to assemble a list of models to stack toegether:
# 1. Train individual models and put them in a list
# 2. Train a grid of models
# 3. Train several grids of models
# Note: All base models must have the same cross-validation folds and
# the cross-validated predicted values must be kept.
#
#### 1. Generate a 2-model emsemble (GBM + RF)
#
#Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train,
                  distribution = "bernoulli",
                  ntrees = 10,
                  max_depth = 3,
                  min_rows = 2,
                  learn_rate = 0.2,
                  nfolds = nfolds,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)
#
#Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = training_frame,
                          ntrees = 10,
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)
#
#### Train a stacked ensemble using the GBM and RF above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                model_id = "my_ensemble_binomial",
                                base_models = list(my_gbm, my_rf))
#
#### Evaluate ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)
#
#Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = test)
perf_rf_test <- h2o.performance(my_rf, newdata = test)
baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), h2o.auc(perf_rf_test))
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("BEst Base-learner Test AUC: %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC: %s", ensemble_auc_test))
#
#### Generate predictions on a test
#
pred <- h2o.predict(ensemble, newdata = test)
#
#### 2. Generate a random grid of models and stack them together
#GBM Hyperparameters
learn_rate_opt <- c(0.01, 0.03)
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 3,
                        seed = 1)

gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid_binomial",
                     x = x,
                     y = y,
                     training_frame = train,
                     ntrees = 10,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)
#
#### Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                model_id = "ensemble_gbm_grid_binomial",
                                base_models = gbm_grid@model_ids)
#
#### Evaluate ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)
#
#Compare to base learner performance on the test set
.getauc <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata = test))
baselearner_aucs <- sapply(gbm_grid@model_ids, .getauc)
baselearner_best_auc_test <- max(baselearner_aucs)
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("Best Base-learner Test AUC: %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC: %s", ensemble_auc_test))
#
#
#We actually lose performance by removing the weak learners!  This demonstrates the power of stacking.
#
#At first thought, you may assume that removing less performant models would increase the perforamnce of the ensemble.  However, each learner has it's own unique contribution to the ensemble and the added diversity among learners usually improves performance.  The Super Learner algorithm learns the optimal way of combining all these learners together in a way that is superior to other combination/blending methods.
#
#
### Roadmap for H2O Ensemble
#H2O Ensemble is currently only available using the R API, however, it will be accessible via all our APIs in a future release.  You can follow the progress of H2O Ensemble development on the [H2O JIRA](https://0xdata.atlassian.net/secure/IssueNavigator.jspa?reset=true&jqlQuery=project+%3D+PUBDEV+AND+component+%3D+Ensemble) (tickets with the "Ensemble" tag). 
#
#### All done, shutdown H2O
h2o.shutdown(prompt=FALSE)
