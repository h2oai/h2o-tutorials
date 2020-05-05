# Ensembles: Stacking, Super Learner

- [Overview](#overview)
- [What is Ensemble Learning?](#what-is-ensemble-learning)
	- [Bagging](#bagging)
	- [Boosting](#boosting)
	- [Stacking / Super Learning](#stacking--super-learning)
- [H2O Stacked Ensemble](#h2o-stacked-ensemble)

# Overview

In this tutorial, we will discuss ensemble learning with a focus on a type of ensemble learning called stacking or Super Learning. In this tutorial, we present an H2O implementation of the Super Learner algorithm (aka Stacking, Stacked Ensembles).

H2O’s Stacked Ensemble method is a supervised ensemble machine learning algorithm that finds the optimal combination of a collection of prediction algorithms using a process called stacking. like all supervised models in H2O, Stacked Ensemble supports regression, binary classification, and multiclass classification. The documentation for H2O Stacked Ensembles, including R and Python code examples, can be found [here](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html).



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

Stacking is a broad class of algorithms that involves training a second-level "metalearner" to ensemble a group of base learners. The type of ensemble learning implemented in H2O is called "super learning", "stacked regression" or "stacking."  Unlike bagging and boosting, the goal in stacking is to ensemble strong, diverse sets of learners together.

### Some Background

[Leo Breiman](https://en.wikipedia.org/wiki/Leo_Breiman), known for his work on classification and regression trees and the creator of the Random Forest algorithm, formalized stacking in his 1996 paper, ["Stacked Regressions"](http://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf).  Although the idea originated with [David Wolpert](https://en.wikipedia.org/wiki/David_Wolpert) in 1992 under the name ["Stacked Generalization"](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.1533), the modern form of stacking that uses internal k-fold cross-validation was Dr. Breiman's contribution.

However, it wasn't until 2007 that the theoretical background for stacking was developed, which is when the algorithm took on the name, "Super Learner".  Until this time, the mathematical reasons for why stacking worked were unknown and stacking was considered a "black art."  The Super Learner algorithm learns the optimal combination of the base learner fits. In an article titled, ["Super Learner"](http://dx.doi.org/10.2202/1544-6115.1309), by [Mark van der Laan](http://www.stat.berkeley.edu/~laan/Laan/laan.html) et al., proved that the Super Learner ensemble represents an asymptotically optimal system for learning.

### Super Learner Algorithm

Here is an outline of the tasks involved in training and testing a Super Learner ensemble:

#### Set up the ensemble

- Specify a list of L base algorithms (with a specific set of model parameters).
- Specify a metalearning algorithm.

#### Train the ensemble

- Train each of the L base algorithms on the training set.
- Perform k-fold cross-validation on each of these learners and collect the cross-validated predicted values from each of the L algorithms.
- The N cross-validated predicted values from each of the L algorithms can be combined to form a new N x L matrix.  This matrix, along wtih the original response vector, is called the "level-one" data.  (N = number of rows in the training set)
- Train the metalearning algorithm on the level-one data.
- The "ensemble model" consists of the L base learning models and the metalearning model, which can then be used to generate predictions on a test set.

#### Predict on new data

- To generate ensemble predictions, first generate predictions from the base learners.
- Feed those predictions into the metalearner to generate the ensemble prediction.


# H2O Stacked Ensemble in R

## Install H2O R Package

First you need to install the H2O R package if you don’t already have it installed. It an be downloaded from CRAN or from the H2O website at: [http://h2o.ai/download](http://h2o.ai/download). 

## Higgs Demo

This is an example of binary classification using the `h2o.stackedEnsemble` function. This demo uses a subset of the [HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS), which has 28 numeric features and a binary response.  The machine learning task in this example is to distinguish between a signal process which produces Higgs bosons (Y = 1) and a background process which does not (Y = 0).  The dataset contains approximately the same number of positive vs negative examples.  In other words, this is a balanced, rather than imbalanced, dataset.

To run this script, be sure to `setwd()` to the location of this script. `h2o.init()` starts H2O in R’s current working directory. `h2o.importFile()` looks for files from the perspective of where H2O was started.

### Start H2O Cluster

```r
library(h2o)
h2o.init()
```

### Load Data into H2O Cluster

First, import a sample binary outcome train and test set into the H2O cluster.

```r
train <- h2o.importFile("https://s3.amazonaws.com/h2o-public-test-data/smalldata/testng/higgs_train_5k.csv")
test <- h2o.importFile("https://s3.amazonaws.com/h2o-public-test-data/testng/higgs_test_5k.csv")
```

Identify predictors and response:

```
y <- "response"
x <- setdiff(names(train), y)
```

For binary classification, the response should be encoded as a [factor](http://stat.ethz.ch/R-manual/R-patched/library/base/html/factor.html) type (also known as the [enum](https://docs.oracle.com/javase/tutorial/java/javaOO/enum.html) type in Java or [categorial](http://pandas.pydata.org/pandas-docs/stable/categorical.html) in Python Pandas).  The user can specify column types in the `h2o.importFile` command, or you can convert the response column as follows:

```r
train[,y] <- as.factor(train[,y])  
test[,y] <- as.factor(test[,y])
```

Number of CV folds (to generate level-one data for stacking):

```r
nfolds <- 5
```

### Train an Ensemble

There are a few ways to assemble a list of models to stack together: 

1. Train individual models and put them in a list 
2. Train a grid of models
3. Train several grids of models 

We demonstrate some of these methods below.

**Note:** In order to use a model for stacking you must set `keep_cross_validation_predctions = TRUE` because the Stacked Ensemble algorithm requires the cross-validation predictions to train the metalaerner algorithm (unless you use a [blending frame](docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/blending_frame.html).

#### 1. Generate a 2-model ensemble (GBM + RF)

```r
# Train & cross-validate a GBM:
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train,
                  distribution = “bernoulli”,
                  ntrees = 10,
                  max_depth = 3,
                  min_rows = 2,
                  learn_rate = 0.2,
                  nfolds = nfolds,
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)

# Train & cross-validate a RF:
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train,
                          ntrees = 50,
                          nfolds = nfolds,
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)

# Train a stacked ensemble using the GBM and RF above:
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                base_models = list(my_gbm, my_rf))
```

##### Eval the ensemble performance on a test set:

Since the the response is binomial, we can use Area Under the ROC Curve (AUC) to evaluate the model performance. Compute test set performance, and sort by AUC (the default metric that is printed for a binomial classification):

```r
perf <- h2o.performance(ensemble, newdata = test)
ensemble_auc_test <- h2o.auc(perf)
```

##### Compare to the base learner performance on the test set.

We can compare the performance of the ensemble to the performance of the individual learners in the ensemble.
 
```r
perf_gbm_test <- h2o.performance(my_gbm, newdata = test)
perf_rf_test <- h2o.performance(my_rf, newdata = test)
baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), 
								 h2o.auc(perf_rf_test))

print(sprintf(“Best Base-learner Test AUC: %s”, baselearner_best_auc_test))
print(sprintf(“Ensemble Test AUC: %s”, ensemble_auc_test))
# [1] "Best Base-learner Test AUC:  0.76979821502548"
# [1] "Ensemble Test AUC:  0.773501212640419"
```

So we see the best individual algorithm in this group is the GBM with a test set AUC of 0.7735, as compared to 0.7698 for the ensemble. At first thought, this might not seem like much, but in many industries like medicine or finance, this small advantage can be highly valuable.

To increase the performance of the ensemble, we have several options. One of them is to increase the number of cross-validation folds using the `nfolds` argument. The other options are to change the base learner library or the metalearning algorithm.


##### Generate predictions on a test set (if necessary):

```r
pred <- h2o.predict(ensemble, newdata = test)
```

### 2. Generate a Random Grid of Models and Stack Them Together

```r
# GBM Hyperparamters
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
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                base_models = gbm_grid@model_ids)

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

# Compare to base learner performance on the test set
.getauc <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata = test))
baselearner_aucs <- sapply(gbm_grid@model_ids, .getauc)
baselearner_best_auc_test <- max(baselearner_aucs)
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))
# [1] "Best Base-learner Test AUC:  0.748146530400473"
# [1] "Ensemble Test AUC:  0.773501212640419"

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = test)
```

### All done, shutdown H2O

```r
h2o.shutdown()
```

## Roadmap for H2O Stacked Ensemble

Open tickets for the native H2O version of Stacked Ensembles can be found [here](https://0xdata.atlassian.net/issues/?filter=19301) (JIRA tickets with the "StackedEnsemble" tag).
