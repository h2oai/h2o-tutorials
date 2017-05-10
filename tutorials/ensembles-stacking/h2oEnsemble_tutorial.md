# h2oEnsemble R Package: Super Learning in H2O

The Super Learner algorithm (aka Stacking) has been implemented as a stand-alone R package called [h2oEnsemble](https://github.com/h2oai/h2o-3/tree/master/h2o-r/ensemble).  The package is an extension to the [h2o](https://cran.r-project.org/web/packages/h2o/index.html) R package that allows the user to train an ensemble in the H2O cluster using any of the supervised machine learning algorithms H2O.  As in the **h2o** R package, all of the actual computation in **h2oEnsemble** is performed inside the H2O cluster, rather than in R memory.

The main computational tasks in the Super Learner ensemble algorithm are the training and cross-validation of the base learners and metalearner.  Therefore, implementing the "plumbing" of the ensemble in R (rather than in Java) does not incur a loss of performance.  All training and data processing are performed in the high-performance H2O cluster.

H2O Ensemble currently supports regression and binary classification.  Multi-class support will be added in a [future release](https://0xdata.atlassian.net/issues/?filter=14900) of **h2oEnsemble**.  

_Note:_ Super Learning / Stacking is how available in the regular **h2o** R package as the `h2o.stackedEnsemble()` method (development [here](https://0xdata.atlassian.net/issues/?filter=19301)).


## Install H2O Ensemble

To install the **h2oEnsemble** package, you just need to follow the installation instructions on the [README](https://github.com/h2oai/h2o-3/blob/master/h2o-r/ensemble/README.md#install) file, also documented here for convenience.

### H2O R Package

First you need to install the H2O R package if you don't already have it installed.  The R installation instructions are at: [http://h2o.ai/download](http://h2o.ai/download)


### H2O Ensemble R Package

The recommended way of installing the **h2oEnsemble** R package is directly from GitHub using the [devtools](https://cran.r-project.org/web/packages/devtools/index.html) package (however, [H2O World](http://h2oworld.h2o.ai/) tutorial attendees should install the package from the provided USB stick).

#### Install from GitHub
```r
library(devtools)
install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
```


## Higgs Demo

This is an example of binary classification using the `h2o.ensemble` function, which is available in **h2oEnsemble**.  This demo uses a subset of the [HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS), which has 28 numeric features and a binary response.  The machine learning task in this example is to distinguish between a signal process which produces Higgs bosons (Y = 1) and a background process which does not (Y = 0).  The dataset contains approximately the same number of positive vs negative examples.  In other words, this is a balanced, rather than imbalanced, dataset.
   
If run from plain R, execute R in the directory of this script. If run from RStudio, be sure to setwd() to the location of this script. h2o.init() starts H2O in R's current working directory. h2o.importFile() looks for files from the perspective of where H2O was started.

### Start H2O Cluster
```r
library(h2oEnsemble)  # This will load the `h2o` R package as well
h2o.init(nthreads = -1)  # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # (Optional) Remove all objects in H2O cluster
```


### Load Data into H2O Cluster

First, import a sample binary outcome train and test set into the H2O cluster.

```r
train <- h2o.importFile("https://s3.amazonaws.com/erin-data/higgs/higgs_train_5k.csv")
test <- h2o.importFile("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")
y <- "response"
x <- setdiff(names(train), y)
family <- "binomial"
```

For binary classification, the response should be encoded as a [factor](http://stat.ethz.ch/R-manual/R-patched/library/base/html/factor.html) type (also known as the [enum](https://docs.oracle.com/javase/tutorial/java/javaOO/enum.html) type in Java or [categorial](http://pandas.pydata.org/pandas-docs/stable/categorical.html) in Python Pandas).  The user can specify column types in the `h2o.importFile` command, or you can convert the response column as follows:

```r
train[,y] <- as.factor(train[,y])  
test[,y] <- as.factor(test[,y])
```


### Specify Base Learners & Metalearner
For this example, we will use the default base learner library for `h2o.ensemble`, which includes the default H2O GLM, Random Forest, GBM and Deep Neural Net (all using default model parameter values).  We will also use the default metalearner, the H2O GLM.

```r
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "h2o.glm.wrapper"
```


### Train an Ensemble
Train the ensemble (using 5-fold internal CV) to generate the level-one data.  Note that more CV folds will take longer to train, but should increase performance.

```r
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = train, 
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))
```


### Evaluate Model Performance

Since the response is binomial, we can use Area Under the ROC Curve ([AUC](https://www.kaggle.com/wiki/AUC)) to evaluate the model performance.  Compute test set performance, and sort by AUC (the default metric that is printed for a binomial classification):

```r
perf <- h2o.ensemble_performance(fit, newdata = test)
```

Print the base learner and ensemble performance:

```r
> perf

Base learner performance, sorted by specified metric:
                   learner       AUC
1          h2o.glm.wrapper 0.6824304
4 h2o.deeplearning.wrapper 0.7006335
2 h2o.randomForest.wrapper 0.7570211
3          h2o.gbm.wrapper 0.7780807


H2O Ensemble Performance on <newdata>:
----------------
Family: binomial

Ensemble performance (AUC): 0.781580655670451
```


We can compare the performance of the ensemble to the performance of the individual learners in the ensemble.

So we see the best individual algorithm in this group is the GBM with a test set AUC of 0.778, as compared to 0.782 for the ensemble.  At first thought, this might not seem like much, but in many industries like medicine or finance, this small advantage can be highly valuable. 

To increase the performance of the ensemble, we have several options.  One of them is to increase the number of internal cross-validation folds using the `cvControl` argument.  The other options are to change the base learner library or the metalearning algorithm.

Note that the ensemble results above are not reproducible since `h2o.deeplearning` is not reproducible when using multiple cores, and we did not set a seed for `h2o.randomForest.wrapper`.

                    
If we want to evaluate the model by a different metric, say "MSE", then we can pass that metric to the `print` method for and ensemble performance object as follows:

```r
> print(perf, metric = "MSE")

Base learner performance, sorted by specified metric:
                   learner       MSE
4 h2o.deeplearning.wrapper 0.2305775
1          h2o.glm.wrapper 0.2225176
2 h2o.randomForest.wrapper 0.2014339
3          h2o.gbm.wrapper 0.1916273


H2O Ensemble Performance on <newdata>:
----------------
Family: binomial

Ensemble performance (MSE): 0.1898735479034431
```

                    

### Predict 

If you actually need to generate the predictions (instead of looking only at model performance), you can use the `predict()` function with a test set.  Generate predictions on the test set and store as an H2O Frame:

```r
pred <- predict(fit, newdata = test)
```

If you need to bring the predictions back into R memory for futher processing, you can convert `pred` to a local R data.frame as follows:

```r
predictions <- as.data.frame(pred$pred)[,3]  #third column is P(Y==1)
labels <- as.data.frame(test[,y])[,1]
```

The `predict` method for an `h2o.ensemble` fit will return a list of two objects.  The `pred$pred` object contains the ensemble predictions, and `pred$basepred` is a matrix of predictions from each of the base learners.  In this particular example where we used four base learners, the `pred$basepred` matrix has four columns.  Keeping the base learner predictions around is useful for model inspection and will allow us to calculate performance of each of the base learners on the test set (for comparison to the ensemble).



### Specifying new learners

Now let's try again with a more extensive set of base learners.  The **h2oEnsemble** packages comes with four functions by default that can be customized to use non-default parameters. 

Here is an example of how to generate a custom learner wrappers:

```r
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
```


Let's grab a subset of these learners for our base learner library and re-train the ensemble.

### Customized base learner library

```r
learner <- c("h2o.glm.wrapper",
             "h2o.randomForest.1", "h2o.randomForest.2",
             "h2o.gbm.1", "h2o.gbm.6", "h2o.gbm.8",
             "h2o.deeplearning.1", "h2o.deeplearning.6", "h2o.deeplearning.7")
```

Train with new library:

```r
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = train,
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))
```

Evaluate the test set performance: 

```r
perf <- h2o.ensemble_performance(fit, newdata = test)
```

We see an increase in performance by including a more diverse library.

Base learner test AUC (for comparison)

```r
> perf

Base learner performance, sorted by specified metric:
             learner       AUC
1    h2o.glm.wrapper 0.6824304
7 h2o.deeplearning.1 0.6897187
8 h2o.deeplearning.6 0.6998472
9 h2o.deeplearning.7 0.7048874
2 h2o.randomForest.1 0.7668024
3 h2o.randomForest.2 0.7697849
4          h2o.gbm.1 0.7751240
6          h2o.gbm.8 0.7752852
5          h2o.gbm.6 0.7771115


H2O Ensemble Performance on <newdata>:
----------------
Family: binomial

Ensemble performance (AUC): 0.780924502576107

```

So what happens to the ensemble if we remove some of the weaker learners?  Let's remove the GLM and DL from the learner library and see what happens...

Here is a more stripped down version of the base learner library used above:

```r
learner <- c("h2o.randomForest.1", "h2o.randomForest.2",
             "h2o.gbm.1", "h2o.gbm.6", "h2o.gbm.8")
```

Again re-train the ensemble and evaluate the performance:

```r
fit <- h2o.ensemble(x = x, y = y, 
                     training_frame = train,
                     family = family, 
                     learner = learner, 
                     metalearner = metalearner,
                     cvControl = list(V = 5))

perf <- h2o.ensemble_performance(fit, newdata = test)
```

We actually lose ensemble performance by removing the weak learners!  This demonstrates the power of stacking with a large and diverse set of base learners.


```r
> perf

Base learner performance, sorted by specified metric:
             learner       AUC
1 h2o.randomForest.1 0.7668024
2 h2o.randomForest.2 0.7697849
3          h2o.gbm.1 0.7751240
5          h2o.gbm.8 0.7752852
4          h2o.gbm.6 0.7771115


H2O Ensemble Performance on <newdata>:
----------------
Family: binomial

Ensemble performance (AUC): 0.778853964308554

```

At first thought, you may assume that removing less performant models would increase the perforamnce of the ensemble.  However, each learner has it's own unique contribution to the ensemble and the added diversity among learners usually improves performance.  The Super Learner algorithm learns the optimal way of combining all these learners together in a way that is superior to other combination/blending methods.

### Stacking Existing Model Sets

You can also use an existing (cross-validated) list of H2O models as the starting point and use the `h2o.stack()` function to ensemble them together via a specified metalearner.  The base models must have been trained on the same dataset with same response and for cross-validation, must have all used the same folds.

An example follows.  As above, start up the H2O cluster and load the training and test data.

```r
library(h2oEnsemble)
h2o.init(nthreads = -1) # Start H2O cluster using all available CPU threads


# Import a sample binary outcome train/test set into R
train <- h2o.importFile("https://s3.amazonaws.com/erin-data/higgs/higgs_train_5k.csv")
test <- h2o.importFile("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")
y <- "response"
x <- setdiff(names(train), y)
family <- "binomial"

#For binary classification, response should be a factor
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])
```


Cross-validate and train a handful of base learners and then use the `h2o.stack()` function to create the ensemble:

```r
# The h2o.stack function is an alternative to the h2o.ensemble function, which
# allows the user to specify H2O models individually and then stack them together
# at a later time.  Saved models, re-loaded from disk, can also be stacked.

# The base models must use identical cv folds; this can be achieved in two ways:
# 1. they be specified explicitly by using the fold_column argument, or
# 2. use same value for `nfolds` and set `fold_assignment = "Modulo"`

nfolds <- 5  

glm1 <- h2o.glm(x = x, y = y, family = family, 
                training_frame = train,
                nfolds = nfolds,
                fold_assignment = "Modulo",
                keep_cross_validation_predictions = TRUE)

gbm1 <- h2o.gbm(x = x, y = y, distribution = "bernoulli",
                training_frame = train,
                seed = 1,
                nfolds = nfolds,
                fold_assignment = "Modulo",
                keep_cross_validation_predictions = TRUE)

rf1 <- h2o.randomForest(x = x, y = y, # distribution not used for RF
                        training_frame = train,
                        seed = 1,
                        nfolds = nfolds,
                        fold_assignment = "Modulo",
                        keep_cross_validation_predictions = TRUE)

dl1 <- h2o.deeplearning(x = x, y = y, distribution = "bernoulli",
                        training_frame = train,
                        nfolds = nfolds,
                        fold_assignment = "Modulo",
                        keep_cross_validation_predictions = TRUE)

models <- list(glm1, gbm1, rf1, dl1)
metalearner <- "h2o.glm.wrapper"

stack <- h2o.stack(models = models,
                   response_frame = train[,y],
                   metalearner = metalearner, 
                   seed = 1,
                   keep_levelone_data = TRUE)


# Compute test set performance:
perf <- h2o.ensemble_performance(stack, newdata = test)
```

Print base learner and ensemble test set performance:

```r
> print(perf)

Base learner performance, sorted by specified metric:
                                   learner       AUC
1          GLM_model_R_1480128759162_16643 0.6822933
4 DeepLearning_model_R_1480128759162_18909 0.7016809
3          DRF_model_R_1480128759162_17790 0.7546005
2          GBM_model_R_1480128759162_16661 0.7780807


H2O Ensemble Performance on <newdata>:
----------------
Family: binomial

Ensemble performance (AUC): 0.781241759877087
```



## Roadmap for H2O Ensemble

You can follow the progress of H2O Ensemble development on the [H2O JIRA](https://0xdata.atlassian.net/issues/?filter=14900) (tickets with the "h2oEnsemble" tag). 

**Update:** Ensembles have been implemented in the H2O Java core, and exposed via the R and Python APIs for H2O.  
