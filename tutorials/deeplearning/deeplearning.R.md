# Classification and Regression with H2O Deep Learning

>**Note**: This tutorial is still in the process of being updated; as such, it may not be fully functional as of yet. 

###### This tutorial shows how a [Deep Learning](http://en.wikipedia.org/wiki/Deep_learning) model can be used to do supervised classification and regression. This file is both valid R and markdown code.

###R Documentation
###### The `h2o.deeplearning` function fits H2O's Deep Learning models from within R.

    library(h2o)
    args(h2o.deeplearning)

###### The R documentation (man page) for H2O's Deep Learning can be opened from within R using the `help` or `?` functions:
  
    help(h2o.deeplearning)

######As you can see, there are a lot of parameters!  Luckily, as you'll see later, you only need to know a few to get the most out of Deep Learning. More information can be found in the [H2O Deep Learning booklet](https://t.co/kWzyFMGJ2S) and in our [slides](http://www.slideshare.net/0xdata/presentations).   

###### We can run the example from the man page using the `example` function:

    example(h2o.deeplearning)

###### And run a longer demonstration from the `h2o` package using the `demo` function (requires an internet connection):

    demo(h2o.deeplearning)

### Start H2O and load the MNIST data
###### For the rest of this tutorial, we will use the well-known [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of hand-written digits, where each row contains the 28^2=784 raw gray-scale pixel values from 0 to 255 of the digitized digits (0 to 9). 

######Initialize the H2O server and import the MNIST training/testing datasets.

    library(h2o)
    h2oServer <- h2o.init(nthreads=-1)
    homedir <- "/data/h2o-training/mnist/"
    TRAIN = "train.csv.gz"
    TEST = "test.csv.gz"
    train_hex <- h2o.importFile(h2oServer, path = paste0(homedir,TRAIN), header = F, sep = ',', key = 'train.hex')
    test_hex <- h2o.importFile(h2oServer, path = paste0(homedir,TEST), header = F, sep = ',', key = 'test.hex')


######While H2O Deep Learning has many parameters, it was designed to be just as easy to use as the other supervised training methods in H2O. Automatic data standardization and handling of categorical variables and missing values and per-neuron adaptive learning rates reduce the amount of parameters the user has to specify. Often, it's just the number and sizes of hidden layers, the number of epochs and the activation function and maybe some regularization techniques.

    dlmodel <- h2o.deeplearning(x=1:784, y=785, training_frame=train_hex, validation_frame=test_hex,
                                hidden=c(50,50), epochs=0.1, activation="Tanh")
    
######Let's look at the model summary, and the confusion matrix and classification error (on the validation set, since it was provided) in particular:

    dlmodel
    dlmodel@model$confusion
    dlmodel@model$valid_class_error

######To confirm that the reported confusion matrix on the validation set (here, the test set) was correct, we make a prediction on the test set and compare the confusion matrices explicitly:

    pred_labels <- h2o.predict(dlmodel, test_hex)[,1]
    actual_labels <- test_hex[,785]
    cm <- h2o.confusionMatrix(pred_labels, actual_labels)
    cm
    dlmodel@model$confusion
    dlmodel@model$confusion == cm

######To see the model parameters again:
    
    dlmodel@model$params 

    
### Hyper-parameter Tuning with Grid Search
###### Since there are a lot of parameters that can impact model accuracy, hyper-parameter tuning is especially important for Deep Learning:

    grid_search <- h2o.deeplearning(x=c(1:784), y=785, data=train_hex, validation=test_hex, 
                                    hidden=list(c(10,10),c(20,20)), epochs=0.1,
                                    activation=c("Tanh", "Rectifier"), l1=c(0,1e-5))
                                
######Let's see which model had the lowest validation error (grid search automatically sorts the models by validation error):

    grid_search
    
    best_model <- grid_search@model[[1]]
    best_model
    best_params <- best_model@model$params
    best_params$activation
    best_params$hidden
    best_params$l1
    
### Hyper-parameter Tuning with Random Search
###### Multi-dimensional hyper-parameter optimization (more than 4 parameters) can be more efficient with random parameter search than with a Cartesian product of pre-given values (grid search). For a random parameter search, we do a loop over models with parameters drawn uniformly from a given range, thus sampling the high-dimensional space uniformly. This code is for demonstration purposes only:

    models <- c()
    for (i in 1:10) {
      rand_activation <- c("TanhWithDropout", "RectifierWithDropout")[sample(1:2,1)]
      rand_numlayers <- sample(2:5,1)
      rand_hidden <- c(sample(10:50,rand_numlayers,T))
      rand_l1 <- runif(1, 0, 1e-3)
      rand_l2 <- runif(1, 0, 1e-3)
      rand_dropout <- c(runif(rand_numlayers, 0, 0.6))
      rand_input_dropout <- runif(1, 0, 0.5)
      dlmodel <- h2o.deeplearning(x=1:784, y=785, data=train_hex, validation=test_hex, epochs=0.1,
                                  activation=rand_activation, hidden=rand_hidden, l1=rand_l1, l2=rand_l2,
                                  input_dropout_ratio=rand_input_dropout, hidden_dropout_ratios=rand_dropout)                                
      models <- c(models, dlmodel)
    }

######We can then find the model with the lowest validation error:
    best_err <- best_model@model$valid_class_error #best model from grid search above
    for (i in 1:length(models)) {
      err <- models[[i]]@model$valid_class_error
      if (err < best_err) {
        best_err <- err
        best_model <- models[[i]]
      }
    }
    best_model
    best_params <- best_model@model$params
    best_params$activation
    best_params$hidden
    best_params$l1
    best_params$l2
    best_params$input_dropout_ratio
    best_params$hidden_dropout_ratios
    
###Checkpointing
######Let's continue training the best model, for 2 more epochs. Note that since many parameters such as `epochs, l1, l2, max_w2, score_interval, train_samples_per_iteration, score_duty_cycle, classification_stop, regression_stop, variable_importances, force_load_balance` can be modified between checkpoint restarts, it is best to specify as many parameters as possible explicitly.

    dlmodel_continued <- h2o.deeplearning(x=c(1:784), y=785, data=train_hex, validation=test_hex,
                                checkpoint = best_model, l1=best_params$l1, l2=best_params$l2, epochs=0.5)

    dlmodel_continued@model$valid_class_error

######Once we are satisfied with the results, we can save the model to disk:

    h2o.saveModel(dlmodel_continued, dir="/tmp", name="mybest_mnist_model", force=T)

######It can be loaded later with
    
    dlmodel_loaded <- h2o.loadModel(h2oServer, "/tmp/mybest_mnist_model")
    
######Of course, you can continue training this model as well (with the same `x`, `y`, `data`, `validation`)

    dlmodel_continued_again <- h2o.deeplearning(x=c(1:784), y=785, data=train_hex, validation=test_hex,
                                checkpoint = dlmodel_loaded, l1=best_params$l1, epochs=0.5)
    
    dlmodel_continued_again@model$valid_class_error

###World-record results on MNIST
######With the parameters shown below, [H2O Deep Learning matched the current world record](https://twitter.com/ArnoCandel/status/533870196818079744) of [0.83% test set error](http://research.microsoft.com/pubs/204699/MNIST-SPM2012.pdf) for models without pre-processing, unsupervised learning, convolutional layers or data augmentation after running for 8 hours on 10 nodes:

    #   > record_model <- h2o.deeplearning(x = 1:784, y = 785, data = train_hex, validation = test_hex,
    #                                      activation = "RectifierWithDropout", hidden = c(1024,1024,2048),
    #                                      epochs = 2000, l1 = 1e-5, input_dropout_ratio = 0.2,
    #                                      train_samples_per_iteration = -1, classification_stop = -1)
    #   > record_model@model$confusion
    #              Predicted
    #     Actual     0    1    2    3   4   5   6    7   8    9 Error
    #       0      974    1    1    0   0   0   2    1   1    0 0.00612
    #       1        0 1135    0    1   0   0   0    0   0    0 0.00088
    #       2        0    0 1028    0   1   0   0    3   0    0 0.00388
    #       3        0    0    1 1003   0   0   0    3   2    1 0.00693
    #       4        0    0    1    0 971   0   4    0   0    6 0.01120
    #       5        2    0    0    5   0 882   1    1   1    0 0.01121
    #       6        2    3    0    1   1   2 949    0   0    0 0.00939
    #       7        1    2    6    0   0   0   0 1019   0    0 0.00875
    #       8        1    0    1    3   0   4   0    2 960    3 0.01437
    #       9        1    2    0    0   4   3   0    2   0  997 0.01189
    #       Totals 981 1142 1038 1013 977 891 956 1031 964 1007 0.00830
    
######Note: results are not 100% reproducible and also depend on the number of nodes, cores, due to thread race conditions and model averaging effects, which can by themselves be useful to avoid overfitting. Often, it can help to run the initial convergence with more `train_samples_per_iteration` (automatic values of -2 or -1 are good choices), and then continue from a checkpoint with a smaller number (automatic value of 0 or a number less than the total number of training rows are good choices here).

###Regression
######If the response column is numeric and non-integer, regression is enabled by default. For integer response columns, as in this case, you have to specify `classification=FALSE` to force regression. In that case, there will be only 1 output neuron, and the loss function and error metric will automatically switch to the MSE (mean square error).

    regression_model <- h2o.deeplearning(x=1:784, y=785, data=train_hex, validation=test_hex,
                                          hidden=c(50,50), epochs=0.1, activation="Rectifier",
                                          classification=FALSE)
                                
######Let's look at the model summary (i.e., the training and validation set MSE values):

    regression_model
    regression_model@model$train_sqr_error
    regression_model@model$valid_sqr_error
    
######We can confirm these numbers explicitly:
  
    mean((h2o.predict(regression_model, train_hex)-train_hex[,785])^2)
    mean((h2o.predict(regression_model, test_hex)-test_hex[,785])^2)
    h2o.mse(h2o.predict(regression_model, test_hex), test_hex[,785])

###### The difference in the training MSEs is because only a subset of the training set was used for scoring during model building (10k rows), see section "Scoring on Training/Validation Sets During Training" below.

###Cross-Validation
###### For N-fold cross-validation, specify nfolds instead of a validation frame, and N+1 models will be built: 1 model on the full training data, and N models with each one successive 1/N-th of the data held out. Those N models then score on the held out data, and the stitched-together predictions are scored to get the cross-validation error.
    
    dlmodel <- h2o.deeplearning(x=1:784, y=785, data=train_hex,
                                hidden=c(50,50), epochs=0.1, activation="Tanh",
                                nfolds=5)
    dlmodel
    dlmodel@model$valid_class_error
    
###### Note: There is a [RUnit test](https://github.com/0xdata/h2o/blob/master/R/tests/testdir_misc/runit_nfold.R) to demonstrate the correctness of N-fold cross-validation results reported by the model.

###### The N individual cross-validation models can be accessed as well:

    dlmodel@xval
    sapply(dlmodel@xval, function(x) (x@model$valid_class_error))

#####Cross-Validation is especially useful for hyperparameter optimizations such as grid searches.

    dlmodel <- h2o.deeplearning(x=1:784, y=785, data=train_hex,
                                hidden=c(10,10), epochs=0.1, activation=c("Tanh","Rectifier"),
                                nfolds=2)
    dlmodel

### Variable Importances
######Variable importances for Neural Network models are notoriously difficult to compute, and there are many [pitfalls](ftp://ftp.sas.com/pub/neural/importance.html). H2O Deep Learning has implemented the method of [Gedeon](http://cs.anu.edu.au/~./Tom.Gedeon/pdfs/ContribDataMinv2.pdf), and returns relative variable importances in descending order of importance.

    dlmodel <- h2o.deeplearning(x=1:784, y=785, data=train_hex,
                                hidden=c(10,10), epochs=0.5, activation="Tanh",
                                variable_importances=TRUE)
    dlmodel@model$varimp[1:10]

### Adaptive Learning Rate
#####By default, H2O Deep Learning uses an adaptive learning rate ([ADADELTA](http://arxiv.org/pdf/1212.5701v1.pdf)) for its stochastic gradient descent optimization. There are only two tuning parameters for this method: `rho` and `epsilon`, which balance the global and local search efficiencies. `rho` is the similarity to prior weight updates (similar to momentum), and `epsilon` is a parameter that prevents the optimization to get stuck in local optima. Defaults are `rho=0.99` and `epsilon=1e-8`. For cases where convergence speed is very important, it might make sense to perform a few grid searches to optimize these two parameters (e.g., with `rho=c(0.9,0.95,0.99,0.999)` and `epsilon=c(1e-10,1e-8,1e-6,1e-4)`). Of course, as lways with grid searches, caution has to be applied when extrapolating grid search results to a different parameter regime (e.g., for more epochs or different layer topologies or activation functions, etc.).

    dlmodel <- h2o.deeplearning(x=seq(1,784,10), y=785, data=train_hex, hidden=c(10,10), epochs=0.5,
                                rho=c(0.9,0.95,0.99,0.999), epsilon=c(1e-10,1e-8,1e-6,1e-4))
    dlmodel
    


#####If `adaptive_rate` is disabled, several manual learning rate parameters become important: `rate`, `rate_annealing`, `rate_decay`, `momentum_start`, `momentum_ramp`, `momentum_stable` and `nesterov_accelerated_gradient`, the discussion of which we leave to [H2O Deep Learning booklet](https://t.co/kWzyFMGJ2S).

###H2O Deep Learning Tips & Tricks

#### Activation Functions
######While sigmoids have been used historically for neural networks, H2O Deep Learning implements `Tanh`, a scaled and shifted variant of the sigmoid which is symmetric around 0. Since its output values are bounded by -1..1, the stability of the neural network is rarely endangered. However, the derivative of the tanh function is always non-zero and back-propagation (training) of the weights is more computationally expensive than for rectified linear units, or `Rectifier`, which is `max(0,x)` and has vanishing gradient for `x<=0`, leading to much faster training speed for large networks and is often the fastest path to accuracy on larger problems. In case you encounter instabilities with the `Rectifier` (in which case model building is automatically aborted), try one of these values to re-scale the weights: `max_w2=c(1,10,100)`. The `Maxout` activation function is least computationally effective, and is not well field-tested at this point.

#### Generalization Techniques
###### L1 and L2 penalties can be applied by specifying the `l1` and `l2` parameters. Intuition: L1 lets only strong weights survive (constant pulling force towards zero), while L2 prevents any single weight from getting too big. [Dropout](http://arxiv.org/pdf/1207.0580.pdf) has recently been introduced as a powerful generalization technique, and is available as a parameter per layer, including the input layer. `input_dropout_ratio` controls the amount of input layer neurons that are randomly dropped (set to zero), while `hidden_dropout_ratios` are specified for each hidden layer. The former controls overfitting with respect to the input data (useful for high-dimensional noisy data such as MNIST), while the latter controls overfitting of the learned features. Note that `hidden_dropout_ratios` require the activation function to end with `...WithDropout`.


#### Early stopping and optimizing for lowest validation error
######By default, `override_with_best_model` is set to TRUE and the model returned after training for the specified number of epochs is the model that has the best training set error, or, if a validation set is provided, the lowest validation set error. This is equivalent to early stopping, except that the determination to stop is made in hindsight, similar to a full grid search over the number of epochs at the granularity of the scoring intervals. Note that for N-fold cross-validation, `override_with_best_model` is disabled to give fair results (all N cross-validation models must run to completion to avoid overfitting on the validation set). Also note that the training or validation set errors can be based on a subset of the training or validation data, depending on the values for `score_validation_samples` or `score_training_samples`, see below. For actual early stopping on a predefined error rate on the *training data* (accuracy for classification or MSE for regression), specify `classification_stop` or `regression_stop`.

#### Training Samples per (MapReduce) Iteration
######This parameter is explained in Section 2.2.4 of the [H2O Deep Learning booklet](https://t.co/kWzyFMGJ2S), and becomes important in multi-node operation.

####Categorical Data
######For categorical data, a feature with K factor levels is automatically one-hot encoded (horizontalized) into K-1 input neurons. Hence, the input neuron layer can grow substantially for datasets with high factor counts. In these cases, it might make sense to reduce the number of hidden neurons in the first hidden layer, such that large numbers of factor levels can be handled. In the limit of 1 neuron in the first hidden layer, the resulting model is similar to logistic regression with stochastic gradient descent, except that for classification problems, there's still a softmax output layer, and that the activation function is not necessarily a sigmoid (`Tanh`). If variable importances are computed, it is recommended to turn on `use_all_factor_levels` (K input neurons for K levels). The experimental option `max_categorical_features` uses feature hashing to reduce the number of input neurons via the hash trick at the expense of hash collisions and reduced accuracy.

####Missing Values
######H2O Deep Learning automatically does mean imputation for missing values during training (leaving the input layer activation at 0 after standardizing the values). For testing, missing test set values are also treated the same way by default. See the `h2o.impute` function to do your own mean imputation.

####Reproducibility
######Every run of DeepLearning results in different results since multithreading is done via [Hogwild!](http://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf) that benefits from intentional lock-free race conditions between threads. To get reproducible results at the expense of speed for small datasets, set reproducible=T and specify a seed. This will not work for big data for technical reasons, and is probably also not desired because of the significant slowdown.

    
####Scoring on Training/Validation Sets During Training  
###### The training and/or validation set errors *can* be based on a subset of the training or validation data, depending on the values for `score_validation_samples` (defaults to 0: all) or `score_training_samples` (defaults to 10,000 rows, since the training error is only used for early stopping and monitoring). For large datasets, Deep Learning can automatically sample the validation set to avoid spending too much time in scoring during training, especially since scoring results are not currently displayed in the model returned to R. For example:

    dlmodel <- h2o.deeplearning(x=1:784, y=785, data=train_hex, validation=test_hex,
                                hidden=c(50,50), epochs=0.1, activation="Tanh",
                                score_training_samples=1000, score_validation_samples=1000)
                                
######Note that the default value of `score_duty_cycle=0.1` limits the amount of time spent in scoring to 10%, so a large number of scoring samples won't slow down overall training progress too much, but it will always score once after the first MapReduce iteration, and once at the end of training.

######Stratified sampling of the validation dataset can help with scoring on datasets with class imbalance.  Note that this option also requires `balance_classes` to be enabled (used to over/under-sample the training dataset, based on the max. relative size of the resulting training dataset, `max_after_balance_size`):
    
    dlmodel <- h2o.deeplearning(x=1:784, y=785, data=train_hex, validation=test_hex,
                                hidden=c(50,50), epochs=0.1, activation="Tanh",
                                score_training_samples=1000, score_validation_samples=1000,
                                balance_classes=TRUE,
                                score_validation_sampling="Stratified")
    
####Benchmark against nnet
######We compare H2O Deep Learning against the nnet package for a regression problem (MNIST digit 0-9). For performance reasons, we sample the predictors and use a small single hidden layer.
    
    if (! "nnet" %in% rownames(installed.packages())) { install.packages("nnet") }
    library("nnet")
    train.R <- as.data.frame(train_hex)
    train.R <- train.R[,c(seq(1, 784, by = 20), 785)]
    test.R <- as.data.frame(test_hex)
    
    nn <- nnet(x=train.R[,-ncol(train.R)], y=train.R$C785, size=10, linout=T)
    nn_pred <- predict(nn, test.R)
    nn_mse <- mean((nn_pred-test.R[,785])^2)
    nn_mse
    
    h2o_nn <- h2o.deeplearning(x=seq(1,784,by=20),y=785,data=train_hex,hidden=10,classification=F,activation="Tanh")
    h2o_pred <- h2o.predict(h2o_nn, test_hex)
    h2o_mse <- mean((as.data.frame(h2o_pred)-test.R[,785])^2)
    h2o_mse


### More information can be found in the [H2O Deep Learning booklet](https://t.co/kWzyFMGJ2S) and in our [slides](http://www.slideshare.net/0xdata/presentations).
