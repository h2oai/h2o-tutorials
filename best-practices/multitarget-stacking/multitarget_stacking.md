Multi-Target Stacking
---------------------

Multi-Target stacking is a process used to predict multiple columns.
With typical machine learning approaches, a single target column is
selected. In this example, we will try to predict the columns `bad_loan`
and `int_rate` using our cleaned lending club data:
<https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv>.

The goal of this use case is to determine who will pay back their loan
payments and for those who will pay back their loan payments - what
should be the interest rate. This would typically be done by training
two models:

-   predict the target column: `bad_loan`
-   predict the target column: `int_rate`

We can assume that `bad_loan` and `int_rate` are highly correlated and
the information about `int_rate` may help us predict `bad_loan` and vice
versa. Rather than split the problem up into two models, we propose a
multi-target stacking approach.

Multi-Target Stacking in H2O-3
------------------------------

Multi-Target stacking is performed in three steps:

1.  Train base models on each target column
2.  Extract cross validation predictions to create final model data
3.  Train final models for each target using the cross validation
    predictions as features

### Train Base Models

We will train a base model for each target encoding using cross
validation.

``` r
library('h2o')
h2o.init()
h2o.no_progress()

df <- h2o.importFile("https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv")
df$bad_loan <- as.factor(df$bad_loan)
```

We will randomly split the data into 75% training and 25% testing. We
will use the testing data to evaluate how well the model performs.

``` r
# Split Frame into training and testing
splits <- h2o.splitFrame(df, seed = 1234, destination_frames=c("train.hex", "test.hex"), ratios = 0.75)
train <- splits[[1]]
test <- splits[[2]]
```

In this next step, we train a model for each target column. We will set
`keep_cross_validation_predictions = TRUE` so that we can use the
predictions later for our final models.

``` r
predictors <- c("loan_amnt", "emp_length", "annual_inc", "dti", "delinq_2yrs", "revol_util", "total_acc", 
                "longest_credit_length", "verification_status", "term", "purpose", "home_ownership")

# Train base model to predict bad_loan
gbm_bad_loan_base <- h2o.gbm(x = predictors, y = "bad_loan", 
                             training_frame = train, nfolds = 5, # 5-fold cross validation
                             keep_cross_validation_predictions = TRUE,
                             score_each_iteration = TRUE, ntrees = 500,
                             seed = 1234,
                             stopping_rounds = 5, stopping_metric = "AUC", stopping_tolerance = 0.001,
                             model_id = "gbm-bad_loan-base.hex")

# Train base model to predict int_rate
gbm_int_rate_base <- h2o.gbm(x = predictors, y = "int_rate", 
                             training_frame = train, nfolds = 5, # 5-fold cross validation
                             keep_cross_validation_predictions = TRUE,
                             score_each_iteration = TRUE, ntrees = 500,
                             seed = 1234,
                             stopping_rounds = 5, stopping_metric = "MAE", stopping_tolerance = 0.001,
                             model_id = "gbm-int_rate-base.hex")
```

What would our model performance be if we knew `int_rate` at the time of
`bad_loan` and vice versa? If these additional features help our model
performance, this could indicate that Multi-Target Stacking can help.

``` r
# Train model to predict bad_loan using int_rate
gbm_bad_loan_all_x <- h2o.gbm(x = c(predictors, "int_rate"), y = "bad_loan", 
                              training_frame = train, nfolds = 5, # 5-fold cross validation
                              keep_cross_validation_predictions = TRUE,
                              score_each_iteration = TRUE, ntrees = 500,
                              seed = 1234,
                              stopping_rounds = 5, stopping_metric = "AUC", stopping_tolerance = 0.001,
                              model_id = "gbm-bad_loan-all_x.hex")

# Train model to predict int_rate using bad_loan
gbm_int_rate_all_x <- h2o.gbm(x = c(predictors, "bad_loan"), y = "int_rate", 
                              training_frame = train, nfolds = 5, # 5-fold cross validation
                              keep_cross_validation_predictions = TRUE,
                              score_each_iteration = TRUE, ntrees = 500,
                              seed = 1234,
                              stopping_rounds = 5, stopping_metric = "MAE", stopping_tolerance = 0.001,
                              model_id = "gbm-int_rate-all_x.hex")
```

The performance metrics on the testing data is shown below:

| Model          |  AUC: bad\_loan|  MAE: int\_rate|
|:---------------|---------------:|---------------:|
| baseline       |          0.6829|          2.6608|
| all predictors |          0.7052|          2.6318|

We have much better performance in predicting `bad_loan` when we know
the loan’s interest rate. Likewise, we have much better performance in
predicting `int_rate` when we know if the loan will be fully paid off.
This indicates that stacking our base model predictions may help improve
performance.

### Create Final Model Data

Now that we have our base models, we can add our cross validation hold
out predictions to our training data. We cannot simply predict using the
base models on the training data because then we were in danger of
overfitting. Any prediction we use should be on some hold-out data.

We will extend our dataset with the holdout predictions and use this
extended data to train our final models.

``` r
bad_loan_preds <- h2o.cross_validation_holdout_predictions(gbm_bad_loan_base)$p1
colnames(bad_loan_preds) <- c("bad_loan_pred")

int_rate_preds <- h2o.cross_validation_holdout_predictions(gbm_int_rate_base)$predict
colnames(int_rate_preds) <- c("int_rate_pred")

ext_train <- h2o.cbind(train, bad_loan_preds, int_rate_preds)
head(ext_train)
```

    ##   loan_amnt      term int_rate emp_length home_ownership annual_inc
    ## 1      5000 36 months    10.65         10           RENT      24000
    ## 2      2500 60 months    15.27          0           RENT      30000
    ## 3      2400 36 months    15.96         10           RENT      12252
    ## 4      5000 36 months     7.90          3           RENT      36000
    ## 5      3000 36 months    18.64          9           RENT      48000
    ## 6      5600 60 months    21.28          4            OWN      40000
    ##          purpose addr_state   dti delinq_2yrs revol_util total_acc
    ## 1    credit_card         AZ 27.65           0       83.7         9
    ## 2            car         GA  1.00           0        9.4         4
    ## 3 small_business         IL  8.72           0       98.5        10
    ## 4        wedding         AZ 11.20           0       28.3        12
    ## 5            car         CA  5.35           0       87.5         4
    ## 6 small_business         CA  5.55           0       32.6        13
    ##   bad_loan longest_credit_length verification_status bad_loan_pred
    ## 1        0                    26            verified     0.2827027
    ## 2        1                    12            verified     0.2334299
    ## 3        0                    10        not verified     0.2907301
    ## 4        0                     7            verified     0.1146051
    ## 5        0                     4            verified     0.1959878
    ## 6        1                     7            verified     0.3950101
    ##   int_rate_pred
    ## 1      13.98439
    ## 2      14.03238
    ## 3      16.04234
    ## 4      11.78119
    ## 5      16.23640
    ## 6      16.18289

We can add these same predictions to our test data. Note that for our
test dataset, we do not need to use the holdout cross validation
predictions. Since the test data was not seen during training, we can
simply predict using our base models on the test data to get our
additional columns.

``` r
bad_loan_preds <- h2o.predict(gbm_bad_loan_base, test)$p1
colnames(bad_loan_preds) <- c("bad_loan_pred")

int_rate_preds <- h2o.predict(gbm_int_rate_base, test)$predict
colnames(int_rate_preds) <- c("int_rate_pred")

ext_test <- h2o.cbind(test, bad_loan_preds, int_rate_preds)
```

### Train Final Models

Now that we have our extended training and testing data with predictions
for `bad_loan` and `int_rate`, we can train our final models.

Our final models are the same as our base models, however, the
`bad_loan` model has the additional feature: `int_rate_pred` and the
`int_rate` model has the additional feature: `bad_loan_pred`.

``` r
gbm_bad_loan_final <- h2o.gbm(x = c(predictors, "int_rate_pred"), y = "bad_loan", 
                              training_frame = ext_train, validation_frame = ext_test,
                              ntrees = 500, score_each_iteration = TRUE, 
                              stopping_rounds = 5, stopping_metric = "AUC", stopping_tolerance = 0.001,
                              model_id = "gbm-bad_loan-final.hex")

gbm_int_rate_final <- h2o.gbm(x = c(predictors, "bad_loan_pred"), y = "int_rate", 
                              training_frame = ext_train, validation_frame = ext_test,
                              ntrees = 500, score_each_iteration = TRUE, 
                              stopping_rounds = 5, stopping_metric = "MAE", stopping_tolerance = 0.001,
                              model_id = "gbm-int_rate-final.hex")
```

The performance metrics on the testing data is shown below:

| Model                  |  AUC: bad\_loan|  MAE: int\_rate|
|:-----------------------|---------------:|---------------:|
| baseline               |          0.6829|          2.6608|
| multi-target\_stacking |          0.6809|          2.6595|

We can see that the performance improves when we use the Multi-Target
Stacking method compared to simply training a model per target.

Putting It All Together
-----------------------

We will put the steps together into one function that trains
Multi-Target Stacking and one function that scores Multi-Target
Stacking.

``` r
# Train Multi-Target Stacking

TrainMultiTargetStacking <- function(training_frame, validation_frame, x, y, 
                                     nfolds = 5, seed = -1, score_tree_interval = 1){
  
  message("Train Base Models")
  
  base_models <- list()
  for(i in y){
    base_model <- h2o.gbm(x = x, y = i, 
                          training_frame = training_frame, 
                          nfolds = nfolds, seed = seed, keep_cross_validation_predictions = TRUE,
                          ntrees = 500, score_tree_interval = score_tree_interval, 
                          stopping_rounds = 5, # early stopping
                          model_id = paste0("base-", i, ".hex"))
    base_models <- c(base_models, list(base_model))
  }
  
  message("Create Final Model Data")
  
  final_training_frame <- training_frame
  final_validation_frame <- validation_frame
  final_x <- x
  
  for(i in base_models){
    
    pred_name <- paste0("pred_", i@parameters$y)
    final_x <- c(final_x, pred_name)
    
    # Add Cross Validation Holdout Predictions to Training
    train_preds <- h2o.cross_validation_holdout_predictions(i)
    train_preds <- train_preds[ ,ncol(train_preds)]
    colnames(train_preds) <- pred_name
    final_training_frame <- h2o.cbind(final_training_frame, train_preds)
    
    # Add Predictions to Validation
    valid_preds <- h2o.predict(i, validation_frame)
    valid_preds <- valid_preds[ ,ncol(valid_preds)]
    colnames(valid_preds) <- pred_name
    final_validation_frame <- h2o.cbind(final_validation_frame, valid_preds)
  }
  
  message("Train Final Models")
  
  final_models <- list()
  for(i in y){
    final_model <- h2o.gbm(x = final_x, y = i, 
                           training_frame = final_training_frame, validation_frame = final_validation_frame,
                           ntrees = 500, score_tree_interval = score_tree_interval, 
                           stopping_rounds = 5, # early stopping
                           model_id = paste0("final-", i, ".hex"))
    final_models <- c(final_models, list(final_model))
  }
  
  names(final_models) <- y
  
  return(list('final_models' = final_models, 
              'base_models' = base_models))
}

# Score with Multi-Target Stacking
ScoreMultiTargetStacking <- function(base_models, final_model, newdata){
  
  scoring_data <- newdata
  final_x <- base_models[[1]]@parameters$x
  
  for(i in base_models){
    
    pred_name <- paste0("pred_", i@parameters$y)
    final_x <- c(final_x, pred_name)
    
    # Add Predictions to Scoring Data
    preds <- h2o.predict(i, scoring_data)
    preds <- preds[ ,ncol(preds)]
    colnames(preds) <- pred_name
    scoring_data <- h2o.cbind(scoring_data, preds)
  }
  
  return(h2o.predict(final_model, scoring_data))
}
```

Building our final models using our new function:

``` r
# Train Multi-Target Stacking Models
stacking_models <- TrainMultiTargetStacking(train, test, x  = predictors, y = c("int_rate", "bad_loan"), seed = 1234)

# Predict with int_rate model
predictions <- ScoreMultiTargetStacking(stacking_models$base_models, stacking_models$final_models$int_rate, test)
head(h2o.cbind(test$int_rate, predictions))
```

    ##   int_rate  predict
    ## 1    13.49 13.22648
    ## 2    12.69 13.09169
    ## 3    15.27 14.00207
    ## 4     6.03 10.01169
    ## 5    12.42 14.50770
    ## 6    11.71 13.44240

References
----------

-   [A Survey on Multi-Output
    Regression](http://cig.fi.upm.es/articles/2015/Borchani-2015-WDMKD.pdf)
