library(h2o)
args(h2o.deeplearning)
help(h2o.deeplearning)
example(h2o.deeplearning)
#demo(h2o.deeplearning)
h2o.init(nthreads=-1, max_mem_size = "2G")
spiral.hex <- h2o.importFile("/users/arno/h2o-world-2015-training/tutorials/data/spiral.csv")
spiral <- as.data.frame(spiral.hex)
grid.hex <- h2o.importFile("/users/arno/h2o-world-2015-training/tutorials/data/grid.csv")

# Helper to plot the training data and the probability contours obtained from the grid frame
plotContour <- function(name, model, data, grid.hex) {
  n=0.5*(sqrt(nrow(grid.hex))-1); d <- 1.5; h <- d*(-n:n)/n
  pred <- as.data.frame(h2o.predict(model, grid.hex))  ##make predictions on a square grid
  plot(data[,-3],pch=19,col=data[,3],cex=1.0,xlim=c(-d,d),ylim=c(-d,d),main=name) ##plot data
  contour(h,h,z=array(ifelse(pred[,1]=="Red",0,1),dim=c(2*n+1,2*n+1)),col="blue",lwd=2,add=T)
}
# set up the canvas for 2x2 plots
par(mfrow=c(2,2))
plotContour("DL",  h2o.deeplearning(x=1:2,y=3,training_frame=spiral.hex,epochs=1000), spiral, grid.hex)
plotContour("GBM", h2o.gbm(x=1:2,y=3,training_frame=spiral.hex), spiral, grid.hex)
plotContour("DRF", h2o.randomForest(x=1:2,y=3,training_frame=spiral.hex), spiral, grid.hex)
plotContour("GLM", h2o.glm(x=1:2,y=3,training_frame=spiral.hex, family="binomial"), spiral, grid.hex)
df <- h2o.importFile("/users/arno/h2o-world-2015-training/tutorials/data/covtype.full.csv")
dim(df)
df
splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
train <- h2o.assign(splits[[1]], "train.hex")
valid <- h2o.assign(splits[[2]], "valid.hex")
test <- h2o.assign(splits[[3]], "test.hex")
plot(h2o.tabulate(df, "Elevation",                       "Cover_Type"))
plot(h2o.tabulate(df, "Horizontal_Distance_To_Roadways", "Cover_Type"))
plot(h2o.tabulate(df, "Soil_Type",                       "Cover_Type"))
plot(h2o.tabulate(df, "Horizontal_Distance_To_Roadways", "Elevation"))
plot(h2o.tabulate(df, "Horizontal_Distance_To_Roadways", "Elevation"))
response <- "Cover_Type"
predictors <- setdiff(names(df), response)
predictors
m <- h2o.deeplearning(
  model_id="dl_model_first", 
  training_frame=train, 
  validation_frame=valid,   ## validation dataset: used for scoring and early stopping
  x=predictors,
  y=response,
  #activation="Rectifier",  ## default
  #hidden=c(200,200),       ## default: 2 hidden layers with 200 neurons each
  epochs=1,
  variable_importances=T    ## not enabled by default
)
summary(m)
vi <- as.data.frame(h2o.varimp(m))
hist(vi$relative_importance)
m <- h2o.deeplearning(
  model_id="dl_model_faster", 
  training_frame=train, 
  validation_frame=valid,
  x=predictors,
  y=response,
  hidden=c(32,32,32),               ## small network, runs faster
  epochs=1000000,
  score_validation_samples = 10000, ## sample the valiation dataset, faster and accurate enough
  stopping_rounds=1,
  stopping_metric="misclassification",  ## could be "MSE","logloss","r2"
  stopping_tolerance=0.01
)
summary(m)
plot(m)

## with some tuning: ~8% in 40s
m <- h2o.deeplearning(
  model_id="dl_model_tuned", 
  training_frame = train, 
  validation_frame = valid, 
  x=predictors, 
  y=response, 
  overwrite_with_best_model=F,
  hidden=c(100,100,100),          ## more hidden layers -> more complex interactions
  epochs=10,                     ## long enough to converge
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  rate=0.02, 
  rate_annealing=2e-6,            
  momentum_start = 0.2,           ## manually tuned momentum
  momentum_stable = 0.4, 
  momentum_ramp = 1e7, 
  l1=1e-5,                        ## add some L1/L2 regularization
  l2=1e-5,
  max_w2 = 10                     ## helps stability for Rectifier
) 
summary(m)
h2o.confusionMatrix(h2o.performance(m, train=T)) ## training data
h2o.confusionMatrix(h2o.performance(m, valid=T)) ## sampled validation data
h2o.confusionMatrix(m, valid)                    ## validation datas
h2o.confusionMatrix(m, test)                     ## test data
p <- h2o.predict(m, test)
p
test$Accuracy <- p$predict == test$Cover_Type
1-mean(test$Accuracy)
dlmodel@model$params 
  sampled_train = train[1:10000,]
hyper_params <- list(
  hidden = list(c(32,32,32),c(64,64)),
  input_dropout_ratio = c(0,0.05),
  hidden_dropout_ratios = list(c(0,0,0),c(0.05,0.05)), ## only one will match a given 'hidden' parameter
  rate=c(0.01,0.02),
  rate_annealing=c(1e-8,1e-7,1e-6)
)
hyper_params
grid <- h2o.grid(
  "deeplearning",
  model_id="dl_grid", 
  training_frame = sampled_train,
  validation_frame = valid, 
  x=predictors, 
  y=response,
  epochs=10,
  stopping_metric="misclassification",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  momentum_start = 0.5,           ## manually tuned momentum
  momentum_stable = 0.9, 
  momentum_ramp = 1e7, 
  l1 = 1e-5,
  l2 = 1e-5,
  activation = c("RectifierWithDropout"),
  max_w2 = 10,                    ## can help improve stability for Rectifier
  hyper_params = hyper_params
)
## Find the best model and its full set of parameters (clunky for now, will be improved)
scores <- cbind(as.data.frame(unlist((lapply(grid@model_ids, function(x) 
  { h2o.confusionMatrix(h2o.performance(h2o.getModel(x),valid=T))$Error[8] })) )), unlist(grid@model_ids))
best_err <- scores[1,1]
best_model <- h2o.getModel(as.character(scores[1,2]))
print(paste0("Best misclassification: ", best_err))
print(best_model@allparameters)
models <- c()
for (i in 1:10) {
  rand_activation <- c("TanhWithDropout", "RectifierWithDropout")[sample(1:2,1)]
  rand_numlayers <- sample(2:5,1)
  rand_hidden <- c(sample(10:50,rand_numlayers,T))
  rand_l1 <- runif(1, 0, 1e-3)
  rand_l2 <- runif(1, 0, 1e-3)
  rand_dropout <- c(runif(rand_numlayers, 0, 0.6))
  rand_input_dropout <- runif(1, 0, 0.5)
  dlmodel <- h2o.deeplearning(
    model_id=paste0("dl_random_model_", i),
    training_frame = sampled_train,
    validation_frame = valid, 
    x=predictors, 
    y=response,
    epochs=10,
    stopping_metric="misclassification",
    stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
    stopping_rounds=2,
    score_validation_samples=10000, ## downsample validation set for faster scoring
    score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
    max_w2 = 10,                    ## can help improve stability for Rectifier

    ### Random parameters
    activation=rand_activation, 
    hidden=rand_hidden, 
    l1=rand_l1, 
    l2=rand_l2,
    input_dropout_ratio=rand_input_dropout, 
    hidden_dropout_ratios=rand_dropout
  )                                
  models <- c(models, dlmodel)
}
if (is.null(best_err)) best_err <- 1      ##use the best reference model from the grid search above
for (i in 1:length(models)) {
  err <- h2o.confusionMatrix(h2o.performance(models[[i]],valid=T))$Error[8]
  if (err < best_err) {
    best_err <- err
    best_model <- models[[i]]
  }
}
best_model
best_params <- best_model@parameters
best_params$activation
best_params$hidden
best_params$l1
best_params$l2
best_params$input_dropout_ratio
best_params$hidden_dropout_ratios
#max_epochs <- 1000 ##Takes a few minutes
max_epochs <- 20    ##Takes about 30s
m <- h2o.deeplearning(
  model_id="dl_model_tuned_continued", 
  checkpoint="dl_model_tuned", 
  training_frame = train, 
  validation_frame = valid, 
  x=predictors, 
  y=response, 
  hidden=c(100,100,100),          ## more hidden layers -> more complex interactions
  epochs=max_epochs,              ## hopefully long enough to converge (otherwise restart again)
  stopping_metric="logloss",
  stopping_tolerance=1e-2,        ## stop when validation logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  rate=0.02, 
  rate_annealing=2e-6,            
  momentum_start = 0.2,           ## manually tuned momentum
  momentum_stable = 0.4, 
  momentum_ramp = 1e7, 
  l1=1e-5,                        ## add some L1/L2 regularization
  l2=1e-5,
  max_w2 = 10                     ## helps stability for Rectifier
) 
summary(m)
plot(m)
h2o.saveModel(dlmodel_continued, dir="/tmp", name="mybest_covtype_model", force=T)
dlmodel_loaded <- h2o.loadModel(h2oServer, "/tmp/mybest_covtype_model")
dlmodel_continued_again <- h2o.deeplearning(x=c(1:784), y=785, data=train_hex, validation=test_hex,
                            checkpoint = dlmodel_loaded, l1=best_params$l1, epochs=0.5)

dlmodel_continued_again@model$valid_class_error
dlmodel <- h2o.deeplearning(
  x=predictors,
  y=response, 
  training_frame=train,
  hidden=c(10,10),
  epochs=0.1,
  nfolds=5)
summary(dlmodel)
    dlmodel <- h2o.deeplearning(x=1:784, y=785, data=train_hex, validation=test_hex,
                                hidden=c(50,50), epochs=0.1, activation="Tanh",
                                score_training_samples=1000, score_validation_samples=1000,
                                balance_classes=TRUE,
                                score_validation_sampling="Stratified")
