#start local H2O environment with 2 gigs of memory
library(h2o)
localh2o<-h2o.init(nthreads=-1, max_mem_size= "2G")


spiral <- h2o.importFile("https://raw.githubusercontent.com/h2oai/h2o-world-2015-training/master/tutorials/data/spiral.csv")
grid   <- h2o.importFile("https://raw.githubusercontent.com/h2oai/h2o-world-2015-training/master/tutorials/data/grid.csv")
# Define helper to plot contours
plotC <- function(name, model, data=spiral, g=grid) {
  data <- as.data.frame(data) #pull data local
  pred <- as.data.frame(h2o.predict(model, g))
  n=0.5*(sqrt(nrow(g))-1); d <- 1.5; h <- d*(-n:n)/n
  plot(data[,-3],pch=19,col=data[,3],cex=1.0,
       xlim=c(-d,d),ylim=c(-d,d),main=name)
  contour(h,h,z=array(ifelse(pred[,1]=="Red",0,1),
                      dim=c(2*n+1,2*n+1)),col="blue",lwd=2,add=T)
}
par(mfrow=c(2,2)) # set up the canvas for 2x2 plots
#plotC( "DL", h2o.deeplearning(1:2,3,spiral,epochs=1e1))
#plotC( "DL", h2o.deeplearning(1:2,3,spiral,epochs=1e2))
plotC( "DL", h2o.deeplearning(1:2,3,spiral,epochs=1e3))
plotC("GBM", h2o.gbm         (1:2,3,spiral))
plotC("DRF", h2o.randomForest(1:2,3,spiral))
plotC("GLM", h2o.glm         (1:2,3,spiral,family="binomial"))
par(mfrow = c(1,1)) # reset canvas

df <- h2o.importFile("https://raw.githubusercontent.com/h2oai/h2o-world-2015-training/master/tutorials/data/covtype.full.csv")
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

response <- "Cover_Type"
predictors <- setdiff(names(df), response)
predictors

m1 <- h2o.deeplearning(
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
summary(m1)

head(as.data.frame(h2o.varimp(m1)))

m2 <- h2o.deeplearning(
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
summary(m2)
plot(m2)

m3 <- h2o.deeplearning(
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
summary(m3)

h2o.confusionMatrix(h2o.performance(m3, train=T)) ## sampled training data (from model building)
h2o.confusionMatrix(h2o.performance(m3, valid=T)) ## sampled validation data (from model building)
h2o.confusionMatrix(m3, train)                    ## full training data
h2o.confusionMatrix(m3, valid)                    ## full validation data
h2o.confusionMatrix(m3, test)                     ## full test data

pred <- h2o.predict(m3, test)
pred
test$Accuracy <- pred$predict == test$Cover_Type
1-mean(test$Accuracy)

sampled_train = train[1:10000,]

hyper_params <- list(
  hidden = list(c(32,32,32),c(64,64)),
  input_dropout_ratio = c(0,0.05),
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
  activation = c("Rectifier"),
  max_w2 = 10,                    ## can help improve stability for Rectifier
  hyper_params = hyper_params
)
grid

scores <- cbind(as.data.frame(unlist((lapply(grid@model_ids, function(x) 
{ h2o.confusionMatrix(h2o.performance(h2o.getModel(x),valid=T))$Error[8] })) )), unlist(grid@model_ids))
names(scores) <- c("misclassification","model")
head(scores)
best_model <- h2o.getModel(as.character(scores$model[1]))
print(best_model@allparameters)
best_err <- scores$misclassification[1]
print(best_err)

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
    #    epochs=100,                    ## for real parameters: set high enough to get to convergence
    epochs=1,
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

if (is.null(best_err)) best_err <- 1      ##start with the best reference model from the grid search above, if available
for (i in 1:length(models)) {
  err <- h2o.confusionMatrix(h2o.performance(models[[i]],valid=T))$Error[8]
  if (err < best_err) {
    best_err <- err
    best_model <- models[[i]]
  }
}
h2o.confusionMatrix(best_model,valid=T)
best_params <- best_model@allparameters
best_params$hidden
best_params$l1
best_params$l2
best_params$input_dropout_ratio

max_epochs <- 12 ## Add two more epochs
m_cont <- h2o.deeplearning(
  model_id="dl_model_tuned_continued", 
  checkpoint="dl_model_tuned", 
  training_frame = train, 
  validation_frame = valid, 
  x=predictors, 
  y=response, 
  hidden=c(100,100,100),          ## more hidden layers -> more complex interactions
  epochs=max_epochs,              ## hopefully long enough to converge (otherwise restart again)
  stopping_metric="logloss",      ## logloss is directly optimized by Deep Learning
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
summary(m_cont)
plot(m_cont)

path <- h2o.saveModel(m_cont, path="/mybest_model", force = T)

m_loaded <- h2o.loadModel(path)
summary(m_loaded)

dlmodel <- h2o.deeplearning(
  x=predictors,
  y=response, 
  training_frame=train,
  hidden=c(10,10),
  epochs=0.1,
  nfolds=5)
summary(dlmodel)

train$bin_response <- ifelse(train[,response]=="class_1", 0, 1) ## numerical 0/1

dlmodel <- h2o.deeplearning(
  x=predictors,
  y="bin_response", 
  training_frame=train,
  hidden=c(10,10),
  epochs=0.1
  #balance_classes=T    ## enable this for high class imbalance
)
summary(dlmodel)

train$bin_response <- as.factor(train$bin_response) ## Turn into categorical levels "0"/"1"
dlmodel <- h2o.deeplearning(
  x=predictors,
  y="bin_response", 
  training_frame=train,
  hidden=c(10,10),
  epochs=0.1
  #balance_classes=T    ## enable this for high class imbalance
)
summary(dlmodel) ## Now the model metrics contain AUC for binary classification
plot(h2o.performance(dlmodel)) ## display ROC curve