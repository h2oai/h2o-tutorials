library(h2o)

## Start up a 1-node H2O cluster on the local machine, allow up to 2GB of memory
h2o.init(nthreads=-1, max_mem_size = "2G")
#h2o.init(ip="mr-0xd1", port=53322) ## connect to running cluster

### Part 1: Introduction

### Show the nature of DL vs GBM vs DRF vs GLM with 2D probability contour plots
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



### Part 2: Cover Type
df <- h2o.importFile("/users/arno/h2o-world-2015-training/tutorials/data/covtype.full.csv")
#df <- h2o.importFile("/home/arno/h2o-world-2015-training/tutorials/data/covtype.full.csv")
df

## Split the data 3 ways: train/validation/test
splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
train <- h2o.assign(splits[[1]], "train.hex")
valid <- h2o.assign(splits[[2]], "valid.hex")
test <- h2o.assign(splits[[3]], "test.hex")
response <- "Cover_Type"
predictors <- setdiff(names(df), response)

## ~17% 17s
m <- h2o.deeplearning(model_id="dl_model_defaults", 
                      training_frame = train, 
                      validation_frame = valid, 
                      x=predictors, y=response, 
                      variable_importances=T, 
                      epochs=1)
m
summary(m)
h2o.varimp(m)

## smaller network, run until convergence
## (stop if misclassification on 10k validation rows does not improve by at least 1%)
## ~15% in 30s
m <- h2o.deeplearning(
  model_id="dl_model_faster", 
  training_frame = train, 
  validation_frame = valid,
  x=predictors,
  y=response,
  hidden=c(32,32,32),
  epochs=1000000,
  score_validation_samples = 10000,
  stopping_rounds=1,
  stopping_metric="misclassification",
  stopping_tolerance=0.01
)
summary(m)

## show convergence
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
  l2=1e-5,                        ## add some L2 regularization
  max_w2 = 10                     ## helps stability for Rectifier
) 

## Optional - continue training the previous model
if (FALSE) {
  max_epochs <- 1000 ##Takes a few minutes
} else {
  max_epochs <- 20   ##Takes about 30s
}
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
  l2=1e-5,                        ## add some L2 regularization
  max_w2 = 10                     ## helps stability for Rectifier
) 

## Now score on the full validation set and the test test
summary(m)
h2o.confusionMatrix(h2o.performance(m, train=T)) ## training
h2o.confusionMatrix(h2o.performance(m, valid=T)) ## sampled validation
h2o.confusionMatrix(m, valid) ## full validation
h2o.confusionMatrix(m, test)

## Manually compute test set error from predictions
p <- h2o.predict(m, test)
p
test$Accuracy <- p$predict == test$Cover_Type
1-mean(test$Accuracy)

plot(m)


## Grid search
if (FALSE) {
  hyper_params <- list(
    hidden = list(c(64,64,64),c(128,128,128),c(512,512)),
    l1 = c(0, 1e-5),
    l2 = c(0, 1e-5),
    input_dropout_ratio = c(0,0.05),
    rate=c(0.005,0.01,0.02),
    rate_annealing=c(1e-8,1e-7,1e-6),
    momentum_start=c(0.25,0.5,0.75),
    momentum_stable =c(0.75,0.9,0.99),
    momentum_ramp = c(1e6, 1e7, 1e8)
  )
  hyper_params # 3*2*2*2*3*3*3*3*3 = 5832 combinations
  
  h2o.grid(
    "deeplearning",
    model_id="dl_grid", 
    training_frame = train, 
    validation_frame = valid, 
    x=predictors, 
    y=response,
    epochs=1000,
    stopping_metric="logloss",
    stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
    stopping_rounds=2,
    score_validation_samples=10000, ## downsample validation set for faster scoring
    score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
    adaptive_rate=F,                ## manually tuned learning rate
    activation = c("Rectifier"),
    max_w2 = 10,                    ## can help improve stability for Rectifier
    hyper_params = hyper_params
  )
}

## Part 3 - Some data visualization

## Scalable scatter plots (binned histograms are made by the H2O cluster)
plot(h2o.tabulate(df, "Elevation",                       "Cover_Type"))
plot(h2o.tabulate(df, "Horizontal_Distance_To_Roadways", "Cover_Type"))
plot(h2o.tabulate(df, "Soil_Type",                       "Cover_Type"))
plot(h2o.tabulate(df, "Horizontal_Distance_To_Roadways", "Elevation"))
plot(h2o.tabulate(df, "Horizontal_Distance_To_Roadways", "Elevation"))


## Plot Accuracy vs Elevation
x <- "Elevation"
#x <- "Horizontal_Distance_To_Roadways"
#x <- "Horizontal_Distance_To_Fire_Points"
y <- "Accuracy"
#y <- "Cover_Type"
tb <- h2o.tabulate(data = test, x=x, y=y, nbins_x = 100, nbins_y = 100)
plot(tb)
resp <- tb$response_table
par(mfrow=c(2,1)) 
plot(resp[,1],resp[,2], main=y, xlab=x, ylab=y, type="l")
plot(resp[,1],resp[,3], main="Frequency", xlab=x, ylab="Frequency", type="l") 


