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

// ~17% validation set error
m <- h2o.deeplearning(model_id="dl_model", training_frame = train, validation_frame = valid, 
                      x=predictors, y=response, variable_importances=T, epochs=1)

m
summary(m)

m <- h2o.deeplearning(model_id="dl_model", 
                      hidden=c(10,10,10),
                      training_frame = train, nfold=5,
                      x=predictors, y=response, epochs=1)
m

m <- h2o.deeplearning(training_frame = as.h2o(iris), x=1:4, y=5, nfold=2)

## Which variables were most important?
h2o.varimp(m)

// 7.77
m <- h2o.deeplearning(hidden=c(128,128,128), epochs=20, adaptive_rate = F, rate=0.02, rate_annealing=1e-6, 
                      initial_weight_distribution = "Normal", initial_weight_scale = 1e-3,
                      momentum_start = 0.5, momentum_stable = 0.9, momentum_ramp = 1e7, max_w2 = 10,
                      model_id="dl_model", training_frame = train, validation_frame = valid, 
                      x=predictors, y=response) 
// 7.63
m <- h2o.deeplearning(hidden=c(128,128,128), epochs=20, adaptive_rate = F, rate=0.02, rate_annealing=1e-6, 
                      initial_weight_distribution = "Normal", initial_weight_scale = 1e-2,
                      momentum_start = 0.5, momentum_stable = 0.9, momentum_ramp = 1e7, max_w2 = 10,
                      model_id="dl_model", training_frame = train, validation_frame = valid, 
                      x=predictors, y=response) 

// 11.8
m <- h2o.deeplearning(hidden=c(128,128,128), epochs=20, adaptive_rate = F, rate=0.02, rate_annealing=1e-7, 
                      initial_weight_distribution = "Normal", initial_weight_scale = 1e-2,
                      momentum_start = 0.5, momentum_stable = 0.9, momentum_ramp = 1e7, max_w2 = 10,
                      model_id="dl_model", training_frame = train, validation_frame = valid, 
                      x=predictors, y=response) 
// 8.3
m <- h2o.deeplearning(hidden=c(128,128,128), epochs=20, adaptive_rate = F, rate=0.02, rate_annealing=1e-5, 
                      initial_weight_distribution = "Normal", initial_weight_scale = 1e-2,
                      momentum_start = 0.5, momentum_stable = 0.9, momentum_ramp = 1e7, max_w2 = 10,
                      model_id="dl_model", training_frame = train, validation_frame = valid, 
                      x=predictors, y=response) 
// 7.3
m <- h2o.deeplearning(hidden=c(128,128,128), epochs=20, adaptive_rate = F, rate=0.02, rate_annealing=2e-6, 
                      initial_weight_distribution = "Normal", initial_weight_scale = 1e-2,
                      momentum_start = 0.5, momentum_stable = 0.9, momentum_ramp = 1e7, max_w2 = 10,
                      model_id="dl_model", training_frame = train, validation_frame = valid, 
                      x=predictors, y=response) 
// 8.2
m <- h2o.deeplearning(hidden=c(128,128,128), epochs=20, adaptive_rate = F, rate=0.02, rate_annealing=3e-6, rate_decay = 1.2,
                      initial_weight_distribution = "Normal", initial_weight_scale = 1e-2,
                      momentum_start = 0.5, momentum_stable = 0.9, momentum_ramp = 1e7, max_w2 = 10,
                      model_id="dl_model", training_frame = train, validation_frame = valid, 
                      x=predictors, y=response) 

// 6.96 2m27s
m <- h2o.deeplearning(
  model_id="dl_model", 
  training_frame = train, 
  validation_frame = valid, 
  x=predictors, 
  y=response, 
  train_samples_per_iteration=-1,
  shuffle_training_data=T,
  hidden=c(128,128,128),  ## more hidden layers -> more complex interactions
  epochs=20, ## long enough to converge
  stopping_metric = "misclassification",
  stopping_tolerance=1e-2,
  stopping_rounds=3,
  score_validation_samples=10000,
  adaptive_rate=F,   ## manual tuning of learning rate
  rate=0.02, 
  rate_annealing=2e-6, 
  initial_weight_distribution = "Normal", 
  initial_weight_scale = 1e-2, 
  momentum_start = 0.5, 
  momentum_stable = 0.9, 
  momentum_ramp = 1e7, 
  max_w2 = 10 
) 

//8.3 1m37s
m <- h2o.deeplearning(
  model_id="dl_model", 
  training_frame = train, 
  validation_frame = valid, 
  x=predictors, 
  y=response, 
  train_samples_per_iteration=-1,
  shuffle_training_data=F,
  hidden=c(100,100,100),  ## more hidden layers -> more complex interactions
  epochs=20, ## long enough to converge
  stopping_metric = "misclassification",
  stopping_tolerance=1e-2,
  stopping_rounds=3,
  score_validation_samples=10000,
  score_duty_cycle=0.025,
  adaptive_rate=F,   ## manual tuning of learning rate
  rate=0.02, 
  rate_annealing=2e-6, 
  initial_weight_distribution = "Normal", 
  initial_weight_scale = 1e-2, 
  momentum_start = 0.5, 
  momentum_stable = 0.9, 
  momentum_ramp = 1e7, 
  max_w2 = 10 
) 

m

h2o.performance(m, train=T)
h2o.performance(m, valid=T)
h2o.confusionMatrix(m, test)

## Confirm manually by looking at the predictions
p <- h2o.predict(m, test)
p
test$Accuracy <- p$predict == test$Cover_Type
1-mean(test$Accuracy)

plot(m)


## Scalable scatter plots (binned histograms are made by the H2O cluster)
plot(h2o.tabulate(data = df, x="Elevation",                       y="Cover_Type"))
plot(h2o.tabulate(data = df, x="Horizontal_Distance_To_Roadways", y="Cover_Type"))
plot(h2o.tabulate(data = df, x="Soil_Type",                       y="Cover_Type"))
plot(h2o.tabulate(data = df, x="Horizontal_Distance_To_Roadways", y="Elevation"))
plot(h2o.tabulate(data = df, x="Horizontal_Distance_To_Roadways", y="Elevation"))

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


