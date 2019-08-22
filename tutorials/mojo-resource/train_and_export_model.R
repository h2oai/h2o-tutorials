library(h2o)
iris.hex <- as.h2o(iris)
iris.gbm <- h2o.gbm(y="Species", training_frame=iris.hex, model_id="irisgbm")
h2o.download_mojo(model=iris.gbm, path="/Users/nkkarpov/ws", get_genmodel_jar=TRUE)