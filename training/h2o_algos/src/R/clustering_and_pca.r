
# import libs and allow h2o to use max. threads
library(ggplot2)
library(h2o)
h2o.init(nthreads = -1)

# location of clean data file
path <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"

# import file
frame <- h2o.importFile(path)

# strings automatically parsed as enums (categorical)
# numbers automatically parsed as numeric
# bad_loan is numeric, but categorical
frame$bad_loan <- as.factor(frame$bad_loan)

# find missing numeric and impute
for (name in names(frame)) {
  if (any(is.na(frame[name]))) {
      h2o.impute(frame, name, "median")
  }
}

h2o.describe(frame) # summarize table, check for missing

# assign target and inputs
y <- 'bad_loan'
X <- names(frame)[names(frame) != y]
print(y)
print(X)

# train k-means cluster model
# print summary
loan_clusters <- h2o.kmeans(training_frame = frame, 
                            x = X, 
                            model_id = "loan_clusters",
                            k = 3,
                            estimate_k = FALSE, 
                            standardize = TRUE,
                            seed=12345)

loan_clusters

# view detailed results at http://ip:port/flow/index.html

# join cluster labels to original data for further analysis
labels = predict(loan_clusters, frame)
labeled_frame = h2o.cbind(frame, labels)
labeled_frame["predict"]

# profile clusters by means
grouped <- h2o.group_by(labeled_frame, by = "predict", mean("int_rate"), mean("annual_inc"))
as.data.frame(grouped)

# project numeric training data onto 2-D using principal components
# join with clusters labels
numerics <- h2o.columns_by_type(frame, type = "numeric")
loan_pca <- h2o.prcomp(frame, numerics, k = 2, transform = "STANDARDIZE") # project onto 2 PCs
features <- h2o.cbind(labeled_frame, predict(loan_pca, labeled_frame))
features <- as.data.frame(features[, c("predict", "PC1", "PC2")])
features[1:10,]

# plot clusters with labels in 2-D space
features$predict <- as.factor(features$predict)
ggplot(features, aes(x=PC1, y=PC2, color=predict)) + geom_point()

h2o.shutdown(prompt = FALSE)
