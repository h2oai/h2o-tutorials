# setwd("/Users/tomk/0xdata/ws/h2o-world-2015-training/tutorials/pojo_webapp")
# getwd()


#
# If necessary, install H2O
#
# This example was prepared with H2O 3.2.0.1, although any later build should work:
#
# Install from CRAN:
#   install.packages("h2o")
#
# Install from H2O's download site:
#   http://h2o-release.s3.amazonaws.com/h2o/rel-slater/1/index.html
#   install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/rel-slater/1/R")))
#

library(h2o)

# Start H2O
h2o.init(nthreads = -1)

# Load data
df = h2o.importFile("data/titanic.csv")
dim(df)
df$survived = as.factor(df$survived)
df$pclass = as.factor(df$pclass)
summary(df)

# Build model
model = h2o.gbm(y = "survived",
                x = c("pclass", "sex", "age", "fare"),
                training_frame = df,
                model_id = "MyModel")
print(model)

# Download generated POJO for model
if (! file.exists("tmp")) {
  dir.create("tmp")
}
h2o.download_pojo(model, path = "tmp")
