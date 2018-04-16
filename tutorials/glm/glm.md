* Introduction
  * Installation and Startup
* Cover Type Dataset
* Multinomial Model
* Binomial Model
  * Adding extra features
* Multinomial Model Revisited


## Introduction
This tutorial shows how a H2O [GLM](http://en.wikipedia.org/wiki/Generalized_linear_model) model can be used to do binary and multi-class classification. This tutorial covers usage of H2O from R. A python version of this tutorial will be available as well in a separate document. This file is available in plain R, R markdown and regular markdown formats, and the plots are available as PDF files. All documents are available [on Github](https://github.com/h2oai/h2o-world-2015-training/raw/master/tutorials/glm/).

If run from plain R, execute R in the directory of this script. If run from RStudio, be sure to `setwd()` to the location of this script. `h2o.init()` starts H2O in R's current working directory. h2o.importFile() looks for files from the perspective of where H2O was started.

More examples and explanations can be found in our [H2O GLM booklet](http://h2o.ai/resources/) and on our [H2O Github Repository](http://github.com/h2oai/h2o-3/). 

### H2O R Package

Load the H2O R package:

```{r}
## R installation instructions are at http://h2o.ai/download
library(h2o)
```

### Start H2O
Start up a 1-node H2O server on your local machine, and allow it to use all CPU cores and up to 2GB of memory:

```{r}
h2o.init(nthreads=-1, max_mem_size="2G")
h2o.removeAll() ## clean slate - just in case the cluster was already running
```
## Cover Type Data
Predicting forest cover type from cartographic variables only (no remotely sensed data).
Let's import the dataset:

```{r}
D = h2o.importFile(path = normalizePath("../data/covtype.full.csv"))
h2o.summary(D)
```
We have 11 numeric and two categorical features. Response is "Cover_Type" and has 7 classes.
Let's split the data into Train/Test/Validation with train having 70% and Test and Validation 15% each:

```{r}
data = h2o.splitFrame(D,ratios=c(.7,.15),destination_frames = c("train","test","valid"))
names(data) <- c("Train","Test","Valid")
y = "Cover_Type"
x = names(data$Train)
x = x[-which(x==y)]
```
## Multinomial Model
 
We imported our data, so let's run GLM. As we mentioned previously, Cover_Type is the response and we use all other columns as predictors.
We have multi-class problem so we pick family=multinomial. L-BFGS solver tends to be faster on multinomial problems, so we pick L-BFGS for our first try. 
The rest can use the default settings.

```{r}
m1 = h2o.glm(training_frame = data$Train, validation_frame = data$Valid, x = x, y = y,family='multinomial',solver='L_BFGS')
h2o.confusionMatrix(m1, valid=TRUE)
```
The model predicts only the majority class so it's not useful at all! Maybe we regularized it too much, let's try again without regularization: 

```{r}
m2 = h2o.glm(training_frame = data$Train, validation_frame = data$Valid, x = x, y = y,family='multinomial',solver='L_BFGS', lambda = 0)
h2o.confusionMatrix(m2, valid=FALSE) # get confusion matrix in the training data
h2o.confusionMatrix(m2, valid=TRUE)  # get confusion matrix in the validation data
```
No overfitting (as train and test performance are the same), regularization is not needed in this case. 

This model is actually useful. It got 28% classification error, down from 51% obtained by predicting majority class only.

## Binomial Model
Since multinomial models are difficult and time consuming, let's try a simpler binary classification. 
We'll take a subset of the data with only `class_1` and `class_2` (the two majority classes) and build a binomial model deciding between them.

```{r}
D_binomial = D[D$Cover_Type %in% c("class_1","class_2"),]
h2o.setLevels(D_binomial$Cover_Type,c("class_1","class_2"))
# split to train/test/validation again
data_binomial = h2o.splitFrame(D_binomial,ratios=c(.7,.15),destination_frames = c("train_b","test_b","valid_b"))
names(data_binomial) <- c("Train","Test","Valid")
```
We can run a binomial model now: 

```{r}
m_binomial = h2o.glm(training_frame = data_binomial$Train, validation_frame = data_binomial$Valid, x = x, y = y, family='binomial',lambda=0)
h2o.confusionMatrix(m_binomial, valid = TRUE)
h2o.confusionMatrix(m_binomial, valid = TRUE)
```
The output for a binomial problem is slightly different from multinomial. The confusion matrix now has a threshold attached to it.

The model produces probability of `class_1` and `class_2` similarly to multinomial example earlier. However, this time we only have two classes and we can tune the classification to our needs. 

The classification errors in binomial cases have a particular meaning: we call them false-positive and false negative. In reality, each can have a different cost associated with it, so we want to tune our classifier accordingly. 

The common way to evaluate a binary classifier performance is to look at its [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). The ROC curve plots the true positive rate versus false positive rate. We can plot it from the H2O model output:

```{r}
fpr = m_binomial@model$training_metrics@metrics$thresholds_and_metric_scores$fpr
tpr = m_binomial@model$training_metrics@metrics$thresholds_and_metric_scores$tpr
fpr_val = m_binomial@model$validation_metrics@metrics$thresholds_and_metric_scores$fpr
tpr_val = m_binomial@model$validation_metrics@metrics$thresholds_and_metric_scores$tpr
plot(fpr,tpr, type='l')
title('AUC')
lines(fpr_val,tpr_val,type='l',col='red')
legend("bottomright",c("Train", "Validation"),col=c("black","red"),lty=c(1,1),lwd=c(3,3))                             
```

The area under the ROC curve (AUC) is a common "good fit" metric for binary classifiers. For this example, the results were:

```{r}
h2o.auc(m_binomial,valid=FALSE) # on train                   
h2o.auc(m_binomial,valid=TRUE)  # on test
```

The default confusion matrix is computed at thresholds that optimize the [F1 score](https://en.wikipedia.org/wiki/F1_score). We can choose different thresholds - the H2O output shows optimal thresholds for some common metrics.

```{r}
m_binomial@model$training_metrics@metrics$max_criteria_and_metric_scores                  
```

The model we just built gets 23% classification error at the F1-optimizing threshold, so there is still room for improvement.
Let's add some features:

* There are 11 numerical predictors in the dataset, we will cut them into intervals and add a categorical variable for each
* We can add interaction terms capturing interactions between categorical variables

Let's make a convenience function to cut the column into intervals working on all three of our datasets (Train/Validation/Test). 
We'll use `h2o.hist` to determine interval boundaries (but there are many more ways to do that!) on the Train set.  
We'll take only the bins with non-trivial support:  

```{r}
cut_column <- function(data, col) {
  # need lower/upper bound due to h2o.cut behavior (points < the first break or > the last break are replaced with missing value) 
  min_val = min(data$Train[,col])-1
  max_val = max(data$Train[,col])+1
  x = h2o.hist(data$Train[, col])
  # use only the breaks with enough support
  breaks = x$breaks[which(x$counts > 1000)]
  # assign level names 
  lvls = c("min",paste("i_",breaks[2:length(breaks)-1],sep=""),"max")
  col_cut <- paste(col,"_cut",sep="")
  data$Train[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Train[,col],breaks=c(min_val,breaks,max_val)),lvls)
  # now do the same for test and validation, but using the breaks computed on the training!
  if(!is.null(data$Test)) {
    min_val = min(data$Test[,col])-1
    max_val = max(data$Test[,col])+1
    data$Test[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Test[,col],breaks=c(min_val,breaks,max_val)),lvls)
  }
  if(!is.null(data$Valid)) {
    min_val = min(data$Valid[,col])-1
    max_val = max(data$Valid[,col])+1
    data$Valid[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Valid[,col],breaks=c(min_val,breaks,max_val)),lvls)
  }
  data
}
```
Now let's make a convenience function generating interaction terms on all three of our datasets. We'll use `h2o.interaction`:

```{r}
interactions <- function(data, cols, pairwise = TRUE) {
  iii = h2o.interaction(data = data$Train, destination_frame = "itrain",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
  data$Train <- h2o.cbind(data$Train,iii)
  if(!is.null(data$Test)) {
    iii = h2o.interaction(data = data$Test, destination_frame = "itest",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
    data$Test <- h2o.cbind(data$Test,iii)
  }
  if(!is.null(data$Valid)) {
    iii = h2o.interaction(data = data$Valid, destination_frame = "ivalid",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
    data$Valid <- h2o.cbind(data$Valid,iii)
  }
  data
}
```
Finally, let's wrap addition of the features into a separate function call, as we will use it again later.
We'll add intervals for each numeric column and interactions between each pair of binary columns.

```{r}
# add features to our cover type example
# let's cut all the numerical columns into intervals and add interactions between categorical terms
add_features <- function(data) {
  names(data) <- c("Train","Test","Valid")
  data = cut_column(data,'Elevation')
  data = cut_column(data,'Hillshade_Noon')
  data = cut_column(data,'Hillshade_9am')
  data = cut_column(data,'Hillshade_3pm')
  data = cut_column(data,'Horizontal_Distance_To_Hydrology')
  data = cut_column(data,'Slope')
  data = cut_column(data,'Horizontal_Distance_To_Roadways')
  data = cut_column(data,'Aspect')
  # pairwise interactions between all categorical columns
  interaction_cols = c("Elevation_cut","Wilderness_Area","Soil_Type","Hillshade_Noon_cut","Hillshade_9am_cut","Hillshade_3pm_cut","Horizontal_Distance_To_Hydrology_cut","Slope_cut","Horizontal_Distance_To_Roadways_cut","Aspect_cut")
  data = interactions(data, interaction_cols)
  # interactions between Hillshade columns
  interaction_cols2 = c("Hillshade_Noon_cut","Hillshade_9am_cut","Hillshade_3pm_cut")
  data = interactions(data, interaction_cols2,pairwise = FALSE)
  data
}
```
Now we generate new features and add them to the dataset. We'll also need to generate column names again, as we added more columns:

```{r}
# Add Features
data_binomial_ext <- add_features(data_binomial)
data_binomial_ext$Train <- h2o.assign(data_binomial_ext$Train,"train_b_ext")
data_binomial_ext$Valid <- h2o.assign(data_binomial_ext$Valid,"valid_b_ext")
data_binomial_ext$Test <- h2o.assign(data_binomial_ext$Test,"test_b_ext")
y = "Cover_Type"
x = names(data_binomial_ext$Train)
x = x[-which(x==y)]
```
Let's build the model! We should add some regularization this time because we added correlated variables, so let's try the default:

```{r}
m_binomial_1_ext = try(h2o.glm(training_frame = data_binomial_ext$Train, validation_frame = data_binomial_ext$Valid, x = x, y = y, family='binomial'))
```
Oops, doesn't run - well, we know have more features than the default method can solve with 2GB of RAM. Let's try L-BFGS instead.

```{r}
m_binomial_1_ext = h2o.glm(training_frame = data_binomial_ext$Train, validation_frame = data_binomial_ext$Valid, x = x, y = y, family='binomial', solver='L_BFGS')
h2o.confusionMatrix(m_binomial_1_ext)
h2o.auc(m_binomial_1_ext,valid=TRUE)
```
Not much better, maybe too much regularization? Let's pick a smaller lambda and try again.

```{r}
m_binomial_2_ext = h2o.glm(training_frame = data_binomial_ext$Train, validation_frame = data_binomial_ext$Valid, x = x, y = y, family='binomial', solver='L_BFGS', lambda=1e-4)
h2o.confusionMatrix(m_binomial_2_ext, valid=TRUE)
h2o.auc(m_binomial_2_ext,valid=TRUE)
```
Way better, we got an AUC of .91 and classification error of 0.180838. 
We picked our regularization strength arbitrarily. Also, we used only the l2 penalty but we added lot of extra features, some of which may be useless. 
Maybe we can do better with an l1 penalty. 
So now we want to run a lambda search to find optimal penalty strength and we want to have a non-zero l1 penalty to get sparse solution.
We'll use the IRLSM solver this time as it does much better with lambda search and l1 penalty. 
Recall we were not able to use it before. We can use it now as we are running a lambda search that will filter out a large portion of the inactive (coefficient==0) predictors. 

```{r}
m_binomial_3_ext = h2o.glm(training_frame = data_binomial_ext$Train, validation_frame = data_binomial_ext$Valid, x = x, y = y, family='binomial', lambda_search=TRUE)
h2o.confusionMatrix(m_binomial_3_ext, valid=TRUE)
h2o.auc(m_binomial_3_ext,valid=TRUE)
```
Better yet, we have 17% error and we used only 3000 out of 7000 features.
Ok, our new features improved the binomial model significantly, so let's revisit our former multinomial model and see if they make a difference there (they should!):

```{r}
# Multinomial Model 2
# let's revisit the multinomial case with our new features
data_ext <- add_features(data)
data_ext$Train <- h2o.assign(data_ext$Train,"train_m_ext")
data_ext$Valid <- h2o.assign(data_ext$Valid,"valid_m_ext")
data_ext$Test <- h2o.assign(data_ext$Test,"test_m_ext")
y = "Cover_Type"
x = names(data_ext$Train)
x = x[-which(x==y)]
m2 = h2o.glm(training_frame = data_ext$Train, validation_frame = data_ext$Valid, x = x, y = y,family='multinomial',solver='L_BFGS',lambda=1e-4)
# 21% err down from 28%
h2o.confusionMatrix(m2, valid=TRUE)
```
Improved considerably, 21% instead of 28%.


