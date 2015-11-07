# Predicting Airline Delays with H2O in R

### Note: If run from plain R, execute R in the directory of this script. If run from RStudio, 
### be sure to setwd() to the location of this script. h2o.init() starts H2O in R's current 
### working directory. h2o.importFile() looks for files from the perspective of where H2O was 
### started.

### Load the H2O R package and start an local H2O cluster
###### Connection to an H2O cloud is established through the `h2o.init` function from the `h2o` package. 
###### When starting H2O from R, specify `nthreads` equal to -1, in order to utilize all the cores on your machine.
###### To connect to a pre-existing H2O cluster make sure to edit the H2O location with argument `myIP` and `myPort`.

    library(h2o)
    h2o.init(nthreads = -1)

### Import Data into H2O
###### We will use the `h2o.importFile` function to do a parallel read of the data into the H2O distributed key-value store. 
###### During import of the data, features Year, Month, DayOfWeek, and FlightNum were set to be parsed as enumerator or categorical rather than numeric columns.

    airlines.hex <- h2o.importFile(path = normalizePath("../data/allyears2k.csv"), destination_frame = "allyears2k.hex")

###### Get an overview of the airlines dataset quickly by running `summary`.

    summary(airlines.hex)

### Building a GLM Model
###### Run a logistic regression model using function `h2o.glm` and selecting “binomial” for parameter `Family`.
###### Add some regularization by setting alpha to 0.5 and lambda to 1e-05.
    
    y <- "IsDepDelayed"
    x <- c("Dest", "Origin", "DayofMonth", "Year", "UniqueCarrier", "DayOfWeek", "Month", "Distance")
    glm_model <- h2o.glm(x = x, y = y, training_frame = airlines.hex, model_id = "glm_model_from_R",
                         solver = "IRLSM", standardize = T, link = "logit",
                         family = "binomial", alpha = 0.5, lambda = 1e-05)
    
    auc <- h2o.auc(object = glm_model)
    print(paste0("AUC of the training set : ", round(auc, 4)))
    print(glm_model@model$standardized_coefficient_magnitudes)
    print(glm_model@model$scoring_history)
        
### Building a Deep Learning Model
###### Build a binary classfication model using function `h2o.deeplearning` and selecting “bernoulli” for parameter `Distribution`.
###### Run 100 passes over the data by setting parameter `epoch` to 100.
    
    dl_model <- h2o.deeplearning(x = x, y = y, training_frame = airlines.hex, distribution = "bernoulli", model_id = "deeplearning_model_from_R", 
                                 epochs = 100, hidden = c(200,200), target_ratio_comm_to_comp = 0.02, seed = 6765686131094811000, variable_importances = T)
    auc2 <- h2o.auc(object = dl_model)
    print(paste0("AUC of the training set : ", round(auc2, 4)))
    print(h2o.varimp(dl_model))
    print(h2o.scoreHistory(dl_model))
    
### All done, shutdown H2O    
    h2o.shutdown(prompt=FALSE)
    
