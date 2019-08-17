## Code for scatterplots
scatter_plot <- function(data, x, y, max_points = 1000, fit = F) {
  if (fit) {
    lr <- h2o.glm(x = x, y = y, training_frame = data, family = "gaussian")
    coeff <- lr@model$coefficients_table$standardized_coefficients    
  }
  
  df <- data[,c(x, y)]
  
  
  runif <- h2o.runif(df)
  df.subset <- df[runif < max_points/nrow(data),]
  df.R <- as.data.frame(df.subset)
  
  if (fit) h2o.rm(lr@model_id)
  
  plot(x = df.R[,x], y = df.R[,y], col = "blue", xlab = x, 
       ylab = y, ylim = c(0, 550))
  if (fit) abline(coef = coeff, col = "black")
}

## Load library and initialize h2o
library(h2o)
h2o.init(nthreads = -1)


## Set file path and import data. Drop constant column (23).
pathToAirlines <- "https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv"

airlines.hex <- h2o.importFile(path = pathToAirlines, destination_frame = "airlines.hex")

airlines.hex <- airlines.hex[-23]
dim(airlines.hex)


## Get a summary of the data. Build a histogram examining the "Year" column using h2o.hist()
summary(airlines.hex)

h2o.hist(airlines.hex$Year)


## Scatter plot of airlines dataset examining the relationship between the "Distance" and "AirTime" columns
scatter_plot(data = airlines.hex, x = "Distance", y = "AirTime", max_points = 10000)


## Use h2o.group_by to calcualte the flights in a given month


## Use as.factor to change the "Year," "Month," "DayOfWeek," and "Cancelled" columns to factors
airlines.hex$Year      <- as.factor(airlines.hex$Year)
airlines.hex$Month     <- as.factor(airlines.hex$Month)
airlines.hex$DayOfWeek <- as.factor(airlines.hex$DayOfWeek)
airlines.hex$Cancelled <- as.factor(airlines.hex$Cancelled)

## Calculate and plot travel timef
hour1 <- airlines.hex$CRSArrTime %/% 100
mins1 <- airlines.hex$CRSArrTime %% 100
arrTime <- hour1*60+mins1

hour2 <- airlines.hex$CRSDepTime %/% 100
mins2 <- airlines.hex$CRSDepTime %% 100
depTime <- hour2*60+mins2




## Impute missing travel times by the "Origin" and "Dest" columns and re-plot. 


## Create test/train split


## Set predictor and response variables
myY <- "IsDepDelayed"



## Simple GLM and GBM models - Predict Delays



## Get summary of models


## Get variable importances for both models


