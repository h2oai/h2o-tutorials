
# coding: utf-8

# In[ ]:

import os
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator


# In[ ]:

import h2o


# In[ ]:

# Connect to a cluster
h2o.init()


# In[ ]:

## Define data paths
base_path = os.path.abspath("../data")
flights_path = base_path + "/flights.csv"
weather_path = base_path + "/weather.csv"


# In[ ]:

## Ingest data
flights_hex = h2o.import_file(path = flights_path, destination_frame = "flights_hex")
weather_hex = h2o.import_file(path = weather_path, destination_frame = "weather_hex")


# In[ ]:

## Summary of the flights and weather dataset
flights_hex.show()
weather_hex.show()


# In[ ]:

# Group flights by Year
flights_hex["IsArrDelayedNumeric"] = (flights_hex["IsArrDelayed"] == "YES").ifelse(1,0)
flights_hex["IsWeatherDelayedNumeric"] = (flights_hex["WeatherDelay"] > 0).ifelse(1,0)
flights_group = flights_hex.group_by("Year")
flights_count = flights_group.count().sum("IsArrDelayedNumeric").sum("IsWeatherDelayedNumeric").frame
flights_count.as_data_frame()


# In[ ]:

## Filter flights before 2003
flights_hex = flights_hex[ flights_hex[ "Year"] >= 2003]
## Filter flights that is delayed but not delayed by weather
flights_hex = flights_hex[ (flights_hex["IsArrDelayed"] == "NO") | (flights_hex["WeatherDelay"] > 0)]


# In[ ]:

## Parameter Creation
hour1 = flights_hex["CRSArrTime"] // 100
mins1 = flights_hex["CRSArrTime"] % 100
arrTime = hour1*60+mins1
hour2 = flights_hex["CRSDepTime"] // 100
mins2 = flights_hex["CRSDepTime"] % 100
depTime = hour2*60+mins2

travelTime = (arrTime - depTime > 0).ifelse(arrTime - depTime, 0)
flights_hex[ "TravelTime"]  = travelTime


# In[ ]:

# Set predictor and response variables
myY = "IsArrDelayed"
myX = ["Year", "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "TravelTime"]


# In[ ]:

# Create test/train split
split = flights_hex["Year"].runif()
train = flights_hex[split <= 0.75]
valid = flights_hex[split > 0.75]


# In[ ]:


# GLM - Predict Delays
arr_delay_glm = H2OGeneralizedLinearEstimator(family="binomial",
                                              standardize=True, 
                                              alpha = 0.5, 
                                              lambda_search = True)
arr_delay_glm.train(x               =myX,
                    y               =myY,
                    training_frame  =train,
                    validation_frame=valid)

# GBM
arr_delay_gbm = H2OGradientBoostingEstimator(distribution   ="bernoulli",
                                             ntrees         =50)

arr_delay_gbm.train(x               =myX,
                    y               =myY,
                    training_frame  =train,
                    validation_frame=valid)


# In[ ]:

print "GLM AUC TRAIN=", arr_delay_glm.auc(train = True),", AUC Valid=",arr_delay_glm.auc(valid = True)
print "GBM AUC TRAIN=", arr_delay_gbm.auc(train = True),", AUC Valid=",arr_delay_gbm.auc(valid = True)


# In[ ]:

## Merge with weather data
merged_data = flights_hex.merge(weather_hex)


# In[ ]:

# Create test/train split
split_weather = merged_data["Year"].runif()
train_weather = merged_data[split <= 0.75]
valid_weather = merged_data[split > 0.75]


# In[ ]:

# Set predictor and response variables
myY = "IsArrDelayed"
myX = ["Year", "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "TravelTime"] + weather_hex.columns


# In[ ]:

# GLM - Predict Delays
arr_delay_weather_glm = H2OGeneralizedLinearEstimator(family="binomial",
                                                      standardize=True, 
                                                      alpha = 0.5, 
                                                      lambda_search = True)
arr_delay_weather_glm.train(x               =myX,
                            y               =myY,
                            training_frame  =train_weather,
                            validation_frame=valid_weather)

# GBM
arr_delay_weather_gbm = H2OGradientBoostingEstimator(distribution   ="bernoulli",
                                                     ntrees         =50)

arr_delay_weather_gbm.train(x               =myX,
                            y               =myY,
                            training_frame  =train_weather,
                            validation_frame=valid_weather)



# In[ ]:

print "GLM AUC TRAIN=", arr_delay_weather_glm.auc(train = True),", AUC Valid=",arr_delay_weather_glm.auc(valid = True)
print "GBM AUC TRAIN=", arr_delay_weather_gbm.auc(train = True),", AUC Valid=",arr_delay_weather_gbm.auc(valid = True)

