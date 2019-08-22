## Load H2O library
library(h2o)

## Connect to H2O cluster
h2o.init(nthreads = -1)

## Define data paths
base_path = normalizePath("~/Desktop/H2OTour/intro_r_python_flow/data/")
flights_path = paste0( base_path, "/flights.csv")
weather_path = paste0( base_path, "/weather.csv")

## Ingest data
flights_hex = h2o.importFile(path = flights_path, destination_frame = "flights_hex")
weather_hex = h2o.importFile(path = weather_path, destination_frame = "weather_hex")

## Summary of the flights and weather dataset
h2o.describe(flights_hex)
h2o.describe(weather_hex)

## Plot flights over the years in the dataset
h2o.hist(x = flights_hex$Year)

## Plot flights over the years using user defined R function
flights_hex$IsArrDelayedNumeric = ifelse(flights_hex$IsArrDelayed == "YES", 1, 0)
flights_hex$IsWeatherDelayedNumeric = ifelse(flights_hex$WeatherDelay > 0, 1, 0)
flights_count = h2o.group_by(data = flights_hex, by = "Year", nrow("Year"), sum("IsArrDelayedNumeric"), sum("IsWeatherDelayedNumeric"))
flights_count_df = as.data.frame(flights_count)
flights_count_df2 = t(flights_count_df[, 2:4])
colnames(flights_count_df2) = flights_count_df$Year
flights_count_df2

## Plot flights over the years using user defined R function
barplot(flights_count_df2, beside = T, col = c("dark blue", "red", "purple"))

## Filter flights before 2003
flights_hex = flights_hex[ flights_hex$Year >= 2003, ]
## Filter flights that is delayed but not delayed by weather
flights_hex = flights_hex[ (flights_hex[, "IsArrDelayed"] == "NO") | (flights_hex[ ,"WeatherDelay"] > 0) ,]

## Weather delay only happens 2.17% of the time
responseCount = as.data.frame( h2o.table(flights_hex$IsArrDelayed))
print("Total number of flights in dataset...")
print(prettyNum(nrow(flights_hex), big.mark = ","))
print("Number of flights delayed arriving in Chicago due to weather ...")
print(prettyNum(responseCount[2,2], big.mark = ","))

## Parameter Creation
hour1 <- flights_hex$CRSArrTime %/% 100
mins1 <- flights_hex$CRSArrTime %% 100
arrTime <- hour1*60+mins1

hour2 <- flights_hex$CRSDepTime %/% 100
mins2 <- flights_hex$CRSDepTime %% 100
depTime <- hour2*60+mins2

travelTime <- ifelse(arrTime - depTime > 0, arrTime - depTime, NA)
flights_hex$TravelTime <- travelTime
flights_hex

## Subset frame down to predictors and response
myY = "IsArrDelayed"
myX = c("Year", "Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "TravelTime")

## Split frame into test/train split
split = h2o.splitFrame(data = flights_hex[, c(myY, myX)], destination_frames = c("train_hex", "valid_hex"))
train = split[[1]]
valid = split[[2]]

arr_delay_glm = h2o.glm(x = myX,
                        y = myY, 
                        training_frame = train, 
                        validation_frame = valid, 
                        family = "binomial", 
                        alpha = 0.5,
                        lambda_search = T)
arr_delay_gbm = h2o.gbm(x = myX,
                        y = myY,
                        training_frame = train, 
                        validation_frame = valid, 
                        distribution = "bernoulli",
                        ntrees = 50)

## Report AUC on preliminary glm and gbm models
glm_model = arr_delay_glm
gbm_model = arr_delay_gbm
auc_table1 = data.frame(GLM_AUC = c(h2o.auc(glm_model, train = T), h2o.auc(glm_model, valid = T)),
                       GBM_AUC = c(h2o.auc(gbm_model, train = T), h2o.auc(gbm_model, valid = T)))
row.names(auc_table1) = c("Training Set", "Validation Set")
auc_table1


## Join with weather data by Year, Month, and DayofMonth
merged_data = h2o.merge(x = flights_hex, y = weather_hex, by = colnames(flights_hex)[1:3])

## Split frame into test/train split
split_weather = h2o.splitFrame(data = merged_data,
                               destination_frames = c("train_weather_hex", "valid_weather_hex"))
train_weather = split_weather[[1]]
valid_weather = split_weather[[2]]

myX_weather = c(names(weather_hex), "DayOfWeek", "UniqueCarrier", "Origin", "TravelTime")
arr_delay_weather_glm = h2o.glm(x = myX_weather, 
                        y = myY, 
                        training_frame = train_weather, 
                        validation_frame = valid_weather, 
                        family = "binomial", 
                        alpha = 0.5, 
                        lambda_search = T)
arr_delay_weather_gbm = h2o.gbm(x = myX_weather,
                        y = myY,
                        training_frame = train_weather, 
                        validation_frame = valid_weather, 
                        distribution = "bernoulli",
                        ntrees = 50)

## Report AUC on preliminary glm and gbm models
glm_model = arr_delay_weather_glm
gbm_model = arr_delay_weather_gbm
auc_table2 = data.frame(GLM_AUC = c(h2o.auc(glm_model, train = T), h2o.auc(glm_model, valid = T)),
                       GBM_AUC = c(h2o.auc(gbm_model, train = T), h2o.auc(gbm_model, valid = T)))

row.names(auc_table2) = c("Training Set", "Validation Set")
auc_table2

