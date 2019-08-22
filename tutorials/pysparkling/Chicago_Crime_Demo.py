# Spark Context
sc

# Start H2O Context
from pysparkling import *
sc
hc= H2OContext(sc).start()

# H2o Context
hc

# Import H2O Python library
import h2o
# View all the available H2O python functions
#dir(h2o)

# Parse Chicago Crime dataset into H2O
column_type = ['Numeric','String','String','Enum','Enum','Enum','Enum','Enum','Enum','Enum','Numeric','Numeric','Numeric','Numeric','Enum','Numeric','Numeric','Numeric','Enum','Numeric','Numeric','Enum']
f_crimes = h2o.import_file(path ="../data/chicagoCrimes10k.csv",col_types =column_type)

print(f_crimes.shape)
f_crimes.summary()

# Look at the distribution of IUCR column
f_crimes["IUCR"].table()

# Look at distribution of Arrest column
f_crimes["Arrest"].table()

# Modify column names to replace blank spaces with underscore
col_names = map(lambda s: s.replace(' ', '_'), f_crimes.col_names)
f_crimes.set_names(col_names)

# Set time zone to UTC for date manipulation
h2o.set_timezone("Etc/UTC")

## Refine date column 
def refine_date_col(data, col, pattern):
    data[col]         = data[col].as_date(pattern)
    data["Day"]       = data[col].day()
    data["Month"]     = data[col].month()    # Since H2O indexes from 0
    data["Year"]      = data[col].year()
    data["WeekNum"]   = data[col].week()
    data["WeekDay"]   = data[col].dayOfWeek()
    data["HourOfDay"] = data[col].hour()
    
    # Create weekend and season cols
    data["Weekend"] = (data["WeekDay"] == "Sun" or data["WeekDay"] == "Sat").ifelse(1, 0)[0]
    data["Season"] = data["Month"].cut([0, 2, 5, 7, 10, 12], ["Winter", "Spring", "Summer", "Autumn", "Winter"])
    
refine_date_col(f_crimes, "Date", "%m/%d/%Y %I:%M:%S %p")
f_crimes = f_crimes.drop("Date")

# Parse Census data into H2O
f_census = h2o.import_file("../data/chicagoCensus.csv",header=1)

## Update column names in the table
col_names = map(lambda s: s.strip().replace(' ', '_'), f_census.col_names)
f_census.set_names(col_names)
f_census = f_census[1:78,:]
print(f_census.dim)
#f_census.summary()

# Parse Weather data into H2O
f_weather = h2o.import_file("../data/chicagoAllWeather.csv")
f_weather = f_weather[1:]
print(f_weather.dim)
#f_weather.summary()

# Look at all the null entires in the Weather table
f_weather[f_weather["meanTemp"].isna()]

# Look at the help on as_spark_frame
hc.as_spark_frame?
f_weather

# Copy data frames to Spark from H2O
df_weather = hc.as_spark_frame(f_weather,)
df_census = hc.as_spark_frame(f_census)
df_crimes = hc.as_spark_frame(f_crimes)

# Look at the weather data as parsed in Spark
df_weather.show(2)

# Join columns from Crime, Census and Weather DataFrames in Spark

## Register DataFrames as tables in SQL context
sqlContext.registerDataFrameAsTable(df_weather, "chicagoWeather")
sqlContext.registerDataFrameAsTable(df_census, "chicagoCensus")
sqlContext.registerDataFrameAsTable(df_crimes, "chicagoCrime")


crimeWithWeather = sqlContext.sql("""SELECT
a.Year, a.Month, a.Day, a.WeekNum, a.HourOfDay, a.Weekend, a.Season, a.WeekDay,
a.IUCR, a.Primary_Type, a.Location_Description, a.Community_Area, a.District,
a.Arrest, a.Domestic, a.Beat, a.Ward, a.FBI_Code,
b.minTemp, b.maxTemp, b.meanTemp,
c.PERCENT_AGED_UNDER_18_OR_OVER_64, c.PER_CAPITA_INCOME, c.HARDSHIP_INDEX,
c.PERCENT_OF_HOUSING_CROWDED, c.PERCENT_HOUSEHOLDS_BELOW_POVERTY,
c.PERCENT_AGED_16__UNEMPLOYED, c.PERCENT_AGED_25__WITHOUT_HIGH_SCHOOL_DIPLOMA
FROM chicagoCrime a
JOIN chicagoWeather b
ON a.Year = b.year AND a.Month = b.month AND a.Day = b.day
JOIN chicagoCensus c
ON a.Community_Area = c.Community_Area_Number""")

# Print the crimeWithWeather data table from Spark
crimeWithWeather.show(2)

#Copy table from Spark to H2O
hc.as_h2o_frame?
crimeWithWeatherHF = hc.as_h2o_frame(crimeWithWeather,framename="crimeWithWeather")

crimeWithWeatherHF.summary()

# Assign column types to the CrimeWeatherHF data table in H2O
crimeWithWeatherHF["Season"]= crimeWithWeatherHF["Season"].asfactor()
crimeWithWeatherHF["WeekDay"]= crimeWithWeatherHF["WeekDay"].asfactor()
crimeWithWeatherHF["IUCR"]= crimeWithWeatherHF["IUCR"].asfactor()
crimeWithWeatherHF["Primary_Type"]= crimeWithWeatherHF["Primary_Type"].asfactor()
crimeWithWeatherHF["Location_Description"]= crimeWithWeatherHF["Location_Description"].asfactor()
crimeWithWeatherHF["Arrest"]= crimeWithWeatherHF["Arrest"].asfactor()
crimeWithWeatherHF["Domestic"]= crimeWithWeatherHF["Domestic"].asfactor()
crimeWithWeatherHF["FBI_Code"]= crimeWithWeatherHF["FBI_Code"].asfactor()
crimeWithWeatherHF["Season"]= crimeWithWeatherHF["Season"].asfactor()

crimeWithWeatherHF.summary()

# Split final H2O data table into train test and validation sets
ratios = [0.6,0.2]
frs = crimeWithWeatherHF.split_frame(ratios,seed=12345)
train = frs[0]
train.frame_id = "Train"
valid = frs[2]
valid.frame_id = "Validation"
test = frs[1]
test.frame_id = "Test"

# Import Model Builders from H2O Python
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# Inspect the avialble gbm parameters
H2OGradientBoostingEstimator?

# Define Preditors
predictors = crimeWithWeatherHF.names[:]
response = "Arrest"
predictors.remove(response)

#Simple GBM model - Predict Arrest
model_gbm = H2OGradientBoostingEstimator(ntrees         =50,
                                        max_depth      =6,
                                        learn_rate     =0.1, 
                                        #nfolds         =2,
                                        distribution   ="bernoulli")

model_gbm.train(x               =predictors,
               y               ="Arrest",
               training_frame  =train,
               validation_frame=valid
               )

# Simple Deep Learning - Predict Arrest
model_dl = H2ODeepLearningEstimator(variable_importances=True,
                                   loss                ="Automatic")

model_dl.train(x                =predictors,
              y                ="Arrest",
              training_frame  =train,
              validation_frame=valid)

# Print confusion matrices for the train and validation set
print(model_gbm.confusion_matrix(train = True))
print(model_gbm.confusion_matrix(valid = True))

print(model_gbm.auc(train=True))
print(model_gbm.auc(valid=True))
model_gbm.plot(metric="AUC")

#Print variable importance
model_gbm.varimp(True)

# Inspect Deep Learning model output
model_dl

# Predict on the test set using the gbm model
predictions = model_gbm.predict(test)
predictions.show()

# Look at performance on test set (if it includes true lables)
test_performance = model_gbm.model_performance(test)
test_performance

#Plots
# Create Plots of Crime type vs Arrest Rate and Proportion of reported Crime

# Create table to report Crimetype, Arrest count per crime, total reported count per Crime  
sqlContext.registerDataFrameAsTable(df_crimes, "df_crimes")
allCrimes = sqlContext.sql("""SELECT Primary_Type, count(*) as all_count FROM df_crimes GROUP BY Primary_Type""")
crimesWithArrest = sqlContext.sql("SELECT Primary_Type, count(*) as crime_count FROM chicagoCrime WHERE Arrest = 'true' GROUP BY Primary_Type")

sqlContext.registerDataFrameAsTable(crimesWithArrest, "crimesWithArrest")
sqlContext.registerDataFrameAsTable(allCrimes, "allCrimes")

crime_type = sqlContext.sql("Select a.Primary_Type as Crime_Type, a.crime_count, b.all_count \
FROM crimesWithArrest a \
JOIN allCrimes b \
ON a.Primary_Type = b.Primary_Type ")

crime_type.show(12)

#Copy Crime_type table from Spark to H2O
crime_typeHF = hc.as_h2o_frame(crime_type,framename="crime_type")

# Create Additional columns Arrest_rate and Crime_propotion 
crime_typeHF["Arrest_rate"] = crime_typeHF["crime_count"]/crime_typeHF["all_count"]
crime_typeHF["Crime_proportion"] = crime_typeHF["all_count"]/crime_typeHF["all_count"].sum()
crime_typeHF["Crime_Type"] = crime_typeHF["Crime_Type"].asfactor()
# h2o.assign(crime_typeHF,crime_type)
crime_typeHF.frame_id = "Crime_type"

crime_typeHF

hc

# Plot the below in Flow 
#plot (g) -> g(
#  g.rect(
#    g.position "Crime_Type", "Arrest_rate"
#    g.fillColor g.value 'blue'
#    g.fillOpacity g.value 0.75
#  )
#  g.rect(
#    g.position "Crime_Type", "Crime_proportion"
#    g.fillColor g.value 'red'
#    g.fillOpacity g.value 0.65
#  )
#  g.from inspect "data", getFrame "Crime_type"
#)