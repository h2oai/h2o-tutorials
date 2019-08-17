# Hive JDBC import into H2O

H2O can ingest data from Hive through the Hive v2 JDBC driver by providing H2O with the JDBC driver for your Hive version.
**Note**: Since Hive only support OFFSET in SELECT statements since Hive 2.0 for earlier versions of Hive it is necessary to
import data with `fetch_mode="SINGLE"` option.

## Motivation
Reading data in Hive via the JDBC interface provides a convenient way how to transform and filter data using Hive Query engine before importing
it into H2O.

## Prerequisities
* Running Hive instance

## Setting up table with data
You can skip these steps if you already have your data prepared in a Hive table. 

1. Get the AirlinesTest dataset from [S3](https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/AirlinesTest.csv.zip).

2. Run the CLI client for Hive:
`$ beeline -u jdbc:hive2://hive-host:10000/db-name`

3. Create the DB table:

		CREATE EXTERNAL TABLE IF IT DOES NOT EXIST AirlinesTest(
		  fYear STRING ,
		  fMonth STRING ,
		  fDayofMonth STRING ,
		  fDayOfWeek STRING ,
		  DepTime INT ,
		  ArrTime INT ,
		  UniqueCarrier STRING ,
		  Origin STRING ,
		  Dest STRING ,
		  Distance INT ,
		  IsDepDelayed STRING ,
		  IsDepDelayed_REC INT
		)
		    COMMENT 'stefan test table'
		    ROW FORMAT DELIMITED
		    FIELDS TERMINATED BY ','
		    LOCATION '/tmp';


4. Import the data from the dataset:
`LOAD DATA INPATH '/tmp/AirlinesTest.csv' OVERWRITE INTO TABLE AirlinesTest;`

## Enable Hive JDBC Driver in H2O

### Retrieve the Hive JDBC Client jar

* For Hortonworks, Hive JDBC client jars can be found on one of the edge nodes after you have installed HDP: ``/usr/hdp/current/hive-client/lib/hive-jdbc-<version>-standalone.jar``. More information is available in [Hortonworks Documentation](https://docs.hortonworks.com/HDPDocuments/HDP2/HDP-2.6.4/bk_data-access/content/hive-jdbc-odbc-drivers.html)
* For Cloudera, install the Hive JDBC package for your operating system, and then add ``/usr/lib/hive/lib/hive-jdbc-<version>-standalone.jar`` to your classpath. More information is available in [Cloudera Documentation](https://www.cloudera.com/documentation/enterprise/5-3-x/topics/cdh_ig_hive_jdbc_install.html)
* You can also retrieve this from Maven for the desire version using ``mvn dependency:get -Dartifact=groupId:artifactId:version``.

### Provide H2O with the JDBC Driver

Based on the setup you can:

* Add the Hive JDBC driver to H2O's classpath for running stand-alone H2O from terminal: 
   `java -cp hive-jdbc-standalone.jar:<path_to_h2o_jar> water.H2OApp`  

* Init from python:
   `h2o.init(extra_classpath = ["hive-jdbc-standalone.jar"])`

* Init from R
   `h2o.init(extra_classpath=["hive-jdbc-standalone.jar"])`

* Add the Hive JDBC driver to H2O's classpath for running clustered H2O on Hadoop from terminal: 
   `hadoop jar h2odriver.jar -libjars hive-jdbc-standalone.jar -nodes 3 -mapperXmx 6g`
After the jar file with JDBC driver is added, then data from the Hive databases can be pulled into H2O using the ``import_sql_table`` and ``import_sql_select`` functions. 

## Putting it all together
You need to define the data that would be used as well as the credentials and the connection url:

```
# python
import h2o
connection_url = "jdbc:hive2://localhost:10000/default"
select_query = "SELECT * FROM AirlinesTest;"
username = "root"
password = "changeit"

# Load the dataset
airlines_dataset = h2o.import_sql_select(connection_url, select_query, username, password)

# And then utilize it
airlines_dataset["table_for_h2o_import.origin"] = airlines_dataset["table_for_h2o_import.origin"].asfactor()
airlines_dataset["table_for_h2o_import.fdayofweek"] = airlines_dataset["table_for_h2o_import.fdayofweek"].asfactor()
airlines_dataset["table_for_h2o_import.uniquecarrier"] = airlines_dataset["table_for_h2o_import.uniquecarrier"].asfactor()
airlines_dataset["table_for_h2o_import.dest"] = airlines_dataset["table_for_h2o_import.dest"].asfactor()
airlines_dataset["table_for_h2o_import.fyear"] = airlines_dataset["table_for_h2o_import.fyear"].asfactor()
airlines_dataset["table_for_h2o_import.fdayofmonth"] = airlines_dataset["table_for_h2o_import.fdayofmonth"].asfactor()
airlines_dataset["table_for_h2o_import.isdepdelayed"] = airlines_dataset["table_for_h2o_import.isdepdelayed"].asfactor()
airlines_dataset["table_for_h2o_import.fmonth"] = airlines_dataset["table_for_h2o_import.fmonth"].asfactor()

airlines_X_col_names = airlines_dataset.col_names[:-2]
airlines_y_col_name = airlines_dataset.col_names[-2]

train, valid, test = airlines_dataset.split_frame([0.6, 0.2], seed=1234)

from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm_v1 = H2OGradientBoostingEstimator(model_id="gbm_airlines_v1", seed=2000000)
gbm_v1.train(airlines_X_col_names, airlines_y_col_name, training_frame=train, validation_frame=valid)
gbm_v1.predict(test)
```
