# Direct Hive import into H2O

H2O can ingest data directly from Hive tables when running on Hadoop.

## Motivation

Reading data directly from Hive tables on HDFS provides an alternative to JDBC, which may be slower (especially on Hive 1.x and older, which do not support parallel JDBC import).

## Prerequisities

* A running Hive instance
* Data in a Hive table

### Start H2O with the Hive Metastore Client on Classpath

Starting H2O with the Hive Metastore client varies based on your setup:

* If your Hadoop is configured with Hive configuration and client jars on classpath (i.e. via `yarn.application.classpath` property), you can start an H2O cluster as you normally would:
   
	 `hadoop jar h2odriver.jar -nodes 3 -mapperXmx 6g`

* Otherwise you will need to provide Hive metastore classes and configuraion to the h2odriver:

	 ```
     HIVE_CP=$(find /usr/hdp/current/hive-client/lib/ -type f | grep jar | tr '\n' ',')
	 HIVE_CP=/etc/hive/conf/hive-site.xml,$HIVE_JARS
	 hadoop jar h2odriver.jar -libjars $HIVE_CP -nodes 3 -mapperXmx 4g
	 ```

After H2O is running on Hadoop, data from the Hive databases can be pulled into H2O using the ``import_hive_table` function. 

## Putting it all together

```
# python
import h2o

h2o.connect(ip="localhost", port=54330, https=True, verify_ssl_certificates=False)

# Load the dataset
airlines_dataset = h2o.import_hive_table("default", "airlinestest")

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
