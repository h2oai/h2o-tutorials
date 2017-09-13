## Overview

The code in this repo allows you to group H2O MOJO models together into a Hive UDF for scoring on Hive tables.

Supports:
- Different model types
- Different predictors

## Requirements

- Clone the h2o-tutorials repository and navigate to this folder:
```
git clone https://github.com/h2oai/h2o-tutorials.git
cd h2o-tutorials
cd tutorials
cd hive_udf_template
cd hive_multimojo_udf_template
```
- Hadoop/Hive
- Java & Maven (to compile & package UDF jar)
- R with h2o (if using the R script)

## Workflow

### 1. Run R Script
```
R < multimojo.R --no-save
cp -r generated_models/. src/main/resources/models/
```

This runs the R script which will build five GBM models and five Random Forest models on our adult data set. This is then copied into the resources directory (required). This will also show predictions using h2o which we can compare later to validate results.

Note: The MOJOs are included in the repo if you wish to skip running the R script section (but then you won't be able to see predictions)

### 2. Cleanup Unnecessary Folders
```
rm -rf generated_models/
rm -rf pums2013/
```

### 3. Download H2O MOJOs & Dependency JAR
- Download h2o-genmodel.jar
- Create a folder named localjars/ in your root folder
  - ```mkdir localjars```
- Place h2o-genmodel.jar into localjars/

### 4. Modify pom.xml as Needed
Change artifactId = [argument] to the name of your function, in this case it is called ScoreData

### 5. Compile & Package UDF JAR
From the root directory of this project, run the following build commands:
```
mvn clean
mvn compile
mvn package -Dmaven.test.skip=true
java -cp ./localjars/h2o-genmodel.jar:./target/ScoreData-1.0-SNAPSHOT.jar ai.h2o.hive.udf.ScoreDataHQLGenerator > ScoreData.hql
```
This cleans any current builds, compiles & packages (skipping tests), & runs ScoreDataHQLGenerator -- which outputs the HQL that should be run in Hive. Then run `source ScoreData.hql` in Hive to apply (**check to make sure the paths to the two JARS are correct!**)

Upload the localjars/h2o-genmodel.jar & target/MyModels-1.0-SNAPSHOT.jar somewhere you can access from Hive. You can keep it on the local filesystem or put it on the Hadoop FS - either way will work as long as you keep in mind the paths when running "ADD JAR ..."

### 5. HQL Overview
The above command will generate HQL similar to below.

```
-- add jars
ADD JAR localjars/h2o-genmodel.jar;
ADD JAR target/ScoreData-1.0-SNAPSHOT.jar;

-- create fn definition
CREATE TEMPORARY FUNCTION fn AS "ai.h2o.hive.udf.MojoUDF";

-- column names reference
set hivevar:colnames=AGEP,COW,SCHL,MAR,INDP,RELP,RAC1P,SEX,WKHP,POBP,LOG_CAPGAIN,LOG_CAPLOSS;

-- sample query, returns nested array
-- select fn(${colnames}) from adult_data_set

```

TABLEWITHAPPROPRIATEDATA MUST have **ALL** columns required by the models. If they are not -- the UDF will fail to score!!

### 6. Lets Try This in Hive!

Get the data into hive first.

```
$ hadoop fs -mkdir hdfs://my-name-node:/user/myhomedir/UDFtest
$ hadoop fs -put adult_2013_test.csv.gz  hdfs://my-name-node:/user/myhomedir/UDFtest/.
$ hive
```

Now that we are in hive, lets source our hql file.

```
source ScoreData.hql;
```

Now, lets run a query on this data using the following command:

```
SELECT fn(${colnames}) FROM adult_data_set limit 5;
```
