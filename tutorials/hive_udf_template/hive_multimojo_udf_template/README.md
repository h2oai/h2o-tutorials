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

Cleanup unnecessary folders

```
rm -rf generated_models/
rm -rf pums2013/
```

### 2. Download H2O MOJOs & Dependency JAR
- Download h2o-genmodel.jar
- Create a folder named localjars/ in your root folder
  - ```mkdir localjars```
- Place h2o-genmodel.jar into localjars/

### 3. Modify pom.xml as Needed
Change artifactId = [argument] to the name of your function, in this case it is called ScoreData

### 4. Compile & Package UDF JAR
From the root directory of this project, run the following build commands:
```
mvn clean
mvn compile
mvn package -Dmaven.test.skip=true
java -cp ./localjars/h2o-genmodel.jar:./target/ScoreData-1.0-SNAPSHOT.jar ai.h2o.hive.udf.ScoreDataHQLGenerator > ScoreData.hql
```
This cleans any current builds, compiles & packages (skipping tests), & runs ScoreDataHQLGenerator -- which outputs the HQL that should be run in Hive.

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

TABLE BEING USED MUST have **ALL** columns required by the models. If they are not -- the UDF will fail to score!!

### 6. Lets Try This in Hive!

Get the data into hive first.

```
$ hadoop fs -mkdir hdfs://my-name-node:/user/myhomedir/UDFtest
$ hadoop fs -put adult_2013_test.csv.gz  hdfs://my-name-node:/user/myhomedir/UDFtest/.
$ hive
```

Now that we are in hive, lets create our table:

```
CREATE EXTERNAL TABLE adult_data_set (AGEP INT, COW STRING, SCHL STRING, MAR STRING, INDP STRING, RELP STRING, RAC1P STRING, SEX STRING, WKHP INT, POBP STRING, LOG_CAPGAIN DOUBLE, LOG_CAPLOSS DOUBLE) COMMENT 'PUMS 2013 test data' ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE location '/user/myhomedir/UDFtest' tblproperties ("skip.header.line.count"="1");
```

Now lets source our hql file.

```
source ScoreData.hql;
```

Now, lets run a query on this data using the following command:

```
SELECT fn(${colnames}) FROM adult_data_set limit 5;
```

Output should be similar to the following:
```
[11.303953296359124,11.057000205731766,11.021330896004384,10.978755236432063,11.07756025198769,10.592100446874445,10.692826612790425,10.671051590259259,10.718797093346005,10.625670267188031]
[10.905440697982568,10.075020095715173,10.327614423276556,9.823622201550917,10.319411297159332,10.028055143356323,10.238343759377797,10.100672340393066,10.1159210840861,9.881835547737452]
[11.302769005385654,11.053856151814854,11.01137647276819,10.978755236432063,11.08063776878562,10.592100446874445,10.692826612790425,10.671051590259259,10.718797093346005,10.625670267188031]
[10.986364208685428,10.864133171151208,10.666882435961908,10.956532157378843,10.5719872002971,10.566539564999667,10.529270648956299,10.564353135915903,10.544568116324289,10.585966674141261]
[11.274469472771171,11.045313182494011,10.98560974605398,10.856955215687504,11.082147184997394,10.592100446874445,10.68827363650004,10.671051590259259,10.704244086855933,10.625670267188031]
```

Compare this with our previous R script to validate results.

Final Notes: This example is for a regression problem. User will have to make changes based on what type of problem they are solving (Regression, Binomial, Multinomial etc). These changes would be implemented in the ```scoreAll()``` method of ```ModelGroup.java```.
