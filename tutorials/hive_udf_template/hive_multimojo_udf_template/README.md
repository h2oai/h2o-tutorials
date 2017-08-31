## Overview

The code in this repo allows you to group H2O MOJO models together into a Hive UDF for scoring on Hive tables.

Supports:
- Different model types
- Different predictors

## Requirements

- Clone the h2o-tutorials repository and navigate to this folder:
```
git clone https://github.com/h2oai/h2o-tutorials/tree/master/tutorials
cd h2o-tutorials
cd tutorials
cd hive_udf_template
cd hive_multimojo_udf_template
```
- Hadoop/Hive
- Java & Maven (to compile & package UDF jar)

## Workflow

### 1. Train H2O Models
Train your H2O models as you would normally using the WebUI (Flow), or Python/R client APIs

### 2. Download H2O MOJOs & Dependency JAR
- Download the MOJOs and the h2o-genmodel.jar dependency
- Place h2o-genmodel.jar into localjars/
- Place the H2O MOJOs into src/main/resources/models

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
This cleans any current builds, compiles & packages (skipping tests), & runs ScoreDataHQLGenerator -- which outputs the HQL that should be run in Hive. Then run `source ScoreData.hql` in Hive to apply (**check to make sure the paths to the two JARS are correct!**)

Upload the localjars/h2o-genmodel.jar & target/MyModels-1.0-SNAPSHOT.jar somewhere you can access from Hive. You can keep it on the local filesystem or put it on the Hadoop FS - either way will work as long as you keep in mind the paths when running "ADD JAR ..."

### 5. Scoring in Hive
The above command will generate HQL similar to below.

```
-- model order (alphabetical)
-- Name: ai.h2o.hive.udf.models.deeplearning_741ae095_5cc4_415c_a726_f4e26762f3fa
--   Category: Regression
--   Hive Select: scores[0][0 - 1]
-- Name: ai.h2o.hive.udf.models.drf_99268734_ddf8_43b3_87e4_84f092df5292
--   Category: Regression
--   Hive Select: scores[1][0 - 1]
-- Name: ai.h2o.hive.udf.models.gbm_094fceb2_48a2_4447_931b_2aeed114c08a
--   Category: Regression
--   Hive Select: scores[2][0 - 1]
-- Name: ai.h2o.hive.udf.models.glm_a00273fc_04a0_4c0c_b5a6_22fba753ca1b
--   Category: Regression
--   Hive Select: scores[3][0 - 1]

-- add jars
ADD JAR localjars/h2o-genmodel.jar;
ADD JAR target/ScoreData-1.0-SNAPSHOT.jar;

-- create fn definition
CREATE TEMPORARY FUNCTION fn AS "ai.h2o.hive.udf.ScoreDataUDF";

-- column names reference
set hivevar:scoredatacolnames=C1,C2,C3,C4,C5,C6,C7,C8

-- sample query, returns nested array
-- select fn(${scoredatacolnames}) from TABLEWITHAPPROPRIATEDATA
```

TABLEWITHAPPROPRIATEDATA MUST have **ALL** columns required by the models. If they are not -- the UDF will fail to score!!
