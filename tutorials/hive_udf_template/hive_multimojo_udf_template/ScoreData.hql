-- model order (alphabetical)
-- Name: GBMModel1.zip
--   Category: Regression
--   Hive Select: scores[0][0 - 1]
-- Name: GBMModel2.zip
--   Category: Regression
--   Hive Select: scores[1][0 - 1]
-- Name: GBMModel3.zip
--   Category: Regression
--   Hive Select: scores[2][0 - 1]
-- Name: GBMModel4.zip
--   Category: Regression
--   Hive Select: scores[3][0 - 1]
-- Name: GBMModel5.zip
--   Category: Regression
--   Hive Select: scores[4][0 - 1]
-- Name: RFModel1.zip
--   Category: Regression
--   Hive Select: scores[5][0 - 1]
-- Name: RFModel2.zip
--   Category: Regression
--   Hive Select: scores[6][0 - 1]
-- Name: RFModel3.zip
--   Category: Regression
--   Hive Select: scores[7][0 - 1]
-- Name: RFModel4.zip
--   Category: Regression
--   Hive Select: scores[8][0 - 1]
-- Name: RFModel5.zip
--   Category: Regression
--   Hive Select: scores[9][0 - 1]

-- add jars
ADD JAR localjars/h2o-genmodel.jar;
ADD JAR target/ScoreData-1.0-SNAPSHOT.jar;

-- create fn definition
CREATE TEMPORARY FUNCTION fn AS "ai.h2o.hive.udf.MojoUDF";

-- column names reference
set hivevar:colnames=AGEP,COW,SCHL,MAR,INDP,RELP,RAC1P,SEX,WKHP,POBP,LOG_CAPGAIN,LOG_CAPLOSS;

-- sample query, returns nested array
-- select fn(${colnames}) from adult_data_set
