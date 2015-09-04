## Introduction

This tutorial describes how to use a model created in H2O to create a Hive UDF (user-defined function) for scoring data.   While the fastest scoring typically results from ingesting data files in HDFS directly into H2O for scoring, there may be several motivations not to do so.  For example, the clusters used for model building may be research clusters, and the data to be scored may be on "production" clusters.  In other cases, the final data set to be scored may be too large to reasonably score in-memory.  To help with these kinds of cases, this document walks through how to take a scoring model from H2O, plug it into a template UDF project, and use it to score in Hive.  All the code needed for this walkthrough can be found in this repository branch.

##The Goal
The desired work flow for this task is:

1. Load training and test data into H2O
2. Create several models in H2O
3. Export the best model as a [POJO](https://en.wikipedia.org/wiki/Plain_Old_Java_Object)
4. Compile the H2O model as a part of the UDF project
5. Copy the UDF to the cluster and load into Hive
6. Score with your UDF

For steps 1-3, we will give instructions scoring the data through R.  We will add a step between 4 and 5 to load some test data for this example.

## Requirements ##

This tutorial assumes the following:

1. Some familiarity with using H2O in R.  Getting started tutorials can be found [here](http://docs.0xdata.com/newuser/top.html).
2. The ability to compile Java code.  The repository provides a pom.xml file, so using Maven will be the simplest way to compile, but IntelliJ IDEA will also read in this file.  If another build system is preferred, it is left to the reader to figure out the compilation details.
3. A working Hive install to test the results.

## The Data

For this post, we will be using a 0.1% sample of the Person-Level 2013 Public Use Microdata Sample (PUMS) from United States Census Bureau.  75% of that sample is designated as the training data set and 25% as the test data set. This data set is intended as an update to the [UCI Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult).  The two datasets are available [here](https://h2o-training.s3.amazonaws.com/pums2013/adult_2013_train.csv.gz) and [here](https://h2o-training.s3.amazonaws.com/pums2013/adult_2013_test.csv.gz).

The goal of the analysis in this demo is to predict if an income exceeds $50K/yr based on census data.  The columns we will be using are:

* AGEP:  age
* COW: class of worker
* SCHL: educational attainment
* MAR: marital status
* INDP: Industry code
* RELP: relationship
* RAC1P: race
* SEX: gender
* WKHP: hours worked per week
* POBP: Place of birth code
* LOG_CAPGAIN: log of capital gains
* LOG_CAPLOSS: log of capital losses
* LOG_WAGP: log of wages or salary


## Building the Model in R
No need to cut and paste code: the complete R script described below is part of this git repository (GBM-example.R).
### Load the training and test data into H2O
Since we are playing with a small data set for this example, we will start H2O locally and load the datasets:

```r
> myIP <- "localhost"
> myPort <- 54321

> library(h2o)
> h2oServer <- h2o.init(ip = myIP, port = myPort, startH2O = TRUE)
```

Load the datasets (change the directory to reflect where you stored these files):

```r
> pumsdir <- file.path("/Users/myhomedir/data/pums2013")
> trainfile <- "adult_2013_train.csv.gz"
> testfile  <- "adult_2013_test.csv.gz"

> adult_2013_train <- h2o.importFile(h2oServer,
                                   path = file.path(pumsdir, trainfile),
                                   key = "adult_2013_train", sep = ",")

> adult_2013_test <- h2o.importFile(h2oServer,
                                  path = file.path(pumsdir, testfile),
                                  key = "adult_2013_test", sep = ",")
```
Looking at the data, we can see that 8 columns are using integer codes to represent different categorical levels.  Let's tell H2O to treat those columns as factors.

```r
> for (j in c("COW", "SCHL", "MAR", "INDP", "RELP", "RAC1P", "SEX", "POBP")) {
  adult_2013_train[[j]] <- as.factor(adult_2013_train[[j]])
  adult_2013_test[[j]]  <- as.factor(adult_2013_test[[j]])
}
```

### Creating several models in H2O
Now that the data has been prepared, let's build a set of models using [GBM](http://h2o-release.s3.amazonaws.com/h2o/rel-simons/7/docs-website/h2o-docs/index.html#Data%20Science%20Algorithms-GBM).  Here we will select the columns used as predictors and results, specify the validation data set, and then build a model.

```r
> predset <- c(("RELP", "SCHL", "COW", "MAR", "INDP", "RAC1P", "SEX", "POBP", "AGEP", "WKHP", "LOG_CAPGAIN", "LOG_CAPLOSS")

> log_wagp_gbm_grid <- h2o.gbm(x = predset),
                             y = "LOG_WAGP",
                             training_frame = adult_2013_train,
                             model_id  = "GBMModel",
                             distribution = "gaussian",
                             max_depth = 5,
                             ntrees = 110,
                             learning_rate = 0.25,
                             validation_frame = adult_2013_test,
                             importance = TRUE)
> log_wagp_gbm

Model Details:
==============

H2ORegressionModel: gbm
Model ID:  GBMModel 
Model Summary:
  number_of_trees model_size_in_bytes min_depth max_depth mean_depth min_leaves max_leaves mean_leaves
1      110.000000       111698.000000  5.000000  5.000000    5.00000  14.000000  32.000000    27.93636


H2ORegressionMetrics: gbm
** Reported on training data. **

MSE:  0.4626122
R2 :  0.7362828
Mean Residual Deviance :  0.4626122


H2ORegressionMetrics: gbm
** Reported on validation data. **

MSE:  0.6605266
R2 :  0.6290677
Mean Residual Deviance :  0.6605266
```


###Export the best model as a POJO
From here, we can download this model as a Java [POJO](https://en.wikipedia.org/wiki/Plain_Old_Java_Object) to a local directory called `generated_model`.

```r
> tmpdir_name <- "generated_model"
> dir.create(tmpdir_name)
> h2o.download_pojo(log_wagp_gbm, tmpdir_name)
[1] "POJO written to: generated_model/GBMModel.java"
```
At this point, the Java POJO is available for scoring data outside of H2O.  As the last step in R, let's take a look at the scores this model gives on the test data set. We will use these to confirm the results in Hive.

```r
> h2o.predict(log_wagp_gbm, adult_2013_test)
H2OFrame with 37345 rows and 1 column

First 10 rows:
     predict
1  10.432787
2  10.244159
3  10.432688
4   9.604912
5  10.285979
6  10.356251
7  10.261413
8  10.046026
9  10.766078
10  9.502004
```

##Compile the H2O model as a part of UDF project

All code for this section can be found in this git repository.  To simplify the build process, I have included a pom.xml file.  For Maven users, this will automatically grab the dependencies you need to compile.

To use the template:

1. Copy the Java from H2O into the project
2. Update the POJO to be part of the UDF package
3. Update the pom.xml to reflect your version of Hadoop and Hive
4. Compile

### Copy the java from H2O into the project

```bash
$ cp generated_model/h2o-genmodel.jar localjars
$ cp generated_model/GBMModel.java src/main/java/ai/h2o/hive/udf/GBMModel.java
```

### Update the POJO to Be a Part of the Same Package as the UDF ###

To the top of `GBMModel.java`, add:

```Java
package ai.h2o.hive.udf;
```

### Update the pom.xml to Reflect Hadoop and Hive Versions ###

Get your version numbers using:

```bash
$ hadoop version
$ hive --version
```

And plug these into the `<properties>`  section of the `pom.xml` file.  Currently, the configuration is set for pulling the necessary dependencies for Hortonworks.  For other Hadoop distributions, you will also need to update the `<repositories>` section to reflect the respective repositories (a commented-out link to a Cloudera repository is included).

###Compile

> Caution:  This tutorial was written using Maven 3.0.4.  Older 2.x versions of Maven may not work.

```bash
$ mvn compile
$ mvn package
```

As with most Maven builds, the first run will probably seem like it is downloading the entire Internet.  It is just grabbing the needed compile dependencies.  In the end, this process should create the file `target/ScoreData-1.0-SNAPSHOT.jar`.

As a part of the build process, Maven is running a unit test on the code. If you are looking to use this template for your own models, you either need to modify the test to reflect your own data, or run Maven without the test (`mvn package -Dmaven.test.skip=true`).  

## Loading test data in Hive
Now load the same test data set into Hive.  This will allow us to score the data in Hive and verify that the results are the same as what we saw in H2O.

```bash
$ hadoop fs -mkdir hdfs://my-name-node:/user/myhomedir/UDFtest
$ hadoop fs -put adult_2013_test.csv.gz  hdfs://my-name-node:/user/myhomedir/UDFtest/.
$ hive
```

Here we mark the table as `EXTERNAL` so that Hive doesn't make a copy of the file needlessly.  We also tell Hive to ignore the first line, since it contains the column names.

```hive
> CREATE EXTERNAL TABLE adult_data_set (AGEP INT, COW STRING, SCHL STRING, MAR STRING, INDP STRING, RELP STRING, RAC1P STRING, SEX STRING, WKHP INT, POBP STRING, WAGP INT, CAPGAIN INT, CAPLOSS INT, LOG_CAPGAIN DOUBLE, LOG_CAPLOSS DOUBLE, LOG_WAGP DOUBLE, CENT_WAGP STRING, TOP_WAG2P INT, RELP_SCHL STRING) COMMENT 'PUMS 2013 test data' ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE location '/user/myhomedir/UDFtest' tblproperties ("skip.header.line.count"="1");
> ANALYZE TABLE adult_data_set COMPUTE STATISTICS;
```


##Copy the UDF to the cluster and load into Hive
```bash
$ hadoop fs -put localjars/h2o-genmodel.jar  hdfs://my-name-node:/user/myhomedir/
$ hadoop fs -put target/ScoreData-1.0-SNAPSHOT.jar  hdfs://my-name-node:/user/myhomedir/
$ hive
```
Note that for correct class loading, you will need to load the h2o-model.jar before the ScoredData jar file.

```hive
> ADD JAR h2o-genmodel.jar;
> ADD JAR ScoreData-1.0-SNAPSHOT.jar;
> CREATE TEMPORARY FUNCTION scoredata AS 'ai.h2o.hive.udf.ScoreDataUDF';
```

Keep in mind that your UDF is only loaded in Hive for as long as you are using it.  If you `quit;` and then join Hive again, you will have to re-enter the last three lines.

##Score with your UDF
Now the moment we've been working towards:

```r
 hive> SELECT scoredata(AGEP, COW, SCHL, MAR, INDP, RELP, RAC1P, SEX, WKHP, POBP, LOG_CAPGAIN, LOG_CAPLOSS) FROM adult_data_set LIMIT 10;
OK
10.476669
10.201586
10.463915
9.709603
10.175115
10.3576145
10.256757
10.050725
10.759903
9.316141
Time taken: 0.063 seconds, Fetched: 10 row(s)
```


<a name="Limitations"></a>
##Limitations

This solution is fairly quick and easy to implement.  Once you've run through things once, going through steps 1-5 should be pretty painless.  There are, however, a few things to be desired here.

The major trade-off made in this template has been a more generic design over strong input checking.   To be applicable for any POJO, the code only checks that the user-supplied arguments have the correct count and they are all at least primitive types.  Stronger type checking could be done by generating Hive UDF code on a per-model basis.

Also, while the template isn't specific to any given model, it isn't completely flexible to the incoming data either.  If you used 12 of 19 fields as predictors (as in this example), then you must feed the scoredata() UDF only those 12 fields, and in the order that the POJO expects. This is fine for a small number of predictors, but can be messy for larger numbers of predictors.  Ideally, it would be nicer to say `SELECT scoredata(*) FROM adult_data_set;` and let the UDF pick out the relevant fields by name.  While the H2O POJO does have utility functions for this, Hive, on the other hand, doesn't provide UDF writers the names of the fields (as mentioned in [this](https://issues.apache.org/jira/browse/HIVE-3491) Hive feature request) from which the arguments originate.

Finally, as written, the UDF only returns a single prediction value.  The H2O POJO actually returns an array of float values.  The first value is the main prediction and the remaining values hold probability distributions for classifiers.  This code can easily be expanded to return all values if desired.

## A Look at the UDF Template

The template code starts with some basic annotations that define the nature of the UDF and display some simple help output when the user types `DESCRIBE scoredata` or `DESCRIBE EXTENDED scoredata`.

```Java
@UDFType(deterministic = true, stateful = false)
@Description(name="scoredata", value="_FUNC_(*) - Returns a score for the given row",
        extended="Example:\n"+"> SELECT scoredata(*) FROM target_data;")
```

Rather than extend the plain UDF class, this template extends GenericUDF.  The plain UDF requires that you hard code each of your input variables.  This is fine for most UDFs, but for a function like scoring the number of columns used in scoring may be large enough to make this cumbersome.   Note the declaration of an array to hold ObjectInspectors for each argument, as well as the instantiation of the model POJO.

```Java
class ScoreDataUDF extends GenericUDF {
  private PrimitiveObjectInspector[] inFieldOI;
  GBMModel p = new GBMModel();

  @Override
  public String getDisplayString(String[] args) {
    return "scoredata("+Arrays.asList(p.getNames())+").";
  }
```

All GenericUDF children must implement initialize() and evaluate().  In initialize(), we see very basic argument type checking, initialization of ObjectInspectors for each argument, and declaration of the return type for this UDF.  The accepted primitive type list here could easily be expanded if needed. BOOLEAN, CHAR, VARCHAR, and possibly TIMESTAMP and DATE might be useful to add.

```Java
  @Override
    public ObjectInspector initialize(ObjectInspector[] args) throws UDFArgumentException {
    // Basic argument count check
    // Expects one less argument than model used; results column is dropped
    if (args.length != p.getNumCols()) {
      throw new UDFArgumentLengthException("Incorrect number of arguments." +
              "  scoredata() requires: "+ Arrays.asList(p.getNames())
              +", in the listed order. Received "+args.length+" arguments.");
    }

    //Check input types
    inFieldOI = new PrimitiveObjectInspector[args.length];
    PrimitiveObjectInspector.PrimitiveCategory pCat;
    for (int i = 0; i < args.length; i++) {
      if (args[i].getCategory() != ObjectInspector.Category.PRIMITIVE)
        throw new UDFArgumentException("scoredata(...): Only takes primitive field types as parameters");
      pCat = ((PrimitiveObjectInspector) args[i]).getPrimitiveCategory();
      if (pCat != PrimitiveObjectInspector.PrimitiveCategory.STRING
              && pCat != PrimitiveObjectInspector.PrimitiveCategory.DOUBLE
              && pCat != PrimitiveObjectInspector.PrimitiveCategory.FLOAT
              && pCat != PrimitiveObjectInspector.PrimitiveCategory.LONG
              && pCat != PrimitiveObjectInspector.PrimitiveCategory.INT
              && pCat != PrimitiveObjectInspector.PrimitiveCategory.SHORT)
        throw new UDFArgumentException("scoredata(...): Cannot accept type: " + pCat.toString());
      inFieldOI[i] = (PrimitiveObjectInspector) args[i];
    }

    // the return type of our function is a double, so we provide the correct object inspector
    return PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
  }
```

The real work is done in the evaluate() method.  Again, some quick sanity checks are made on the arguments, then each argument is converted to a double.  All H2O models take an array of doubles as their input.  For integers, a simple casting is enough.  For strings/enumerations, the double quotes are stripped, then the enumeration value for the given string/field index is retrieved, and then it is cast to a double.   Once all the arguments have been made into doubles, the model's predict() method is called to get a score.  The main prediction for this row is then returned.

```Java
  @Override
    public Object evaluate(DeferredObject[] record) throws HiveException {
    // Expects one less argument than model used; results column is dropped
    if (record != null) {
      if (record.length == p.getNumCols()) {
        double[] data = new double[record.length];
        //Sadly, HIVE UDF doesn't currently make the field name available.
        //Thus this UDF must depend solely on the arguments maintaining the same
        // field order seen by the original H2O model creation.
        for (int i = 0; i < record.length; i++) {
          try {
            Object o = inFieldOI[i].getPrimitiveJavaObject(record[i].get());
            if (o instanceof java.lang.String) {
              // Hive wraps strings in double quotes, remove
              data[i] = p.mapEnum(i, ((String) o).replace("\"", ""));
              if (data[i] == -1)
                throw new UDFArgumentException("scoredata(...): The value " + (String) o
                    + " is not a known category for column " + p.getNames()[i]);
            } else if (o instanceof Double) {
              data[i] = ((Double) o).doubleValue();
            } else if (o instanceof Float) {
              data[i] = ((Float) o).doubleValue();
            } else if (o instanceof Long) {
              data[i] = ((Long) o).doubleValue();
            } else if (o instanceof Integer) {
              data[i] = ((Integer) o).doubleValue();
            } else if (o instanceof Short) {
              data[i] = ((Short) o).doubleValue();
            } else if (o == null) {
              return null;
            } else {
              throw new UDFArgumentException("scoredata(...): Cannot accept type: "
                  + o.getClass().toString() + " for argument # " + i + ".");
            }
          } catch (Throwable e) {
            throw new UDFArgumentException("Unexpected exception on argument # " + i + ". " + e.toString());
          }
        }
        // get the predictions
        try {
          double[] preds = new double[p.getPredsSize()];
          p.score0(data, preds);
          return preds[0];
        } catch (Throwable e) {
          throw new UDFArgumentException("H2O predict function threw exception: " + e.toString());
        }
      } else {
        throw new UDFArgumentException("Incorrect number of arguments." +
            "  scoredata() requires: " + Arrays.asList(p.getNames()) + ", in order. Received "
            +record.length+" arguments.");
      }
    } else { // record == null
      return null; //throw new UDFArgumentException("scoredata() received a NULL row.");
    }
  }
```

Really, almost all the work is in type detection and conversion.

## Summary

That's it.  The given template should work for most cases.  As mentioned in the [limitations](#Limitations) section, two major modifications could be done.  Some users may desire handling for a few more primitive types.  Other users might want stricter type checking.  There are two options for the latter: either use the template as the basis for auto-generating type checking UDF code on a per model basis, or create a Hive client application and call the UDF from the client.  A Hive client could handle type checking and field alignment, since it would both see the table level information and invoke the UDF.
