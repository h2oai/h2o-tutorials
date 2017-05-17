/**
 * Launch following commands:
 *   export SPARK_HOME="/path/to/spark/installation" 
 *   export SPARK_LOCAL_IP="127.0.0.1"
 *   export MASTER="local[*]"
 *   bin/sparkling-shell --conf "spark.executor.memory=1g"
 *
  * When running using spark shell or using scala rest API:
  *    SQLContext is available as sqlContext
  *     - if you want to use sqlContext implicitly, you have to redefine it like: implicit val sqlContext = sqlContext,
  *      but better is to use it like this: implicit val sqlContext = SQLContext.getOrCreate(sc)
  *    SparkContext is available as sc
  */
// Create an environment
import _root_.hex.genmodel.utils.DistributionFamily
import _root_.hex.deeplearning.DeepLearningModel
import _root_.hex.tree.gbm.GBMModel
import _root_.hex.{Model, ModelMetricsBinomial}
import org.apache.spark.SparkFiles
import org.apache.spark.examples.h2o.{Crime, RefineDateColumn}
import org.apache.spark.h2o._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import water.fvec.{H2OFrame, Vec}
import water.parser.ParseSetup
import water.support.{H2OFrameSupport, ModelMetricsSupport, SparkContextSupport}
import water.support.H2OFrameSupport._

// Create SQL support
implicit val sqlContext = spark.sqlContext

// Start H2O services
implicit val h2oContext = H2OContext.getOrCreate(sc)
import h2oContext._
import h2oContext.implicits._


//
// Add files to Spark Cluster
//
SparkContextSupport.addFiles(sc,
  "examples/smalldata/chicagoAllWeather.csv",
  "examples/smalldata/chicagoCensus.csv",
  "examples/smalldata/chicagoCrimes10k.csv"
)


//
// H2O Data loader using H2O API
//
def loadData(datafile: String, modifyParserSetup: ParseSetup => ParseSetup = identity[ParseSetup]): H2OFrame = {
  val uri = java.net.URI.create(datafile)
  val parseSetup = modifyParserSetup(water.fvec.H2OFrame.parserSetup(uri))
  new H2OFrame(parseSetup, new java.net.URI(datafile))
}

//
// Loader for weather data
//
def createWeatherTable(datafile: String): H2OFrame = {
  val table = loadData(datafile)
  withLockAndUpdate(table){
    // Remove first column since we do not need it
    _.remove(0).remove()
  }
}

// Load weather data
val weatherTable = asDataFrame(createWeatherTable(SparkFiles.get("chicagoAllWeather.csv")))(sqlContext)
weatherTable.show()

//
// Loader for census data
//
def createCensusTable(datafile: String): H2OFrame = {
  val table = loadData(datafile)
  // Rename columns: replace ' ' by '_'
  val colNames = table.names().map( n => n.trim.replace(' ', '_').replace('+','_'))
  withLockAndUpdate(table){_._names = colNames}
}


// Load census data
val censusTable = asDataFrame(createCensusTable(SparkFiles.get("chicagoCensus.csv")))(sqlContext)
censusTable.show()

//
// Load and modify crime data
//
def createCrimeTable(datafile: String, datePattern:String, dateTimeZone:String): H2OFrame = {
  val table = loadData(datafile, (parseSetup: ParseSetup) => {
    val colNames = parseSetup.getColumnNames
    val typeNames = parseSetup.getColumnTypes
    colNames.indices.foreach { idx =>
      if (colNames(idx) == "Date") typeNames(idx) = Vec.T_STR
    }
    parseSetup
  })

  withLockAndUpdate(table){ fr =>
    // Refine date into multiple columns
    val dateCol = table.vec(2)
    fr.add(new RefineDateColumn(datePattern, dateTimeZone).doIt(dateCol))
    // Update names, replace all ' ' by '_'
    val colNames = fr.names().map( n => n.trim.replace(' ', '_'))
    fr._names = colNames
    // Remove Date column
    fr.remove(2).remove()
  }
}


// Load crime table
val crimeTable  = asDataFrame(createCrimeTable(SparkFiles.get("chicagoCrimes10k.csv"), "MM/dd/yyyy hh:mm:ss a", "Etc/UTC"))(sqlContext)
crimeTable.show()

//
// Use Spark SQL to join datasets
//


// Register DataFrames as tables in SQL context
weatherTable.createOrReplaceTempView("chicagoWeather")
censusTable.createOrReplaceTempView("chicagoCensus")
crimeTable.createOrReplaceTempView("chicagoCrime")

//
// Join crime data with weather and census tables
//
val crimeWeather = sqlContext.sql(
  """SELECT
    |a.Year, a.Month, a.Day, a.WeekNum, a.HourOfDay, a.Weekend, a.Season, a.WeekDay,
    |a.IUCR, a.Primary_Type, a.Location_Description, a.Community_Area, a.District,
    |a.Arrest, a.Domestic, a.Beat, a.Ward, a.FBI_Code,
    |b.minTemp, b.maxTemp, b.meanTemp,
    |c.PERCENT_AGED_UNDER_18_OR_OVER_64, c.PER_CAPITA_INCOME, c.HARDSHIP_INDEX,
    |c.PERCENT_OF_HOUSING_CROWDED, c.PERCENT_HOUSEHOLDS_BELOW_POVERTY,
    |c.PERCENT_AGED_16__UNEMPLOYED, c.PERCENT_AGED_25__WITHOUT_HIGH_SCHOOL_DIPLOMA
    |FROM chicagoCrime a
    |JOIN chicagoWeather b
    |ON a.Year = b.year AND a.Month = b.month AND a.Day = b.day
    |JOIN chicagoCensus c
    |ON a.Community_Area = c.Community_Area_Number""".stripMargin)

crimeWeather.show()


//
// Publish Spark DataFrame as H2O Frame with given name
crimeWeather.printSchema()
val crimeWeatherDF:H2OFrame = crimeWeather
// Transform all string columns into categorical
withLockAndUpdate(crimeWeatherDF){ allStringVecToCategorical(_) }
// Transform weekday to categorical
withLockAndUpdate(crimeWeatherDF){ fr => fr.replace(fr.find("WeekDay"), fr.vec("WeekDay").toCategoricalVec)}

//
// Split final data table
//
val keys = Array[String]("train.hex", "test.hex")
val ratios = Array[Double](0.8, 0.2)
val frs = H2OFrameSupport.splitFrame(crimeWeatherDF, keys, ratios)
val (train, test) = (frs(0), frs(1))

//
// Show results
//
openFlow

//
// Build GBM model
//

def GBMModel(train: H2OFrame, test: H2OFrame, response: String,
             ntrees:Int = 50, depth:Int = 3, distribution: DistributionFamily = DistributionFamily.bernoulli)
            (implicit h2oContext: H2OContext) : GBMModel = {
  import h2oContext.implicits._
  import _root_.hex.tree.gbm.GBM
  import _root_.hex.tree.gbm.GBMModel.GBMParameters

  val gbmParams = new GBMParameters()
  gbmParams._train = train
  gbmParams._valid = test
  gbmParams._response_column = response
  gbmParams._ntrees = ntrees
  gbmParams._max_depth = depth
  gbmParams._distribution = distribution

  val gbm = new GBM(gbmParams)
  val model = gbm.trainModel.get
  model
}

val gbmModel = GBMModel(train, test, 'Arrest)



//
// Build Deep Learning model
//

def DLModel(train: H2OFrame, test: H2OFrame, response: String)
           (implicit h2oContext: H2OContext) : DeepLearningModel = {
  import _root_.hex.deeplearning.DeepLearning
  import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters

  val dlParams = new DeepLearningParameters()
  dlParams._train = train
  dlParams._valid = test
  dlParams._response_column = response
  dlParams._variable_importances = true
  // Create a job
  val dl = new DeepLearning(dlParams)
  val model = dl.trainModel.get
  model
}

val dlModel = DLModel(train, test, 'Arrest)

// Collect model metrics
def binomialMetrics[M <: Model[M,P,O], P <: _root_.hex.Model.Parameters, O <: _root_.hex.Model.Output]
(model: Model[M,P,O], train: H2OFrame, test: H2OFrame):(ModelMetricsBinomial, ModelMetricsBinomial) = {
  import water.support.ModelMetricsSupport._
  (modelMetrics(model,train), modelMetrics(model, test))
}

val (trainMetricsGBM, testMetricsGBM) = binomialMetrics(gbmModel, train, test)
val (trainMetricsDL, testMetricsDL) = binomialMetrics(dlModel, train, test)

//
// Print Scores of GBM & Deep Learning
//
println(
  s"""Model performance:
     |  GBM:
     |    train AUC = ${trainMetricsGBM.auc}
      |    test  AUC = ${testMetricsGBM.auc}
      |  DL:
      |    train AUC = ${trainMetricsDL.auc}
      |    test  AUC = ${testMetricsDL.auc}
      """.stripMargin)

//
// Create a predictor
//
def scoreEvent(crime: Crime, model: Model[_,_,_], censusTable: DataFrame)
              (implicit sqlContext: SQLContext, h2oContext: H2OContext): Float = {
  import h2oContext.implicits._
  import sqlContext.implicits._
  // Create a single row table
  val srdd:DataFrame = sqlContext.sparkContext.parallelize(Seq(crime)).toDF()
  // Join table with census data
  val row: H2OFrame = censusTable.join(srdd).where('Community_Area === 'Community_Area_Number) //.printSchema

  withLockAndUpdate(row){ allStringVecToCategorical(_) }
  withLockAndUpdate(row){ fr => fr.replace(fr.find("WeekDay"), fr.vec("WeekDay").toCategoricalVec)}

  val predictTable = model.score(row)
  val probOfArrest = predictTable.vec("true").at(0)

  probOfArrest.toFloat
}

// Score some crimes

// Define crimes
val crimeExamples = Seq(
  Crime("02/08/2015 11:43:58 PM", 1811, "NARCOTICS", "STREET",false, 422, 4, 7, 46, 18),
  Crime("02/08/2015 11:00:39 PM", 1150, "DECEPTIVE PRACTICE", "RESIDENCE",false, 923, 9, 14, 63, 11))

// Score
for (crime <- crimeExamples) {
  val arrestProbGBM = 100 * scoreEvent(crime, gbmModel, censusTable)
  val arrestProbDL = 100 * scoreEvent(crime, dlModel, censusTable)
  println(
    s"""
       |Crime: $crime
       |  Probability of arrest based on DeepLearning: ${arrestProbDL} %
       |  Probability of arrest based on GBM: ${arrestProbGBM} %
        """.stripMargin)
}

// stop H2O and Spark services
h2oContext.stop(stopSparkContext = true)
