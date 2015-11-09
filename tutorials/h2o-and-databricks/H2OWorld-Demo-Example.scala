// Databricks notebook source exported at Mon, 9 Nov 2015 04:31:10 UTC
// Ensure you have included the table smsData

// COMMAND ----------

// Representation of a training message
import org.apache.spark.mllib.linalg.Vector
case class SMS(target: String, fv: Vector)

// COMMAND ----------

// Define tokenizer function
def tokenize(data: RDD[String]): RDD[Seq[String]] = {
  val ignoredWords = Seq("the", "a", "", "in", "on", "at", "as", "not", "for")
  val ignoredChars = Seq(',', ':', ';', '/', '<', '>', '"', '.', '(', ')', '?', '-', '\'','!','0', '1')

  val texts = data.map( r=> {
    var smsText = r.toLowerCase
    for( c <- ignoredChars) {
      smsText = smsText.replace(c, ' ')
    }

    val words =smsText.split(" ").filter(w => !ignoredWords.contains(w) && w.length>2).distinct

    words.toSeq
  })
  texts
}

// COMMAND ----------

// Define function which builds an IDF model
import org.apache.spark.mllib.feature._

def buildIDFModel(tokens: RDD[Seq[String]],
                  minDocFreq:Int = 4,
                  hashSpaceSize:Int = 1 << 10): (HashingTF, IDFModel, RDD[Vector]) = {
  // Hash strings into the given space
  val hashingTF = new HashingTF(hashSpaceSize)
  val tf = hashingTF.transform(tokens)
  // Build term frequency-inverse document frequency
  val idfModel = new IDF(minDocFreq = minDocFreq).fit(tf)
  val expandedText = idfModel.transform(tf)
  (hashingTF, idfModel, expandedText)
}

// COMMAND ----------

// Define function which builds a DL model
import org.apache.spark.h2o._
import water.Key
import _root_.hex.deeplearning.DeepLearning
import _root_.hex.deeplearning.DeepLearningParameters
import _root_.hex.deeplearning.DeepLearningModel

def buildDLModel(train: Frame, valid: Frame,
               epochs: Int = 10, l1: Double = 0.001, l2: Double = 0.0,
               hidden: Array[Int] = Array[Int](200, 200))
              (implicit h2oContext: H2OContext): DeepLearningModel = {
import h2oContext._
// Build a model

val dlParams = new DeepLearningParameters()
dlParams._model_id = Key.make("dlModel.hex")
dlParams._train = train
dlParams._valid = valid
dlParams._response_column = 'target
dlParams._epochs = epochs
dlParams._l1 = l1
dlParams._hidden = hidden

// Create a job
val dl = new DeepLearning(dlParams)
val dlModel = dl.trainModel.get

// Compute metrics on both datasets
dlModel.score(train).delete()
dlModel.score(valid).delete()

dlModel
}

// COMMAND ----------

// Create SQL support
import org.apache.spark.sql._
implicit val sqlContext = SQLContext.getOrCreate(sc)
import sqlContext.implicits._

// Start H2O services
import org.apache.spark.h2o._
@transient val h2oContext = new H2OContext(sc).start()



// COMMAND ----------

// Open H2O UI
h2oContext.openFlow

// COMMAND ----------

// Build the application

import org.apache.spark.rdd.RDD
import org.apache.spark.examples.h2o.DemoUtils._
import scala.io.Source

// load both columns from the table
val data = sqlContext.sql("SELECT * FROM smsData")
// Extract response spam or ham
val hamSpam = data.map( r => r(0).toString)
val message = data.map( r => r(1).toString)
// Tokenize message content
val tokens = tokenize(message)
// Build IDF model
var (hashingTF, idfModel, tfidf) = buildIDFModel(tokens)

// Merge response with extracted vectors
val resultRDD: DataFrame = hamSpam.zip(tfidf).map(v => SMS(v._1, v._2)).toDF

// Publish Spark DataFrame as H2OFrame
// This H2OFrame has to be transient because we do not want it to be serialized.  When calling for example sc.parallelize(..) the object which we are trying to parallelize takes with itself all variables in its surroundings scope - apart from those marked as serialized.
// 
@transient val table = h2oContext.asH2OFrame(resultRDD)
println(sc.parallelize(Array(1,2)))
// Transform target column into categorical
table.replace(table.find("target"), table.vec("target").toCategoricalVec()).remove()
table.update(null)

// Split table
val keys = Array[String]("train.hex", "valid.hex")
val ratios = Array[Double](0.8)
@transient val frs = split(table, keys, ratios)
@transient val train = frs(0)
@transient val valid = frs(1)
table.delete()

// Build a model
@transient val dlModel = buildDLModel(train, valid)(h2oContext)



// COMMAND ----------

dlModel

// COMMAND ----------

// Evaluate model equality

// Collect model metrics and evaluate model quality
import water.app.ModelMetricsSupport
val trainMetrics = ModelMetricsSupport.binomialMM(dlModel, train)
println(trainMetrics.auc._auc)

// COMMAND ----------

// Collect model metrics and evaluate model quality
import water.app.ModelMetricsSupport
val validMetrics = ModelMetricsSupport.binomialMM(dlModel, valid)
println(validMetrics.auc._auc)

// COMMAND ----------

// Create a spam detector - a method which will return SPAM or HAM for given text message
import water.DKV._
// Spam detector
def isSpam(msg: String,
       modelId: String,
       hashingTF: HashingTF,
       idfModel: IDFModel,
       h2oContext: H2OContext,
       hamThreshold: Double = 0.5):String = {
val dlModel: DeepLearningModel = water.DKV.getGet(modelId)
val msgRdd = sc.parallelize(Seq(msg))
val msgVector: DataFrame = idfModel.transform(
                            hashingTF.transform (
                              tokenize (msgRdd))).map(v => SMS("?", v)).toDF
val msgTable: H2OFrame = h2oContext.asH2OFrame(msgVector)
msgTable.remove(0) // remove first column
val prediction = dlModel.score(msgTable)
//println(prediction)
if (prediction.vecs()(1).at(0) < hamThreshold) "SPAM DETECTED!" else "HAM"
}   

// COMMAND ----------

// Try do detect spam

isSpam("Michal, h2oworld party tonight in MV?", dlModel._key.toString, hashingTF, idfModel, h2oContext)

// COMMAND ----------

isSpam("We tried to contact you re your reply to our offer of a Video Handset? 750 anytime any networks mins? UNLIMITED TEXT?", dlModel._key.toString, hashingTF, idfModel, h2oContext)

// COMMAND ----------


