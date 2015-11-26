# H2OWorld - Building Machine Learning Applications with Sparkling Water

## Requirements
 
 - Oracle Java 7+ ([USB](../../))
 - [Spark 1.5.1](http://spark.apache.org/downloads.html) ([USB](../../Spark))
 - [Sparkling Water 1.5.6](http://h2o-release.s3.amazonaws.com/sparkling-water/rel-1.5/6/index.html) ([USB](../../SparklingWater))
 - [SMS dataset](https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/smsData.txt) ([USB](../data/smsData.txt))
 
## Provided on USB
 - [Binaries](../../)
 - [SMS dataset](../data/smsData.txt)
 - [Slides](SparklingWater.pdf)
 - [Scala Script](h2oworld.script.scala)

## Machine Learning Workflow

**Goal**: For a given text message, identify if it is spam or not.

  1. Extract data
  2. Transform & tokenize messages
  3. Build Spark's [Tf-IDF model](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and expand messages to feature vectors
  4. Create and evaluate [H2O's Deep Learning model](https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/dl/dl.md)
  5. Use the models to detect spam messages

### Prepare environment

1. Run Sparkling shell with an embedded Spark cluster:
  ```bash
  cd "path/to/sparkling/water"
  export SPARK_HOME="/path/to/spark/installation"
  export MASTER="local-cluster[3,2,4096]"
  bin/sparkling-shell --conf spark.executor.memory=2G 
  ```

  > Note: To avoid flooding output with Spark INFO messages, I recommend editing your `$SPARK_HOME/conf/log4j.properties` and configuring the log level to `WARN`.

2. Open Spark UI: Go to [http://localhost:4040/](http://localhost:4040/) to see the Spark status.

3. Prepare the environment:
  ```scala
  // Input data
  val DATAFILE="../data/smsData.txt"
  // Common imports from H2O and Sparks
  import _root_.hex.deeplearning.{DeepLearningModel, DeepLearning}
  import _root_.hex.deeplearning.DeepLearningParameters
  import org.apache.spark.examples.h2o.DemoUtils._
  import org.apache.spark.h2o._
  import org.apache.spark.mllib
  import org.apache.spark.mllib.feature.{IDFModel, IDF, HashingTF}
  import org.apache.spark.rdd.RDD
  import water.Key
  ```
  
4. Define the representation of the training message:
   ```scala
   // Representation of a training message
   case class SMS(target: String, fv: mllib.linalg.Vector)
   ```

5. Define the data loader and parser:
  ```scala
  def load(dataFile: String): RDD[Array[String]] = {
    // Load file into memory, split on TABs and filter all empty lines
    sc.textFile(dataFile).map(l => l.split("\t")).filter(r => !r(0).isEmpty)
  }
  ```
  
6. Define the input messages tokenizer:
  ```scala
  // Tokenizer
  // For each sentence in input RDD it provides array of string representing individual interesting words in the sentence
  def tokenize(dataRDD: RDD[String]): RDD[Seq[String]] = {
    // Ignore all useless words
    val ignoredWords = Seq("the", "a", "", "in", "on", "at", "as", "not", "for")
    // Ignore all useless characters
    val ignoredChars = Seq(',', ':', ';', '/', '<', '>', '"', '.', '(', ')', '?', '-', '\'','!','0', '1')

	// Invoke RDD API and transform input data
    val textsRDD = dataRDD.map( r => {
      // Get rid of all useless characters
      var smsText = r.toLowerCase
      for( c <- ignoredChars) {
        smsText = smsText.replace(c, ' ')
      }
	  // Remove empty and uninteresting words
      val words = smsText.split(" ").filter(w => !ignoredWords.contains(w) && w.length>2).distinct

      words.toSeq
    })
    textsRDD
  }
  ```

7. Configure Spark's Tf-IDF model builder: 
  ```scala
  def buildIDFModel(tokensRDD: RDD[Seq[String]],
                    minDocFreq:Int = 4,
                    hashSpaceSize:Int = 1 << 10): (HashingTF, IDFModel, RDD[mllib.linalg.Vector]) = {
    // Hash strings into the given space
    val hashingTF = new HashingTF(hashSpaceSize)
    val tf = hashingTF.transform(tokensRDD)
    
    // Build term frequency-inverse document frequency model
    val idfModel = new IDF(minDocFreq = minDocFreq).fit(tf)
    val expandedTextRDD = idfModel.transform(tf)
    (hashingTF, idfModel, expandedTextRDD)
  }
  ```
  
  > **Wikipedia** defines TF-IDF as: "tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval and text mining. The tf-idf value increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general."
  
8. Configure H2O's DeepLearning model builder:
  ```scala
  def buildDLModel(trainHF: Frame, validHF: Frame,
                 epochs: Int = 10, l1: Double = 0.001, l2: Double = 0.0,
                 hidden: Array[Int] = Array[Int](200, 200))
                (implicit h2oContext: H2OContext): DeepLearningModel = {
    import h2oContext._
    import _root_.hex.deeplearning.DeepLearning
    import _root_.hex.deeplearning.DeepLearningParameters
    // Create algorithm parameres
    val dlParams = new DeepLearningParameters()
    // Name for target model
    dlParams._model_id = Key.make("dlModel.hex")
    // Training dataset
    dlParams._train = trainHF
    // Validation dataset
    dlParams._valid = validHF
    // Column used as target for training
    dlParams._response_column = 'target
    // Number of passes over data
    dlParams._epochs = epochs
    // L1 penalty
    dlParams._l1 = l1
    // Number internal hidden layers
    dlParams._hidden = hidden

    // Create a DeepLearning job
    val dl = new DeepLearning(dlParams)
    // And launch it
    val dlModel = dl.trainModel.get

    // Force computation of model metrics on both datasets
    dlModel.score(trainHF).delete()
    dlModel.score(validHF).delete()

	// And return resulting model
    dlModel
  }
  ```

9. Initialize `H2OContext` and start H2O services on top of Spark:
  ```scala
   // Create SQL support
   import org.apache.spark.sql._
   implicit val sqlContext = SQLContext.getOrCreate(sc)
   import sqlContext.implicits._

   // Start H2O services
   import org.apache.spark.h2o._
   val h2oContext = new H2OContext(sc).start()
  ```

10. Open H2O UI and verify that H2O is running: 
  ```scala
  h2oContext.openFlow
  ```

  > At this point, you can use the H2O UI and see the status of the H2O cloud by typing `getCloud`.

11. Build the final workflow using all building pieces:
  ```scala
  // Data load
  val dataRDD = load(DATAFILE)
  // Extract response column from dataset
  val hamSpamRDD = dataRDD.map( r => r(0))
  // Extract message from dataset
  val messageRDD = dataRDD.map( r => r(1))
  // Tokenize message content
  val tokensRDD = tokenize(messageRDD)

  // Build IDF model on tokenized messages
  // It returns
  //   - hashingTF: hashing function to hash a word to a vector space
  //   - idfModel: a model to transform hashed sentence to a feature vector
  //   - tfidf: transformed input messages
  var (hashingTF, idfModel, tfidfRDD) = buildIDFModel(tokensRDD)

  // Merge response with extracted vectors
  val resultDF = hamSpamRDD.zip(tfidfRDD).map(v => SMS(v._1, v._2)).toDF

  // Publish Spark DataFrame as H2OFrame  
  val tableHF = h2oContext.asH2OFrame(resultDF, "messages_table")
  
  // Transform target column into categorical!
  tableHF.replace(tableHF.find("target"), tableHF.vec("target").toCategoricalVec()).remove()
  tableHF.update(null)

  // Split table into training and validation parts
  val keys = Array[String]("train.hex", "valid.hex")
  val ratios = Array[Double](0.8)
  val frs = split(tableHF, keys, ratios)
  val (trainHF, validHF) = (frs(0), frs(1))
  tableHF.delete()

  // Build final DeepLearning model
  val dlModel = buildDLModel(trainHF, validHF)(h2oContext)
  ```
  
12. Evaluate the model's quality:
   ```scala
   // Collect model metrics and evaluate model quality
   import water.app.ModelMetricsSupport
   val trainMetrics = ModelMetricsSupport.binomialMM(dlModel, trainHF)
   val validMetrics = ModelMetricsSupport.binomialMM(dlModel, validHF)
   println(trainMetrics.auc._auc)
   println(validMetrics.auc._auc)
   ```

   > You can also open the H2O UI and type `getPredictions` to visualize the model's performance or type `getModels` to see model output.
   
13. Create a spam detector:
   ```scala
   // Spam detector
   def isSpam(msg: String,
           dlModel: DeepLearningModel,
           hashingTF: HashingTF,
           idfModel: IDFModel,
           h2oContext: H2OContext,
           hamThreshold: Double = 0.5):String = {
      val msgRdd = sc.parallelize(Seq(msg))
      val msgVector: DataFrame = idfModel.transform(
                                  hashingTF.transform (
                                    tokenize (msgRdd))).map(v => SMS("?", v)).toDF
      val msgTable: H2OFrame = h2oContext.asH2OFrame(msgVector)
      msgTable.remove(0) // remove first column
      val prediction = dlModel.score(msgTable)

      if (prediction.vecs()(1).at(0) < hamThreshold) "SPAM DETECTED!" else "HAM"
  }   
  ```
  
14. Try to detect spam:
   ```scala
   isSpam("Michal, h2oworld party tonight in MV?", dlModel, hashingTF, idfModel, h2oContext)
   // 
   isSpam("We tried to contact you re your reply to our offer of a Video Handset? 750 anytime any networks mins? UNLIMITED TEXT?", dlModel, hashingTF, idfModel, h2oContext)
   ```

15. At this point, you have finished your 1st Sparkling Water Machine Learning application. Hack and enjoy! Thank you!   

