package org.apache.storm.starter;

import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.StormSubmitter;
import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.topology.base.BaseRichBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;
import org.apache.storm.utils.Utils;
import org.testng.annotations.Test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

/**
 * This is a basic example of embedding an H2O scoring POJO into a Storm topology.
 */
public class H2OStormStarter {


  /**
   * The ScoreBolt is responsible for obtaining class probabilities from the score pojo.
   * It emits these probabilities to a ClassifierBolt, which classifies the observation as "cat" or "dog".
   */
  public static class PredictionBolt extends BaseRichBolt {
    OutputCollector _collector;

    @Override
    public void prepare(Map conf, TopologyContext context, OutputCollector collector) {
      _collector = collector;
    }

    @Override public void execute(Tuple tuple) {

      GBMPojo p = new GBMPojo();

      // get the input tuple as a String[]
      ArrayList<String> vals_string = new ArrayList<String>();
      for (Object v : tuple.getValues()) vals_string.add((String)v);
      String[] raw_data = vals_string.toArray(new String[vals_string.size()]);

      // the score pojo requires a single double[] of input.
      // We handle all of the categorical mapping ourselves
      double data[] = new double[raw_data.length-1]; //drop the Label

      String[] colnames = tuple.getFields().toList().toArray(new String[tuple.size()]);

      // if the column is a factor column, then look up the value, otherwise put the double
      for (int i = 1; i < raw_data.length; ++i) {
        data[i-1] = p.getDomainValues(colnames[i]) == null
                ? Double.valueOf(raw_data[i])
                : p.mapEnum(p.getColIdx(colnames[i]), raw_data[i]);
      }

      // get the predictions
      double[] preds = new double [GBMPojo.NCLASSES+1];
      //p.predict(data, preds);
      p.score0(data, preds);

      // emit the results
      _collector.emit(tuple, new Values(raw_data[0], preds[1]));
      _collector.ack(tuple);
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
      declarer.declare(new Fields("expected_class", "dogProbability"));
    }
  }

  /**
   * The ClassifierBolt receives the input probabilities and then makes a classification.
   * It uses a threshold value to determine how to classify the observation, which is computed based on the validation
   * done during model fitting.
   */
  public static class ClassifierBolt extends BaseRichBolt {
    OutputCollector _collector;
    final double _thresh = 0.54;

    @Override
    public void prepare(Map conf, TopologyContext context, OutputCollector collector) {
      _collector = collector;
    }

    @Override
    public void execute(Tuple tuple) {
      String expected=tuple.getString(0);
      double dogProb = tuple.getDouble(1);
      String content = expected + "," + (dogProb <= _thresh ? "dog" : "cat");
      try {
        File file = new File("/Users/ludirehak/apache/h2o-training/tutorials/streaming/storm/web/out"); // EDIT ME TO YOUR PATH!
        if (!file.exists())  file.createNewFile();
        FileWriter fw = new FileWriter(file.getAbsoluteFile());
        BufferedWriter bw = new BufferedWriter(fw);
        bw.write(content);
        bw.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
      _collector.emit(tuple, new Values(expected, dogProb <= _thresh ? "dog" : "cat"));
      _collector.ack(tuple);
    }

    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
      declarer.declare(new Fields("expected_class", "class"));
    }
  }

  @Test
  public static void h2o_storm() throws Exception {
    TopologyBuilder builder = new TopologyBuilder();

    builder.setSpout("input_row", new TestH2ODataSpout(), 10);
    builder.setBolt("score_probabilities", new PredictionBolt(), 3).shuffleGrouping("input_row");
    builder.setBolt("classify", new ClassifierBolt(), 3).shuffleGrouping("score_probabilities");

    Config conf = new Config();
    conf.setDebug(true);

    String[] args = null;
    if (args != null && args.length > 0) {
      conf.setNumWorkers(3);

      StormSubmitter.submitTopologyWithProgressBar(args[0], conf, builder.createTopology());
    }
    else {

      LocalCluster cluster = new LocalCluster();
      cluster.submitTopology("test", conf, builder.createTopology());
      Utils.sleep(1000 * 60 * 60); // run for 1 hour
      cluster.killTopology("test");
      cluster.shutdown();
    }
  }
}
