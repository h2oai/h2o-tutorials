package storm.starter;

import backtype.storm.Config;
import backtype.storm.spout.SpoutOutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichSpout;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;
import backtype.storm.utils.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;


public class TestH2ODataSpout extends BaseRichSpout {
  public static Logger LOG = LoggerFactory.getLogger(TestH2ODataSpout.class);
  boolean _isDistributed;
  SpoutOutputCollector _collector;
  AtomicInteger _cnt = new AtomicInteger(0);


  public TestH2ODataSpout() {
    this(true);
  }

  public TestH2ODataSpout(boolean isDistributed) {
    _isDistributed = isDistributed;
  }

  public void open(Map conf, TopologyContext context, SpoutOutputCollector collector) {
    _collector = collector;
  }

  public void close() {

  }

  public void nextTuple() {
    Utils.sleep(1000);
    File file = new File("/Users/ludirehak/apache/h2o-training/tutorials/streaming/storm/live_data.csv");  // EDIT ME TO YOUR PATH!
    String[] observation=null;
    int i = 0;
    try {
      String line="";
      BufferedReader br = new BufferedReader(new FileReader(file));
      while (i++<=_cnt.get()) line = br.readLine(); // stream thru to next line
      observation = line.split(",");
    } catch (Exception e) {
      e.printStackTrace();
      _cnt.set(0);
    }
    _cnt.getAndIncrement();
    if (_cnt.get() == 1000) _cnt.set(0); // force reset, for demo only!!!
    _collector.emit(new Values(observation));
  }

  public void ack(Object msgId) {
    //empty
  }

  public void fail(Object msgId) {
    //empty
  }

  public void declareOutputFields(OutputFieldsDeclarer declarer) {
    LinkedList<String> fields_list = new LinkedList<String>(Arrays.asList(GBMPojo.NAMES));
    fields_list.add(0,"Label");                            // put label, shift right

    String[] fields = fields_list.toArray(new String[fields_list.size()]); // emit these fields
    declarer.declare(new Fields(fields));
  }

  @Override
  public Map<String, Object> getComponentConfiguration() {
    if(!_isDistributed) {
      Map<String, Object> ret = new HashMap<String, Object>();
      ret.put(Config.TOPOLOGY_MAX_TASK_PARALLELISM, 1);
      return ret;
    } else {
      return null;
    }
  }
}