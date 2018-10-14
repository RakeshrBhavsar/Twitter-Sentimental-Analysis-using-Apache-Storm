import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.BasicOutputCollector;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseBasicBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;
import twitter4j.Status;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

public class TwitterBolt extends BaseBasicBolt {

    private PrintWriter writer;
    Map<Integer,String> map;
    Integer i;

    public void prepare(Map stormConf, TopologyContext context) {

        String fileName = "output"+"-"+context.getThisTaskId()+"-"+context.getThisComponentId()+".csv";
        try{
            this.writer = new PrintWriter(stormConf.get("dirToWrite").toString()+fileName, "UTF-8");
            this.map = new HashMap<Integer, String>();
            this.i = 0;
        }
        catch (Exception e){}
    }

    public void execute(Tuple tuple, BasicOutputCollector basicOutputCollector) {


        Status status = (Status) tuple.getValueByField("tweet");
        String tweetText = status.getText();
        map.put(i++,tweetText);


        basicOutputCollector.emit(new Values(tweetText));
    }

    public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {

        outputFieldsDeclarer.declare(new Fields("tweet"));

    }

    public void cleanup() {

        for(String tweet:map.values()){
            writer.println(tweet);
        }
        writer.close();
    }
}
