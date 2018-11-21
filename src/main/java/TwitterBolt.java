import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.BasicOutputCollector;
import org.apache.storm.topology.IRichBolt;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseBasicBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;
import twitter4j.Status;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

public class TwitterBolt implements IRichBolt {

    private OutputCollector collector;

    public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
        this.collector = collector;
    }

    public void execute(Tuple tuple) {

        Status status = (Status) tuple.getValueByField("tweet");
        String url = "";
        String tweetText = "";
        if (status.getURLEntities() != null && status.getURLEntities().length > 0) {
            url = status.getURLEntities()[0].getURL().trim();
        } else {
            url = "Not Available";
        }

        if(status.isRetweet()){
            tweetText = status.getRetweetedStatus().getText();
            System.out.println("RAKESH ::: "+tweetText);
        } else{
            tweetText = status.getText();
        }
        collector.emit(new Values(status.getUser().getScreenName(), status.getUser().getName(),
                status.getId(), url, tweetText, status.getCreatedAt(), status.getGeoLocation()));

    }

    public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {

        outputFieldsDeclarer.declare(new Fields("screenName", "userName", "statusID",
                "url", "title", "publishedDate", "geoLocation"));
    }

    public Map<String, Object> getComponentConfiguration() {
        return null;
    }


    public void cleanup() {

    }
}
