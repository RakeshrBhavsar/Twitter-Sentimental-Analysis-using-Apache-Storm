import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.IRichBolt;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.tuple.Tuple;
import twitter4j.GeoLocation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Date;
import java.util.Map;

public class TwitterFileWriterBolt implements IRichBolt {

  private File file = null;
  private FileWriter fileWriter = null;

  public void prepare(Map map, TopologyContext topologyContext, OutputCollector outputCollector) {
    try {
      file = new File("userTweetOct29-stacey.csv");
      fileWriter = new FileWriter(file.getAbsoluteFile(), true);
      StringBuilder sb = new StringBuilder();
      sb.append("screen_name" + "," + "user_name" + "," + "status_id"
        + "," + "url" + ","  + "title"
        + "," + "published_date" + "," + "geoLocation");
      sb.append("\n");
      fileWriter.write(sb.toString());
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void execute(Tuple tuple) {
    Long statusID = null;
    String screenName = "";
    String userName = "";
    String url = "";
    String tweetText = "";
    Date publishedDate = null;
    GeoLocation geoLocation = null;

    statusID = (Long) tuple.getValueByField("statusID");
    screenName = (String) tuple.getValueByField("screenName");
    userName = (String) tuple.getValueByField("userName");
    url = (String) tuple.getValueByField("url");
    tweetText = (String) tuple.getValueByField("title");
    publishedDate = (Date) tuple.getValueByField("publishedDate");
    geoLocation = (GeoLocation) tuple.getValueByField("geoLocation");

    Tweet tweet = new Tweet();
    tweet.setStatusID(String.valueOf(statusID));
    tweet.setScreenName(screenName);
    tweet.setUserName(userName);
    tweet.setUrl(url);
    tweet.setTitle(tweetText);
    tweet.setPublishedDate(String.valueOf(publishedDate));
    tweet.setGeoLocation(geoLocation);
    tweet.setTweetComplete(true);

    this.write(tweet);


  }

  private void write(Tweet tweet) {
        if (tweet != null) {
          if (tweet.isWritable()) {
            try {
              BufferedWriter bufferedWriter = new BufferedWriter(
                fileWriter);
              System.out.println("tweet==> "+tweet.toString());
              bufferedWriter.write(tweet.toString());
              bufferedWriter.newLine();
              bufferedWriter.flush();

            } catch (Exception e) {
              e.printStackTrace();
            }
          }
        }
  }

  public void cleanup() {

  }

  public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {

  }

  public Map<String, Object> getComponentConfiguration() {
    return null;
  }
}
