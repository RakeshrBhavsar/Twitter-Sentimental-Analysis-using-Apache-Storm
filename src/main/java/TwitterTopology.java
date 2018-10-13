import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.topology.TopologyBuilder;

public class TwitterTopology {
    public static void main(String args[]) throws InterruptedException{

        TopologyBuilder topologyBuilder = new TopologyBuilder();
        topologyBuilder.setSpout("twitter-data-collector", new TwitterSpout());
        topologyBuilder.setBolt("twitter-data-extractor", new TwitterBolt())
                .shuffleGrouping("twitter-data-collector");

        LocalCluster cluster = new LocalCluster();
        Config conf = new Config();
        conf.setDebug(true);
        cluster.submitTopology("twitter-topology", conf, topologyBuilder.createTopology());
        Thread.sleep(10000);
        cluster.shutdown();

    }
}
