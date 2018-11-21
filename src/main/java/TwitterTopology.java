import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.StormSubmitter;
import org.apache.storm.generated.AlreadyAliveException;
import org.apache.storm.generated.AuthorizationException;
import org.apache.storm.generated.InvalidTopologyException;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.tuple.Fields;

public class TwitterTopology {
    public static void main(String args[]) throws AlreadyAliveException, InvalidTopologyException, AuthorizationException {

        TopologyBuilder topologyBuilder = new TopologyBuilder();
        topologyBuilder.setSpout("twitter-data-collector", new TwitterSpout());
        topologyBuilder.setBolt("twitter-data-extractor", new TwitterBolt(),2)
                .shuffleGrouping("twitter-data-collector");
        topologyBuilder.setBolt("twitter-file-writer", new TwitterFileWriterBolt())
                .fieldsGrouping("twitter-data-extractor",new Fields("title"));



        Config conf = new Config();
        if (args != null && args.length > 0) {
            conf.setNumWorkers(1);
            StormSubmitter.submitTopology(args[0], conf, topologyBuilder.createTopology());
        }
        else {
            final LocalCluster cluster = new LocalCluster();
            cluster.submitTopology("twitter-topology", conf, topologyBuilder.createTopology());
            Runtime.getRuntime().addShutdownHook(new Thread()	{
                @Override
                public void run()	{
                    cluster.killTopology("twitter-topology");
                    cluster.shutdown();
                }
            });
        }
    }
}
