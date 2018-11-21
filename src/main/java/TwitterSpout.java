import org.apache.storm.spout.SpoutOutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseRichSpout;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;
import org.apache.storm.utils.Utils;
import twitter4j.*;
import twitter4j.conf.ConfigurationBuilder;

import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;

public class TwitterSpout extends BaseRichSpout {

    private SpoutOutputCollector spoutOutputCollector;
    private LinkedBlockingQueue<Status> queue;
    private TwitterStream twitterStream;

    public void open(Map map, TopologyContext topologyContext, SpoutOutputCollector spoutOutputCollector) {
        this.spoutOutputCollector = spoutOutputCollector;

        ConfigurationBuilder configurationBuilder = new ConfigurationBuilder();

        configurationBuilder
                .setOAuthConsumerKey("jSRNPoKRdCklZksI1F2Zgf5dT")
                .setOAuthConsumerSecret("JLNvnBSWYTNcsBA5bgqC0AEbHb99SJ7tj7rJkLkhHV6wmwa5Pf")
                .setOAuthAccessToken("1052765523195367424-x8IvDifs2W0GT3J1JhirnyFKsHAtcp")
                .setOAuthAccessTokenSecret("8VJmnued9KpadDvyGYhXKhMH5kWZXCWbNa0W91NPVkVhP")
                .setTweetModeExtended(true);


        this.twitterStream = new TwitterStreamFactory(configurationBuilder.build()).getInstance();

        this.queue = new LinkedBlockingQueue<Status>();

        final StatusListener statusListener = new StatusListener() {
            public void onStatus(Status status) {
                queue.offer(status);
            }

            public void onDeletionNotice(StatusDeletionNotice statusDeletionNotice) {

            }

            public void onTrackLimitationNotice(int i) {

            }

            public void onScrubGeo(long l, long l1) {

            }

            public void onStallWarning(StallWarning stallWarning) {

            }

            public void onException(Exception e) {

            }
        };

        twitterStream.addListener(statusListener);
        final FilterQuery query = new FilterQuery();
        query.track(new String[]{"staceyabrams","imwithstacey"});
//        query.track(new String[]{"gapol","imwithstacey","GAGov","BrianKempGA","staceyabrams","KempForGovernor"});

        twitterStream.filter(query);
    }

    public void nextTuple() {
        final Status status = queue.poll();

        if (status == null) {
            Utils.sleep(60);
        } else {
            spoutOutputCollector.emit(new Values(status));
        }
    }

    public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {
        outputFieldsDeclarer.declare(new Fields("tweet"));
    }

    public void close() {
        twitterStream.shutdown();
    }

}
