import twitter4j.GeoLocation;

public class Tweet {

    private String statusID;
    private String screenName;
    private String userName;
    private String url;
    private String title;
    private String publishedDate;
    private GeoLocation geoLocation;
    private boolean isTweetComplete = false;

    public String getStatusID() {
      return statusID;
    }

    public void setStatusID(String statusID) {
      this.statusID = statusID;
    }

    public String getScreenName() {
      return screenName;
    }

    public void setScreenName(String screenName) {
      this.screenName = screenName;
    }

    public String getUserName() {
      return userName;
    }

    public void setUserName(String userName) {
      this.userName = userName;
    }

    public String getUrl() {
      return url;
    }

    public void setUrl(String url) {
      this.url = url;
    }

    public String getTitle() {
      return title;
    }

    public void setTitle(String title) {
      this.title = title;
    }

    public String getPublishedDate() {
      return publishedDate;
    }

    public void setPublishedDate(String publishedDate) {
      this.publishedDate = publishedDate;
    }

    public boolean isTweetComplete() {
      return isTweetComplete;
    }

    public void setTweetComplete(boolean tweetComplete) {
      isTweetComplete = tweetComplete;
    }

    public boolean isWritable() {
      if(this.isTweetComplete) {
        return true;
      }
      return false;
    }

    public GeoLocation getGeoLocation() {
      return geoLocation;
    }

    public void setGeoLocation(GeoLocation geoLocation) {
      this.geoLocation = geoLocation;
    }

    @Override
    public String toString() {

      StringBuilder sb = new StringBuilder();
      sb.append("\""+this.screenName+"\",");
      sb.append("\""+this.userName+"\",");
      sb.append("\""+this.statusID+"\",");
      sb.append("\""+this.url+"\",");
      sb.append("\""+this.title+"\",");
      sb.append("\""+this.geoLocation+"\",");
      sb.append("\""+this.publishedDate+"\",");


      return sb.toString();

    }
}
