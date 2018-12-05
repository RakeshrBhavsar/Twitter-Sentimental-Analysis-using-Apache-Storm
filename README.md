# Predicting Georgia General Election Results with Tweet Sentimental Analysis
-----------

   Election empowers citizens to choose their leaders. The public sentiments towards the leader highly influence the decision of the general public. People use Twitter as the social media platform for speaking their views out. Twitter will transform democracy, allowing citizens and politicians to communicate, connect and interact in ways never before thought possible. Social media tools such as Twitter is now considered as politically transformative communication technologies as radio and television. Twitter influences elections and public opinion poll results. Therefore, it is needed to understand how citizens and politicians worldwide share political information and opinion via social media. 

   This paper presents Predicting Georgia Election Results with Tweet Sentiment Analysis to analyze citizen tweets which can be used as a good predictor of public opinion regarding the Georgia elections and the two candidates (Brian Kemp vs Stacey Abrams). To perform sentiment analysis of tweets we have implemented various machine learning algorithms such as Dictionary based, LSTM using Glove & Bidirectional LSTM using Glove. Our results show that LSTM achieves an accuracy of 94% in predicting the sentiment of a tweet as positive or negative for the respective candidate.  

### System Architecture:
-----------

![alt text](https://github.com/RakeshrBhavsar/Twitter-Sentimental-Analysis-using-Apache-Storm/blob/master/images/Picture1.png "System Architecture")

### Conclusion:

- Collected live tweets using apache storm consisting of mixture of words, emoticons, URLs, hashtags, user mentions and symbols. Performed Pre-Processing of tweets to make it suitable for feeding into model.

- We have used machine learning algorithms such as dictionary based, LSTM using Glove and BiDirectional LSTM using Glove. 

| Model                          | Accuracy  |
| ------------------------------ |:---------:|
| Dictionary Based               | 63.38%    |
| LSTM using Glove               | 94.00%    |
| BiDirectional LSTM using Glove | 94.5%     |

- We have Trained & Tested our Model on AWS EC2 Instance (p2.xlarge). Our Best LSTM model achieved an accuracy of 94%.


### Requirements:
    - Python
    - Java
    - Maven
    - Apache Storm
    - Keras 
    - Tensorflow
    - Numpy
    - AWS EC2 (p2.xlarge)
    - Springboot framework. 


