
# coding: utf-8

# In[1]:


import utils
import re
import sys
import csv


# In[2]:


def preprocess_word(word):
    word = word.strip('\'"?!,.():;')
    word = re.sub(r'(.)\1+', r'\1\1', word)
    word = re.sub(r'(-|\')', '', word)
    return word


# In[4]:


def handle_emojis(tweet):
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


# In[3]:


def is_valid_word(word):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


# In[5]:


def preprocess_tweet(tweet):
    processed_tweet = []
    tweet = tweet.lower()
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    tweet = re.sub(r'@briankempga', 'briankempga', tweet)
    tweet = re.sub(r'@staceyabrams', 'staceyabrams', tweet)
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    hashtagIgnoreList = ['#kempforgovernor','#votekemp','#briankempga','#teamkemp','#voteredtosaveamerica',
                        '#voteforstaceyabrams','#rednationrising','#staceyabrams','#votedemsout',
                        '#staceyabramsforgovernor','#walkawayfromdemocrats','#voteredsaveamerica2018','#canttrustkemp',
                        '#crookedkemp','#corruptkemp','#votersuppression','#voteredtosaveamerica','#georgiavoteforkemp',
                        '#demorats','#redwaverising','#votered','#voteredtosavegeorgia','#georgiagovernorsrace','#votered',
                        '#backtheblue','#votebigred','#turngablue','#voteblue2018','#bluewave2018','#teamabrams','#gagov',
                        '#bluegov','#blackwomenlead','#blackwomenvote','#voteblue','#votebluetosaveamerica','#voteabrams',
                        '#bluewave','#staceyabrams','#turngablue','#votersuppression','#gadems','#iamwithstacey','#votingwhileblack',
                        '#georgiagovernorsrace','#votebluetoendthisnightmare','#stopkemp','#alllivesmatter','#staceyabrams4gov','#savedemocracy',
                        '#flipitblue','#female','#mutetrump','#winblue','#demswork4usa']
    for hashtag in hashtagIgnoreList:
        tweet = re.sub(r'#'+re.escape(hashtag[1:]),hashtag[1:], tweet)
    tweet = re.sub(r'#[\S]+', 'Hashtag', tweet)
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    tweet = tweet.strip(' "\'')
    tweet = handle_emojis(tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            processed_tweet.append(word)
    return ' '.join(processed_tweet)


# In[23]:


def preprocess_csv(csv_file_name, processed_file_name):
    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweet = row['text']
            #sentiment = row['Sentiment']
            processed_tweet = preprocess_tweet(tweet)
            save_to_file.write('%s \n' %(processed_tweet))
    save_to_file.close()
    print ('\nSaved processed tweets to: %s' % processed_file_name)
    return processed_file_name
if __name__ == '__main__':
    csv_file_name = '/Users/ayushjaiswal/Documents/UGA_Fall18/TwitterSentiment/NovDataset/Kemp3.csv'
    processed_file_name = '/Users/ayushjaiswal/Documents/UGA_Fall18/TwitterSentiment/NovDataset/Kemp3-processed.csv'
    preprocess_csv(csv_file_name, processed_file_name)

