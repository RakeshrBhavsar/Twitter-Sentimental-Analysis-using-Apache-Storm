
# coding: utf-8

# In[109]:


import utils
import csv


# In[110]:


TRAIN_PROCESSED_FILE = '/Users/ayushjaiswal/Documents/UGA_Fall18/TwitterSentiment/train.csv'
TEST_PROCESSED_FILE = '/Users/ayushjaiswal/Documents/UGA_Fall18/TwitterSentiment/test.csv'
POSITIVE_WORDS_FILE = '/Users/ayushjaiswal/Documents/UGA_Fall18/TwitterSentiment/positive-words.txt'
NEGATIVE_WORDS_FILE = '/Users/ayushjaiswal/Documents/UGA_Fall18/TwitterSentiment/negative-words.txt'
TRAIN = False


# In[111]:


def classify(processed_csv, test_file=True, **params):
    positive_words = utils.file_to_wordset(params.pop('positive_words'))
    negative_words = utils.file_to_wordset(params.pop('negative_words'))
    predictions = []
    
    with open(processed_csv, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if test_file:
                tweet_id = row['ItemID']
                tweet = row['SentimentText']
            else:
                tweet_id = row['ItemID']
                sentiment = row['Sentiment']
                tweet = row['SentimentText']
            pos_count, neg_count = 0, 0
            for word in tweet.split():
                if word in positive_words:
                    pos_count += 1
                elif word in negative_words:
                    neg_count += 1
            # print pos_count, neg_count
            prediction = 1 if pos_count >= neg_count else 0
            if test_file:
                predictions.append((tweet_id, prediction))
            else:
                predictions.append((tweet_id,sentiment, prediction))
    return predictions


# In[112]:


if __name__ == '__main__':
    if TRAIN:
        predictions = classify(TRAIN_PROCESSED_FILE, test_file=(not TRAIN), positive_words=POSITIVE_WORDS_FILE, negative_words=NEGATIVE_WORDS_FILE)
        count=0
        for p in predictions:
              if (int(p[1]) == int(p[2])):
                count+=1
        correct = count*100/len(predictions)
        print("Correct %s" %correct)
    else:
        predictions = classify(TEST_PROCESSED_FILE, test_file=(not TRAIN), positive_words=POSITIVE_WORDS_FILE, negative_words=NEGATIVE_WORDS_FILE)
        utils.save_results_to_csv(predictions, '/Users/ayushjaiswal/Documents/UGA_Fall18/TwitterSentiment/baseline.csv')

