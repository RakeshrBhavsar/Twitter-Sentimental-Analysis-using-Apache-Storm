
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Input,Bidirectional,GlobalMaxPool1D,Dropout
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.utils.np_utils import to_categorical
import re


# In[15]:


tweets_pd=pd.read_csv('FinalDataSet.csv')


# In[16]:


tweets_pd=tweets_pd[['Tweets']]


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


tweets_pd


# In[19]:


tweets_pd['Tweets']=tweets_pd.astype('str')


# In[5]:


maxlen=100
max_fatures = 2000


# In[20]:


max_fatures = 2000
maxlen=100
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(tweets_pd['Tweets'].values)
X = tokenizer.texts_to_sequences(tweets_pd['Tweets'].values)
X = pad_sequences(X,maxlen=maxlen)


# In[21]:


X.shape


# In[7]:


embed_dim = 100
lstm_out = 196


# In[22]:


def check(x):
    x=np.array(x)
    return np.argmax(x)
     


# In[23]:


def actual(x):
    for index,i in enumerate(x):
        if i==1:
            return index


# In[20]:


from matplotlib.pyplot import figure


# In[71]:


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.CMRmap):
    a4_dims = (10.7, 8.27)
    plt.figure(figsize=(10,5))
    plt.matshow(df_confusion, cmap=cmap,fignum=1) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


# In[24]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[9]:


EMBEDDING_FILE_100=f'glove.6B.100d.txt'


# In[10]:


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
# embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


# ##### Changing the Glove dimension

# In[11]:


embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE_100))


# In[12]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[25]:


word_index = tokenizer.word_index
nb_words = min(max_fatures, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_dim))
for word, i in word_index.items():
    if i >= max_fatures: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


nb_words


# In[26]:


embedding_matrix.shape


# In[27]:


inp = Input(shape=(maxlen,))
x = Embedding(max_fatures, embed_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(392, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(4, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[28]:


model.summary()


# In[29]:


model.load_weights('glove100d2.h5')


# In[30]:


y_pred=model.predict(X)


# In[31]:


pred_df=pd.DataFrame(y_pred,columns=['kemp_negative','kemp_positive','stacy_negative','stacy_positive'])


# In[32]:


pred_df['prediction']=pred_df.apply(lambda x:check(x),axis=1)


# In[33]:


pred_df=pred_df[['prediction']]

# acc=(len(pred_df[pred_df['prediction']==pred_df['actual']])/len(pred_df))*100


# In[39]:


pred_df.dtypes


# In[1]:


pred_df_cnt=pred_df.groupby('prediction').size()


# In[43]:


pred_df_cnt=pd.DataFrame(pred_df_cnt)


# In[46]:


pred_df_cnt=pred_df_cnt.reset_index()


# In[48]:


pred_df_cnt=pred_df_cnt.rename(columns={0:'votes_count'})


# In[51]:


pred_df_cnt['candidate']=['Kemp_Neg','Kemp_Pos','Stacey_Neg','Stacey_Pos']


# In[44]:


import seaborn as sns


# In[52]:


a4_dims = (10.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x=pred_df_cnt.candidate, y=pred_df_cnt.votes_count, data=pred_df_cnt)

