
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


# In[2]:


tweets_pd=pd.read_csv('LabelledDataSet-processed.csv')


# In[3]:


tweets_pd=tweets_pd[['Text','Label']]


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


tweets_pd['Label']=tweets_pd['Label'].astype('int32')


# In[6]:


tweets_pd


# In[7]:


tweets_pd['Text']=tweets_pd.astype('str')


# In[8]:


tweets_pd_grp=tweets_pd.groupby('Label')


# In[9]:


tweets_cnt=pd.DataFrame(tweets_pd_grp.count())


# In[10]:


tweets_cnt=tweets_cnt.rename(columns={'Text':'Tweet Count'})


# In[11]:


tweets_cnt=tweets_cnt.reset_index()


# In[12]:


a4_dims = (10.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x="Label", y="Tweet Count", data=tweets_cnt)


# In[13]:


max_fatures = 2000
maxlen=100
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(tweets_pd['Text'].values)
X = tokenizer.texts_to_sequences(tweets_pd['Text'].values)
X = pad_sequences(X,maxlen=maxlen)


# In[14]:


X.shape


# In[15]:


embed_dim = 100
lstm_out = 196


# In[16]:


Y = pd.get_dummies(tweets_pd['Label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[60]:


word_to_id = {k:(v) for k,v in tokenizer.word_index.items()}


# In[62]:


X_test[11]


# In[58]:


X_test[11]


# In[46]:


def check(x):
    x=np.array(x)
    return np.argmax(x)
     


# In[51]:


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


# In[22]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[ ]:


EMBEDDING_FILE=f'glove/glove.6B.50d.txt'


# In[23]:


EMBEDDING_FILE_100=f'glove/glove.6B.100d.txt'


# In[24]:


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
# embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


# ##### Changing the Glove dimension

# In[25]:


embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE_100))


# In[26]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[27]:


word_index = tokenizer.word_index
nb_words = min(max_fatures, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_dim))
for word, i in word_index.items():
    if i >= max_fatures: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


nb_words


# In[28]:


embedding_matrix.shape


# In[29]:


model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1],weights=[embedding_matrix]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[30]:


batch_size = 32
model.fit(X_train, Y_train, epochs = 50, batch_size=batch_size, verbose = 1)


# In[31]:


output = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)


# In[33]:


Y_test.shape


# In[32]:


print("score: %.2f" % (output[0]))
print("acc: %.2f" % (output[1]))


# In[35]:


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


# In[36]:


model.summary()


# In[37]:


batch_size = 32
model.fit(X_train, Y_train, epochs = 50, batch_size=batch_size, verbose = 1)


# In[38]:


output = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.3f" % (output[0]))
print("acc: %.3f" % (output[1]))


# In[39]:


model.save_weights('glove100d2.h5')


# In[40]:


y_pred=model.predict(X_test)


# In[42]:


pred_df=pd.DataFrame(y_pred,columns=['kemp_negative','kemp_positive','stacy_negative','stacy_positive'])


# In[47]:


pred_df['prediction']=pred_df.apply(lambda x:check(x),axis=1)


# In[49]:


actual_df=pd.DataFrame(Y_test,columns=['kemp_negative','kemp_positive','stacy_negative','stacy_positive'])


# In[52]:


actual_df['actual']=actual_df.apply(lambda x:actual(x),axis=1)


# In[54]:


pred_df['actual']=actual_df['actual']


# In[56]:


pred_df[pred_df['prediction']!=pred_df['actual']]


# In[63]:


pred_df=pred_df[['prediction','actual']]

acc=(len(pred_df[pred_df['prediction']==pred_df['actual']])/len(pred_df))*100


# In[64]:


print(f1_score(pred_df['actual'], pred_df['prediction'], average="macro"))
print(precision_score(pred_df['actual'], pred_df['prediction'], average="macro"))
print(recall_score(pred_df['actual'], pred_df['prediction'], average="macro"))   


# In[75]:


from sklearn.metrics import confusion_matrix
multiclass=confusion_matrix(pred_df['actual'], pred_df['prediction'])


# In[66]:


df_confusion = pd.crosstab(pred_df['actual'], pred_df['prediction'], rownames=['Actual'], colnames=['Predicted'], margins=False)


# In[67]:


df_confusion


# In[74]:


from mlxtend.plotting import plot_confusion_matrix


# In[88]:



fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                figsize=(11,11),
                                
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True)
plt.show()


# #### Testing on real data

# In[89]:


model.load_weights('glove100d2.h5')

