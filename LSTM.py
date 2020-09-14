"""
Filename: LSTM.py
Author: Akash Desai, Vaibhav Joshi
Description: Implementation of LSTM network
"""
import praw
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding, Dropout
from keras.datasets import imdb


def reddit_test(API_client, API_secret, user_agent, user, passwd):
    """
    This method is used to fetch data from the subreddit and perform various
    data mining methods on the data.
    :param API_client: API_client
    :param API_secret: API_secrect
    :param user_agent: User Agent
    :param user: Username of application
    :param passwd: Password of Application
    :return:None
    """
    reddit = praw.Reddit(client_id=API_client,
                         client_secret=API_secret,
                         user_agent=user_agent,
                         username=user,
                         password=passwd)

    sub1 = reddit.subreddit('todayilearned').top(limit=900)
    sub2 = reddit.subreddit('history').top(limit=900)
    fields = ["title"]
    all_posts = []

    for post in sub1:
        to_dict = vars(post)
        sub_dict = {field: to_dict[field] for field in fields}
        sub_dict["target"] = 0
        all_posts.append(sub_dict)

    for post in sub2:
        to_dict = vars(post)
        sub_dict = {field: to_dict[field] for field in fields}
        sub_dict["target"] = 1
        all_posts.append(sub_dict)


    #Feature Extraction
    dataframe = pd.DataFrame(data=all_posts)
    tfidf_transformer = TfidfVectorizer(stop_words=sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
    X_train_title_counts = tfidf_transformer.fit_transform(dataframe["title"].values).todense()
    X_train_title_counts = pd.DataFrame(X_train_title_counts)


    Y = pd.DataFrame(dataframe["target"].values)
    train_X, test_X, train_Y, test_Y = train_test_split(X_train_title_counts, Y, train_size=0.50,random_state=1)
    dev_X, test_X, dev_Y, test_Y = train_test_split(test_X, test_Y, train_size=0.50,random_state=1)

    #Reshape into 3-D
    A=train_X.values
    train1_X = A.reshape(train_X.shape[0], 1, train_X.shape[1])

    # Reshape into 3-D
    A = dev_X.values
    dev1_X = A.reshape(dev_X.shape[0], 1, dev_X.shape[1])

    # Reshape into 3-D
    A = test_X.values
    test1_X = A.reshape(test_X.shape[0], 1, test_X.shape[1])

    #LSTM model
    model=Sequential()
    model.add(LSTM(1,input_shape=( 1,train_X.shape[1]),return_sequences=True))
    model.add(Dense(10))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    model.fit(train1_X,train_Y,epochs=100,validation_data=(dev1_X,dev_Y))
    scores = model.evaluate(test1_X, test_Y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

def main():
    API_client = 'JmV8nK1GpgmTdA'
    API_secret = 'Ka8kIzZZKiX2HBA_bm9HR4aY_-k'
    user_agent = 'FIS Project'
    user = 'vj_34'
    passwd = 'Myreddit123'

    reddit_test(API_client, API_secret, user_agent, user, passwd)


if __name__ == '__main__':
    main()
