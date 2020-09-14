"""
Filename: MultiClassClassifier.py
Author: Akash Desai, Vaibhav Joshi
Description: Implementation of Multi Class classifier using various data mining techniques
"""

import praw
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import sklearn
import sklearn.utils as util
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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

    # Make authorized instances of reddit
    sub1 = reddit.subreddit('soccer').top(limit=900)
    sub2 = reddit.subreddit('britishproblems').top(limit=900)
    sub3 = reddit.subreddit('learnprogramming').top(limit=900)
    fields = ["title"]  # Fields to be used for data mining methods
    all_posts = []  # Stores the list of posts

    # Create dataframe and assign value to the fields accordingly
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

    for post in sub3:
        to_dict = vars(post)
        sub_dict = {field: to_dict[field] for field in fields}
        sub_dict["target"] = 2
        all_posts.append(sub_dict)

    dataframe = pd.DataFrame(all_posts)
    cols = list(dataframe.columns)
    cols[cols.index('target')], cols[-1] = cols[-1], cols[cols.index('target')]
    dataframe = dataframe[cols]

    # Creates an instance of TF-IDF vector for feature extraction. Stop words will be ignored.
    tfidf_transformer = TfidfVectorizer(stop_words=sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
    # Creates a feature vector for post's title
    X_train_title_counts = pd.DataFrame((tfidf_transformer.fit_transform(dataframe["title"].values)).todense())
    Y = pd.DataFrame(dataframe["target"].values)

    # Splits the data into training set, development set and test set
    train_X, train_Y, dev_X,dev_Y,test_X, test_Y = get_training_and_testing_sets(X_train_title_counts, Y)


    Accuracy = {}

    # Perform various data mining methods on the data
    Train_SVC(train_X, train_Y, test_X, test_Y, Accuracy)
    Train_RandomForest(train_X, train_Y, test_X, test_Y, Accuracy)
    train_X=train_X.append(dev_X)



def Train_RandomForest(train_X, train_Y, test_X, test_Y, Accuracy):
    """
    Implementation of Random Forest
    :param train_X: Training Set Features
    :param train_Y: Training Set Label
    :param test_X: Test Set Feature
    :param test_Y: Test Set Lable
    :param Accuracy: List which stores accuracy
    :return: None
    """
    rdf = RandomForestClassifier(n_estimators=100)
    rdf.fit(train_X, train_Y)
    Y_pred = rdf.predict(test_X)
    Accuracy["Random Forest"] = metrics.accuracy_score(test_Y, Y_pred)
    print("Accuracy of Random Forest: %.2f%%" % (Accuracy["Random Forest"] * 100))
    #Confusion Matrix
    cm_rf = confusion_matrix(test_Y, Y_pred)
    rf_cm = pd.DataFrame(cm_rf, range(3), range(3))
    sn.set(font_scale=1.7)
    sn.heatmap(rf_cm, annot=True, annot_kws={"size": 15}, fmt='g', cmap="Greens")

def Train_SVC(train_X, train_Y, test_X, test_Y, Accuracy):
    """
    Implementation of Support Vector Classifier
    :param train_X: Training Set Features
    :param train_Y: Training Set Label
    :param test_X: Test Set Feature
    :param test_Y: Test Set Lable
    :param Accuracy: List which stores accuracy
    :return:None
    """
    clf = SVC(probability=True, C=5, gamma='auto', kernel='linear')
    clf.fit(train_X, train_Y)
    y_predict = clf.predict(test_X)
    Accuracy["SVM"] = metrics.accuracy_score(test_Y, y_predict)
    print("Accuracy of SVM: %.2f%%" % (Accuracy["SVM"] * 100))
    # Confusion Matrix
    cm_rf = confusion_matrix(test_Y, y_predict)
    rf_cm = pd.DataFrame(cm_rf, range(3), range(3))
    sn.set(font_scale=1.7)
    sn.heatmap(rf_cm, annot=True, annot_kws={"size": 15}, fmt='g', cmap="Greens")


def get_training_and_testing_sets(data, Y):
    """
    Splits the data into training set, development set and test set
    :param data: Features of the data
    :param Y: Target Label of the data
    :return: Training Set, Development Set and Test Set
    """
    data = pd.concat([data, Y], axis=1)
    x,y=data.shape
    train_X_sub1=data[0:x//6]
    dev_X_sub1 = data[x//6:x//6 + x//12]
    test_X_sub1 = data[x//6 + x//12:x//3]

    train_X_sub2 = data[x//3:x//3+x//6]
    dev_X_sub2 = data[x//6 + x//3:x//3 + x//6 + x//12]
    test_X_sub2 = data[x//3 + x//6 + x//12:2*x//3]

    train_X_sub3 = data[2*x//3:(2*x//3) +x//6]
    dev_X_sub3 = data[x//6 + 2*x//3: (2*x//3) + x//6 + x//12]
    test_X_sub3 = data[2*x//3 + x//6 + x//12:x]

    train_X=train_X_sub1.append(train_X_sub2,ignore_index = True)
    train_X =train_X.append(train_X_sub3,ignore_index = True)
    dev_X= dev_X_sub1.append(dev_X_sub2,ignore_index = True)
    dev_X = dev_X.append(dev_X_sub3,ignore_index = True)
    test_X = test_X_sub1.append(test_X_sub2,ignore_index = True)
    test_X = test_X.append(test_X_sub3,ignore_index = True)


    train_X = util.shuffle(train_X)
    train_X = train_X.reset_index(drop=True)

    dev_X = util.shuffle(dev_X)
    dev_X = dev_X.reset_index(drop=True)

    test_X = util.shuffle(test_X)
    test_X = test_X.reset_index(drop=True)

    train_X_final=train_X
    dev_X_final = dev_X
    test_X_final = test_X
    x, y = train_X_final.shape
    train_X = train_X_final.iloc[:, 0:y - 1]
    train_Y = train_X_final.iloc[:, y - 1]

    x, y = test_X_final.shape
    test_X = test_X_final.iloc[:, 0:y - 1]
    test_Y = test_X_final.iloc[:, y - 1]

    x, y = dev_X_final.shape
    dev_X = dev_X_final.iloc[:, 0:y - 1]
    dev_Y = dev_X_final.iloc[:, y - 1]

    return train_X, train_Y, dev_X,dev_Y,test_X, test_Y



def main():
    API_client = 'JmV8nK1GpgmTdA'
    API_secret = 'Ka8kIzZZKiX2HBA_bm9HR4aY_-k'
    user_agent = 'FIS Project'
    user = 'vj_34'
    passwd = 'Myreddit123'

    reddit_test(API_client, API_secret, user_agent, user, passwd)


if __name__ == '__main__':
    main()
