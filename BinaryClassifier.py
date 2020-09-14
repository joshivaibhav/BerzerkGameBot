"""
Filename: BinaryClassifier.py
Author: Akash Desai, Vaibhav Joshi
Description: Implementation of binary classifier using various data mining techniques
"""

import praw
import math
import numpy as np
import pandas as pd
import sklearn
import seaborn as sn
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
from sklearn.metrics import confusion_matrix


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

    #Make authorized instances of reddit
    sub1 = reddit.subreddit('soccer').top(limit=900)
    sub2 = reddit.subreddit('britishproblems').top(limit=900)
    fields = ["title"] #Fields to be used for data mining methods
    all_posts = [] #Stores the list of posts

    #Create dataframe and assign value to the fields accordingly
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

    dataframe = pd.DataFrame(all_posts)
    cols = list(dataframe.columns)
    cols[cols.index('target')], cols[-1] = cols[-1], cols[cols.index('target')]
    dataframe = dataframe[cols]

    #Creates an instance of TF-IDF vector for feature extraction. Stop words will be ignored.
    tfidf_transformer = TfidfVectorizer(stop_words=sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
    #Creates a feature vector for post's title
    X_train_title_counts = pd.DataFrame((tfidf_transformer.fit_transform(dataframe["title"].values)).todense())
    Y = pd.DataFrame(dataframe["target"].values)

    #Splits the data into training set, development set and test set
    train_X, train_Y, test_X, test_Y,dev_X,dev_Y = get_training_and_testing_sets(X_train_title_counts, Y)

    ROC_List = {}
    Accuracy = {}

    #Perform various data mining methods on the data
    Train_SVC(train_X, train_Y, test_X, test_Y, ROC_List, Accuracy)
    Train_NaiveBayes(train_X, train_Y, test_X, test_Y, ROC_List, Accuracy)
    Train_Logistic(train_X, train_Y, test_X, test_Y, ROC_List, Accuracy)
    Train_RandomForest(train_X, train_Y, test_X, test_Y, ROC_List, Accuracy)

    #Plots ROC curve
    plot_roc_curve_and_accuracy(ROC_List, Accuracy)


def Train_RandomForest(train_X, train_Y, test_X, test_Y, ROC_List, Accuracy):
    """
    Implementation of Random Forest
    :param train_X: Training Set Features
    :param train_Y: Training Set Label
    :param test_X: Test Set Feature
    :param test_Y: Test Set Lable
    :param ROC_List: List which stores parameters for ROC curve
    :param Accuracy: List which stores accuracy
    :return: None
    """
    rdf = RandomForestClassifier(n_estimators=100)
    rdf.fit(train_X, train_Y)
    Y_pred = rdf.predict(test_X)
    Accuracy["Random Forest"] = metrics.accuracy_score(test_Y, Y_pred)
    print("Accuracy of Random Forest: %.2f%%" % (Accuracy["Random Forest"] * 100))
    y_pred_proba = rdf.predict_proba(test_X)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(test_Y, y_pred_proba)
    ROC_List["Random Forest"] = [fpr, tpr]

    #Confusion Matrix
    cm_rf = confusion_matrix(test_Y, Y_pred)
    rf_cm = pd.DataFrame(cm_rf, range(2), range(2))
    sn.set(font_scale=1.7)
    sn.heatmap(rf_cm, annot=True, annot_kws={"size": 15}, fmt='g', cmap="Greens")


    auc = metrics.roc_auc_score(test_Y, y_pred_proba)
    plt.plot(fpr, tpr, label="Random Forest AUC=" + str(auc))
    plt.legend(loc=4)
    plt.title("Random Forest")
    plt.show()



def Train_SVC(train_X, train_Y, test_X, test_Y, ROC_List, Accuracy):
    """
    Implementation of Support Vector Classifier
    :param train_X: Training Set Features
    :param train_Y: Training Set Label
    :param test_X: Test Set Feature
    :param test_Y: Test Set Lable
    :param ROC_List: List which stores parameters for ROC curve
    :param Accuracy: List which stores accuracy
    :return:None
    """
    clf = SVC(probability=True,C=5, gamma='auto', kernel='linear')
    clf.fit(train_X, train_Y)
    y_predict = clf.predict(test_X)
    Accuracy["SVM"] = metrics.accuracy_score(test_Y, y_predict)
    print("Accuracy of SVM: %.2f%%" % (Accuracy["SVM"] * 100))
    y_pred_proba = clf.predict_proba(test_X)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(test_Y, y_pred_proba)
    ROC_List["SVM"] = [fpr, tpr]
    #Confusion Matrix
    cm_rf = confusion_matrix(test_Y, y_predict)
    rf_cm = pd.DataFrame(cm_rf, range(2), range(2))
    sn.set(font_scale=1.7)
    sn.heatmap(rf_cm, annot=True, annot_kws={"size": 15}, fmt='g', cmap="Greens")

    auc = metrics.roc_auc_score(test_Y, y_pred_proba)
    plt.plot(fpr, tpr, label="SVC AUC=" + str(auc))
    plt.legend(loc=4)
    plt.title("SVC")
    plt.show()



def Train_Logistic(train_X, train_Y, test_X, test_Y, ROC_List, Accuracy):
    """
    Implementation of Logistic Regression
    :param train_X: Training Set Features
    :param train_Y: Training Set Label
    :param test_X: Test Set Feature
    :param test_Y: Test Set Lable
    :param ROC_List: List which stores parameters for ROC curve
    :param Accuracy: List which stores accuracy
    :return:None
    """
    logistic = LogisticRegression()
    logistic.fit(train_X, train_Y)
    y_predict=logistic.predict(test_X)
    Accuracy["Logistic"] = logistic.score(test_X, test_Y)
    print("Accuracy of Logistic Regression: %.2f%%" % (Accuracy["Logistic"] * 100))
    y_pred_proba = logistic.predict_proba(test_X)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(test_Y, y_pred_proba)
    ROC_List["Logistic Regression"] = [fpr, tpr]
    # Confusion Metrix
    cm_rf = confusion_matrix(test_Y, y_predict)
    rf_cm = pd.DataFrame(cm_rf, range(2), range(2))
    sn.set(font_scale=1.7)
    sn.heatmap(rf_cm, annot=True, annot_kws={"size": 15}, fmt='g', cmap="Greens")

    """
    auc = metrics.roc_auc_score(test_Y, y_pred_proba)
    plt.plot(fpr, tpr, label="Logistic AUC=" + str(auc))
    plt.legend(loc=4)
    plt.title("Logistic Regression")
    plt.show()
    """


def Train_NaiveBayes(train_X, train_Y, test_X, test_Y, ROC_List, Accuracy):
    """
    Implementation of Naive Bayes
    :param train_X: Training Set Features
    :param train_Y: Training Set Label
    :param test_X: Test Set Feature
    :param test_Y: Test Set Lable
    :param ROC_List: List which stores parameters for ROC curve
    :param Accuracy: List which stores accuracy
    :return:None
    """
    nb = MultinomialNB().fit(train_X, train_Y)
    y_predict = nb.predict(test_X)
    Accuracy["Naive Bayes"] = metrics.accuracy_score(test_Y, y_predict)
    print("Accuracy of Naive Bayes: %.2f%%" % (Accuracy["Naive Bayes"] * 100))
    y_pred_proba = nb.predict_proba(test_X)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(test_Y, y_pred_proba)
    ROC_List["Naive Bayes"] = [fpr, tpr]
    # Confusion Metrix
    cm_rf = confusion_matrix(test_Y, y_predict)
    rf_cm = pd.DataFrame(cm_rf, range(2), range(2))
    sn.set(font_scale=1.7)
    sn.heatmap(rf_cm, annot=True, annot_kws={"size": 15}, fmt='g', cmap="Greens")


    auc = metrics.roc_auc_score(test_Y, y_pred_proba)
    plt.plot(fpr, tpr, label="NB AUC=" + str(auc))
    plt.legend(loc=4)
    plt.title("Naive Bayes")
    plt.show()

def get_training_and_testing_sets(data, Y):
    data = pd.concat([data, Y], axis=1)
    training_split = 0.5
    development_split = 0.25
    test_split = 0.5
    split_index_1 = math.floor(len(data) * training_split)
    split_index_2 = math.floor(len(data) * test_split)
    sub1 = data.iloc[:split_index_1]
    sub2 = data.iloc[split_index_1: split_index_2 + split_index_1]
    training_1 = sub1.head(len(sub1) // 2)
    training_2 = sub2.head(len(sub2) // 2)

    development_split1 = data.iloc[split_index_1 // 2: split_index_1 // 2 + split_index_1 // 4]

    development_split2 = data.iloc[
                         split_index_2 + split_index_2 // 2: split_index_2 + split_index_2 // 2 + split_index_2 // 4]

    development_split_final = development_split1.append(development_split2, ignore_index=True)
    test1 = sub1.tail(len(sub1) // 4)
    test2 = sub2.tail(len(sub2) // 4)
    training_final = training_1.append(training_2, ignore_index=True)
    test_final = test1.append(test2, ignore_index=True)

    training_final = util.shuffle(training_final)
    training_final = training_final.reset_index(drop=True)

    development_split_final = util.shuffle(development_split_final)
    development_split_final = development_split_final.reset_index(drop=True)

    test_final = util.shuffle(test_final)
    test_final = test_final.reset_index(drop=True)

    x, y = training_final.shape
    train_X = training_final.iloc[:, 0:y - 1]
    train_Y = training_final.iloc[:, y - 1]
    x, y = test_final.shape
    test_X = test_final.iloc[:, 0:y - 1]
    test_Y = test_final.iloc[:, y - 1]
    x, y = development_split_final.shape
    dev_X = development_split_final.iloc[:, 0:y - 1]
    dev_Y = development_split_final.iloc[:, y - 1]
    return train_X, train_Y, test_X, test_Y, dev_X, dev_Y

def plot_roc_curve_and_accuracy(fpr_tpr_all, Accuracy):
    algo_list = []
    acc_list = []
    for x in Accuracy.keys():
        acc_list.append(Accuracy[x])
        algo_list.append(x)
        y_pos = np.arange(len(acc_list))
        plt.bar(y_pos, acc_list, align='center', alpha=0.5)
        plt.xticks(y_pos, algo_list)
        plt.title('Accuracy achieved -  r/soccer and r/britishproblems')
    plt.show()
    plt.close()

    keys = []
    for x in fpr_tpr_all.keys():
        keys.append(x)

    i = 0
    for x in fpr_tpr_all.values():
        plt.plot(x[0], x[1], lw=1, label=keys[i])
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        i += 1
    plt.show()


def curve(train_X, train_Y):
    # rdf= RandomForestClassifier(n_estimators=100).fit(train_X,train_Y)
    logreg = LogisticRegression().fit(train_X, train_Y)
    print(logreg)
    svm = SVC(gamma=0.001, kernel='linear', probability=True).fit(train_X, train_Y)
    train_sizes, train_scores, test_scores = learning_curve(svm,
                                                            train_X,
                                                            train_Y,
                                                            # Number of folds in cross-validation
                                                            cv=5,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=-1,
                                                            train_sizes=[100, 200, 500, 700])

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    API_client = 'JmV8nK1GpgmTdA'
    API_secret = 'Ka8kIzZZKiX2HBA_bm9HR4aY_-k'
    user_agent = 'FIS Project'
    user = 'vj_34'
    passwd = 'Myreddit123'

    reddit_test(API_client, API_secret, user_agent, user, passwd)


if __name__ == '__main__':
    main()
