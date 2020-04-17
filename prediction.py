import numpy as np
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import json
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_squared_error
import os
cwd = os.getcwd()
import statistics
import sys



def dic_to_feature(all_con, length, part):
    """
    This function trims the text from the tuples in the dictionary, and
    reformat it as numpy array. If we want the first half, the length
    would be 2, and the part would be 'head', and If we want the whole
    transcript, the length would be 1, and the part would be 'head'.

    :param all_con: The dictionary from `text_to_dic`
    :param length: An integer decides how to split the text
    :param part: An string can only be 'head' or 'tail'.
    :return: Two numpy array. The first one is data features, and the 
    second one is label.
    """
    all_string = []
    all_label = []
    #Iterate the dictionary and get conversation and label in it.
    for conv, label in all_con.values():
        #Slice of conversation
        if "head" in part:
            all_string.append(conv[:len(conv)//int(length)])
        else:
            all_string.append(conv[len(conv)//int(length):])
        all_label.append(label)

    all_string = np.asarray(all_string)
    all_label = np.asarray(all_label)
    return all_string, all_label

def bagofword_vectorize(x_train, x_test):
    """
    This function implements bag-of-word model. It vectorizes the whole text
    from train data into word counts. The vector format then be apply to test
    set.

    :param x_train: Numpy array of training set (text).
    :param x_test: Numpy array of test set (text).
    :return: Numpy arrays but with word counts (vectors of integer).
    """
    #BagofWord
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    
    return x_train, x_test

def train_and_dev(train, label, model):
    """
    This function trains machine learning model and test it on development data.

    :param train: Numpy array from `dic_to_feature`
    :param label: Numpy array from `dic_to_feature`
    :param model: A string should be either 'logistic' or 'linear'
    :return: three numpy array of label, gold label of development set, predictions,
    and baseline prediction.
    """
    x_train, x_dev, y_train, y_dev = train_test_split(train, label, test_size=0.2, random_state=42)
    x_train, x_dev = bagofword_vectorize(x_train, x_dev)        
    y_pred = modeling(model, x_train, y_train, x_dev)
    
    #Baseline
    if 'linear' in model:
        y = statistics.mean(list(y_train))
        y_base = [y]*len(y_dev)
    else:
        if statistics.mean(list(y_train)) < 0.5:
            y_base = np.zeros(len(y_dev))
        else:
            y_base = np.ones(len(y_dev))

    return y_dev, y_pred, y_base

def predict_test(train, label, test, label_t, model):
    """
    This functions operate the same training and then apply it on test data.
    The scores would be printed.

    :param train: Numpy array from `dic_to_feature`
    :param label: Numpy array from `dic_to_feature`
    :param test: Numpy array from `dic_to_featur`
    :param label_t: Numpy array from `dic_to_feature`
    :param model: A string should be either 'logistic' or 'linear'
    """
    train, test = bagofword_vectorize(train, test)
    pred = modeling(model, train, label, test)

    if 'logistic' in model:
        acc = accuracy_score(label_t, pred)
        print("test score is: ", acc)
       #Baseline 
        if statistics.mean(list(label)) < 0.5:
            y_base = np.zeros((len(label_t)))
        else:
            y_base = np.ones(len(label_t))
    
        acc_base = accuracy_score(label_t, y_base)
        print("baseline score is:", acc_base)

    else:
        acc = mean_squared_error(label_t, pred, squared=False)
        print("test score is: ", acc)
        #Baseline
        y = statistics.mean(list(label))
        y_base = [y]*len(label_t)

        acc_base = mean_squared_error(label_t, y_base, squared=False)
        print("baseline score is: ", acc_base)

def evaluation(model, y_gold, y_pred, y_base):
    """
    The function print out the evaluation of model 
    performance on development set. The score would
    be printed without return,

    :param model: A string should be either "logistic" or "linear"
    :param y_gold: Development label. y_dev from `train__and_dev`
    :param y_pred: Predictions. y_pred from `train_and_dev`
    :param y_base: Baeline. y_base from `train_and_dev`
    """
    if "logistic" in model:
        if not y_base:
            acc = accuracy_score(y_gold, y_pred)
            print("prediction score is: ", acc)
        else:
            acc_base = accuracy_score(y_gold, y_pred)
            print("baseline score is: ", acc_base)

    else:
        if not y_base:
            acc = mean_squared_error(y_gold, y_pred, squared = False)
            print("RMSE is: ", acc)
        else:
            acc_base = mean_squared_error(y_gold, y_pred, squared = False)
            print("baseline of RMSE: ", acc_base)
    

def modeling(model, x_train, y_train, x_test):
    """
    This functions builds the machine learning model and feeds
    data in it.

    :param model: A string should be either "logistic" or "linear"
    :param x_train: Numpy array of training features
    :param y_train: Numpy array of training label
    :param x_test: Numpy array of testing feature
    :return: Predictions of the label.
    """
    if "logistic" in model:
        #LogisticRegression
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=1000, max_iter=300)
    elif "linear" in model:
        clf = LinearRegression()

    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return pred
