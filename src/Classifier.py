#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

import lightgbm as lgb

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import string

df = pd.read_csv('../data/analytics/train_2kmZucJ.csv')

test = pd.read_csv('../data/analytics/test_oJQbWVk.csv')

submission = pd.read_csv('../data/analytics/sample_submission_LnhVWA4.csv')

lemm = WordNetLemmatizer()

def clean_text(sentence):
    sentence = " ".join([x for x in sentence.split() if x not in stopwords.words('english')])
    sentence = "".join(x for x in sentence if x not in string.punctuation)
    sentence = " ".join([lemm.lemmatize(x, 'v') for x in sentence.split()])
    
    return sentence


df['clean_text'] = df['tweet'].apply(lambda word : clean_text(word))

test['clean_text'] = test['tweet'].apply(lambda word : clean_text(word))


def pre_process_data(train, test, vec):
    
    frame = [train['clean_text'], test['clean_text']]
    data = pd.concat(frame)
    
    print("Fitting vectorizer to data")
    cvec = vec(analyzer='word')
    cvec.fit(data)
    
    train_x, valid_x, train_y, valid_y = train_test_split(df['clean_text'], df['label'])
    
    print('Transformif text to vector')
    train_count = cvec.transform(train_x).astype('float64')
    valid_count = cvec.transform(valid_x).astype('float64')
    test_count = cvec.transform(test['clean_text']).astype('float64')
    
    return train_count, valid_count, train_y, valid_y, test_count


def convert_bol(pred, threhold=0.5):
    for i in range(len(pred)):
        if pred[i] >= threhold:
            pred[i] = 1
        else:
            pred[i] = 0
            
    return pred


def train_model(X, y, num_runs=2500):
    params = {}
    params['objective'] = 'binary'
    params['learning_rate'] = 0.05
    params['metric'] = 'binary_logloss'
    params['ntree'] = 500
    params['max_depth'] = 9
    params['min_data'] = 95
    
    train_count_df = lgb.Dataset(X, label=y)
    watchlist = [train_count_df]
    
    clf = lgb.train(params, train_count_df, num_runs, watchlist, verbose_eval=100)
    
    return clf
    

def evaluate_model(X, actual, model, threhold):
    
    pred = convert_bol(model.predict(X), threhold=threhold)
    print("Accuracy of the model    : {:}".format(accuracy_score(pred, actual)))
    print("F1 score of the model    : {:}".format(f1_score(pred, actual)))
    print("Kappa Score of the model : {:}".format(cohen_kappa_score(pred, actual)))


train_count, valid_count, train_y, valid_y, test_count = pre_process_data(df, test, vec=TfidfVectorizer)


train_count_c, valid_count_c, train_y, valid_y, test_count_c = pre_process_data(df, test, vec=CountVectorizer)


model = train_model(X=train_count_c, y=train_y, num_runs=4000)

evaluate_model(X=valid_count_c, actual=valid_y, model=model, threhold=0.5)

'''
Accuracy of the model    : 0.847979797979798
F1 score of the model    : 0.7108549471661865
Kappa Score of the model : 0.6077478971685819
'''

test_y = convert_bol(model.predict(test_count), threhold=0.5)
submission['label'] = test_y

submission.to_csv('../data/sub.csv', index=False)
