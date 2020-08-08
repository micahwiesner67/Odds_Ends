# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:29:46 2020

@author: new
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
    
print(os.getcwd())
os.chdir('Downloads')

#This data is from a kaggle challenge called real or not
df = pd.read_csv('train.csv')
X = df['text'].values
y = df['target'].values

#%%
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

porter = PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

df['stemmed'] = [stemSentence(str(tweet)) for tweet in df['text']]

#%% Split data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['stemmed'], y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%% Stemmed MNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=1, 
                                           stop_words='english')

X_train_vec = unigram_count_vectorizer.fit_transform(X_train)
X_test_vec = unigram_count_vectorizer.transform(X_test)



nb_clf = MultinomialNB()
nb_clf.fit(X_train_vec,y_train)
y_pred = nb_clf.predict(X_test_vec)
print(f1_score(y_pred,y_test))

#%%
feature_ranks = sorted(zip(nb_clf.feature_log_prob_[0], unigram_count_vectorizer.get_feature_names()))
#very_negative_features = feature_ranks[10:]
#print(very_negative_features)

print(feature_ranks[-10:])

#%% BernoulliNB
from sklearn.naive_bayes import BernoulliNB

unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=3, 
                                           stop_words=None)

X_train_vec = unigram_count_vectorizer.fit_transform(X_train)
X_test_vec = unigram_count_vectorizer.transform(X_test)

Bnb_clf = BernoulliNB()
Bnb_clf.fit(X_train_vec,y_train)
y_pred = Bnb_clf.predict(X_test_vec)
print(f1_score(y_pred,y_test))

#%% LinearSVC
from sklearn.svm import LinearSVC

unigram_bool_vectorizer = CountVectorizer(
        encoding='latin-1', binary=True, 
        min_df=14, stop_words=None, max_df=0.95)

X_train_vec = unigram_bool_vectorizer.fit_transform(X_train)
X_test_vec = unigram_bool_vectorizer.transform(X_test)

# initialize the MNB model
SVM_clf = LinearSVC(C=1)

SVM_clf.fit(X_train_vec,y_train)

y_pred = SVM_clf.predict(X_test_vec)

print(f1_score(y_pred,y_test))






