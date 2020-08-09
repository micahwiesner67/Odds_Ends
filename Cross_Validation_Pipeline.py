# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 11:34:11 2020

@author: new
"""

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report

#%% 
#Remember 'vect' is an alias for countvectorizer
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'vect__stop_words': [None, 'english'],
    'tfidf__use_idf': [True, False],
}

x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, random_state=42)
fone_scorer = make_scorer(f1_score)

clf = GridSearchCV(text_clf, tuned_parameters, cv=5, scoring=fone_scorer)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))
#%%
print(clf.best_estimator_)