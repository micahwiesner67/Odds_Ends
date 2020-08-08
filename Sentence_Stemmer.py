# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:23:59 2020

@author: new
"""

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