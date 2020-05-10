# coding:utf-8

import os
import sys
import xlrd
import string
import pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer


if __name__ == "__main__":
    wnl = nltk.WordNetLemmatizer()
    apidir = '/home/pengqianyang/nlp/nlp/api.xlsx'
    apidata = xlrd.open_workbook(apidir)
    apitable = apidata.sheet_by_index(0)
    corpus = []
    for i in range(1,12141):
            corpus.append(apitable.row_values(i)[8])

    raw_material = [line.strip() for line in corpus]
    texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in raw_material]

    english_stopwords = stopwords.words('english')
    texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
    texts_filtered = [[word for word in document if not word in string.punctuation] for document in texts_filtered_stopwords]

    texts_lemmatized = [[wnl.lemmatize(t) for t in document]  for document in texts_filtered]

    corpus = [' '.join(document) for document in texts_lemmatized]

    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    result = []
    for i in range(len(weight)):
        cur = []
        cur.append(apitable.row_values(i+1)[2])
        cate = [wrd.lower() for wrd in word_tokenize(apitable.row_values(i+1)[3])]
        cate_filtered = [wrd for wrd in cate if not wrd in string.punctuation]
        for cat in cate_filtered:
            cur.append((cat,1.0))
        print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
        for j in range(len(word)):
            if (weight[i][j] > 0.25):
                cur.append((word[j],weight[i][j]))
        print cur
        raw_input()
        result.append(cur)