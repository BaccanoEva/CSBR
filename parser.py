import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import os
import string
import pickle

from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

raw_material = [line.strip() for line in open('material.txt','r').read().split('-=this is the spread line=-')]
texts_tokenized = [[word.lower() for word in word_tokenize(document.decode('utf-8'))] for document in raw_material]

english_stopwords = stopwords.words('english')
texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]

texts_filtered = [[word for word in document if not word in string.punctuation] for document in texts_filtered_stopwords]

st = LancasterStemmer()
texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]

#for i in range(len(texts_filtered)):
#    if texts_filtered[i] != texts_stemmed[i]:
#        print texts_filtered[i] + ' --- ' + texts_stemmed[i]

#all_stems = sum(texts_stemmed, [])
#stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
#texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
#To be further considered
texts = texts_stemmed

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)

fd=open('dictionary.dic','w')  
pickle.dump(dictionary,fd,0) 
fd.close()  
fl=open('lsi.lsi','w')  
pickle.dump(lsi,fl,0) 
fl.close()  
