import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import os
import string
import pickle
import re
import xlrd

from gensim import corpora, models, similarities
import logging

f=open('dictionary.dic','r')  
dictionary=pickle.load(f)  
f.close()  
f=open('lsi.lsi','r')  
lsi=pickle.load(f)  
f.close() 

a = open('out.txt','r') .read()
apiNum = [[i for i in api.split(' ') if  not(i == '')] for api in a.split('\n')]

apidir = '/home/pengqianyang/nlp/nlp/api.xlsx'
apidata = xlrd.open_workbook(apidir)
apitable = apidata.sheet_by_index(0)
f=open('rawresult.txt','w')  
for num in range(0,7000):
    dir = "/home/pengqianyang/nlp/nlp/result/"
    file = str(num)+".txt"
    print file
    if os.path.exists(dir+file):
        print>>f, file,
        raw_text = open(dir+file,'r').read()
        pattern = re.compile(r'text _!(.*?)_!',re.M)
        match = pattern.findall(raw_text)
        raw_material = [line.strip() for line in match]
        texts_tokenized = [[word.lower() for word in word_tokenize(document.decode('utf-8'))] for document in raw_material]

        english_stopwords = stopwords.words('english')
        texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
        texts_filtered = [[word for word in document if not word in string.punctuation] for document in texts_filtered_stopwords]

        st = LancasterStemmer()
        texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]
        texts = texts_stemmed
        corpus = [dictionary.doc2bow(text) for text in texts]
        index = similarities.MatrixSimilarity(lsi[corpus])
        #------------------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------------
        res2 = []
        for kapi in apiNum[num]:
            target =  unicode(apitable.row_values(int(kapi))[3]).strip()
            raw_material = target.strip()
            texts_tokenized = [word.lower() for word in word_tokenize(raw_material)]

            english_stopwords = stopwords.words('english')
            texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords]

            texts_filtered = [word for word in texts_filtered_stopwords if not word in string.punctuation] 

            st = LancasterStemmer()
            texts_stemmed = [st.stem(word) for word in texts_filtered]

            text = texts_stemmed

            ml_bow = dictionary.doc2bow(text)
            ml_lsi = lsi[ml_bow]
            sims = index[ml_lsi]

            res2.append(sims)
            sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        if res2 != []:
            max = [-1]*len(res2[0])
            min = [1]*len(res2[0])
            for i in res2:
                for j in range(len(i)):
                    if (i[j]>max[j]):
                        max[j] = i[j]
                    if (i[j]<min[j]):
                        min[j] = i[j]
            for i in range(len(res2)):
                for j in range(len(res2[i])):
                    if (max[j]!=min[j]):
                        res2[i][j] = (res2[i][j]-min[j])/(max[j]-min[j])
        #------------------------------------------------------------------------------------------------------------------------------------------
        res3 = []
        for kapi in apiNum[num]:
            target =  unicode(apitable.row_values(int(kapi))[6]).strip() #Here may exist problem when checking
            raw_material = target.strip()
            texts_tokenized = [word.lower() for word in word_tokenize(raw_material)]

            english_stopwords = stopwords.words('english')
            texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords]

            texts_filtered = [word for word in texts_filtered_stopwords if not word in string.punctuation] 

            st = LancasterStemmer()
            texts_stemmed = [st.stem(word) for word in texts_filtered]

            #for i in range(len(texts_filtered)):
            #    if texts_filtered[i] != texts_stemmed[i]:
            #        print texts_filtered[i] + ' --- ' + texts_stemmed[i]

            text = texts_stemmed

            ml_bow = dictionary.doc2bow(text)
            ml_lsi = lsi[ml_bow]
            sims = index[ml_lsi]

            res3.append(sims)
            sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        if res3!= []:
            max = [-1]*len(res3[0])
            min = [1]*len(res3[0])
            for i in res3:
                for j in range(len(i)):
                    if (i[j]>max[j]):
                        max[j] = i[j]
                    if (i[j]<min[j]):
                        min[j] = i[j]
            for i in range(len(res3)):
                for j in range(len(res3[i])):
                    if (max[j]!=min[j]):
                         res3[i][j] = (res3[i][j]-min[j])/(max[j]-min[j])
        #------------------------------------------------------------------------------------------------------------------------------------------
        for i in range(len(res2)):
            for j in range(len(res2[i])):
                res2[i][j] = res2[i][j]+res3[i][j]
            res2[i] = sorted(enumerate(res2[i]), key=lambda item: -item[1])
            print>>f, res2[i][0][0],
        print>>f,''
f.close()