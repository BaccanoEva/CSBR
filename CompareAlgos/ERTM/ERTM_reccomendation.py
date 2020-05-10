import json

from gensim import models, similarities, corpora
from collections import defaultdict
import os

import community
import networkx as nx

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def cal_distance(service_name,lines):
	#stoplist = set('for a of the and to in you your on is are am that by this or and can from with mashup as an so their if within api service apis services at such than it into be let lets all about using uses use more'.split())
	stoplist = stop_words
	punctions = [' ', '\n','\t', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-']
	filter_words = []
	for item in stoplist:
		filter_words.append(item)
	for item in punctions:
		filter_words.append(item)

	texts = [[word for word in document.lower().split() if word not in filter_words] for document in lines]
	#print(texts)
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token]+=1
	#texts1 = [[token for token in text if frequency[token] > 1] for text in texts]
	#print(texts1)
	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
	corpus_lsi = lsi_model[corpus_tfidf]
	index = similarities.MatrixSimilarity(lsi_model[corpus])

	G = nx.Graph()

	for i in range(len(lines)):
		#print(i)
		en_str = lines[i]
		en_str_vec = dictionary.doc2bow(en_str.lower().split())
		lsi_str_vec1 = lsi_model[en_str_vec]
		sims = index[lsi_str_vec1]
		for item in list(enumerate(sims)):
			if item[0]<=i or item[1]<0.8:
				continue
			G.add_edge(i, item[0], weight = item[1])

	partition = community.best_partition(G)
	size = float(len(set(partition.values())))
	pos = nx.spring_layout(G)
	semantics = {}
	for com in set(partition.values()) :
		semantics[com] = {}
		list_nodes = [nodes for nodes in partition.keys()if partition[nodes] == com]
		for node in list_nodes:
			for w in texts[node]:
				if contain_number(w):
					continue
				if w not in semantics[com]:
					semantics[com][w] = 0
				semantics[com][w]+=1
		#print(com, list_nodes)
		#print(com, semantics[com])
	json.dump(semantics,open("data/service_semantics/"+service_name+".json","w"))

def contain_number(s):
	for i in ['0','1','2','3','4','5','6','7','8','9','\\','\'s','/']:
		if i in s:
			return True
	return False

def ERTM(service_name,lines):
	stoplist = set('for a of the and to in you your on is are am that by this or and can from with mashup as an so their if within api service apis services at such than it into be let lets all about using uses use more'.split())
	punctions = [' ', '\n','\t', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-']
	filter_words = []
	for item in stoplist:
		filter_words.append(item)
	for item in punctions:
		filter_words.append(item)

	texts = [[word for word in document.lower().split() if word not in filter_words] for document in lines]
	#print(texts)
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			if(contain_number(token)):
				continue
			frequency[token]+=1
	frequency_sorted = sorted(frequency.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
	#print(frequency_sorted)
	
	semantics = []
	for item in frequency_sorted:
		if(item[1]>=(len(lines)/50)):
			semantics.append(item[0])
	#print(semantics)
	return semantics
	#json.dump(semantics,open("data/service_semantics/"+service_name+".json","w"))

service_semantics = json.load(open("data/service.semantics.json","r"))
print(len(service_semantics))
#supplement_semantics = {}
index = 0
for k,v in service_semantics.items():
	print(index,k)
	if(k == "1"):
		#continue
	#supplement_semantics[k] = cal_distance_sort(k,v)
		cal_distance(k,v)	
		index+=1
#json.dump(supplement_semantics,open("data/service_semantics/supplement_semantics.json","w"))
