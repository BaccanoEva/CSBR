import pandas
import csv
import json
import os

if False:
	result_path = "/Users/liuyancen/Desktop/nlp_lxtong/nlp/result/" 
	sresult_path = "/Users/liuyancen/Desktop/nlp_lxtong/nlp/sresult/"
	mashup_edus = {}
	for i in os.listdir(result_path):
		l = i.split(".")
		mashup_edus[l[0]] = []
		f = open(result_path+i,"r")
		if "error" not in i:
			for line in f:
				if "_!" in line:
					tmp = line.split("_!")
					mashup_edus[l[0]].append(tmp[1])
		else:
			for line in f:
				mashup_edus[l[0]].append(line)
	json.dump(mashup_edus,open("mashup_edus.json","w"))

	service_edus = {}
	for i in os.listdir(sresult_path):
		l = i.split(".")
		service_edus[l[0]] = []
		f = open(sresult_path+i,"r")
		if "error" not in i:
			for line in f:
				if "_!" in line:
					tmp = line.split("_!")
					service_edus[l[0]].append(tmp[1])
		else:
			for line in f:
				service_edus[l[0]].append(line)
	json.dump(service_edus,open("service_edus.json","w"))

if False:
	groundtruth = {}
	groundtruth_reverse = {}
	ground_truth_path = "/Users/liuyancen/Desktop/API-Prefer/data/groundtruth.csv"
	index = 0
	reader = csv.reader(open(ground_truth_path,"r"))
	for line in reader:
		groundtruth[str(index)] = line
		for ind in line:
			if ind not in groundtruth_reverse:
				groundtruth_reverse[ind] = []
			groundtruth_reverse[ind].append(str(index))
		index+=1
	json.dump(groundtruth,open("groundtruth.json","w"))
	json.dump(groundtruth_reverse,open("groundtruth.reverse.json","w"))

if False:
	groundtruth = json.load(open("groundtruth.reverse.json","r"))
	mashup_edus = json.load(open("mashup_edus.json","r"))
	service_edus = json.load(open("service_edus.json","r"))

	service_semantics = {}
	for service,mashups in groundtruth.items():
		service_semantics[service] = []
		for item in service_edus[service]:
			service_semantics[service].append(item)
		for mashup in mashups:
			if mashup in mashup_edus:
				for item in mashup_edus[mashup]:
					service_semantics[service].append(item)
	json.dump(service_semantics,open("service.semantics.json","w"))
	print(len(service_semantics))

groundtruth = json.load(open("groundtruth.reverse.json","r"))
service_supplement = json.load(open("/Users/liuyancen/Desktop/API-Prefer/data/service_semantics/supplement_semantics.json","r"))
service_edus = json.load(open("service_edus.json","r"))

service_sentences = {}
for service in groundtruth.keys():
	text = ""
	for s in service_edus[service]:
		text+=s
		text+=" "
	if service in service_supplement:
		for w in service_supplement[service]:
			text+=w
			text+=" "
	service_sentences[service] = text
json.dump(service_sentences,open("service.sentences.json","w"))




