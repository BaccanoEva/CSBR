import csv

csvreader = csv.reader(open("mashup.csv","r"))
csvwriter = csv.writer(open("mashup.lda.csv","w"))
csvwriter.writerow(["","name","desc"])
index = 0
for line in csvreader:
	if index == 0:
		index+=1
		continue
	
	csvwriter.writerow([index,line[2],line[6]])
	index+=1
	
