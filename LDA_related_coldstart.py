import os
import re
import csv
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import random
#process the data to stanford_tmt_input.csv and get the result from tmt_0.4.0_.jar
'''
f1 = open('stanford_tmt_input.csv',"w")
writer1 = csv.writer(f1)

def SaveLeaves(content):
    l = content.split('\n')
    result = []
    for i in l:
        if 'text' in i:
            t = i.split('_!')
            result.append(t[1])
    return result

def extract_mashup_content(mashupid):
    mashup_content = []
    dir = "/Users/liuyancen/Desktop/nlp_lxtong/nlp/result/"
    file = str(mashupid)+".txt"
    try:
        f = open(dir+file,'r')
        content = f.read()
        mashup_content = SaveLeaves(content)
    except:
        #continue
        f = open("/Users/liuyancen/Desktop/nlp_lxtong/nlp/mashup_in/"+file,'r')
        content = f.read()
        mashup_content.append(content)
    for i in mashup_content:
        #if len(i.split(' '))<3 or len(i.split(' '))==3:
        #    continue
        l = []
        l.append("mashup_"+str(mashupid))
        l.append(i)
        writer1.writerow(l)

def extract_service_content(serviceid):
    service_content = []
    dir = "/Users/liuyancen/Desktop/nlp_lxtong/nlp/services-in/"
    file = str(serviceid)+".txt"
    f = open(dir+file,'r')
    content = f.read()
    #print "content"
    #print content
    s = ""
    l = content.split('.')
    if(len(l)>3):
        s = l[0]+l[1]+l[2]
    else:
        s = content
    service_content.append(s)
    for i in service_content:
        #if len(i.split(' '))<3 or len(i.split(' '))==3:
        #    continue
        l = []
        l.append("service_"+str(serviceid))
        l.append(i)
        writer1.writerow(l)

for i in range(6976):
    print i
    extract_mashup_content(i)
for i in range(12140):
    print i
    extract_service_content(i)

f1.close()
'''
#parameters
combination_method = "one_or_two_or_three_combination_for_each_mashup.result"
set_num_threshold = 10
cluster_number = 5
extract_relationship_threshold = 0.5
Test_set_size = 100

#f = open("Test_set_"+str(Test_set_size))
f = open("Test_set_100_cold_start")

print "0. Test set"
Test_set = []

#generate random number for Test set
'''
while len(Test_set)<(Test_set_size+1):
    n = random.randint(0,6975)
    if n in Test_set:
        continue
    else:
        Test_set.append(n)
Test_set.sort()

for i in Test_set:
    f.write(str(i))
    f.write("|")
'''
l = f.readline()
t = l.split("|")
for i in range(len(t)-1):
    Test_set.append(int(t[i]))

f.close()

print "1. Initialize the mashup_service_lda_vector"

f = open('/Users/liuyancen/Desktop/stanford_tmt/stanford_tmt_input/document-topic-distributions.csv',"r")
reader = csv.reader(f)

mashup_service_lda_vector = {}
for item in reader:
    vector = []
    for i in range(1,51):
        vector.append(item[i])
    if mashup_service_lda_vector.has_key(item[0]):
        mashup_service_lda_vector[item[0]].append(vector)
    else:
        mashup_service_lda_vector[item[0]] = []
        mashup_service_lda_vector[item[0]].append(vector)

#print mashup_service_lda_vector["service_20"]

f.close()

def calculate_sim_lda(mashup_id,service_set):
    mashup_content = []
    if mashup_service_lda_vector.has_key("mashup_"+str(mashup_id)):
        mashup_content = mashup_service_lda_vector["mashup_"+str(mashup_id)]
    else:
        mashup_content = []

    service_content = []
    for i in service_set:
        service = []
        for j in i:
            if mashup_service_lda_vector.has_key("service_"+str(j)):
                service.append(mashup_service_lda_vector["service_"+str(j)][0])
            else:
                service.append([0]*50)
        service_content.append(service)

    sim_result = []
    #print "mashup_content_size,",len(mashup_content)
    #print "service_set_content_size,",len(service_content)
    for j in mashup_content:
        sim_set = []
        for service_set in service_content:
            sim = []
            for service in service_set:
                if service == [0]*50:
                    s = 0
                else:
                    s = np.dot(np.array(service,dtype=float),np.array(j,dtype=float))/(np.linalg.norm(np.array(service,dtype=float))*np.linalg.norm(np.array(j,dtype=float)))
                #try:
                #    s = np.dot(np.array(i),np.array(j))/(np.linalg.norm(np.array(i))*np.linalg.norm(np.array(j)))
                #except ValueError:
                #    s = 0
                sim.append(s)
            sim_set.append(sim)
        sim_result.append(sim_set)
    #print sim_result
    return sim_result

#mashup_id = 5430
#service_set = [[44,47],[136,164]]
#sim_result = calculate_sim_lda(mashup_id,service_set)

#5421 [[44,47],[1,141],[136,316],[26,63]]
#5426 [[1,2],[0,8],[124],[382]]
#5430 [[44,47],[136,164]]
#5432 [[0,20],[447]]
#5440 [[3,4197],[7769]]
#5266 [[44,47],[128,164]]
#define the relationship between sentence and service set
def extract_relationship(mashup_id,service_set,sim_result):
    #print "mashup_id",mashup_id
    #print "service_set",service_set

    relation = []
    for i in service_set:
        relation.append([])

    for i in range(len(sim_result)):
        l = []
        for j in sim_result[i]:
            j.sort()
            l.append(j[-1])

        #print l
        flag = True
        for j in range(len(l)):
            if l[j] > extract_relationship_threshold or l[j] == extract_relationship_threshold:
                relation[j].append(i)
                flag = False
        if flag:
            for item in relation:
                item.append(i)
    #how to ensure that every service set been allocated at least one sentence vector
    #but this part low the recall

    '''
    F = False
    for i in relation:
        if len(i)==0:
            F = True
    threshold = extract_relationship_threshold + 0.1
    while F and threshold < 1.1:
        relation = []
        for i in service_set:
            relation.append([])

        for i in range(len(sim_result)):
            l = []
            for j in sim_result[i]:
                j.sort()
                l.append(j[-1])

            #print l
            flag = True
            for j in range(len(l)):
                if l[j] > threshold or l[j] == threshold:
                    relation[j].append(i)
                    flag = False
            if flag:
                for item in relation:
                    item.append(i)

        for i in relation:
            if len(i)==0:
                F = True
        threshold += 0.1
    '''
    #print "relation:"
    #print relation

    return relation

#extract_relationship(mashup_id,service_set,sim_result)
print "2. Cold Start Service"
g = {}
csvfile = open('groundtruth.csv','r')
reader = csv.reader(csvfile)

index = 0
for item in reader:
    g[index] = item
    index += 1
csvfile.close()
print "index = ",index

gt_reverse = {}
items = g.items()

for key,value in items:
    for i in value:
        if gt_reverse.has_key(i):
            if key in gt_reverse[i]:
                n = 1
            else:
                gt_reverse[i].append(key)
        else:
            gt_reverse[i] = []
            gt_reverse[i].append(key)

one_time_service = []

items = gt_reverse.items()
for key,value in items:
    if len(value)==1:
        if value[0] in one_time_service:
            n=1
        else:
            one_time_service.append(int(key))
print "len(one_time_service)",len(one_time_service)

one_time_service_sim = {}
one_time_service_no_sim = []

for i in one_time_service:
    if mashup_service_lda_vector.has_key("service_"+str(i)):
        v1 = mashup_service_lda_vector["service_"+str(i)][0]
        similar_one = []
        for j in range(12140):
            if mashup_service_lda_vector.has_key("service_"+str(j)) and j!=i:
                v2 = mashup_service_lda_vector["service_"+str(j)][0]
                s = np.dot(np.array(v1,dtype=float),np.array(v2,dtype=float))/(np.linalg.norm(np.array(v1,dtype=float))*np.linalg.norm(np.array(v2,dtype=float)))
                if s > 0.9:
                    similar_one.append(j)
        print len(similar_one)
        if len(similar_one)>5:
            one_time_service_sim[i]=random.sample(similar_one,5)
        elif len(similar_one)<=5 and len(similar_one)>0:
            one_time_service_sim[i]=similar_one
        else:
            one_time_service_no_sim.append(i)
#os._exit(0)
print "3. Calculate sim and extract the relation between sentence and service set"
items_t = one_time_service_sim.items()
one_time_service_sim_reverse = {}
for key,value in items_t:
    for i in value:
        if one_time_service_sim_reverse.has_key(i):
            one_time_service_sim_reverse[i].append(key)
        else:
            one_time_service_sim_reverse[i]=[]
            one_time_service_sim_reverse[i].append(key)
print one_time_service_sim_reverse
#os._exit(0)

f = open(combination_method)
service_set_sentence = {}#{(service_set):{mashup_id:[0,1,2...],...},...}
line = f.readline()
index = 0
while line:
    #print "index = ",index

    ser_list = []
    l1 = line.split('#')
    for i in range(0,len(l1)-1):
        l = []
        l2 = l1[i].split(' ')
        for j in range(0,len(l2)-1):
            l.append(int(l2[j]))
        ser_list.append(l)

    ser_list_total = []
    ser_list_total.append(ser_list)
    for i in range(len(ser_list)):
        for j in range(len(ser_list[i])):
            if one_time_service_sim_reverse.has_key(ser_list[i][j]):
                for k in one_time_service_sim_reverse[ser_list[i][j]]:
                    t = ser_list
                    t[i][j] = k
                    ser_list_total.append(t)


    #print ser_list
    for ser_list in ser_list_total:
        sim = calculate_sim_lda(index,ser_list)
        relation = extract_relationship(index,ser_list,sim)
        for i in range(len(ser_list)):
            k = tuple(ser_list[i])
            if service_set_sentence.has_key(k) == False:
                service_set_sentence[k]={}
            service_set_sentence[k][index]=relation[i]

    line = f.readline()
    index +=1

f.close()
print"len(service_set_sentence):", len(service_set_sentence)

service_set_vector = {}
items = service_set_sentence.items()

count = 0
for key,value in items:
    service_set_vector[key]=[]
    t = value.items()
    for k,v in t:

        if k in Test_set:#devide out the mashup in Test_set
            continue

        if mashup_service_lda_vector.has_key("mashup_"+str(k)):
            for i in v:
                service_set_vector[key].append(mashup_service_lda_vector["mashup_"+str(k)][i])
    #print len(service_set_vector[key])
    if len(service_set_vector[key]) > 3:
        count+=1

#print "count = ",count
#print int(5/2)

#print len(service_set_vector[(2,57)])
#s = np.array(service_set_vector[(2,57)])
#kmeans = KMeans(n_clusters=11, random_state=0).fit(s)
#print "kmeans.labels_",kmeans.labels_
#print "kmeans.cluster_centers_",kmeans.cluster_centers_

#f1 = open("service_set_semantic_based_on_lda.kb","w")
print "4. Using kmeans construct knowledge base"
knowledge_base = {}
items = service_set_vector.items()

#statistic the amount of services that are in the knowledge base
count_used = {}
for key,vector in items:
    for i in key:
        if count_used.has_key(i):
            count_used[i]+=1
        else:
            count_used[i] = 1
print "the amount of services in the knowledge base is ",len(count_used)

for key,vector in items:
    if len(vector) > set_num_threshold:
        knowledge_base[key]=[]
        s = np.array(vector)
        kmeans = KMeans(n_clusters = cluster_number,random_state = 0).fit(s)
        record = {}
        for i in kmeans.labels_:
            if record.has_key(i):
                record[i]+=1
            else:
                record[i]=1
        r = []
        t = record.items()
        for k,v,in t:
            if v > (len(vector)/5) or v == (len(vector)/5):
                r.append(k)

        record = {}
        for i in range(len(kmeans.labels_)):
            if kmeans.labels_[i] in r:
                if record.has_key(kmeans.labels_[i]):
                    record[kmeans.labels_[i]].append(vector[i])
                else:
                    record[kmeans.labels_[i]]=[]
                    record[kmeans.labels_[i]].append(vector[i])
        t = record.items()
        for k,v in t:
            knowledge_base[key].append(v)

    else:
        #if the set is not big enough for clustring, add the original sentence vector to it
        knowledge_base[key] = []
        l = []
        for i in vector:
            l.append(i)
        #for i in key:
        #    if mashup_service_lda_vector.has_key("service_"+str(i)):
        #        knowledge_base[key].append(mashup_service_lda_vector["service_"+str(i)])
        for i in key:
            if mashup_service_lda_vector.has_key("service_"+str(i)):
                l.append(mashup_service_lda_vector["service_"+str(i)][0])
        knowledge_base[key].append(l)

#suplement the simple service's description (just for the service in the used set)
for key,vector in items:
    for i in key:
        k = []
        k.append(i)
        k = tuple(k)
        if (knowledge_base.has_key(k)) == False:
            l = []
            if mashup_service_lda_vector.has_key("service_"+str(i)):
                l.append(mashup_service_lda_vector["service_"+str(i)][0])
            knowledge_base[k] = []
            knowledge_base[k].append(l)

for i in one_time_service_no_sim:
    k = []
    k.append(i)
    k = tuple(k)

    l = []
    if mashup_service_lda_vector.has_key("service_"+str(i)):
        l.append(mashup_service_lda_vector["service_"+str(i)][0])
    knowledge_base[k] = []
    knowledge_base[k].append(l)


print "5. Using knowledge base to do recommendation"
#recommendation method 1: calculate a total point and sort to select
def evaluate(value,content):
    if value == [[]]:
        return 0
    else:
        result = []
        for i in content:
            res = []
            for item in value:
                r = []
                for j in item:
                    if j == [0]*50:
                        s = 0
                    else:
                        s = np.dot(np.array(i,dtype=float),np.array(j,dtype=float))/(np.linalg.norm(np.array(i,dtype=float))*np.linalg.norm(np.array(j,dtype=float)))
                    r.append(s)
                r.sort()
                r.reverse()
                res.append(r[0])
            res.sort()
            res.reverse()
            result.append(res[0])
        return sum(result)

def evaluate_knowledge_base_between_test(index,t):
    content = []
    if mashup_service_lda_vector.has_key("mashup_"+str(index)):
        content = mashup_service_lda_vector["mashup_"+str(index)]
        point = {}
        items = knowledge_base.items()
        for key,value in items:
            point[key] = evaluate(value,content)
        res = sorted(point.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

        r = []
        for i in res:
            for j in i[0]:
                if j in r:
                    continue
                else:
                    r.append(j)
            if len(r)>50 or len(r)==50:
                break

        tmp = []
        rank = []
        for i in res:
            for j in i[0]:
                if j in tmp:
                    n=1
                else:
                    tmp.append(j)
                    if j in t:
                        rank.append(len(tmp))
                if len(rank)==len(t):
                    break
        if(len(rank)==0):
            ra = 0
        else:
            ra = float(sum(rank))/len(rank)
        return r,ra
    else:
        res = []
        return res,-1

#recommendation method 2: recommend bundle and the algorithm is under considering
#initialize the set combination for next step to use
set_combination = {}
f = open(combination_method)
line = f.readline()
index = 0
while line:
    ser_list = []
    l1 = line.split('#')
    for i in range(0,len(l1)-1):
        l = []
        l2 = l1[i].split(' ')
        for j in range(0,len(l2)-1):
            l.append(int(l2[j]))
        ser_list.append(l)

    if len(ser_list)>1:
        for i in range(len(ser_list)-1):
            t1 = tuple(ser_list[i])
            for j in range(i+1,len(ser_list)):
                t2 = tuple(ser_list[j])
                if set_combination.has_key(t1):
                    if set_combination[t1].has_key(t2):
                        set_combination[t1][t2] += 1
                    else:
                        set_combination[t1][t2] = 1
                else:
                    set_combination[t1] = {}
                    set_combination[t1][t2] = 1

    line = f.readline()
    index +=1

f.close()

set_combination_statistic = {}
items = set_combination.items()
for key, value in items:
    t = set_combination[key]
    r = sorted(t.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    l = []
    for i in r:
        if i[1] < 2:
            break
        l.append(i[0])
    if len(l)!=0:
        set_combination_statistic[key] = l


def evaluate_bundle(value,content):
    if value == [[]]:
        return 0
    else:
        result = []
        for i in content:
            res = []
            for item in value:
                r = []
                for j in item:
                    if j == [0]*50:
                        s = 0
                    else:
                        s = np.dot(np.array(i,dtype=float),np.array(j,dtype=float))/(np.linalg.norm(np.array(i,dtype=float))*np.linalg.norm(np.array(j,dtype=float)))
                    r.append(s)
                r.sort()
                r.reverse()
                res.append(r[0])
            res.sort()
            res.reverse()
            result.append(res[0])
        return list(result),sum(result)

def recommendation_bundle(index,bundle_amount):
    content = []
    if mashup_service_lda_vector.has_key("mashup_"+str(index)):
        content = mashup_service_lda_vector["mashup_"+str(index)]
        #print "len(content):",len(content)
        point = {} #{key:[point1,point2,...]}
        p = {}
        items = knowledge_base.items()
        for key,value in items:
            n1,n2 = evaluate_bundle(value,content)
            point[key]=n1
            p[key] = n2
        res = sorted(p.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

        res_bundle = []
        index = 0

        while len(res_bundle) < bundle_amount:
            rec = []
            flag = False
            c = [0]*len(content)
            v = point[res[index][0]]
            print v

            for i in res[index][0]:
                rec.append(i)

            for i in range(len(v)):
                if v[i] > 0.85:
                    c[i] = 1
                else:
                    flag = True

            #if the set has other set that always combined, consider them first
            if set_combination_statistic.has_key(tuple(res[index][0])) and flag:
                print "situation in"
                for s in set_combination_statistic[tuple(res[index][0])]:
                    print s
                    if flag == False:
                        break

                    v1 = point[s]
                    print v1
                    f = True
                    for i in range(len(v1)):
                        if v1[i]>0.85 and c[i] == 0:
                            f = False
                    if f:
                        continue
                    for i in s:
                        rec.append(i)

                    flag = False
                    for i in range(len(v1)):
                        if v1[i]>0.85:
                            c[i] = 1
                        elif c[i] == 0:
                            flag = True


            while flag:
                p = {}
                items = point.items()
                for key, vector in items:
                    p[key] = 0
                    for i in range(len(vector)):
                        if c[i] == 0:
                            p[key]+=vector[i]
                r = sorted(p.items(),lambda x,y : cmp(x[1],y[1]),reverse = True)
                v1 = point[r[0][0]]
                #print v1

                #if the newly find vector cannot change the situation, quit from the cicle
                #but problem may be : [0.7,0.85] [0.9,0.5]
                f = True
                for i in range(len(v1)):
                    if v1[i]>0.85 and c[i] == 0:
                        f = False
                if f:
                    break
                # this part is temporary, should be thought and complement(quan yi zhi ji)

                for i in r[0][0]:
                    rec.append(i)
                flag = False
                for i in range(len(v1)):
                    if v1[i]>0.85:
                        c[i] = 1
                    elif c[i] == 0:
                        flag = True
            print "rec = ",rec
            index += 1
            res_bundle.append(rec)

        res = []
        for i in res_bundle:
            for j in i:
                if j in res:
                    res = res
                else:
                    res.append(j)
        #print "res:",res
        return res

    else:
        res = []
        return res

#recommendation_bundle(375,10)
def recommendation_bundle_method2(index,bundle_amount):
    content = []
    if mashup_service_lda_vector.has_key("mashup_"+str(index)):
        content = mashup_service_lda_vector["mashup_"+str(index)]
        #print "len(content):",len(content)
        point = {} #{key:[point1,point2,...]}
        p = {}
        items = knowledge_base.items()
        for key,value in items:
            n1,n2 = evaluate_bundle(value,content)
            point[key]=n1
            p[key] = n2

        res = sorted(p.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

        res_bundle = []
        index = 0

        while len(res_bundle) < bundle_amount:
            rec = []
            flag = False
            c = [0]*len(content)
            v = point[res[index][0]]
            print v

            for i in res[index][0]:
                rec.append(i)

            for i in range(len(v)):
                if v[i] > 0.85:
                    c[i] = 1
                else:
                    flag = True

            #if the set has other set that always combined, consider them first
            if set_combination_statistic.has_key(tuple(res[index][0])) and flag:
                print "situation in"
                for s in set_combination_statistic[tuple(res[index][0])]:
                    print s
                    if flag == False:
                        break

                    v1 = point[s]
                    print v1
                    f = True
                    for i in range(len(v1)):
                        if v1[i]>0.85 and c[i] == 0:
                            f = False
                    if f:
                        continue
                    for i in s:
                        rec.append(i)

                    flag = False
                    for i in range(len(v1)):
                        if v1[i]>0.85:
                            c[i] = 1
                        elif c[i] == 0:
                            flag = True


            while flag:
                p = {}
                items = point.items()
                for key, vector in items:
                    p[key] = 0
                    for i in range(len(vector)):
                        if c[i] == 0 and vector[i] > 0.85:
                            p[key]+=vector[i]
                r = sorted(p.items(),lambda x,y : cmp(x[1],y[1]),reverse = True)
                v1 = point[r[0][0]]
                #print v1

                #if the newly find vector cannot change the situation, quit from the cicle
                #but problem may be : [0.7,0.85] [0.9,0.5]
                f = True
                for i in range(len(v1)):
                    if v1[i]>0.85 and c[i] == 0:
                        f = False
                if f:
                    break
                # this part is temporary, should be thought and complement(quan yi zhi ji)

                for i in r[0][0]:
                    rec.append(i)
                flag = False
                for i in range(len(v1)):
                    if v1[i]>0.85:
                        c[i] = 1
                    elif c[i] == 0:
                        flag = True
            print "rec = ",rec
            index += 1
            res_bundle.append(rec)

        res = []
        for i in res_bundle:
            for j in i:
                if j in res:
                    res = res
                else:
                    res.append(j)
        #print "res:",res
        return res

    else:
        res = []
        return res

def recommendation_bundle_method3(index,bundle_amount):
    content = []
    if mashup_service_lda_vector.has_key("mashup_"+str(index)):
        content = mashup_service_lda_vector["mashup_"+str(index)]
        #print "len(content):",len(content)
        point = {} #{key:[point1,point2,...]}
        p = {}
        items = knowledge_base.items()
        for key,value in items:
            n1,n2 = evaluate_bundle(value,content)
            point[key]=n1
            p[key] = n2

        res = sorted(p.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

        res_bundle = []
        index = 0

        while len(res_bundle) < bundle_amount:
            rec = []
            flag = False
            c = [0]*len(content)
            v = point[res[index][0]]
            #print v

            #prevent the newly set have all already in the result
            duplicate_flag = True
            for i in res[index][0]:
                r = False
                for j in res_bundle:
                    if i in j:
                        r = True
                if r == False:
                    duplicate_flag = False
                    break
            if duplicate_flag:
                index += 1
                continue

            for i in res[index][0]:
                rec.append(i)

            for i in range(len(v)):
                if v[i] > 0.85:
                    c[i] = 1
                else:
                    flag = True

            #if the set has other set that always combined, consider them first
            if set_combination_statistic.has_key(tuple(res[index][0])) and flag:
                print "situation 1"
                for s in set_combination_statistic[tuple(res[index][0])]:
                    #print s
                    if flag == False:
                        break

                    v1 = point[s]
                    #print v1
                    f = True
                    for i in range(len(v1)):
                        if v1[i]>0.85 and c[i] == 0:
                            f = False
                    if f:
                        continue
                    for i in s:
                        rec.append(i)

                    flag = False
                    for i in range(len(v1)):
                        if v1[i]>0.85:
                            c[i] = 1
                        elif c[i] == 0:
                            flag = True


            while flag:
                print "situation 2"
                p = {}
                items = point.items()
                for key, vector in items:
                    p[key] = 0
                    for i in range(len(vector)):
                        if c[i] == 0 and vector[i] > 0.85:
                            p[key]+=vector[i]
                r = sorted(p.items(),lambda x,y : cmp(x[1],y[1]),reverse = True)

                #prevent duplicate
                ind = 0
                '''
                #the effect of this part is relatively small and a little bit down so delete this part temporarily
                duplicate_flag = True
                while duplicate_flag and ind < 10:#ind<10? is this ok?
                    print r[ind][0]
                    for i in r[ind][0]:
                        n = False
                        for j in res_bundle:
                            if i in j:
                                n = True
                        if n==False:
                            duplicate_flag = False
                    ind += 1
                '''
                v1 = point[r[ind][0]]
                #print v1

                #if the newly find vector cannot change the situation, quit from the cicle
                #but problem may be : [0.7,0.85] [0.9,0.5]
                f = True
                for i in range(len(v1)):
                    if v1[i]>0.85 and c[i] == 0:
                        f = False
                if f:
                    break
                # this part is temporary, should be thought and complement(quan yi zhi ji)

                for i in r[ind][0]:
                    rec.append(i)
                flag = False
                for i in range(len(v1)):
                    if v1[i]>0.85:
                        c[i] = 1
                    elif c[i] == 0:
                        flag = True

            print "rec = ",rec
            index += 1
            res_bundle.append(rec)

        res = []
        for i in res_bundle:
            for j in i:
                if j in res:
                    res = res
                else:
                    res.append(j)
        #print "res:",res
        return res

    else:
        res = []
        return res

#combine the bundle recommendation with prediction
#the prediction is about whether the mashup is respond to one service or a service bundle
#I think first I should ensure if this kind of prediction is really matter.
#Then think about raise the acc of prediction

#store gt to groundtruth
groundtruth = []
csvfile = open('groundtruth.csv','r')
reader = csv.reader(csvfile)

for item in reader:
    groundtruth.append(item)
csvfile.close()

def recommendation_bundle_method4(index,bundle_amount):
    #before the recommendation step, check if the mashup in gt is consisted of one or more services
    gt = []
    for j in groundtruth[index]:
        gt.append(int(j))
    isSingle = False#is the mashup just contain one service or not
    if len(gt) == 1:
        isSingle = True
    print "isSingle = ",isSingle

    content = []
    if mashup_service_lda_vector.has_key("mashup_"+str(index)):
        if isSingle:
            content = mashup_service_lda_vector["mashup_"+str(index)]
            #print "len(content):",len(content)
            point = {} #{key:[point1,point2,...]}
            p = {}
            items = knowledge_base.items()
            for key,value in items:
                n1,n2 = evaluate_bundle(value,content)
                point[key]=n1
                p[key] = n2

            res = sorted(p.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            res_bundle = []
            index = 0
            while len(res_bundle) <bundle_amount:
                if len(res[index][0]) == 1:
                    rec = []
                    for i in res[index][0]:
                        rec.append(i)
                    res_bundle.append(rec)
                    index += 1
                    continue
                else:
                    index += 1
                    continue

            res = []
            for i in res_bundle:
                for j in i:
                    if j in res:
                        res = res
                    else:
                        res.append(j)
            #print "res:",res
            return res

        else:
            content = mashup_service_lda_vector["mashup_"+str(index)]
            #print "len(content):",len(content)
            point = {} #{key:[point1,point2,...]}
            p = {}
            items = knowledge_base.items()
            for key,value in items:
                n1,n2 = evaluate_bundle(value,content)
                point[key]=n1
                p[key] = n2

            res = sorted(p.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

            res_bundle = []
            index = 0

            while len(res_bundle) < bundle_amount:
                rec = []
                flag = False
                c = [0]*len(content)
                v = point[res[index][0]]
                #print v

                #prevent the newly set have all already in the result
                duplicate_flag = True
                for i in res[index][0]:
                    r = False
                    for j in res_bundle:
                        if i in j:
                            r = True
                    if r == False:
                        duplicate_flag = False
                        break
                if duplicate_flag:
                    index += 1
                    continue

                for i in res[index][0]:
                    rec.append(i)

                for i in range(len(v)):
                    if v[i] > 0.85:
                        c[i] = 1
                    else:
                        flag = True

                #if the set has other set that always combined, consider them first
                if set_combination_statistic.has_key(tuple(res[index][0])) and flag:
                    print "situation 1"
                    for s in set_combination_statistic[tuple(res[index][0])]:
                        #print s
                        if flag == False:
                            break

                        v1 = point[s]
                        #print v1
                        f = True
                        for i in range(len(v1)):
                            if v1[i]>0.85 and c[i] == 0:
                                f = False
                        if f:
                            continue
                        for i in s:
                            rec.append(i)

                        flag = False
                        for i in range(len(v1)):
                            if v1[i]>0.85:
                                c[i] = 1
                            elif c[i] == 0:
                                flag = True


                while flag:
                    print "situation 2"
                    p = {}
                    items = point.items()
                    for key, vector in items:
                        p[key] = 0
                        for i in range(len(vector)):
                            if c[i] == 0 and vector[i] > 0.85:
                                p[key]+=vector[i]
                    r = sorted(p.items(),lambda x,y : cmp(x[1],y[1]),reverse = True)

                    #prevent duplicate
                    ind = 0
                    '''
                    #the effect of this part is relatively small and a little bit down so delete this part temporarily
                    duplicate_flag = True
                    while duplicate_flag and ind < 10:#ind<10? is this ok?
                        print r[ind][0]
                        for i in r[ind][0]:
                            n = False
                            for j in res_bundle:
                                if i in j:
                                    n = True
                            if n==False:
                                duplicate_flag = False
                        ind += 1
                    '''
                    v1 = point[r[ind][0]]
                    #print v1

                    #if the newly find vector cannot change the situation, quit from the cicle
                    #but problem may be : [0.7,0.85] [0.9,0.5]
                    f = True
                    for i in range(len(v1)):
                        if v1[i]>0.85 and c[i] == 0:
                            f = False
                    if f:
                        break
                    # this part is temporary, should be thought and complement(quan yi zhi ji)

                    for i in r[ind][0]:
                        rec.append(i)
                    flag = False
                    for i in range(len(v1)):
                        if v1[i]>0.85:
                            c[i] = 1
                        elif c[i] == 0:
                            flag = True

                print "rec = ",rec
                index += 1
                res_bundle.append(rec)

            res = []
            for i in res_bundle:
                for j in i:
                    if j in res:
                        res = res
                    else:
                        res.append(j)
            #print "res:",res
            return res

    else:
        res = []
        return res

#apply a cf algorithm to complete the proposed algorithm
def cf(index,amount,test_set):
    print "index = ",index
    print mashup_service_lda_vector.has_key("mashup_"+str(index))
    content = []
    if mashup_service_lda_vector.has_key("mashup_"+str(index)):
        content = mashup_service_lda_vector["mashup_"+str(index)]

    if content == [[]] or content == []:
        print "null"
        return []

    com_content = {}
    for i in range(0,6976):
        if i in test_set:
            continue
        if mashup_service_lda_vector.has_key("mashup_"+str(i)):
            m = 0
            for j in mashup_service_lda_vector["mashup_"+str(i)]:
                for k in content:
                    s = np.dot(np.array(j,dtype=float),np.array(k,dtype=float))/(np.linalg.norm(np.array(j,dtype=float))*np.linalg.norm(np.array(k,dtype=float)))
                    if s > m:
                        m = s
            com_content[i] = m
    res = sorted(com_content.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    cf_res = []
    index = 0
    while len(cf_res)<amount:
        for i in groundtruth[res[index][0]]:
            if i in cf_res:
                cf_res = cf_res
            else:
                cf_res.append(i)
        index += 1
    r = []
    for i in cf_res:
        r.append(int(i))
    return r


# this part is for the first recommendation method
p_1 = []
r_1 = []
p_2 = []
r_2 = []
p_3 = []
r_3 = []
p_4 = []
r_4 = []
p_5 = []
r_5 = []
p_10 = []
r_10 = []
p_20 = []
r_20 = []
p_30 = []
r_30 = []
p_40 = []
r_40 = []
p_50 = []
r_50 = []
c = 0 #gt = 0
m = 0 #gt > 1

rank = 0
ra_amount = 0

#calculate the presition and recall
for i in range(len(Test_set)):
    gt = []
    for j in groundtruth[Test_set[i]]:
        gt.append(int(j))

    t = []
    for j in gt:
        if j in one_time_service:
            t.append(j)

    print i
    print "t=",t
    res,ra = evaluate_knowledge_base_between_test(Test_set[i],t)
    print res

    if ra!=-1 and ra!=0:
        rank+=ra
        ra_amount+=1

    print gt

    if len(gt)==0 or len(res)==0:
        c+=1
    if len(gt)>1:
        m+=1

    hit_1 = 0
    hit_2 = 0
    hit_3 = 0
    hit_4 = 0
    hit_5 = 0
    hit_10 = 0
    hit_20 = 0
    hit_30 = 0
    hit_40 = 0
    hit_50 = 0

    for j in range(len(res)):
        if res[j] in gt:
            if j < 1:
                hit_1 += 1
            if j < 2:
                hit_2 += 1
            if j < 3:
                hit_3 += 1
            if j < 4:
                hit_4 += 1
            if j < 5:
                hit_5 += 1
            if j < 10:
                hit_10 += 1
            if j < 20:
                hit_20 += 1
            if j < 30:
                hit_30 += 1
            if j < 40:
                hit_40 += 1
            if j < 50:
                hit_50 += 1

    if len(res)==0 or len(gt)==0:
        presition_1 = 0
        recall_1 = 0
        presition_2 = 0
        recall_2 = 0
        presition_3 = 0
        recall_3 = 0
        presition_4 = 0
        recall_4 = 0
        presition_5 = 0
        recall_5 = 0
        presition_10 = 0
        recall_10 = 0
        presition_20 = 0
        recall_20 = 0
        presition_30 = 0
        recall_30 = 0
        presition_40 = 0
        recall_40 = 0
        presition_50 = 0
        recall_50 = 0

    else:
        presition_1 = float(hit_1)/1
        recall_1 = float(hit_1)/len(gt)
        presition_2 = float(hit_2)/2
        recall_2 = float(hit_2)/len(gt)
        presition_3 = float(hit_3)/3
        recall_3 = float(hit_3)/len(gt)
        presition_4 = float(hit_4)/4
        recall_4 = float(hit_4)/len(gt)
        presition_5 = float(hit_5)/5
        recall_5 = float(hit_5)/len(gt)
        presition_10 = float(hit_10)/10
        recall_10 = float(hit_10)/len(gt)
        presition_20 = float(hit_20)/20
        recall_20 = float(hit_20)/len(gt)
        presition_30 = float(hit_30)/30
        recall_30 = float(hit_30)/len(gt)
        presition_40 = float(hit_40)/40
        recall_40 = float(hit_40)/len(gt)
        presition_50 = float(hit_50)/len(res)
        recall_50 = float(hit_50)/len(gt)


    print "presition@1",presition_1
    print "recall@1",recall_1
    print "presition@2",presition_2
    print "recall@2",recall_2
    print "presition@3",presition_3
    print "recall@3",recall_3
    print "presition@4",presition_4
    print "recall@4",recall_4
    print "presition@5",presition_5
    print "recall@5",recall_5
    print "presition@10",presition_10
    print "recall@10",recall_10
    print "presition@20",presition_20
    print "recall@20",recall_20
    print "presition@30",presition_30
    print "recall@30",recall_30
    print "presition@40",presition_40
    print "recall@40",recall_40
    print "presition@50",presition_50
    print "recall@50",recall_50

    p_1.append(presition_1)
    r_1.append(recall_1)
    p_2.append(presition_2)
    r_2.append(recall_2)
    p_3.append(presition_3)
    r_3.append(recall_3)
    p_4.append(presition_4)
    r_4.append(recall_4)
    p_5.append(presition_5)
    r_5.append(recall_5)
    p_10.append(presition_10)
    r_10.append(recall_10)
    p_20.append(presition_20)
    r_20.append(recall_20)
    p_30.append(presition_30)
    r_30.append(recall_30)
    p_40.append(presition_40)
    r_40.append(recall_40)
    p_50.append(presition_50)
    r_50.append(recall_50)

print "Parameters:"

print "combination_method,",combination_method
print "set_num_threshold,",set_num_threshold
print "cluster_number,",cluster_number
print "extract_relationship_threshold,",extract_relationship_threshold
print "Test_set_size,",Test_set_size

print "rank:",float(rank)/ra_amount
print "mean presition@1",float(sum(p_1))/(len(p_1)-c)
print "mean recall@1",float(sum(r_1))/(len(r_1)-c)
print "mean presition@2",float(sum(p_2))/(len(p_2)-c)
print "mean recall@2",float(sum(r_2))/(len(r_2)-c)
print "mean presition@3",float(sum(p_3))/(len(p_3)-c)
print "mean recall@3",float(sum(r_3))/(len(r_3)-c)
print "mean presition@4",float(sum(p_4))/(len(p_4)-c)
print "mean recall@4",float(sum(r_4))/(len(r_4)-c)
print "mean presition@5",float(sum(p_5))/(len(p_5)-c)
print "mean recall@5",float(sum(r_5))/(len(r_5)-c)
print "mean presition@10",float(sum(p_10))/(len(p_10)-c)
print "mean recall@10",float(sum(r_10))/(len(r_10)-c)
print "mean presition@20",float(sum(p_20))/(len(p_20)-c)
print "mean recall@20",float(sum(r_20))/(len(r_20)-c)
print "mean presition@30",float(sum(p_30))/(len(p_30)-c)
print "mean recall@30",float(sum(r_30))/(len(r_30)-c)
print "mean presition@40",float(sum(p_40))/(len(p_40)-c)
print "mean recall@40",float(sum(r_40))/(len(r_40)-c)
print "mean presition@50",float(sum(p_50))/(len(p_50)-c)
print "mean recall@50",float(sum(r_50))/(len(r_50)-c)
print "gt = 0 in Test_set:",c
print "gt > 1 in Test_set:",m

#this part is for the second recommendation method based on bundle
'''
p = []
r = []
l = []
c = 0
m = 0
for i in range(len(Test_set)):
    print i
    #bundle recommendation with CF
    res1 = recommendation_bundle_method4(Test_set[i],1)
    #res2 = cf(Test_set[i],5,Test_set)
    res = res1
    #for t in res2:
    #    if t in res:
    #        res = res
    #    else:
    #        res.append(t)
    print res

    gt = []
    for j in groundtruth[Test_set[i]]:
        gt.append(int(j))
    print gt

    if len(gt)==0 or len(res)==0:
        c+=1
    if len(gt)>1:
        m+=1

    hit=0

    for j in range(len(res)):
        if res[j] in gt:
            hit += 1

    if len(res)==0 or len(gt)==0:
        presition = 0
        recall = 0
    else:
        presition = float(hit)/len(res)
        recall = float(hit)/len(gt)



    print "presition",presition
    print "recall",recall


    p.append(presition)
    r.append(recall)
    l.append(len(res))


print "Parameters:"

print "combination_method,",combination_method
print "set_num_threshold,",set_num_threshold
print "cluster_number,",cluster_number
print "extract_relationship_threshold,",extract_relationship_threshold
print "Test_set_size,",Test_set_size
print "bundle recommendation@10 bundle with mean length,",float(sum(l))/(len(l)-c)

print "mean presition",float(sum(p))/(len(p)-c)
print "mean recall",float(sum(r))/(len(r)-c)
print "gt = 0 in Test_set:",c
print "gt > 1 in Test_set:",m
'''
