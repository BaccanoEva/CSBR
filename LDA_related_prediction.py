import os
import re
import csv
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import random


#parameters
combination_method_list = ["one_for_each_mashup.result","one_or_two_combination_for_each_mashup.result","one_or_two_or_three_combination_for_each_mashup.result","one_or_two_or_three_or_four_combination_for_each_mashup.result"]
combination_method = "one_or_two_combination_for_each_mashup.result"
is_combine_sim_length = True
just_consider_one_length = True

set_num_threshold = 10
cluster_number = 5
extract_relationship_threshold = 0.5
Test_set_size = 100

f = open("Test_set_"+str(Test_set_size))
#f = open("Test_set_100_cold_start")

print "0. Test set"
Test_set = []


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

print "2. Calculate sim and extract the relation between sentence and service set"

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
    #print ser_list
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
print "3. Using kmeans construct knowledge base"
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
        #if the set is big enough for clustring, add the original sentence vector to it
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

print "4. Using knowledge base to do recommendation"
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
                        print "rank=",len(tmp)
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

def combine_sim_length(p):
    #p{key:point}
    new_p = {}
    for key, value in p.items():
        length = len(key)
        new_p[key] = 1/float(length)+value
    return new_p


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
            p = {} #{key:sum(points)}
            items = knowledge_base.items()
            for key,value in items:
                n1,n2 = evaluate_bundle(value,content)
                point[key]=n1
                p[key] = n2

            if is_combine_sim_length:
                p = combine_sim_length(p)

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

            if is_combine_sim_length:
                p = combine_sim_length(p)

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


#this part is for the second recommendation method based on bundle
bundle_size_list = [1,2,3,4,5,6,7,8,9,10]
for bundle_size_ in bundle_size_list:
    p = []
    r = []
    l = []
    c = 0
    m = 0
    for i in range(len(Test_set)):
        gt = []
        for j in groundtruth[Test_set[i]]:
            gt.append(int(j))
        print gt
        
        if just_consider_one_length and len(gt)>1:
            continue

        print i
        res = recommendation_bundle_method4(Test_set[i],10)
        print res

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
    print "bundle recommendation@",bundle_size_,"bundle with mean length,",float(sum(l))/(len(l)-c)

    print "mean presition",float(sum(p))/(len(p)-c)
    print "mean recall",float(sum(r))/(len(r)-c)
    print "gt = 0 in Test_set:",c
    print "gt > 1 in Test_set:",m
