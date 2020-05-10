import matplotlib.pyplot as plt
import numpy as np

#-----------100 test
x = [5,10,20,30,40,50]
#baseline
recall_WVSM =        [0.151981050818,0.195521102498,0.303799064846,0.380703211517,0.444349362954,0.488594136269]
precision_WVSM =     [0.0697674418605,0.0472163495419,0.0398671096346,0.0340085021255,0.0306296086217,0.0270557835537]
recall_Wjaccard =    [0.11632213609,0.17459086994,0.243140150117,0.295625692137,0.363126279405,0.41321241205]
precision_Wjaccard = [0.0542635658915,0.0422832980973,0.032746031746,0.0272568142036,0.0259574588769,0.0229518163855]
recall_ERTM =        [0.113484205563,0.161515794437,0.20974322707,0.313447031516,0.361961883001,0.384156602473]
precision_ERTM =     [0.0531485148515,0.0413762376238,0.0275841584158,0.030603960396,0.0251485148515,0.020504950495]
recall_CF =          [0.150219417546,0.188211293657,0.265780836325,0.386610996264,0.430340369202,0.490583179197]
precision_CF =       [0.061004243281,0.0447147796012,0.0355035994235,0.034358431731,0.0301813958507,0.027869380857]
recall_TopicCF =     [0.288621296196,0.403145105719,0.539625539477,0.582645522794,0.608878431799,0.624524353534]
precision_TopicCF =  [0.12099009901,0.0841584158416,0.0574257425743,0.0509240924092,0.0336633663366,0.0308910891089]


recall_one_two_three_four_5_clusters = [0.453588935574,0.484070086368,0.587733426704,0.666563083567,0.70710638422,0.720995273109]
precision_one_two_three_four_5_clusters = [0.1775,0.1270833333333,0.07,0.0651944444444,0.0425,0.03533952363]

plt.figure(figsize=(10,5))
#plt.title("Precision of baselines and Ranking Method")
plt.xlabel("N")
plt.ylabel("precision")

new_ticks = np.linspace(0, 50, 11)
print(new_ticks)
plt.xticks(new_ticks)

x = np.array(x)
a = np.array(precision_WVSM)
b = np.array(precision_Wjaccard)
c = np.array(precision_ERTM)
d = np.array(precision_CF)
e = np.array(precision_TopicCF)
f = np.array(precision_one_two_three_four_5_clusters)


total_width, n = 4, 6
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='WVSM')
plt.bar(x + width, b, width=width, label='Wjaccard')
plt.bar(x + 2 * width, c, width=width, label='ERTM')
plt.bar(x + 3 * width, d, width=width, label='CF')
plt.bar(x + 4 * width, e, width=width, label='AFUPR')
plt.bar(x + 5 * width, f, width=width, label='CSBR')


plt.legend()
plt.grid(True)
plt.savefig("compare_between_baseline_ranking_method_precision.png")
plt.show()

plt.figure(figsize=(10,5))
#plt.title("Recall of baselines and Ranking Method")
plt.xlabel("N")
plt.ylabel("recall")

new_ticks = np.linspace(0, 50, 11)
print(new_ticks)
plt.xticks(new_ticks)

x = np.array(x)
a = np.array(recall_WVSM)
b = np.array(recall_Wjaccard)
c = np.array(recall_ERTM)
d = np.array(recall_CF)
e = np.array(recall_TopicCF)
f = np.array(recall_one_two_three_four_5_clusters)


total_width, n = 4, 6
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='WVSM')
plt.bar(x + width, b, width=width, label='WJaccard')
plt.bar(x + 2 * width, c, width=width, label='ERTM')
plt.bar(x + 3 * width, d, width=width, label='CF')
plt.bar(x + 4 * width, e, width=width, label='AFUPR')
plt.bar(x + 5 * width, f, width=width, label='CSBR')


plt.legend()
plt.grid(True)
plt.savefig("compare_between_baseline_ranking_method_recall.png")
plt.show()

x = [1,2,3,4,5]
#baseline
recall_WVSM =                       [0.04,0.08,0.10,0.12,0.15]
precision_WVSM =                    [0.05,0.07,0.07,0.066,0.07]
recall_Wjaccard =                   [0.05,0.07,0.08,0.10,0.12]
precision_Wjaccard =                [0.06,0.06,0.059,0.058,0.054]
recall_ERTM =                       [0.03,0.05,0.078,0.09,0.113]
precision_ERTM =                    [0.04,0.04,0.057,0.054,0.053]
recall_CF =                         [0.03,0.05,0.07,0.11,0.15]
precision_CF =                      [0.04,0.05,0.05,0.05,0.06]
recall_TopicCF =                    [0.12,0.17,0.21,0.25,0.29]
precision_TopicCF =                [0.16,0.135,0.125,0.113,0.12]

recall_one_two_three_5_clusters =    [0.31,0.37,0.41,0.44,0.47]
precision_one_two_three_5_clusters = [0.46,0.28,0.22,0.19,0.18]


plt.figure(figsize=(10,5))
#plt.title("Precision of baselines and Ranking Method")
plt.xlabel("N")
plt.ylabel("precision")

new_ticks = np.linspace(0, 5, 6)
print(new_ticks)
plt.xticks(new_ticks)

x = np.array(x)
a = np.array(precision_WVSM)
b = np.array(precision_Wjaccard)
c = np.array(precision_ERTM)
d = np.array(precision_CF)
e = np.array(precision_TopicCF)
f = np.array(precision_one_two_three_5_clusters)


total_width, n = 0.4, 5
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='WVSM')
plt.bar(x + width, b, width=width, label='WJaccard')
plt.bar(x + 2 * width, c, width=width, label='ERTM')
plt.bar(x + 3 * width, d, width=width, label='CF')
plt.bar(x + 4 * width, e, width=width, label='AFUPR')
plt.bar(x + 5 * width, f, width=width, label='CSBR')


plt.legend()
plt.grid(True)
plt.savefig("compare_between_baseline_ranking_method_precision_2.png")
plt.show()

plt.figure(figsize=(10,5))
#plt.title("Recall of baselines and Ranking Method")
plt.xlabel("N")
plt.ylabel("recall")

new_ticks = np.linspace(0, 5, 6)
print(new_ticks)
plt.xticks(new_ticks)

x = np.array(x)
a = np.array(recall_WVSM)
b = np.array(recall_Wjaccard)
c = np.array(recall_ERTM)
d = np.array(recall_CF)
e = np.array(recall_TopicCF)
f = np.array(recall_one_two_three_5_clusters)


total_width, n = 0.4, 5
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='WVSM')
plt.bar(x + width, b, width=width, label='WJaccard')
plt.bar(x + 2 * width, c, width=width, label='ERTM')
plt.bar(x + 3 * width, d, width=width, label='CF')
plt.bar(x + 4 * width, e, width=width, label='AFUPR')
plt.bar(x + 5 * width, f, width=width, label='CSBR')


plt.legend()
plt.grid(True)
plt.savefig("compare_between_baseline_ranking_method_recall_2.png")
plt.show()

'''
plt.figure(figsize=(12,6))
plt.title("100 Test set")
plt.xlabel("recommendation number")
plt.ylabel("recall")
plt.plot(x,recall_WVSM,'.:',label = "WVSM")
plt.plot(x,recall_Wjaccard,',:',label = "Wjaccard")
plt.plot(x,recall_ERTM,'o:',label = "ERTM")
plt.plot(x,recall_TopicCF,'v:',label = "TopicCF")
plt.plot(x,recall_CF,'^:',label = "CF")
plt.plot(x,recall_one_5_clusters,'<:',label = "one_5_clusters")
plt.plot(x,recall_one_two_5_clusters,'>:',label = "two_5_clusters")
plt.plot(x,recall_one_two_three_5_clusters,'*:',label = "three_5_clusters")
plt.plot(x,recall_one_two_three_four_5_clusters,'D:',label = "four_5_clusters")
plt.legend()
plt.grid(True)
plt.savefig("compare_with_baseline_recall.jpg")
plt.show()

plt.figure(figsize=(12,6))
plt.title("100 Test set")
plt.xlabel("recommendation number")
plt.ylabel("precision")
plt.plot(x,precision_WVSM,'.:',label = "WVSM")
plt.plot(x,precision_Wjaccard,',:',label = "Wjaccard")
plt.plot(x,precision_ERTM,'o:',label = "ERTM")
plt.plot(x,precision_TopicCF,'v:',label = "TopicCF")
plt.plot(x,precision_CF,'^:',label = "CF")
plt.plot(x,precision_one_5_clusters,'<:',label = "one_5_clusters")
plt.plot(x,precision_one_two_5_clusters,'>:',label = "two_5_clusters")
plt.plot(x,precision_one_two_three_5_clusters,'*:',label = "three_5_clusters")
plt.plot(x,precision_one_two_three_four_5_clusters,'D:',label = "four_5_clusters")
plt.legend()
plt.grid(True)
plt.savefig("compare_with_baseline_precision.jpg")
plt.show()

plt.figure(figsize=(12,6))
plt.title("100 Test set")
plt.xlabel("recommendation number")
plt.ylabel("recall")
plt.plot(x,recall_one_3_clusters,'.:',label = "one_3_clusters")
plt.plot(x,recall_one_two_3_clusters,',:',label = "two_3_clusters")
plt.plot(x,recall_one_two_three_3_clusters,'o:',label = "three_3_clusters")
plt.plot(x,recall_one_two_three_four_3_clusters,'v:',label = "four_3_clusters")
plt.plot(x,recall_one_5_clusters,'^:',label = "one_5_clusters")
plt.plot(x,recall_one_two_5_clusters,'<:',label = "two_5_clusters")
plt.plot(x,recall_one_two_three_5_clusters,'>:',label = "three_5_clusters")
plt.plot(x,recall_one_two_three_four_5_clusters,'*:',label = "four_5_clusters")
plt.legend()
plt.grid(True)
plt.savefig("compare_between_3_clusters_and_5_clusters_recall.jpg")
plt.show()

plt.figure(figsize=(12,6))
plt.title("100 Test set")
plt.xlabel("recommendation number")
plt.ylabel("precision")
plt.plot(x,precision_one_3_clusters,'.:',label = "one_3_clusters")
plt.plot(x,precision_one_two_3_clusters,',:',label = "two_3_clusters")
plt.plot(x,precision_one_two_three_3_clusters,'o:',label = "three_3_clusters")
plt.plot(x,precision_one_two_three_four_3_clusters,'v:',label = "four_3_clusters")
plt.plot(x,precision_one_5_clusters,'^:',label = "one_5_clusters")
plt.plot(x,precision_one_two_5_clusters,'<:',label = "two_5_clusters")
plt.plot(x,precision_one_two_three_5_clusters,'>:',label = "three_5_clusters")
plt.plot(x,precision_one_two_three_four_5_clusters,'*:',label = "four_5_clusters")
plt.legend()
plt.grid(True)
plt.savefig("compare_between_3_clusters_and_5_clusters_precision.jpg")
plt.show()
'''

#-------------100 test with bundle recommendation
#parameters: >10 5 clusters  bundle algorithm with 0.85
x = [1,2,3,4,5,6,7,8,9,10]
recall_one = [0.260781395892,0.313477474323,0.333442752101,0.353408029879,0.376845529879,0.391949696545,0.405838585434,0.409310807656,0.41451914099,0.425283029879]
precision_one = [0.338541666667,0.201736111111,0.145138888889,0.116716269841,0.100806051587,0.0893518518519,0.0811879960317,0.0732410818348,0.0661802607115,0.0622348320539]
recall_two = [0.284466911765,0.366932189542,0.407309173669,0.426406395892,0.464427229225,0.490468895892,0.500885562558,0.500885562558,0.52171889589,0.525811157796]
precision_two = [0.302951388889,0.191493055556,0.15248015873,0.124066558442,0.109302936647,0.0961704094517,0.0835352147852,0.0739491190318,0.0695716031654,0.0650935688462]
recall_three = [0.299340569561,0.351423902894,0.412882236228,0.44760445845,0.469925887021,0.490759220355,0.501175887021,0.501175887021,0.513080648926,0.513080648926]
precision_three = [0.303819444444,0.18621031746,0.152517361111,0.131477723665,0.10992413258,0.0965176316739,0.0843305565962,0.0751540982989,0.0709515022015,0.064676868727]
recall_four = [0.27789449113,0.350290324463,0.388484768908,0.42320699113,0.445528419701,0.485458975257,0.48754230859,0.490146475257,0.491634570495,0.495106792717]
precision_four = [0.315972222222,0.192013888889,0.152595899471,0.122197420635,0.105249669312,0.0956698683261,0.0838390962931,0.0747447864635,0.0677494990515,0.0622621038752]

plt.figure(figsize=(10,5))
#plt.title("Precision of Bundle Recommendation")
plt.xlabel("Recommendation Number")
plt.ylabel("precision")

new_ticks = np.linspace(0, 10, 11)
print(new_ticks)
plt.xticks(new_ticks)

x = np.array(x)
a = np.array(precision_one)
b = np.array(precision_two)
c = np.array(precision_three)
d = np.array(precision_four)

total_width, n = 0.8, 4
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='g = 1')
plt.bar(x + width, b, width=width, label='g = 2')
plt.bar(x + 2 * width, c, width=width, label='g = 3')
plt.bar(x + 3 * width, d, width=width, label='g = 4')

plt.legend()
plt.grid(True)
plt.savefig("bundle_recommendation_precision.png")
plt.show()


plt.figure(figsize=(10,5))
#plt.title("Recall of Bundle Recommendation")
plt.xlabel("Recommendation Number")
plt.ylabel("recall")

new_ticks = np.linspace(0, 10, 11)
print(new_ticks)
plt.xticks(new_ticks)

x = np.array(x)
a = np.array(recall_one)
b = np.array(recall_two)
c = np.array(recall_three)
d = np.array(recall_four)


total_width, n = 0.8, 4
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='g = 1')
plt.bar(x + width, b, width=width, label='g = 2')
plt.bar(x + 2 * width, c, width=width, label='g = 3')
plt.bar(x + 3 * width, d, width=width, label='g = 4')


plt.legend()
plt.grid(True)
plt.savefig("bundle_recommendation_recall.png")
plt.show()


'''
plt.figure(figsize=(12,6))
plt.title("100 Test set")
plt.xlabel("recommendation bundle number")
plt.ylabel("recall")
plt.plot(x,recall_one,'^:',label = "one_5_clusters")
plt.plot(x,recall_two,'<:',label = "two_5_clusters")
plt.plot(x,recall_three,'>:',label = "three_5_clusters")
plt.plot(x,recall_four,'*:',label = "four_5_clusters")
plt.legend()
plt.grid(True)
plt.savefig("bundle_recommendation_recall.jpg")
plt.show()

plt.figure(figsize=(12,6))
plt.title("100 Test set")
plt.xlabel("recommendation bundle number")
plt.ylabel("precision")
plt.plot(x,precision_one,'^:',label = "one_5_clusters")
plt.plot(x,precision_two,'<:',label = "two_5_clusters")
plt.plot(x,precision_three,'>:',label = "three_5_clusters")
plt.plot(x,precision_four,'*:',label = "four_5_clusters")
plt.legend()
plt.grid(True)
plt.savefig("bundle_recommendation_precision.jpg")
plt.show()
'''
#-------------500 test
recall_one_5_clusters = [0.326059830252,0.377507690828,0.44930579245,0.509870053265,0.555574003533,0.594601849822]
precision_one_5_clusters = [0.108713692946,0.0655601659751, 0.0403526970954,0.0311894882434,0.0255705394191,0.0221161825726]
recall_two_5_clusters = [0.378017485322,0.434399623239,0.529298293034,0.590112010478,0.633237418956,0.661158717615]
precision_two_5_clusters = [0.126141078838,0.0751037344398,0.0477178423237,0.0362378976487,0.0293049792531,0.0245952322838]
recall_three_5_clusters = [0.37372086562,0.450069428578,0.548211205238,0.599022738103,0.634664447708,0.658530778472]
precision_three_5_clusters = [0.121576763485,0.0780082987552,0.0486514522822,0.0367219917012,0.0296680497925,0.0246953993854]
recall_four_5_clusters = [0.371988918453,0.443586026088,0.544439095605,0.593797499994,0.632697914932,0.656965884169]
precision_four_5_clusters = [0.121161825726,0.0782157676349,0.0488589211618,0.0363762102351,0.0294605809129,0.0245696976524]

#-------------100 test with (ensure that every service set been allocated at least one sentence vector)
recall_three_5_clusters = [0.448993347339,0.473398109244,0.54581144958,0.609976073763,0.647302462652,0.679420518207]
precision_three_5_clusters = [0.1375,0.0760416666667,0.0458333333333,0.0357638888889,0.0283854166667,0.0241060206134]
recall_four_5_clusters = [0.450116713352,0.489278419701,0.555615371148,0.611528361345,0.668656629318,0.701387429972]
precision_four_5_clusters = [0.135416666667,0.0770833333333,0.0479166666667,0.0354166666667,0.0296875,0.0253676470588]

#-------------statistic about each mashup respond to how much services
plt.figure(figsize=(4,6))
labels = ["n = 1","n = 2","n = 3","n = 4","n >= 5"]
sizes = [3706,1634,664,333,495]
#colors = ['yellow','yellowgreen','lightskyblue','purple','red']
explode = (0,0,0,0,0.05)
patches,text1,text2 = plt.pie(sizes,
                      explode=explode,
                      labels=labels,
                      #colors=colors,
                      autopct = '%3.2f%%',
                      shadow = False,
                      startangle =90,
                      pctdistance = 0.6)
plt.axis('equal')
plt.show()
#------------proposed method with CF Method(mehtod 3 0.85 5clusters)---1b1a 2b2a 3b3a 4b4a 5b5a
recall_one_bundle_amount = [0.35,0.40,0.42,0.45,0.48]
precision_one_bundle_amount = [0.27,0.17,0.12,0.10,0.09]
recall_two_bundle_amount = [0.40,0.44,0.45,0.49,0.53]
precision_two_bundle_amount = [0.27,0.16,0.12,0.10,0.09]
recall_three_bundle_amount = [0.39,0.40,0.44,0.48,0.51]
precision_three_bundle_amount = [0.27,0.16,0.12,0.11,0.09]
recall_four_bundle_amount = [0.38,0.42,0.44,0.47,0.50]
precision_four_bundle_amount = [0.28,0.16,0.12,0.10,0.09]


plt.figure(figsize=(10,5))
#plt.title("Recall of Bundle Recommendation")
plt.xlabel("recommendation bundle number and CF amount")
plt.ylabel("recall")

new_ticks = np.linspace(0, 10, 11)
print(new_ticks)
plt.xticks(new_ticks)

x = [1,2,3,4,5]
x = np.array(x)
a = np.array(recall_one_bundle_amount)
b = np.array(recall_two_bundle_amount)
c = np.array(recall_three_bundle_amount)
d = np.array(recall_four_bundle_amount)


total_width, n = 0.8, 4
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='g = 1')
plt.bar(x + width, b, width=width, label='g = 2')
plt.bar(x + 2 * width, c, width=width, label='g = 3')
plt.bar(x + 3 * width, d, width=width, label='g = 4')


plt.legend()
plt.grid(True)
plt.savefig("bundle_cf_recommendation_recall.png")
plt.show()

plt.figure(figsize=(10,5))
#plt.title("Recall of Bundle Recommendation")
plt.xlabel("recommendation bundle number and CF amount")
plt.ylabel("precision")

new_ticks = np.linspace(0, 10, 11)
print(new_ticks)
plt.xticks(new_ticks)

x = [1,2,3,4,5]
x = np.array(x)
a = np.array(precision_one_bundle_amount)
b = np.array(precision_two_bundle_amount)
c = np.array(precision_three_bundle_amount)
d = np.array(precision_four_bundle_amount)


total_width, n = 0.8, 4
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='g = 1')
plt.bar(x + width, b, width=width, label='g = 2')
plt.bar(x + 2 * width, c, width=width, label='g = 3')
plt.bar(x + 3 * width, d, width=width, label='g = 4')


plt.legend()
plt.grid(True)
plt.savefig("bundle_cf_recommendation_precision.png")
plt.show()

#------------------------bundle recommendation with amount prediction
x = [1,2,3,4,5,6,7,8,9,10]
recall_one =      [0.26,0.31,0.33,0.35,0.38,0.39,0.41,0.41,0.41,0.43]
precision_one =   [0.35,0.21,0.15,0.12,0.11,0.09,0.08,0.08,0.07,0.06]
recall_two =      [0.22,0.31,0.37,0.41,0.42,0.43,0.45,0.45,0.47,0.48]
precision_two =   [0.28,0.19,0.16,0.13,0.11,0.09,0.09,0.08,0.07,0.07]
recall_three =    [0.25,0.33,0.38,0.42,0.45,0.46,0.47,0.48,0.50,0.50]
precision_three = [0.30,0.20,0.17,0.14,0.12,0.11,0.09,0.08,0.08,0.07]
recall_four =     [0.25,0.31,0.36,0.39,0.40,0.44,0.46,0.47,0.47,0.47]
precision_four =  [0.32,0.20,0.16,0.13,0.11,0.10,0.09,0.08,0.07,0.07]

plt.figure(figsize=(10,5))
#plt.title("Precision of Bundle Recommendation")
plt.xlabel("recommendation bundle number")
plt.ylabel("precision")

new_ticks = np.linspace(0, 10, 11)
print(new_ticks)
plt.xticks(new_ticks)

x = np.array(x)
a = np.array(precision_one)
b = np.array(precision_two)
c = np.array(precision_three)
d = np.array(precision_four)

total_width, n = 0.8, 4
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='g = 1')
plt.bar(x + width, b, width=width, label='g = 2')
plt.bar(x + 2 * width, c, width=width, label='g = 3')
plt.bar(x + 3 * width, d, width=width, label='g = 4')

plt.legend()
plt.grid(True)
plt.savefig("bundle_recommendation_with_prediction_precision.png")
plt.show()


plt.figure(figsize=(10,5))
#plt.title("Recall of Bundle Recommendation")
plt.xlabel("recommendation bundle number")
plt.ylabel("recall")

new_ticks = np.linspace(0, 10, 11)
print(new_ticks)
plt.xticks(new_ticks)

x = np.array(x)
a = np.array(recall_one)
b = np.array(recall_two)
c = np.array(recall_three)
d = np.array(recall_four)


total_width, n = 0.8, 4
width = float(total_width) / n
x = x - (total_width - width) / 2

plt.bar(x, a,  width=width, label='g = 1')
plt.bar(x + width, b, width=width, label='g = 2')
plt.bar(x + 2 * width, c, width=width, label='g = 3')
plt.bar(x + 3 * width, d, width=width, label='g = 4')


plt.legend()
plt.grid(True)
plt.savefig("bundle_recommendation_with_prediction_recall.png")
plt.show()
