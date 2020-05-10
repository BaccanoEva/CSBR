import os
import re
from textblob import TextBlob

pos = 0

class tree:
    def __init__(self):
        self.position = '' #Root/Nucleus/Satelette
        self.character = '' #span/leaf
        self.relationship = '' #span/Backhround/Elaboration etc
        self.serial = -1
        self.n = 0
        self.level = 0
        self.children = []
        self.parent = None

def addNode(lst,t,childrenlist):
    global pos
    pos += 1
    newnode = tree()
    t.children.append(newnode)
    newnode.parent = t
    newnode.level = newnode.parent.level + 1
    part = lst[pos]
    while not ((part.find('span')!=-1 and part.find('rel2par') == -1) or part.find('leaf') != -1):
        pos += 1
        part = lst[pos]
    if part.find('span') != -1:
        newnode.position = lst[pos-1]
        newnode.character = 'span'
        if ('rel2par' in lst[pos+1]):
            newnode.relationship = lst[pos+1][8:]
        newnode.n += int(part.split(' ')[2]) - int(part.split(' ')[1]) + 1
        t.n -= int(part.split(' ')[2]) - int(part.split(' ')[1]) + 1
        while(newnode.n>0):
            addNode(lst,newnode,childrenlist)
    if part.find('leaf') != -1:
        t.n -= 1
        newnode.position = lst[pos-1]
        newnode.character = 'leaf'
        newnode.relationship = lst[pos+1].split(' ')[1]
        newnode.serial = int(part.split(' ')[1])-1
        childrenlist.append(newnode)

def findrelationship(a,b,childrenlist):
    p = childrenlist[a]
    q = childrenlist[b]
    if p.level > q.level:
        for i in range(p.level - q.level):
            p = p.parent
    elif p.level < q.level:
        for i in range(q.level - p.level):
            q = q.parent
    while (p.parent != q.parent):
        p = p.parent
        q = q.parent
    return p.relationship + '-' + q.relationship

def SaveLeaves(content):
    l = content.split('\n')
    result = []
    for i in l:
        if 'text' in i:
            t = i.split('_!')
            result.append(t[1])
    for i in result:
        print i
    #for i in result:
#        wiki = TextBlob(i)
#        print wiki.tags


if __name__ == '__main__' :
    #for i in range(0,14000):
    for i in [528, 2196, 2366, 2764,2952]:
        print i
        pos = 0
        dir = "/Users/liuyancen/Desktop/nlp_lxtong/nlp/result/"
        file = str(i)+".txt"
        try:
            f = open(dir+file,'r')
        except:
            continue
        content = f.read()

        print "SaveLeaves: "
        SaveLeaves(content)
        p = re.compile(r'text _!(.*?)_!')
        content = p.sub('text',content)
        k = re.split('\(|\)', content)
        while None in k:
            k.remove(None)
        for word in range(len(k)):
            k[word] = k[word].strip()
        while '' in k:
            k.remove('')
        top = tree()
        top.position = 'hokori'
        childrenlist = []
        for i in range (k.count('Root')):
            addNode(k,top,childrenlist)
#        for i in top.children:
#            print i.serial
#            print i.level
#            print i.parent.position
#            print i.position
#            print i.relationship
#            for j in i.children:
#                print 'ch'
#                print j.serial
#                print j.level
#                print j.position
#                print j.relationship
        for i in range(len(childrenlist)):
            for j in range(len(childrenlist)):
                if i != j:
                    print 'relationship of '+str(i)+','+str(j)+'is:'+findrelationship(i,j,childrenlist)
