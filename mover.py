import os
import shutil
import string

if __name__ == "__main__":
    dst = open('material.txt','a')
    rootdir = "/home/pengqianyang/nlp/nlp/mashup_in/"
    for i in range(6976):
            f = open(rootdir+str(i)+'.txt','r')
            dst.write(f.read())
            dst.write('-=this is the spread line=-')
            f.close()
            print i
    rootdir = "/home/pengqianyang/nlp/nlp/services-in/"
    for i in range(12140):
            f = open(rootdir+str(i)+'.txt','r')
            dst.write(f.read())
            dst.write('-=this is the spread line=-')
            f.close()
            print i
    dst.close()
