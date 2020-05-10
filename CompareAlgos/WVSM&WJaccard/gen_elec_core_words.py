#coding=utf-8

def gen_core_word():
    infile = open('knowledge_base_v3.txt')
    outfile = open('elec_core_word', 'w')
    corewords = set()
    for n, line in enumerate(infile):
        try:
            if n == 0:
                continue
            #if n > 50:
            #    break
            line = line.split('\t')
            if line[1] == '': #New question
                #answer = line[2].strip().replace('"', '“').replace(',', '，').replace('[', '').replace(']','')
                #question = line[0].strip().replace('"', '“').replace(',', '，').replace('[', '').replace(']','')
                pass
            else:
                #question = line[1].strip().replace('[', '').replace(']','').replace('|','').replace('"', '“').replace(',', '，')
                question = line[1].strip().split('][')
                for item in question:
                    item = item.strip().split('|')
                    for itemm in item:
                        itemm = itemm.strip('[').strip(']').strip('@')
                        if itemm != '':
                            corewords.add(itemm)
               
        except Exception as e:
            print e
            continue

    for item in corewords:
        print >>outfile, item

def gen_core_qa():
    infile = open('knowledge_base_v3.txt')
    outfile = open('elec_core_qa_v2', 'w')
    core_qa = {}
    question_str = ''
    answer_str = ''
    exception_count = 0
    for n, line in enumerate(infile):
        try:
            if n == 0:
                continue
            #if n > 50:
            #    break
            line = line.split('\t')
            
            if line[1] == '': #New question
                if question_str != '' and answer_str != '':
                    core_qa[question_str] = answer_str
                answer_str = line[2].strip().replace('"', '“').replace(',', '，').replace('[', '').replace(']','')
                question_str = line[0].strip().replace('"', '“').replace(',', '，').replace('[', '').replace(']','')
                #core_qa[question] = answer
                
            else:
                #question = line[1].strip().replace('[', '').replace(']','').replace('|','').replace('"', '“').replace(',', '，')
                corewords = set()
                question = line[1].strip().split('][')
                for item in question:
                    item = item.strip().split('|')
                    for itemm in item:
                        itemm = itemm.strip('[').strip(']').strip('@')
                        if itemm != '':
                            corewords.add(itemm)
                for item in corewords:
                    question_str = question_str + ' ' + item
                #if strtmp != '' and answer != '':
                #    core_qa[strtmp] = answer
               
        except Exception as e:
            #print e
            exception_count += 1
            continue

    for item in core_qa:
        print >> outfile, item + '|' +  core_qa[item]
    print exception_count
        
        
#gen_core_word()
gen_core_qa()



   
