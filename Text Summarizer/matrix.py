import os
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import FreqDist
import numpy as np
import operator,numpy,os,math,csv
import pandas as pd

def performAllFunctions(fn,ofn):
    global file_name
    global original_file_name
    original_file_name = ofn+".csv"
    file_name = fn
    #file_name = input("Enter the Name of File : ")
    readFile(file_name)
    TF()
    computeTF()
    return 

def readFile(file_name):
    global script_dir
    script_dir = os.path.dirname(__file__)
    rel_path = "documents/"+file_name
    abs_file_path = os.path.join(script_dir, rel_path)
    rf = open(abs_file_path,"r")
    global L
    L = list()
    L = rf.readlines()
    return

def doWrite(l,name):
    global file_name
    global original_file_name
    rel_path = "documents/tfidf/"+name+"_document"+original_file_name
    path = os.path.join(script_dir, rel_path)
##    #print(path)
##    
##    wf = open(path,"w")
##    
##    for i in l:
##        wf.write(str(i)+"\n")
    
    f=open(path, "w",newline='')
    writer = csv.writer(f)
    
    writer.writerow(['words', 'frequency','tf_score'])
    k = 0
    for i in range(0,len(l)):
        writer.writerow([l['col0'][k],l['col1'][k],l['col2'][k]])
        k = k+1

    return

def readCSV():
    global original_file_name
    rel_path = "documents/tfidf/"+"tfidf_document"+"1"+".csv"
    path = os.path.join(script_dir, rel_path)
##    rf = open(path,"r")
##    reader = csv.reader(rf, delimiter=';')
##    for row in reader:
##        print(row[0])
##
    df = pd.read_csv(path)
    print(df['tf_score'])
    return
def TF():
    global L
    w = 2
    global length,h
    leng = len(L)
    h = len(L)
    Matrix = [[0 for x in range(w)] for y in range(h)]
    k = 0;
    L = list(map(lambda s: s.strip(), L))
    #print(h)        
    for i in L:
        Matrix[k][0]=L[k]
        k = k+1
    for i in range(0,h):
        count = 0;
        for j in range(0,h):
            if(Matrix[i][0] == Matrix[j][0]):
                count = count + 1
        Matrix[i][1] = count
    for i in range(h-1,-1,-1):
        count = 0;
        for j in range(h-1,-1,-1):
            if(Matrix[i][0] == Matrix[j][0]):
                count = count + 1
        if(count > 1):
            del Matrix[i]
            h = h - 1
    #doWrite(Matrix,"wf")        
    return Matrix

def computeTF():
    global length,h
    leng = len(TF())
    Matrix_tfidf = [[0 for x in range(2)] for y in range(h)]
    Matrix_tfidf = TF()
    words = list()
    frequency = list()
    tf_score = list()
    k = 0;   
    for i in range (0,h):
        words.insert(k,Matrix_tfidf[i][0])
        frequency.insert(k,Matrix_tfidf[i][1])
        Matrix_tfidf[i][1] = (Matrix_tfidf[i][1]/leng)
        Matrix_tfidf[i][1] = math.floor(Matrix_tfidf[i][1]*10000)/10000
        tf_score.insert(k,Matrix_tfidf[i][1])
        k = k + 1
    Matrix_list = pd.DataFrame(
    {'col0': words,
     'col1': frequency,
     'col2': tf_score
    })
    doWrite(Matrix_list,"tfidf")
    return

def calculate_IDF2(word,count):
    global n
    c = 1
    for i in range(0,int(n)):
        b = 0
        if(i==int(count)):
            pass
        else:
            #print("rocky"+str(i))
            y = int(i)+1
            #print(str(y))
            x = str(y)
            dir_path = os.path.dirname(__file__)
            rel_path2 = "documents/tfidf/"+"tfidf_document"+x+".csv"
            path2 = os.path.join(dir_path, rel_path2)
            word_pd_reader = pd.read_csv(path2,encoding = "ISO-8859-1")
            m = 0
            for j in range (0,len(word_pd_reader['words'])):
                if(word_pd_reader['words'][m]== word):
                    b = 1
                m = m + 1    
                
        c = c + b
    
    c = 1+math.log10(int(n)/int(c))
    c = math.floor(c*10000)/10000
    #print(c)
    #print(l)
    return c

def calculateIDF(count):
    x = str(count + 1)
    dir_path = os.path.dirname(__file__)
    rel_path2 = "documents/tfidf/"+"tfidf_document"+x+".csv"
    path2 = os.path.join(dir_path, rel_path2)

    word_pd_reader = pd.read_csv(path2,encoding = "ISO-8859-1")
    k = 0
    l = list()
    for i in range(0,len(word_pd_reader['words'])):
        #print(word_pd_reader['words'][k])
        c = calculate_IDF2(word_pd_reader['words'][k],count)
        l.insert(int(i),c)
        k = k + 1
    
    x = str(count + 1)
    dir_pathx = os.path.dirname(__file__)
    rel_path23 = "documents/tfidf/"+"tfidf_document"+x+".csv"
    path3 = os.path.join(dir_pathx, rel_path23)
    df = pd.read_csv(path3,encoding = "ISO-8859-1")
    new_column = pd.DataFrame({'idf_values':l})
    df = df.merge(new_column, left_index = True, right_index = True)
    df.to_csv(path2,index = False)
    return

def calculateTFIDF(count):
    x = str(count + 1)
    dir_path = os.path.dirname(__file__)
    rel_path2 = "documents/tfidf/"+"tfidf_document"+x+".csv"
    path2 = os.path.join(dir_path, rel_path2)
    word_pd_reader = pd.read_csv(path2,encoding = "ISO-8859-1")
    l = list()
    for i in range(0,len(word_pd_reader)):
        l.insert(int(i),word_pd_reader['tf_score'][i]*word_pd_reader['idf_values'][i])
    new_column = pd.DataFrame({'tfxidf':l})
    df = word_pd_reader.merge(new_column, left_index = True, right_index = True)
    df.to_csv(path2,index = False)
    return

def main():
    global n
    n = input("Enter the number of documents:")
    a = ["" for x in range(int(n))]
    for i in range(0,len(a)):
        
        a[i] = str(i+1)
        x = "tokenize_document"+a[i]+".txt"
        performAllFunctions(x,a[i])
    for i in range(0,len(a)):
        calculateIDF(i)
    for i in range(0,len(a)):
        calculateTFIDF(i)
        print("")    
    return
main()
