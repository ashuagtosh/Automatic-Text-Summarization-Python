import os
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import FreqDist
import numpy as np
import operator
import numpy,os

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

def doTokenize(l):
    global file_name
    rel_path = "documents/"+"kalu"+file_name
    path = os.path.join(script_dir, rel_path)
    print(path)
    wf = open(path,"w")
    
    for i in l:
        wf.write(str(i)+"\n")
        #print(i[0]+ "\n")
    #wf.write(str(word_tokenize(s)))
    return

def TF():
    global L
    w = 2
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
        
    #Matrix = numpy.array(Matrix)
    for i in range(h-1,-1,-1):
        count = 0;
        for j in range(h-1,-1,-1):
            if(Matrix[i][0] == Matrix[j][0]):
                count = count + 1
                #print(count)
        if(count > 1):
            del Matrix[i]
            h = h - 1
    #print(Matrix[2][0])
    #Matrix = numpy.array(Matrix)
    doTokenize(Matrix)        
    #for i in range(0,h):    
        #print(Matrix[i])
    return

def main():
    global file_name
    file_name = input("Enter the Name of File : ")
    readFile(file_name)
    TF()
    #print(L)
    return
main()
