from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import re,string,os
import numpy
import operator,math,csv
import pandas as pd
import time,copy

def createDictionaries(n):
    global dictlist
    dictlist = [dict() for x in range(int(n))]
    for i in range(0,n):
        dir_path = os.path.dirname(__file__)
        rel_path = "documents/tfidf/"+"tfidf_document"+str(i+1)+".csv"
        path = os.path.join(dir_path, rel_path)
        word_pd_reader = pd.read_csv(path,encoding = "ISO-8859-1")
        for j in range(0,len(word_pd_reader['tfxidf'])):
            dictlist[i][word_pd_reader['words'][j]] = word_pd_reader['tfxidf'][j]
    return

def checkTfIdf_Score2(word,document_no):

    dir_path = os.path.dirname(__file__)
    rel_path = "documents/tfidf/"+"tfidf_document"+document_no+".csv"
    path = os.path.join(dir_path, rel_path)
    word_pd_reader = pd.read_csv(path,encoding = "ISO-8859-1")
    k = 0
    for i in range(0,len(word_pd_reader['tfxidf'])):
        if(word_pd_reader['words'][k] == word):
            return word_pd_reader['tfxidf'][k]
        k = k + 1
    return 0

def checkTfIdf_Score(word,document_no,wordss):

    global dictlist
    #global pv,pvs,word_pd_reader,wordss
##    if(pv == document_no):
##        pass
##    else:
##        dir_path = os.path.dirname(__file__)
##        rel_path = "documents/tfidf/"+"tfidf_document"+document_no+".csv"
##        path = os.path.join(dir_path, rel_path)
##        word_pd_reader = pd.read_csv(path,encoding = "ISO-8859-1")
##    if(pvs == sent):
##        pass
##    else:
##        dir_path = os.path.dirname(__file__)
##        rel_path = "documents/"+"preprocessed"+"_document"+document_no+".txt"
##        abs_file_path_write = os.path.join(dir_path, rel_path)
##        rf = open(abs_file_path_write,"r")
##        input_text = rf.read();
##        sentences = sent_tokenize(input_text)
##        rf.close()
##        single_sent = sentences[int(sent)-1]
##        single_sent = re.sub(r'[^\w\s]','',single_sent)
##        wordss = word_tokenize(single_sent)
##    print(wordss)
##    print(word)
##    print("*****")
##    print(dictlist[int(document_no)-1])
    if(word in wordss):
        return dictlist[int(document_no)-1][word]
            
            
##        k = 0
##        for i in range(0,len(word_pd_reader['tfxidf'])):
##            if(word_pd_reader['words'][k] == word):
##                return word_pd_reader['tfxidf'][k]
##            k = k + 1
    pv = document_no
    return 0

def calculate_AvgDoc(doc1,sen1):
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/"+"preprocessed"+"_document"+str(doc1)+".txt"
    abs_file_path_write = os.path.join(dir_path, rel_path)
    rf = open(abs_file_path_write,"r")
    input_text = rf.read();
    sentences = sent_tokenize(input_text)
    rf.close()
    single_sent = sentences[int(sen1)-1]
    single_sent = re.sub(r'[^\w\s]','',single_sent)
    words = word_tokenize(single_sent)
    value = 0
    k = 0
    for i in words:
        value = value+checkTfIdf_Score2(i,doc1)**2
        k = k + 1
    value = value**0.5

    return value
def findDocNo(n):

    global array_for_range
    rf = readFile("","xyz.txt")
    text = rf.read()
    sentences = sent_tokenize(text)
    rf.close()
    if(n == 0):
        return 1
    else:
        for i in range(0,len(array_for_range)):
            if(n<=array_for_range[i]):
                break

    return i
def findSenNo(d_no,x):
    global array_for_range
    upper_bound = array_for_range[d_no]
    lower_bound = array_for_range[d_no-1]+1
    #print("UB : "+str(upper_bound)+"LB : "+str(lower_bound))
    if(d_no == 1):
        lower_bound = 0
    count = 1
    for i in range(lower_bound,upper_bound+1):
        if(x == i):
            return count
        else:
            count = count+1
    return count
def createSentenceAverageList(dimension):
    avg_list = list()
    for i in range(0,dimension):
        doc_no1 = findDocNo(i)
        sen_no1 = findSenNo(doc_no1,i)
        avg_list.append(calculate_AvgDoc(str(doc_no1),str(sen_no1)))
    return avg_list

def returnSentence(doc):
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/"+"preprocessed"+"_document"+doc+".txt"
    abs_file_path_write = os.path.join(dir_path, rel_path)
    rf = open(abs_file_path_write,"r")
    input_text = rf.read();
    sentences = sent_tokenize(input_text)
    rf.close()
    return sentences

def returnWords(sen,sentences):
    single_sent = sentences[int(sen)-1]
    single_sent = re.sub(r'[^\w\s]','',single_sent)
    words = word_tokenize(single_sent)
    return words

def calculate_MulDoc1Doc2(doc1,sen1,doc2,sen2,x,y):

    value1 = x
    value2 = y
    global words1,words2
    global previous_doc1,previous_doc2
    global previous_sent1,previous_sent2
    global sentences1,sentences2
##    print(str(previous_doc1) +"    :    "+str(doc1))
##    print(str(previous_sent1)+"####"+str(sen1))
    if(str(previous_doc1)==str(doc1)):
##        print("^^^^")
        #print(previous_sent1+"####"+sen1)
        if(doc1==doc2):
            sentences2 = sentences1
            words1 = returnWords(sen1,sentences1)
            words2 = returnWords(sen2,sentences2)
##            print(words1)
##            print(words2)
        else:
            words1 = returnWords(sen1,sentences1)
            sentences2 = returnSentence(doc2)
            words2 = returnWords(sen2,sentences2)

    elif(previous_doc2 == doc2):
            sentences1 = returnSentence(doc1)
            words1 = returnWords(sen1,sentences1)
            words2 = returnWords(sen2,sentences2)

    else:    
        if(doc1 == doc2):
            sentences1 = returnSentence(doc1)
            sentences2 = sentences1
            words1 = returnWords(sen1,sentences1)
            words2 = returnWords(sen2,sentences2)
        else:
            sentences1 = returnSentence(doc1)
            words1 = returnWords(sen1,sentences1)
            sentences2 = returnSentence(doc2)
            words2 = returnWords(sen2,sentences2)
    wordsx = list()
    wordsx = copy.deepcopy(words1)
    for i in words2:
        if(i in words1):
            pass
        else:
            wordsx.append(i)

    value = 0
    global pv,pvs
    pv = 0
    pvs = 0
    list1 = list()
    list2 = list()
    for i in wordsx:
        list1.append(checkTfIdf_Score(i,doc1,words1))
        #print(words1)
    for i in wordsx:
        #print(words2)
        list2.append(checkTfIdf_Score(i,doc2,words2))
        #print(words2)
    for i in range(0,len(wordsx)):
        value = value + list1[i]*list2[i]
    value3 = value1*value2
    value = value/value3
    
    previous_doc1 = doc1
    previous_doc2 = doc2
    previous_sent1 = sen1
    previous_sent2 = sen2
    
    return value

def createMatrix(n):
    global matrix
    matrix = numpy.empty((n,n))
    return

def readFile(path,fileName):
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/"
    path = rel_path+path+fileName
    path = os.path.join(dir_path, path)
    rf = open(path,"r")
    return rf

def convertIntoSingle(n):
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/xyz.txt"
    path = os.path.join(dir_path, rel_path)
    wf = open(path,"w+")
    wf.close()
    sum = 0
    global array_for_range
    array_for_range = []
    array_for_range.append(0)
    for i in range(0,n):
        s = str(i+1)+".txt"
        rf = readFile("preprocessed_document",s)
        input_text = rf.read();
        sentences = sent_tokenize(input_text)
        rf.close()
        wf = open(path,"a")
        sum = sum + len(sentences)
        for j in range(0,len(sentences)):
            wf.write(" "+sentences[j])
        array_for_range.append(sum-1)
        
    wf.close()
    print(array_for_range)
    createMatrix(sum)

    dir_path = os.path.dirname(__file__)
    rel_path = "documents/combined.txt"
    path = os.path.join(dir_path, rel_path)
    wf = open(path,"w+")
    wf.close()
    
    
    for i in range(0,n):
        s = str(i+1)+".txt"
        rf = readFile("document",s)
        input_text = rf.read();
        sentences = sent_tokenize(input_text)
        rf.close()
        wf = open(path,"a")
        for j in range(0,len(sentences)):
            wf.write(" "+sentences[j])
        
    wf.close()
    
    
    return sum



def createSimilarityMatrix(dimension):

    global matrix
    global previous_doc1,previous_sent1,previous_sent2,previous_doc2
##    prev1 = 0
##    prev2 = 0
    AverageList = createSentenceAverageList(dimension)
    previous_sent1 = 0
    previous_sent2 = 0
    previous_doc1 = 0
    previous_doc2 = 0
    for i in range(0,dimension):
        doc_no1 = findDocNo(i)
        sen_no1 = findSenNo(doc_no1,i)
        x = AverageList[i]
        for j in range(0,dimension):
            doc_no2 = findDocNo(j)
            sen_no2 = findSenNo(doc_no2,j)
            y = AverageList[j]
            if(i<j):      
                matrix[i][j] = calculate_MulDoc1Doc2(str(doc_no1),str(sen_no1),str(doc_no2),str(sen_no2),x,y)
                matrix[j][i] = matrix[i][j] 
                print(str(doc_no1)+"    :    "+str(sen_no1)+"    :    "+str(doc_no2)+"    :    "+str(sen_no2))
            if(i == j):
                matrix[i][j] = 1.0
                print(str(doc_no1)+"    :    "+str(sen_no1)+"    :    "+str(doc_no2)+"    :    "+str(sen_no2))
                
    print(matrix)
    
    return
def createAdjancyList(dimension):
    global matrix
    global adjancyList
    adjancyList = list()
    for i in range(0,dimension):
        #print(i)
        x = list()
        for j in range(0,dimension):
            if(matrix[i][j]!=1.0 and matrix[i][j]!=0.0):
                x.append(j)
            else:
                pass
        adjancyList.insert(i,x)

    print(adjancyList)
    return
def findNeighbours(node):
    x = list()
    global adjancyList
    x = adjancyList[node]
    return x
def calculateLexScores(dimension):
    global lx_score
    lx_score = list()
    for i in range(0,dimension):
        score = 0
        x = list()
        x = findNeighbours(i)
        #print("***Neighbours***")
        #print(x)
        for j in range(0,len(x)):
            y = list()
            y = findNeighbours(x[j])
##            print("###Neghbours###")
##            print(y)
            v = 0
            for k in range(0,len(y)):
                v = v + matrix[y[k]][x[j]]
##                print("@@@@@@@@@"+str(v)+"@@@@@@")
            v = matrix[i][x[j]]/v
##            print("$$$$"+str(v)+"$$$$")
            score = score+v
##            print(score)
        lx_score.insert(i,0.15*score+0.85/dimension)

    print(lx_score)        
    return
def createSummary(size):
    global index_list
    index_list = list()
    global lx_score
    lx = list()
    lx = copy.deepcopy(lx_score)
    for i in range(0,size):
        for j in range(0,len(lx)):
            if(lx[j]==max(lx)):
                #print(max(lx))
                index_list.append(j)
                lx[j]=0.0
                break
            else:
                pass
    index_list = sorted(index_list)
    rf = readFile("","combined.txt")
    input_text = rf.read();
    sentences = sent_tokenize(input_text)
    rf.close()
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/tfidf/"+"summary.txt"
    path = os.path.join(dir_path, rel_path)
    wf = open(path,"a")

    for i in range(0,len(index_list)):
        wf.write(sentences[i]+"\n")
    
    return 
def main():
    start_time = time.time()
    n = input("Enter the number of documents:")
    size = input("Enter the size of summary:")
    createDictionaries(int(n))
    sum = convertIntoSingle(int(n))
    createSimilarityMatrix(sum)
    print("Similarity Matrix Created in : "+str((time.time() - start_time)/60)+"minutes")
    createAdjancyList(sum)
    calculateLexScores(sum)
    createSummary(int(size))
    print("\n\n")
    print("Total Execution time : "+str((time.time() - start_time)/60)+"minutes")
    return

main()
