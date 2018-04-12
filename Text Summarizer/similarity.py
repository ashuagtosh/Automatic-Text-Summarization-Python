from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import re,string,os
import numpy as np
import operator,math,csv
import pandas as pd

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

def checkTfIdf_Score(word,document_no,sent):

    dir_path = os.path.dirname(__file__)
    rel_path = "documents/tfidf/"+"tfidf_document"+document_no+".csv"
    path = os.path.join(dir_path, rel_path)
    word_pd_reader = pd.read_csv(path,encoding = "ISO-8859-1")

    dir_path = os.path.dirname(__file__)
    rel_path = "documents/"+"preprocessed"+"_document"+document_no+".txt"
    abs_file_path_write = os.path.join(dir_path, rel_path)
    rf = open(abs_file_path_write,"r")
    input_text = rf.read();
    sentences = sent_tokenize(input_text)
    rf.close()
    single_sent = sentences[int(sent)-1]
    single_sent = re.sub('[%s]' % re.escape(string.punctuation), '', single_sent)
    wordss = word_tokenize(single_sent)
    
    if(word in wordss):
        k = 0
        for i in range(0,len(word_pd_reader['tfxidf'])):
            if(word_pd_reader['words'][k] == word):
                return word_pd_reader['tfxidf'][k]
            k = k + 1
    
    return 0

def calculate_AvgDoc(doc1,sen1):
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/"+"preprocessed"+"_document"+doc1+".txt"
    abs_file_path_write = os.path.join(dir_path, rel_path)
    rf = open(abs_file_path_write,"r")
    input_text = rf.read();
    sentences = sent_tokenize(input_text)
    rf.close()
    single_sent = sentences[int(sen1)-1]
    single_sent = re.sub('[%s]' % re.escape(string.punctuation), '', single_sent)
    words = word_tokenize(single_sent)
    value = 0
    k = 0
    for i in words:
        value = value+checkTfIdf_Score2(i,doc1)**2
        k = k + 1
    value = value**0.5

    return value

def calculate_MulDoc1Doc2(doc1,sen1,doc2,sen2):
    value1 = calculate_AvgDoc(doc1,sen1)
    value2 = calculate_AvgDoc(doc2,sen2)
    
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/"+"preprocessed"+"_document"+doc1+".txt"
    abs_file_path_write = os.path.join(dir_path, rel_path)
    rf = open(abs_file_path_write,"r")
    input_text = rf.read();
    sentences = sent_tokenize(input_text)
    rf.close()
    single_sent1 = sentences[int(sen1)-1]
    single_sent1 = re.sub('[%s]' % re.escape(string.punctuation), '', single_sent1)
    words = word_tokenize(single_sent1)
    #print(words1)

    rel_path = "documents/"+"preprocessed"+"_document"+doc2+".txt"
    abs_file_path_write = os.path.join(dir_path, rel_path)
    rf = open(abs_file_path_write,"r")
    input_text = rf.read();
    sentences = sent_tokenize(input_text)
    rf.close()
    single_sent2 = sentences[int(sen2)-1]
    single_sent2 = re.sub('[%s]' % re.escape(string.punctuation), '', single_sent2)
    words2 = word_tokenize(single_sent2)

    for i in words2:
        if(i in words):
            print("")
        else:
            words.append(i)

    value = 0

    for i in words:
        value = value + checkTfIdf_Score(i,doc1,sen1)*checkTfIdf_Score(i,doc2,sen2)
    value3 = value1*value2
    value = value/value3
    return value

def getDocumentDimensions(name):
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/"+name
    abs_file_path_write = os.path.join(dir_path, rel_path)
    rf=open(abs_file_path_write, "r")
    input_text = rf.read();
    sentences = sent_tokenize(input_text)
    rf.close()
    
    return len(sentences)
def createNames():
    n = input("Enter the number of documents:")
    a = ["" for x in range(int(n))]
    L = []
    for i in range(0,len(a)):
        a[i] = str(i+1)
        x = "preprocessed_document"+a[i]+".txt"
        for j in range(0,getDocumentDimensions(x)):
            L.append("d"+str(a[i])+"s"+str(j+1))
        
    return L
def createExcel(list):
    dir_path = os.path.dirname(__file__)
    rel_path = "documents/"+"similarity_matrix.csv"
    path = os.path.join(dir_path, rel_path)
    f=open(path, "w",newline='')
    writer = csv.writer(f)
    new_list = []
    new_list.append("NA")
    new_list.extend(list)
    writer.writerow(new_list)
    
    lili = []
    for i in range(0,len(new_list)):
        lili.append('0')
    for i in range(0,len(new_list)):
        writer.writerow(lili)
    f.close()
    
    word_pd_reader = pd.read_csv(path,encoding = "ISO-8859-1")
    for i in range(0,len(list)):
        word_pd_reader['NA'][i+1]=str(list[i])
    
    return
def main():
    print(calculate_MulDoc1Doc2('6','1','6','1'));    
    createExcel(createNames())
    return
main()
