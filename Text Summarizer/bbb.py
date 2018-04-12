from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import re,string,os
import numpy
import operator,math,csv
import pandas as pd
import time

def createDictionaries(n):
    dictlist = [dict() for x in range(int(n))]

    for i in range(0,n):
        dir_path = os.path.dirname(__file__)
        rel_path = "documents/tfidf/"+"tfidf_document"+str(i+1)+".csv"
        path = os.path.join(dir_path, rel_path)
        word_pd_reader = pd.read_csv(path,encoding = "ISO-8859-1")
        for j in range(0,len(word_pd_reader['tfxidf'])):
            dictlist[i][word_pd_reader['words'][j]] = word_pd_reader['tfxidf'][j]
    return

def main():
    n = input("Enter the number of documents:")
    createDictionaries(int(n))

    l = list()
    g = list()
    g = [2,3,5,7,9]
    l.append(g)
    l.append(g)
    print(l[1][1])
    return
main()
