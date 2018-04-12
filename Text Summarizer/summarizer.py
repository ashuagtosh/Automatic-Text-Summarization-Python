from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import FreqDist

import re,string,os,time,copy
import numpy as np
import operator,math,csv
import pandas as pd

class Preprocessing():

    cue_phrases = {'anyway','by the way','furthermore','first','second','then','now','thus','moreover','therefore','hence','lastly','in summary','finally','on the other hand'}
    pn_count = 0
    sentences = ""
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    script_dir = os.path.dirname(__file__)
    file_name = ""
    L= list()
    preprocessed_sentence = list()
    
    def __init__(self,N):
        self.N = N

    def read__File(self,dname,flname):
        rel_path = "documents/"+dname+flname
        abs_file_path = os.path.join(self.script_dir, rel_path)
        rf = open(abs_file_path,"r")    
        return rf

    def get_AbsolutePath(self,dname,flname):
        rel_path = "documents/"+dname+flname
        afp = os.path.join(self.script_dir, rel_path)
        return afp    
    
    def start(self):
        wwf = open(self.get_AbsolutePath("","cpf.txt"),"w")
        wwf.truncate()
        wwf.close()
        wf = open(self.get_AbsolutePath("","pnf.txt"),"w")
        wf.truncate()
        wf.close()
        wf = open(self.get_AbsolutePath("","slf.txt"),"w")
        wf.truncate()
        wf.close()
        wf = open(self.get_AbsolutePath("","slf.txt"),"a")
        a = ["" for x in range(self.N)]
        for i in range(0,len(a)):
            wf.write(str(self.pn_count)+"\n")
            a[i] = "document"+str(i+1)+".txt"
            self.doAllfunctions(a[i])
            wf.write(str(self.pn_count-1)+"\n")
        return

    def doAllfunctions(self,fn):
        self.file_name = fn
        self.arrangeFunc()
        self.removeStopWords()
        self.preprocessed_sentence = self.doStemming()
        #print(self.preprocessed_sentence)
        self.preprocessed_sentence = self.removePunctuation()
        self.preprocessingFunc()
        self.doTokenize()
        return

    def arrangeFunc(self):
        rf = self.read__File("",self.file_name)
        input_text = rf.read();
        rf.close    
        self.sentences = sent_tokenize(input_text)
        return

    def filterStopWords(self,words):
        filtered_sentence = ""
        for w in words:
            if w not in self.stop_words:
                filtered_sentence = filtered_sentence + w +" "
        return filtered_sentence
    
    def removeStopWords(self):
        x = 0;
        self.L.clear()
        for i in self.sentences:
            words = word_tokenize(i)
            self.L.insert(x,self.filterStopWords(words))
            x = x+1
        #print(self.L)
        return

    def get_wordnet_pos(self,treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''
    
    def performStem(self,sent):
        stem_sent = ""
        k = 0
        word_tokens = word_tokenize(sent)
        pos_tokens = pos_tag(word_tokens)
        #print(word_tokens)
        for w in word_tokens:
            if(self.get_wordnet_pos(pos_tokens[k][1])==''):
                stem_sent = stem_sent + pos_tokens[k][0]+" "
            else:
                stem_sent = stem_sent + self.lemmatizer.lemmatize(pos_tokens[k][0],self.get_wordnet_pos(pos_tokens[k][1]))+" "
            k = k + 1
        return stem_sent

    def doStemming(self):
        self.preprocessed_sentence.clear()
        
        k = 0
        for i in self.L:
            self.preprocessed_sentence.insert(k,self.performStem(i))
            k = k + 1
        return self.preprocessed_sentence
    
    def checkProperNoun(self,senti):
        b_var = False
        tagged_sent = pos_tag(senti.split())
        propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
        if(len(propernouns)==2):
            b_var = True       
        return b_var

    def checkCuePhrase(self,senti):
        b_var = False
        for i in self.cue_phrases:
            if(senti.find(i)!=-1):
                b_var = True
            else:
                pass
        return b_var

    def removePunctuation(self):
        k = 0;
        wff = open(self.get_AbsolutePath("","pnf.txt"),"a")
        wwf = open(self.get_AbsolutePath("","cpf.txt"),"a")
        for i in self.preprocessed_sentence:
            w = re.sub(r'[^\w\s]','',i)
            self.preprocessed_sentence[k] = ""
            if(self.checkProperNoun(w)):
                wff.write(str(self.pn_count)+"\n")
            else:
                pass
            self.preprocessed_sentence[k] = w.lower()
            if(self.checkCuePhrase(w.lower())):
                wwf.write(str(self.pn_count)+"\n")
            k = k+1
            self.pn_count = self.pn_count+1
        wff.close()
        wwf.close()
        return self.preprocessed_sentence

    def preprocessingFunc(self):
        wf = open(self.get_AbsolutePath("preprocessed_",self.file_name),"w")
        x = 1;
        for i in self.preprocessed_sentence:
            wf.write(i+". ")
            x = x + 1
        wf.close()    
        return

    def doTokenize(self):
        rel_path = "documents/"+"tokenize_"+self.file_name
        path = os.path.join(self.script_dir, rel_path)
        wf = open(path,"w")
        k = 0;
        s = ""
        for i in self.preprocessed_sentence:
            s = s+i
            k = k+1
        for st in word_tokenize(s):
            wf.write(str(st)+"\n")    
        return

class createMatrix():

    file_name = ""
    original_file_name = ""
    script_dir = os.path.dirname(__file__)
    L = list()
    leng = 0
    h = 0
    
    def __init__(self,N):
        self.N = N
        
    def start(self):
        a = ["" for x in range(self.N)]
        for i in range(0,len(a)):
            a[i] = str(i+1)
            x = "tokenize_document"+a[i]+".txt"
            self.performAllFunctions(x,a[i])
        for i in range(0,len(a)):
            self.calculateIDF(i)
        for i in range(0,len(a)):
            self.calculateTFIDF(i)
        return
    
    def performAllFunctions(self,fn,ofn):
        self.file_name = fn
        self.original_file_name = ofn+".csv"
        self.readFile()
        self.TF()
        self.computeTF()
        return

    def readFile(self):
        rel_path = "documents/"+self.file_name
        abs_file_path = os.path.join(self.script_dir, rel_path)
        rf = open(abs_file_path,"r")
        self.L = rf.readlines()
        return
    
    def doWrite(self,l,name):
        
        rel_path = "documents/tfidf/"+name+"_document"+self.original_file_name
        path = os.path.join(self.script_dir, rel_path)
        f=open(path, "w",newline='')
        writer = csv.writer(f)
        writer.writerow(['words', 'frequency','tf_score'])
        k = 0
        for i in range(0,len(l)):
            writer.writerow([l['col0'][k],l['col1'][k],l['col2'][k]])
            k = k+1
        return

    def readCSV(self):
        rel_path = "documents/tfidf/"+"tfidf_document"+"1"+".csv"
        path = os.path.join(self.script_dir, rel_path)
        df = pd.read_csv(path)
        print(df['tf_score'])
        return

    def TF(self):
        w = 2
        self.leng = len(self.L)
        self.h = len(self.L)
        Matrix = [[0 for x in range(w)] for y in range(self.h)]
        k = 0;
        self.L = list(map(lambda s: s.strip(), self.L))
        #print(h)        
        for i in self.L:
            Matrix[k][0]=self.L[k]
            k = k+1
        for i in range(0,self.h):
            count = 0;
            for j in range(0,self.h):
                if(Matrix[i][0] == Matrix[j][0]):
                    count = count + 1
            Matrix[i][1] = count
        for i in range(self.h-1,-1,-1):
            count = 0;
            for j in range(self.h-1,-1,-1):
                if(Matrix[i][0] == Matrix[j][0]):
                    count = count + 1
            if(count > 1):
                del Matrix[i]
                self.h = self.h - 1
        #doWrite(Matrix,"wf")
        print(Matrix)
        return Matrix

    def computeTF(self):
        self.leng = len(self.TF())
        Matrix_tfidf = [[0 for x in range(2)] for y in range(self.h)]
        Matrix_tfidf = self.TF()
        words = list()
        frequency = list()
        tf_score = list()
        k = 0;   
        for i in range (0,self.h):
            words.insert(k,Matrix_tfidf[i][0])
            frequency.insert(k,Matrix_tfidf[i][1])
            Matrix_tfidf[i][1] = (Matrix_tfidf[i][1]/self.leng)
            Matrix_tfidf[i][1] = math.floor(Matrix_tfidf[i][1]*10000)/10000
            tf_score.insert(k,Matrix_tfidf[i][1])
            k = k + 1
        Matrix_list = pd.DataFrame(
        {'col0': words,
         'col1': frequency,
         'col2': tf_score
        })
        self.doWrite(Matrix_list,"tfidf")
        return

    def calculate_IDF2(self,word,count):
        c = 1
        for i in range(0,self.N):
            b = 0
            if(i==int(count)):
                pass
            else:
                y = int(i)+1
                x = str(y)
                rel_path2 = "documents/tfidf/"+"tfidf_document"+x+".csv"
                path2 = os.path.join(self.script_dir, rel_path2)
                word_pd_reader = pd.read_csv(path2,encoding = "ISO-8859-1")
                m = 0
                for j in range (0,len(word_pd_reader['words'])):
                    if(word_pd_reader['words'][m]== word):
                        b = 1
                    m = m + 1    
            c = c + b
        c = 1+math.log10(int(n)/int(c))
        c = math.floor(c*10000)/10000
        return c

    def calculateIDF(self,count):
        x = str(count + 1)
        rel_path2 = "documents/tfidf/"+"tfidf_document"+x+".csv"
        path2 = os.path.join(self.script_dir, rel_path2)
        word_pd_reader = pd.read_csv(path2,encoding = "ISO-8859-1")
        k = 0
        l = list()
        for i in range(0,len(word_pd_reader['words'])):
            #print(word_pd_reader['words'][k])
            c = self.calculate_IDF2(word_pd_reader['words'][k],count)
            l.insert(int(i),c)
            k = k + 1
        
        x = str(count + 1)
        rel_path23 = "documents/tfidf/"+"tfidf_document"+x+".csv"
        path3 = os.path.join(self.script_dir, rel_path23)
        df = pd.read_csv(path3,encoding = "ISO-8859-1")
        new_column = pd.DataFrame({'idf_values':l})
        df = df.merge(new_column, left_index = True, right_index = True)
        df.to_csv(path2,index = False)
        return

    def calculateTFIDF(self,count):
        x = str(count + 1)
        rel_path2 = "documents/tfidf/"+"tfidf_document"+x+".csv"
        path2 = os.path.join(self.script_dir, rel_path2)
        word_pd_reader = pd.read_csv(path2,encoding = "ISO-8859-1")
        l = list()
        for i in range(0,len(word_pd_reader)):
            l.insert(int(i),word_pd_reader['tf_score'][i]*word_pd_reader['idf_values'][i])
        new_column = pd.DataFrame({'tfxidf':l})
        df = word_pd_reader.merge(new_column, left_index = True, right_index = True)
        df.to_csv(path2,index = False)
        return
class Similarity():
    
    def __init__(self,N,size):
        self.N = N
        self.size = size
    def start(self):
        self.createDictionaries(self.N)
        sum = self.convertIntoSingle(self.N)
        self.createSimilarityMatrix(sum)
        self.createAdjancyList(sum)
        self.calculateLexScores(sum)
        self.createSummary(self.size)
        return
    
    def createDictionaries(self,n):
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

    def checkTfIdf_Score2(self,word,document_no):
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

    def checkTfIdf_Score(self,word,document_no,wordss):
        global dictlist
        if(word in wordss):
            return dictlist[int(document_no)-1][word]
        pv = document_no
        return 0

    def calculate_AvgDoc(self,doc1,sen1):
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
            value = value+self.checkTfIdf_Score2(i,doc1)**2
            k = k + 1
        value = value**0.5
        return value
    
    def findDocNo(self,n):
        global array_for_range
        rf = self.readFile("","xyz.txt")
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
    
    def findSenNo(self,d_no,x):
        global array_for_range
        upper_bound = array_for_range[d_no]
        lower_bound = array_for_range[d_no-1]+1
        if(d_no == 1):
            lower_bound = 0
        count = 1
        for i in range(lower_bound,upper_bound+1):
            if(x == i):
                return count
            else:
                count = count+1
        return count
    
    def createSentenceAverageList(self,dimension):
        avg_list = list()
        for i in range(0,dimension):
            doc_no1 = self.findDocNo(i)
            sen_no1 = self.findSenNo(doc_no1,i)
            avg_list.append(self.calculate_AvgDoc(str(doc_no1),str(sen_no1)))
        return avg_list

    def returnSentence(self,doc):
        dir_path = os.path.dirname(__file__)
        rel_path = "documents/"+"preprocessed"+"_document"+doc+".txt"
        abs_file_path_write = os.path.join(dir_path, rel_path)
        rf = open(abs_file_path_write,"r")
        input_text = rf.read();
        sentences = sent_tokenize(input_text)
        rf.close()
        return sentences

    def returnWords(self,sen,sentences):
        single_sent = sentences[int(sen)-1]
        single_sent = re.sub(r'[^\w\s]','',single_sent)
        words = word_tokenize(single_sent)
        return words

    def calculate_MulDoc1Doc2(self,doc1,sen1,doc2,sen2,x,y):

        value1 = x
        value2 = y
        global words1,words2
        global previous_doc1,previous_doc2
        global previous_sent1,previous_sent2
        global sentences1,sentences2
        if(str(previous_doc1)==str(doc1)):
            if(doc1==doc2):
                sentences2 = sentences1
                words1 = self.returnWords(sen1,sentences1)
                words2 = self.returnWords(sen2,sentences2)
            else:
                words1 = self.returnWords(sen1,sentences1)
                sentences2 = self.returnSentence(doc2)
                words2 = self.returnWords(sen2,sentences2)

        elif(previous_doc2 == doc2):
                sentences1 = self.returnSentence(doc1)
                words1 = self.returnWords(sen1,sentences1)
                words2 = self.returnWords(sen2,sentences2)

        else:    
            if(doc1 == doc2):
                sentences1 = self.returnSentence(doc1)
                sentences2 = sentences1
                words1 = self.returnWords(sen1,sentences1)
                words2 = self.returnWords(sen2,sentences2)
            else:
                sentences1 = self.returnSentence(doc1)
                words1 = self.returnWords(sen1,sentences1)
                sentences2 = self.returnSentence(doc2)
                words2 = self.returnWords(sen2,sentences2)
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
            list1.append(self.checkTfIdf_Score(i,doc1,words1))
        for i in wordsx:
            list2.append(self.checkTfIdf_Score(i,doc2,words2))
        for i in range(0,len(wordsx)):
            value = value + list1[i]*list2[i]
        value3 = value1*value2
        value = value/value3
        
        previous_doc1 = doc1
        previous_doc2 = doc2
        previous_sent1 = sen1
        previous_sent2 = sen2
        
        return value

    def createMatrix(self,n):
        global matrix
        matrix = np.empty((n,n))
        return

    def readFile(self,path,fileName):
        dir_path = os.path.dirname(__file__)
        rel_path = "documents/"
        path = rel_path+path+fileName
        path = os.path.join(dir_path, path)
        rf = open(path,"r")
        return rf

    def convertIntoSingle(self,n):
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
            rf = self.readFile("preprocessed_document",s)
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
        self.createMatrix(sum)

        dir_path = os.path.dirname(__file__)
        rel_path = "documents/combined.txt"
        path = os.path.join(dir_path, rel_path)
        wf = open(path,"w+")
        wf.close()
        for i in range(0,n):
            s = str(i+1)+".txt"
            rf = self.readFile("document",s)
            input_text = rf.read();
            sentences = sent_tokenize(input_text)
            rf.close()
            wf = open(path,"a")
            for j in range(0,len(sentences)):
                wf.write(" "+sentences[j])
            
        wf.close()
        return sum

    def createSimilarityMatrix(self,dimension):
        global matrix
        global previous_doc1,previous_sent1,previous_sent2,previous_doc2
        AverageList = self.createSentenceAverageList(dimension)
        previous_sent1 = 0
        previous_sent2 = 0
        previous_doc1 = 0
        previous_doc2 = 0
        for i in range(0,dimension):
            doc_no1 = self.findDocNo(i)
            sen_no1 = self.findSenNo(doc_no1,i)
            x = AverageList[i]
            for j in range(0,dimension):
                doc_no2 = self.findDocNo(j)
                sen_no2 = self.findSenNo(doc_no2,j)
                y = AverageList[j]
                if(i<j):      
                    matrix[i][j] = self.calculate_MulDoc1Doc2(str(doc_no1),str(sen_no1),str(doc_no2),str(sen_no2),x,y)
                    matrix[j][i] = matrix[i][j] 
                    print(str(doc_no1)+"    :    "+str(sen_no1)+"    :    "+str(doc_no2)+"    :    "+str(sen_no2))
                if(i == j):
                    matrix[i][j] = 1.0
                    print(str(doc_no1)+"    :    "+str(sen_no1)+"    :    "+str(doc_no2)+"    :    "+str(sen_no2))
                    
        print(matrix)
        return

    def createAdjancyList(self,dimension):
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
    def findNeighbours(self,node):
        x = list()
        global adjancyList
        x = adjancyList[node]
        return x

    def calculateLexScores(self,dimension):
        global lx_score
        lx_score = list()
        for i in range(0,dimension):
            score = 0
            x = list()
            x = self.findNeighbours(i)
            for j in range(0,len(x)):
                y = list()
                y = self.findNeighbours(x[j])
                v = 0
                for k in range(0,len(y)):
                    v = v + matrix[y[k]][x[j]]
                v = matrix[i][x[j]]/v
                score = score+v
            lx_score.insert(i,0.15*score+0.85/dimension)
        print(lx_score)        
        return

    def createSummary(self,size):
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
        rf = self.readFile("","combined.txt")
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

##        self.N = N
##        self.size = size
##    class Similarity()
##    dictlist = [dict() for x in range(self.N)]
##    script_dir = os.path.dirname(__file__)
##    array_for_range = list()
##    matrix = numpy.empty((self.N,self.N))
##    previous_doc1 = 0
##    previous_sent1 = 0
##    previous_sent2 = 0
##    previous_doc2 = 0
##    
##    def __init__(self,N,size):
##        self.N = N
##        self.size = size
##
##    def start(self):
##        self.createDictionaries(self.N)
##        sum = self.convertIntoSingle()
##        createSimilarityMatrix(sum)
##        
##    def createDictionaries(self,n):
##        for i in range(0,n):
##            rel_path = "documents/tfidf/"+"tfidf_document"+str(i+1)+".csv"
##            path = os.path.join(self.script_dir, rel_path)
##            word_pd_reader = pd.read_csv(path,encoding = "ISO-8859-1")
##            for j in range(0,len(word_pd_reader['tfxidf'])):
##                dictlist[i][word_pd_reader['words'][j]] = word_pd_reader['tfxidf'][j]
##        return
##
##    def convertIntoSingle(self):
##        rel_path = "documents/xyz.txt"
##        path = os.path.join(self.script_dir, rel_path)
##        wf = open(path,"w+")
##        wf.close()
##        sum = 0
##        self.array_for_range.append(0)
##        
##        for i in range(0,self.N):
##            s = str(i+1)+".txt"
##            rf = self.readFile("preprocessed_document",s)
##            input_text = rf.read();
##            sentences = sent_tokenize(input_text)
##            rf.close()
##            wf = open(path,"a")
##            sum = sum + len(sentences)
##            for j in range(0,len(sentences)):
##                wf.write(" "+sentences[j])
##            array_for_range.append(sum-1)
##            
##        wf.close()
##        print(self.array_for_range)
##        self.createMatrix(sum)
##
##        rel_path = "documents/combined.txt"
##        path = os.path.join(self.script_dir, rel_path)
##        wf = open(path,"w+")
##        wf.close()
##        
##        for i in range(0,self.N):
##            s = str(i+1)+".txt"
##            rf = self.readFile("document",s)
##            input_text = rf.read();
##            sentences = sent_tokenize(input_text)
##            rf.close()
##            wf = open(path,"a")
##            for j in range(0,len(sentences)):
##                wf.write(" "+sentences[j])
##            
##        wf.close()
##        return sum
##    
##    def readFile(self,path,fileName):
##        rel_path = "documents/"
##        path = rel_path+path+fileName
##        path = os.path.join(self.script_dir, path)
##        rf = open(path,"r")
##        return 
##    def createMatrix(self,n):
##        matrix = numpy.empty((n,n))
##        return
##
##    def createSimilarityMatrix(self,dimension):
##    
##        AverageList = self.createSentenceAverageList(dimension)
##        for i in range(0,dimension):
##            doc_no1 = self.findDocNo(i)
##            sen_no1 = self.findSenNo(doc_no1,i)
##            x = AverageList[i]
##            for j in range(0,dimension):
##                doc_no2 = self.findDocNo(j)
##                sen_no2 = self.findSenNo(doc_no2,j)
##                y = AverageList[j]
##                if(i<j):      
##                    self.matrix[i][j] = self.calculate_MulDoc1Doc2(str(doc_no1),str(sen_no1),str(doc_no2),str(sen_no2),x,y)
##                    self.matrix[j][i] = self.matrix[i][j] 
##                    print(str(doc_no1)+"    :    "+str(sen_no1)+"    :    "+str(doc_no2)+"    :    "+str(sen_no2))
##                if(i == j):
##                    self.matrix[i][j] = 1.0
##                    print(str(doc_no1)+"    :    "+str(sen_no1)+"    :    "+str(doc_no2)+"    :    "+str(sen_no2))
##                
##        print(self.matrix)
##        return
##
##    def createSentenceAverageList(self,dimension):
##        avg_list = list()
##        for i in range(0,dimension):
##            doc_no1 = self.findDocNo(i)
##            sen_no1 = self.findSenNo(doc_no1,i)
##            avg_list.append(self.calculate_AvgDoc(str(doc_no1),str(sen_no1)))
##        return avg_list
##
##    def findDocNo(self,n):
##        rf = self.readFile("","xyz.txt")
##        text = rf.read()
##        sentences = sent_tokenize(text)
##        rf.close()
##        if(n == 0):
##            return 1
##        else:
##            for i in range(0,len(self.array_for_range)):
##                if(n<=self.array_for_range[i]):
##                    break
##        return i 
##
##    def findSenNo(self,d_no,x):
##        upper_bound = self.array_for_range[d_no]
##        lower_bound = self.array_for_range[d_no-1]+1
##        if(d_no == 1):
##            lower_bound = 0
##        count = 1
##        for i in range(lower_bound,upper_bound+1):
##            if(x == i):
##                return count
##            else:
##                count = count+1
##        return count

n = input("Enter the number of documents:")
size = input("Enter the size of Summary")
inst_preprocessing = Preprocessing(int(n))
inst_preprocessing.start()
inst_features = createMatrix(int(n))
inst_features.start()
inst_similarity = Similarity(int(n),int(size))
inst_similarity.start()
