from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import re,string,os
import numpy as np
    
sentences = ""
stop_words = ""
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
script_dir = os.path.dirname(__file__)

def read__File(dname,flname):
    script_dir = os.path.dirname(__file__)
    rel_path = "documents/"+dname+flname
    abs_file_path = os.path.join(script_dir, rel_path)
    rf = open(abs_file_path,"r")    
    return rf

def get_AbsolutePath(dname,flname):
    rel_path = "documents/"+dname+flname
    afp = os.path.join(script_dir, rel_path)
    return afp

def doAllDocumensts_Preprocessing(fn):

    global file_name
    file_name = fn
    arrangeFunc(file_name)
    removeStopWords()
    global preprocessed_sentence
    preprocessed_sentence = doStemming()
    #print(preprocessed_sentence)
    preprocessed_sentence = removePunctuation(preprocessed_sentence)
    preprocessingFunc(preprocessed_sentence)
    doTokenize(preprocessed_sentence)
    return

def arrangeFunc(file_name):
    rf = read__File("",file_name)
    input_text = rf.read();
    rf.close
    
    global sentences    
    sentences = sent_tokenize(input_text)
    return

def filterStopWords(words):

    filtered_sentence = ""
    global stop_words
    for w in words:
        if w not in stop_words:
            filtered_sentence = filtered_sentence + w +" "
            
    return filtered_sentence

def removeStopWords():
    x = 0;
    global sentences
    global L
    L= []
    for i in sentences:
        words = word_tokenize(i)
        L.insert(x,filterStopWords(words))
        x = x+1
    return

def get_wordnet_pos(treebank_tag):
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
def performStem(sent):
    global lemmatizer
    stem_sent = ""
    k = 0
    word_tokens = word_tokenize(sent)
    pos_tokens = pos_tag(word_tokens)
    #print(pos_tokens)
    for w in word_tokens:
        if(get_wordnet_pos(pos_tokens[k][1])==''):
            stem_sent = stem_sent + pos_tokens[k][0]+" "
        else:
            stem_sent = stem_sent + lemmatizer.lemmatize(pos_tokens[k][0],get_wordnet_pos(pos_tokens[k][1]))+" "
        k = k + 1
    #print(stem_sent)    
    return stem_sent       
def checkProperNoun(senti):
    b_var = False
    tagged_sent = pos_tag(senti.split())
    propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
##    print(propernouns)
##    print("*****"+str(senti))
    if(len(propernouns)==2):
        b_var = True
    
    return b_var

def checkCuePhrase(senti):
    b_var = False
    #word_tokens = word_tokenize(senti)
    global cue_phrases
    global pn_count
    for i in cue_phrases:
        if(senti.find(i)!=-1):
            b_var = True
        else:
            #print(str(pn_count)+"**"+senti)
            pass
    return b_var
def doStemming():
    global sentences
    global L
    global preprocessed_sentence
    preprocessed_sentence = list()
    
    k = 0
    
    for i in L:
        preprocessed_sentence.insert(k,performStem(i))
        #print("["+str(k)+"] "+performStem(i))        
        k = k + 1
    return preprocessed_sentence

def removePunctuation(list):
    k = 0;
    global pn_count
    wff = open(get_AbsolutePath("","pnf.txt"),"a")
    wwf = open(get_AbsolutePath("","cpf.txt"),"a")
    for i in list:
        #w = re.sub('[%s]' % re.escape(string.punctuation), '', i)
        w = re.sub(r'[^\w\s]','',i)
        list[k] = ""
        if(checkProperNoun(w)):
            #print("@@@@@@@@"+str(pn_count))
            wff.write(str(pn_count)+"\n")
        else:
            pass
        
        list[k] = w.lower()
        if(checkCuePhrase(w.lower())):
            wwf.write(str(pn_count)+"\n")
        k = k+1
        pn_count = pn_count+1

    wff.close()
    wwf.close()
    return list

def preprocessingFunc(list):
    
    global file_name
    wf = open(get_AbsolutePath("preprocessed_",file_name),"w")
    x = 1;
    
    for i in list:
        wf.write(i+". ")
        x = x + 1
    
    wf.close()    
    return

def doTokenize(list):
    global file_name
    rel_path = "documents/"+"tokenize_"+file_name
    path = os.path.join(script_dir, rel_path)
    wf = open(path,"w")
    k = 0;
    s = ""
    for i in list:
        s = s+i
        k = k+1
    for st in word_tokenize(s):
        wf.write(str(st)+"\n")    
    #wf.write(str(word_tokenize(s)))
    return

def main():
    global cue_phrases
    cue_phrases = {'anyway','by the way','furthermore','first','second','then','now','thus','moreover','therefore','hence','lastly','in summary','finally','on the other hand'}
    
    global pn_count
    pn_count = 0
    wwf = open(get_AbsolutePath("","cpf.txt"),"w")
    wwf.truncate()
    wwf.close()
    wf = open(get_AbsolutePath("","pnf.txt"),"w")
    wf.truncate()
    wf.close()
    wf = open(get_AbsolutePath("","slf.txt"),"w")
    wf.truncate()
    wf.close()
    wf = open(get_AbsolutePath("","slf.txt"),"a")
    n = input("Enter the number of documents:")
    a = ["" for x in range(int(n))]
    for i in range(0,len(a)):
        wf.write(str(pn_count)+"\n")
        a[i] = "document"+str(i+1)+".txt"
        doAllDocumensts_Preprocessing(a[i])
        wf.write(str(pn_count-1)+"\n")
        
    return
main()
