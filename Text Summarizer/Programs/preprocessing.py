from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import re,string,os

sentences = ""
stop_words = ""
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
script_dir = os.path.dirname(__file__)


def arrangeFunc(file_name):
    script_dir = os.path.dirname(__file__)
    rel_path = "documents/"+file_name
    abs_file_path = os.path.join(script_dir, rel_path)
    rf = open(abs_file_path,"r")    
    input_text = rf.read();
    rf.close
    
    global sentences    
    sentences = sent_tokenize(input_text)
    rel_path = "documents/"+"preprocessed_"+file_name
    global abs_file_path_write
    abs_file_path_write = os.path.join(script_dir, rel_path)
    wf = open(abs_file_path_write,"w")
    x =1;
    for i in sentences:
        wf.write("["+str(x)+"] "+i+"\n")
        x = x+1
    wf.close()    
    return

def preprocessingFunc(list):
    wf = open(abs_file_path_write,"w")
    x = 1;
    for i in list:
        wf.write("["+str(x)+"] "+i+"\n")
        x = x + 1
    wf.close()    
    return
def removePunctuation(list):
    k = 0;
    for i in list:
        w = re.sub('[%s]' % re.escape(string.punctuation), '', i)
        list[k] = ""
        list[k] = w
        k = k+1
    return list
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
        #print("["+str(x)+"] "+filterStopWords(words))
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
    for w in word_tokens:
        if(get_wordnet_pos(pos_tokens[k][1])==''):
            stem_sent = stem_sent + pos_tokens[k][0]+" "
        else:
            stem_sent = stem_sent + lemmatizer.lemmatize(pos_tokens[k][0],get_wordnet_pos(pos_tokens[k][1]))+" "
        k = k + 1
    #print(stem_sent)    
    return stem_sent       
    
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
    global file_name
    file_name = input("Enter the name of document for preprocessing : ")    
    arrangeFunc(file_name)
    removeStopWords()
    doStemming()
    global preprocessed_sentence
    preprocessed_sentence = removePunctuation(preprocessed_sentence)
    preprocessingFunc(preprocessed_sentence)
    doTokenize(preprocessed_sentence)
    return
main()
