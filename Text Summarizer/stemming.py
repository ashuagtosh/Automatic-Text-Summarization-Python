from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

ls = WordNetLemmatizer()
ps = PorterStemmer()

text = "This is very importantly chunking tagger taged talking"

words = word_tokenize(text)

for i in words:
    print(ps.stem(i))

