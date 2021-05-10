import nltk
import string 
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords, words
from collections import Counter

#Todos los caracteres de puntuación se eliminan de la cadena
#Un token es cada palabra. Tokenization es una palabra de procesamiento de lenguaje
#stop words son palabras que no dan significado a la palabra en el procesamiento de lenguaje

def subjects_extractor(text):
    tokenized_words = word_tokenize(text, 'english')
    tagged_sent = pos_tag(tokenized_words)
    subjects_list = [word for word, pos in tagged_sent if pos == 'NN']
    return subjects_list

def nouns_extractor(text):
    tokenized_words = word_tokenize(text, 'english')
    tagged_sent = pos_tag(tokenized_words)
    nouns_list = [word for word, pos in tagged_sent if pos == 'NNS']
    return nouns_list

#print(words.words())
#stop_words en 'corpora\stopwords\english'

def emotions_extractor(text):
    emotion_list = []
    tokenized_words = word_tokenize(text, 'english')
    with open('corpora\dolch\emotions', 'r') as file:
        for line in file:
            clean_line = line.replace('\n', '').replace(',', '').replace("'", '').replace(' ','')
            word, emotion = clean_line.split(':')
            if word in tokenized_words:
                emotion_list.append(emotion)
    return emotion_list

clean_text = open('read.txt', encoding='utf-8').read().lower().translate(str.maketrans('', '', string.punctuation))
emotion_list = emotions_extractor(clean_text)
nouns_list = nouns_extractor(clean_text)        
subjects_list = subjects_extractor(clean_text)

w = Counter(emotion_list)
x = Counter(nouns_list)
y = Counter(subjects_list)
w, x, y = dict(w.most_common(15)), dict(x.most_common(15)), dict(y.most_common(15))

figW, ax1W = plt.subplots()
ax1W.bar(w.keys(), w.values())
figW.autofmt_xdate()
plt.savefig('feelingsGraph.png')

figX, ax1X = plt.subplots()
ax1X.bar(x.keys(), x.values())
figX.autofmt_xdate()
plt.savefig('NounsGraph.png')

figY, ax1Y = plt.subplots()
ax1Y.bar(y.keys(), y.values())
figY.autofmt_xdate()
plt.savefig('subjectsGraph.png')