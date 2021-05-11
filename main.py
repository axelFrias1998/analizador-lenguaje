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

"""
Subjects extractor: evalúa el texto y etiqueta el resultado según el tag correspondiente a la librería nltk.tag. Si es 'NN', es un sujeto y se añade
a la lista de sujetos a retornar

args : text -> texto a evaluar
returns: subjects_list -> sujetos con etiqueta NN de clase pos_tag 
"""
def subjects_extractor(text):
    tagged_sent = pos_tag(word_tokenize(text, 'english'))
    subjects_list = [word for word, pos in tagged_sent if pos == 'NN']
    return subjects_list

"""
Nouns extractor: evalúa el texto y etiqueta el resultado según el tag correspondiente a la librería nltk.tag. Si es 'NNS', es un sustantivo y se añade
a la lista de sustativos a retornar

args : text -> texto a evaluar
returns: nouns_list -> sustantivos con etiqueta NNS de clase pos_tag 
"""
def nouns_extractor(text):
    tagged_sent = pos_tag(word_tokenize(text, 'english'))
    nouns_list = [word for word, pos in tagged_sent if pos == 'NNS']
    return nouns_list

"""
Emotions extractor: evalúa el texto, tokeniza las palabras en inglés y, según el documento emotions de la carpeta doc, acumula los resultados según
el tipo de sentimiento de la palabra identificada. Los sentimientos se acumulan en un listado si la palabra corresponde a una emoción.

args : text -> texto a evaluar
returns: emotion_list -> acumulado de emociones del documento emotions 
"""
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

"""
Graph_counters: recibe tres objetos de la clase Counters, los cuales incluyen el listado de emociones, sustantivos y sujetos en el texto evaluado.
Se obtienen los tres valores con más repeticiones y se grafican con la clase pyplot, generando un png con el resultado

args : 
    w -> Counter que tiene el listado de emociones
    x -> Counter que tiene el listado de sustantivos
    y -> Counter que tiene el listado de sujetos
"""
def graph_Counters(w, x, y):
    w, x, y = dict(w.most_common(15)), dict(x.most_common(15)), dict(y.most_common(15))
    
    print("\nSentimientos: ")
    for key, value in w.items():
        print(key, ': ', value)
    print("\nSustantivos: ")
    for key, value in x.items():
        print(key, ': ', value)
    print("\nSentimientos: ")
    for key, value in y.items():
        print(key, ': ', value)
    
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

"""
Main: ejecuta los métodos de gráfica y recolleción de datos, además de extraer el texto y limpiarlo.
"""
def main():
    clean_text = open('read.txt', encoding='utf-8').read().lower().translate(str.maketrans('', '', string.punctuation))
    emotion_list = emotions_extractor(clean_text)
    nouns_list = nouns_extractor(clean_text)        
    subjects_list = subjects_extractor(clean_text)
    graph_Counters(Counter(emotion_list), Counter(nouns_list), Counter(subjects_list))


if __name__ == "__main__":
    main()