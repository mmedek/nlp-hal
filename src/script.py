# preprocessing
import re
import unidecode
import math
# sorting map
import operator
# local imports
import czech_stemmer as stem

############################ CONSTANTS AND LOADING DATA

TRAIN_DATA_PATH = '../data/train.txt'
STOPWORDS_PATH = '../data/stopwords.txt'
REGEX = re.compile('[^a-zA-Z]')
WINDOW_SIZE = 3
LOG_BASE = 2
MAX_TF = 2000

# load data from txt files except stopwords (without accents)
with open(STOPWORDS_PATH, encoding="utf8") as file:
    stopwords = file.readlines()
with open(TRAIN_DATA_PATH, encoding="utf8") as file:
    train_data = file.readlines()

############################ METHODS

# preprocess string    
def to_string(str):
    # remove accents and lower case
    without_accents = unidecode.unidecode(str).lower()
    # keep only a-z
    without_accents = REGEX.sub('', without_accents)
    return without_accents

# check if is string among stopwords if yes returns empty string
# otherwise returns string
def remove_stopwords(str):
    # if is str in our stopwords return '' otherwise return str
    if any(str in s for s in stopwords):
        return ''
    return str

# expected list of sentences which tokenize
# preprocess and returns
def preprocess_sentences(loaded_sentences, use_stopwords = True, 
                         use_stemm = False):
    sentences = []
    for i in range(len(loaded_sentences)):
    #for i in range(1000):
        splitted = loaded_sentences[i].split()
        preprocessed_sentence = []
        for j in range(len(splitted)):
            formatted = to_string(splitted[j])
            without_stopwords = formatted
            if use_stopwords:
                without_stopwords = remove_stopwords(formatted)
            if (without_stopwords == ''):
                continue
            # stemming can be agressive or not according to flag
            stemmed = without_stopwords
            if use_stemm:
                stemmed = stem.cz_stem(stemmed)
            preprocessed_sentence.append(stemmed)
        sentences.append(preprocessed_sentence)
    return sentences

############################ MAIN
    
# remove accents, lower case and eventually stem or remove stopwords
train_data = preprocess_sentences(train_data)

# create vocabulary and map of occurences for computing IDF
voc = {}
occ = {}
num_words = 0
for i in range(len(train_data)):
    for j in range(len(train_data[i])):
        num_words += 1
        voc[train_data[i][j]] = {}
        if train_data[i][j] in occ:
            occ[train_data[i][j]] += 1
        else:
            occ[train_data[i][j]] = 1
# filter according to term frequency
for i in range(len(train_data)):
    removed = 0
    for j in range(len(train_data[i])):
        if occ[train_data[i][j - removed]] >= MAX_TF:
            train_data[i].pop(j - removed)
            removed += 1
# process HAL
for i in range(len(train_data)):
    sentence_size = len(train_data[i])
    for j in range(sentence_size):
        key = train_data[i][j]
        
        start_ind = 0
        if j - WINDOW_SIZE >= 0:
            start_ind = j - WINDOW_SIZE
            
        end_ind = sentence_size
        if j + WINDOW_SIZE + 1 <= sentence_size:
            end_ind = j + WINDOW_SIZE + 1

        for index in range(start_ind, end_ind, 1):
            if j != index:
                word = train_data[i][index]
                weight = 1 / abs(j - index)
                if word in voc[key]:
                    voc[key][word] += 1#weight * idf
                else:
                    voc[key][word] = 0

# test
context_word = 'praha'
# print_nearest_words(voc, context_word, sorted_voc)
sorted_voc = sorted(voc[context_word].items(), key=operator.itemgetter(1), reverse=True)
