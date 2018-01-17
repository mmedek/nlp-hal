# preprocessing
import re
import unidecode
import math
from scipy.sparse import lil_matrix
import operator
# sorting map
import numpy as np
# local imports
import czech_stemmer as stem
from sklearn.metrics.pairwise import cosine_similarity

############################ CONSTANTS AND LOADING DATA

TRAIN_DATA_PATH = '../data/train.txt'
STOPWORDS_PATH = '../data/stopwords.txt'
REGEX = re.compile('[^a-zA-Z]')
WINDOW_SIZE = 2
LOG_BASE = 2
TOP_WORD_OCCUR = 3000
TOP_RESULTS = 10

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

# build scipy sparse matrix which represents ccoocurencest matrix
def build_sparse_matrix(voc, indexes):
    voc_size = len(voc)
    mat = lil_matrix((voc_size, voc_size), dtype=float)
    for key, value in voc.items():
        for key2, value2 in value.items(): 
            mat[indexes[key], indexes[key2]] = value2
    return mat

# compute cosine similarity between pivot matrix and all others words matrix   
def compute_results(indexes, context_word, sparse_matrix):
    # idnex of pivot word in sparse matrix
    ind = indexes[context_word]
    # convert sparse matrix row to long array for pivot
    mat = sparse_matrix[:, ind]
    results = {}
    # iterate through vocabulary
    for key, ind2 in indexes.items():
        # convert sparse matrix row to long array for current item
        mat2 = sparse_matrix[:, ind2]
        results[key] = cosine_similarity(np.transpose(mat), np.transpose(mat2), dense_output=False)
    return results

# select only TOP most frequented words
def select_top_occurs(train_data, occ):
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            if train_data[i][j] in occ:
                occ[train_data[i][j]] += 1
            else:
                occ[train_data[i][j]] = 1
                
    sorted_occ = sorted(occ.items(), key=operator.itemgetter(1), reverse=True)
    sorted_occ = sorted_occ[0:TOP_WORD_OCCUR]
    sorted_occ = dict(sorted_occ)
    smaller_data = []
    for i in range(len(train_data)):
        reduced_sentence = []
        for j in range(len(train_data[i])):
            if train_data[i][j] in sorted_occ:
                reduced_sentence.append(train_data[i][j])
        if len(reduced_sentence) > 0:
            smaller_data.append(reduced_sentence)
    occ = sorted_occ
    train_data = smaller_data
    return train_data


# print results    
def print_results(res):
    # convert data from sparse matrix into single var
    for key, value in res.items(): 
        res[key] = value.toarray().flatten()[0]
    top_results = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
    top_results = top_results[0:TOP_RESULTS]
    print("##### TOP " + str(TOP_RESULTS) + " results for word '" + context_word + "'")
    for i in range(len(top_results)):
        print(top_results[i][0] + ": %0.3f" % top_results[i][1])    

# find counts of neighbours for building coocurency matrix        
def run_hal(train_data):
    for i in range(len(train_data)):
        sentence_size = len(train_data[i])
        for j in range(sentence_size):
            key = train_data[i][j]
            # compute start of sliding window
            start_ind = 0
            if j - WINDOW_SIZE >= 0:
                start_ind = j - WINDOW_SIZE
            # compute end of sliding window
            end_ind = sentence_size
            if j + WINDOW_SIZE + 1 <= sentence_size:
                end_ind = j + WINDOW_SIZE + 1
            for index in range(start_ind, end_ind, 1):
                if j != index:
                    weight = 1 / abs(j - index)
                    idf = math.log((num_words / occ[train_data[i][index]]), LOG_BASE)
                    if train_data[i][index] in voc[key]:
                        voc[key][train_data[i][index]] += weight * idf
                    else:
                        voc[key][train_data[i][index]] = 0
                        
############################ MAIN   
                        
# remove accents, lower case and eventually stem or remove stopwords
train_data = preprocess_sentences(train_data)
# create smaller dataset for faster run
occ = {}
train_data = select_top_occurs(train_data, occ)
# create vocabulary and map of occurences for computing IDF
voc = {}
indexes = {}
counter = 0
num_words = 0
for i in range(len(train_data)):
    for j in range(len(train_data[i])):
        num_words += 1
        voc[train_data[i][j]] = {}
        if train_data[i][j] not in indexes:
            indexes[train_data[i][j]] = counter
            counter += 1
# process HAL
run_hal(train_data)
# building sparse matrix - we save memory space
sparse_matrix = build_sparse_matrix(voc, indexes)
# compute cosine similarity for pivot
context_word = 'washington'
res = compute_results(indexes, context_word, sparse_matrix)
# print results
print_results(res)