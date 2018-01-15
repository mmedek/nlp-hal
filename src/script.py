# preprocessing
import re
import unidecode
# local imports
import czech_stemmer as stem


TRAIN_DATA_PATH = '../data/train.txt'
STOPWORDS_PATH = '../data/stopwords.txt'
REGEX = re.compile('[^a-zA-Z]')

# load data from txt files except stopwords (without accents)
with open(STOPWORDS_PATH, encoding="utf8") as file:
    stopwords = file.readlines()
with open(TRAIN_DATA_PATH, encoding="utf8") as file:
    train_data = file.readlines()


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
    #for i in range(len(loaded_sentences)):
    # LOAD ONLY 100 TEXTS FOR TESTING PURPOSE
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

# remove accents, lower case and eventually stem or remove stopwords
train_data = preprocess_sentences(train_data, use_stopwords = False)