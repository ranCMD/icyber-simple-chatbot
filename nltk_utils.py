import nltk
import numpy as np
from nltk.stem.api import StemmerI
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentance):
    return nltk.word_tokenize(sentance)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    # Returns bag of words array.
    # One for each known word that exists in the sentence otherwise zero.
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    sentence_words = [stem(word) for word in tokenized_sentence] # Stems each word
    bag = np.zeros(len(words), dtype=np.float32) # Initialises bag of words with 0 for each word
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag