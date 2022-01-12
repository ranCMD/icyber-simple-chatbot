import nltk # Platform used for building programs that work with human language data for applying in statistical natural language processing.
from nltk.stem.api import StemmerI # Processing interface for removing morphological affixes from words (known as stemming).
#nltk.download('punkt') # Tokenizer that divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.
from nltk.stem.porter import PorterStemmer # Word stemmer based on the Porter stemming algorithm.
import numpy as np # Numerical Python, is a library consisting of multidimensional array objects and a collection of routines for processing those arrays.

"""
nltk_utils.py: Handles the natural language processing of the program.
"""

stemmer = PorterStemmer()

def tokenize(sentance):
    # Divide strings into lists of substrings.
    # Can be used to find the words and punctuation in a string.
    return nltk.word_tokenize(sentance)

def stem(word):
    # Used to remove morphological affixes from words, leaving only the word stem.
    # Organising, organise, organised = organis.
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    # Returns bag of words array.
    # One for each known word that exists in the sentence otherwise zero.
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    sentence_words = [stem(word) for word in tokenized_sentence] # Stems each word.
    bag = np.zeros(len(words), dtype=np.float32) # Initialises bag of words with 0 for each word.
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag