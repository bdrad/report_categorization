import pickle
from nltk.corpus import stopwords

RADLEX_PATH = '../data/semantic_maps/radlex_replacements'
MISC_PATH = '../data/semantic_maps/misc_replacements'
CLEVER_PATH = '../data/semantic_maps/clever_replacements'

# originally contained 'mm' and 'cm'
REMOVAL_SET = set(['x', 'please', 'is', 'are', 'be', 'been'])


def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))


def read_removal(extra_removal):
    stop_words = set(stopwords.words('english'))
    return stop_words.union(extra_removal)
