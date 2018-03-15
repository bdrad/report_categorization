from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from random import shuffle
import fastText as ft
import numpy as np

FAST_TEXT_DIM = 350

def train_word2vec(corpus_path, out_path, cbow=False):
    sentences = []
    print("Loading files...")
    with open(corpus_path, 'r') as infile:
        for line in infile:
            sentences.append(line.split(" "))

    print("Training model...")
    if cbow:
        model = Word2Vec(sentences, min_count=2, size=400, window=8, hs=0, negative=15, iter=15, sg=0)
    else:
        model = Word2Vec(sentences, min_count=2, size=400, window=8, hs=0, negative=15, iter=15, sg=1)
    print("Trained!")
    print(model)

    model.save(out_path)

def load_word2vec_model(path):
    return Word2Vec.load(path)

class WordVectorizer(TransformerMixin):
    def __init__(self, model):
        self.model = model
    def transform(self, labeled_reports, *_):
        result = []
        for report in labeled_reports:
            new_sentences = []
            for sentence in report[0]:
                wordvecs = []
                for word in sentence.split(" "):
                    try:
                        wordvecs.append(self.model.wv[word])
                    except:
                        pass # print("OOV Word: " + word)
                if len(wordvecs) > 0:
                    new_sentences.append(wordvecs)
            result.append((new_sentences, report[1]))
        return result

def load_fastText_model(path):
    return ft.load_model(path)

class TfidfReportVectorizer(TransformerMixin):
    def transform(self, labeled_reports, *_):
        shuffle(labeled_reports)

        # Create TF-IDF Vectorizer
        documents = [" ".join(lr[0]) for lr in labeled_reports]
        vectorizer = TfidfVectorizer()
        transformed_docs = vectorizer.fit_transform(documents)
        return [(transformed_docs[i].toarray()[0], labeled_reports[i][1]) for i in range(len(labeled_reports))]

padding = np.zeros((FAST_TEXT_DIM,))
class FastTextReportVectorizer(TransformerMixin):
    def __init__(self, model, granularity="report", pad_len=-1):
        if granularity not in ["report", "sentence", "word"]:
            print("Unknown granularity!")
            raise ValueError
        self.granularity = granularity
        self.model = model
        self.pad_len=pad_len
    def transform(self, labeled_reports, *_):
        shuffle(labeled_reports)
        result = []
        for report in labeled_reports:
            if self.granularity == "report":
                report_text = " ".join(report[0])
                out_vector = self.model.get_sentence_vector(report_text)
                result.append((out_vector, report[1]))
            elif self.granularity == "sentence":
                sentence_vectors = [self.model.get_sentence_vector(s) for s in report[0]]
                if len(sentence_vectors) > 0:
                    result.append((sentence_vectors, report[1]))
            elif self.granularity == "word":
                vects = []
                for sentence in report[0]:
                    word_vecs = [self.model.get_word_vector(w) for w in sentence.split(" ")]
                    vects += word_vecs
                if self.pad_len > 0 and len(vects) > 0:
                    if self.pad_len >= len(vects):
                        paddings = (self.pad_len - len(vects)) * [padding]
                        result.append((vects + paddings, report[1]))
                elif len(vects) > 0:
                    result.append((vects, report[1]))
        return result

def train_fastText(corpus_path, out_path, cbow=False, dim=FAST_TEXT_DIM):
    if cbow:
        model = ft.train_unsupervised(input=corpus_path, model='cbow', dim=dim, epoch=24, wordNgrams=3)
    else:
        model = ft.train_unsupervised(input=corpus_path, model='skipgram', dim=dim, epoch=24, wordNgrams=3)
    model.save_model(out_path)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('-c','--corpus_path', nargs=1, required=True)
    parser.add_argument('-o','--out_path', nargs=1, required=True)
    parser.add_argument('-m','--model', nargs='?', default="word2vec", type=str)
    parser.add_argument('-b','--cbow', dest='cbow', action='store_true')
    parser.add_argument('-d','--dim', required=False)

    args = parser.parse_args()

    if args.model == "word2vec":
        train_word2vec(args.corpus_path[0], args.out_path[0], cbow=args.cbow)
    elif args.model == 'fastText':
        train_fastText(args.corpus_path[0], args.out_path[0], cbow=args.cbow, dim=int(args.dim))
    else:
        print("Unsupported model!")
