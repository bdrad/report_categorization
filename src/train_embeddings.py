from gensim.models import Word2Vec
from sklearn.base import TransformerMixin
from random import shuffle
import fastText as ft

def train_word2vec(corpus_path, out_path):
    sentences = []
    print("Loading files...")
    with open(corpus_path, 'r') as infile:
        for line in infile:
            sentences.append(line.split(" "))

    print("Training model...")
    model = Word2Vec(sentences, min_count=2, size=150, hs=0, negative=15, iter=15)
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

class FastTextReportVectorizer(TransformerMixin):
    def __init__(self, model, granularity="report"):
        if granularity not in ["report", "sentence"]:
            print("Unknown granularity!")
            raise ValueError
        self.granularity = granularity
        self.model = model
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
        return result

def train_fastText(corpus_path, out_path):
    model = ft.train_unsupervised(input=corpus_path, model='skipgram', epoch=12)
    model.save_model(out_path)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('-c','--corpus_path', nargs=1, required=True)
    parser.add_argument('-o','--out_path', nargs=1, required=True)
    parser.add_argument('-m','--model', nargs='?', default="word2vec", type=str)

    args = parser.parse_args()

    if args.model == "word2vec":
        train_word2vec(args.corpus_path[0], args.out_path[0])
    elif args.model == 'fastText':
        train_fastText(args.corpus_path[0], args.out_path[0])
    else:
        print("Unsupported model!")
