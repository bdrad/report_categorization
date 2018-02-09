# Contains code for reading from CSVs, normalizing text, and labeling text
import csv
from random import shuffle
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.tokenize import sent_tokenize

def get_reports_from_csv(corpus_path):
    with open(corpus_path) as csvfile:
        reader = csv.DictReader(csvfile)
        while True:
            n = next(reader, None)
            if n is None:
                return
            yield n['Report Text']

class ImpressionExtractor(TransformerMixin):
    def transform(self, reports, *_):
        result = []
        for report in reports:
            a = report.split("END OF IMPRESSION")[0]
            b = a.split("IMPRESSION:")[-1]
            b = b[1:] if len(b) > 0 and b[0] == "\n" else b
            b = b[:-1] if len(b) > 0 and b[-1] == "\n" else b
            result.append(b)
        return result

punct = "!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n"
class SentenceTokenizer(TransformerMixin):
    def transform(self, texts, *_):
        result = []
        for text in texts:
            text = text.replace("Dr.", "Dr")
            sentences = sent_tokenize(text)
            new_sentences = []
            for sentence in sentences:
                if len(sentence) <= 2:
                    continue
                for r in punct:
                    sentence = sentence.replace(r, " ")
                sentence = sentence.replace("  ", " ")
                sentence = sentence[:-1] if sentence[-1] == " " else sentence
                sentence = sentence.lower()
                new_sentences.append(sentence)
            result.append(new_sentences)
        return result

class ReportLabeler(TransformerMixin):
    def transform(self, reports, *_):
        result = []
        for report in reports:
            clean_report = []
            label = 0
            for sentence in report:
                if "discussed with" in sentence:
                    label = 1
                else:
                    clean_report.append(sentence)
            result.append((clean_report, label))
        return result

class ExtraneousSentenceRemover(TransformerMixin):
    def transform(self, impressions, *_):
        result = []
        for i in impressions:
            new_sentences = []
            for sentence in i:
                if not "dictated by" in sentence:
                    new_sentences.append(sentence)
            result.append(new_sentences)
        return result

import argparse
import pickle
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('-i','--in_path', nargs='+', required=True)
    parser.add_argument('-o','--out_path', nargs=1, required=True)

    args = parser.parse_args()

    data = [get_reports_from_csv(ip) for ip in args.in_path]
    merged_data = list(itertools.chain.from_iterable(data))
    pipeline = make_pipeline(ImpressionExtractor(), SentenceTokenizer(), ExtraneousSentenceRemover(), ReportLabeler(), None)
    preprocessed = pipeline.transform(merged_data)
    print("Writing " + str(len(preprocessed)) + " preprocessed reports")
    pickle.dump(preprocessed, open(args.out_path[0], "wb"))
