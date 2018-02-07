# Contains code for reading from CSVs, normalizing text, and semantically mapping text
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

class SentenceTokenizer(TransformerMixin):
    def transform(self, texts, *_):
        result = []
        for text in texts:
            sentences = sent_tokenize(text)
            sentences = [s.lower().replace("\n", " ").replace("?", " ") for s in sentences if len(s) > 2]
            result.append(sentences)
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
path = "../Rnet_files/positive_examples.csv"
a = list(get_reports_from_csv(path))
shuffle(a)

pipeline = make_pipeline(ImpressionExtractor(), SentenceTokenizer(), ReportLabeler(), None)
data = pipeline.transform(a)


print(data[0])
print("///////////////")
print(data[1])
print("///////////////")
print(data[2])
print("///////////////")
print(data[3])
print("///////////////")
