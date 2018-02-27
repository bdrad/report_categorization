from sklearn.base import TransformerMixin
from random import shuffle
from functools import reduce
import numpy as np

class LabelSeparator(TransformerMixin):
    def __init__(self, shuffle=True):
        self.shuffle = shuffle
    def transform(self, labeled_reports, *_):
        if self.shuffle:
            shuffle(labeled_reports)
        return ([l[0] for l in labeled_reports], [l[1] for l in labeled_reports])

# Assumes that each report contains a series of vectors, with each vector representing a single sentence
class SentenceBasedClassifier(TransformerMixin):
    def __init__(self, vector_classifier, combiner=max, use_prob_func=False):
        self.model = vector_classifier
        self.combiner = combiner
        self.use_prob_func = use_prob_func
    def fit(self, X, y):
        # Label each sentence
        sent_vects = []
        sent_labels = []
        for report, label in zip(X, y):
            for sent in report:
                sent_vects.append(sent)
                sent_labels.append(label)
        self.model.fit(sent_vects, sent_labels)

    def decision_function(self, reports):
        predictions = []
        for report in reports:
            if not self.use_prob_func:
                confs = self.model.decision_function(report)
            else:
                class_probs = self.model.predict_proba(report)
                confs = [c[1] for c in class_probs]
            predictions.append(self.combiner(confs))
        return predictions

    # TODO: def predict(reports):

class ReportVectorAverager(TransformerMixin):
    def __init__(self, granularity="report", normalize=True):
        if granularity not in ["report", "sentence"]:
            print("Unknown granularity!")
            raise ValueError
        self.normalize = normalize
        self.granularity = granularity

    def transform(self, labeled_reports, *_):
        errs = 0
        report_vectors = []
        for report in labeled_reports:
            sentence_vecs = []
            for sentence in report[0]:
                sentence_vecs.append(np.mean(sentence, axis=0))
            if self.granularity == "report":
                try:
                    sentence_vecs = np.array(sentence_vecs)
                    report_vec = np.mean(sentence_vecs, axis=0)
                    if self.normalize:
                        report_vec = report_vec / np.linalg.norm(report_vec)
                    if report_vec.shape[0] > 0:
                        report_vectors.append((report_vec, report[1]))
                except:
                    pass #print(sentence_vecs)
            elif self.granularity == "sentence":
                if len(sentence_vecs) > 0:
                    if self.normalize:
                        sentence_vecs = [s / np.linalg.norm(s) for s in sentence_vecs]
                    report_vectors.append((sentence_vecs, report[1]))
        return report_vectors
