from sklearn.base import TransformerMixin
from random import shuffle
import numpy as np

class LabelSeparator(TransformerMixin):
    def transform(self, labeled_reports, *_):
        return ([l[0] for l in labeled_reports], [l[1] for l in labeled_reports])

class ReportVectorAverager(TransformerMixin):
    def __init__(self, normalize=True, shuffle=True, desired_dim=200):
        self.normalize = normalize
        self.shuffle = shuffle
        self.desired_dim = desired_dim

    def transform(self, labeled_reports, *_):
        errs = 0
        shuffle(labeled_reports)
        report_vectors = []
        for report in labeled_reports:
            sentence_vecs = []
            for sentence in report[0]:
                sentence_vecs.append(np.mean(sentence, axis=0))
            try:
                sentence_vecs = np.array(sentence_vecs)
                report_vec = np.mean(sentence_vecs, axis=0)
                if self.normalize:
                    report_vec = report_vec / np.linalg.norm(report_vec)
                if report_vec.shape[0] == self.desired_dim:
                    report_vectors.append((report_vec, report[1]))
            except:
                pass #print(sentence_vecs)
        return report_vectors
