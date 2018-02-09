from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
import pickle
import re

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

def reports_to_corpus(reports, out_file):
    for report in report:
        for sentence in report[0]:
            out_file.write(sentence + "\n")

class SemanticMapper(TransformerMixin):
    def __init__(self, replacements, regex=False):
        self.replacements = replacements
        self.regex = regex

    def transform(self, labeled_report, *_):
        result = []
        for report in labeled_report:
            new_sentences = []
            for sentence in report[0]:
                for r in self.replacements:
                    if self.regex:
                        sentence = re.sub(r[0], r[1], sentence)
                    else:
                        sentence = sentence.replace(" " + r[0] + " ", " " + r[1] + " ")
                new_sentences.append(sentence)
            result.append((new_sentences, report[1]))
        return result

DateTimeMapper = SemanticMapper([(r'[0-9][0-9]? [0-9][0-9]? [0-9][0-9][0-9][0-9]', 'DATE'),
                                 (r'[0-9][0-9]? [0-9][0-9] (am|pm)?', 'TIME')], regex=True)

import argparse
import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('labeled_reports_in_path')
    parser.add_argument('replacement_file_path')
    parser.add_argument('corpus_out_path')
    parser.add_argument('labels_out_path')
    args = parser.parse_args()

    labeled_reports = pickle.load(open(args.labeled_reports_in_path, "rb"))
    replacements = read_replacements(args.replacement_file_path)
    ReplacementMapper = SemanticMapper(replacements)
    pipeline = make_pipeline(ReplacementMapper, DateTimeMapper, None)
    labeled_output = pipeline.transform(labeled_reports)

    reports_to_corpus(labeled_output, open(args.corpus_out_path, "w"))
    pickle.dump(labeled_output, open(args.labels_out_path, "wb"))
