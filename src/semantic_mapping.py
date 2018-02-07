from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
import pickle
import re

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

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

replacement_path = "data/processed/clever_replacements"
replacements = read_replacements(replacement_path)
print(replacements[0])

data = pickle.load(open('src/preproc.txt', 'rb'))
print(data[0])

mapper = SemanticMapper(replacements)
new_data = DateTimeMapper.transform(mapper.transform(data))

print(new_data[4])
print(new_data[5])
print(new_data[6])
