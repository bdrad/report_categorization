from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
import itertools
import pickle
import re


class LowerCaseSentence(TransformerMixin):
    def transform(self, labeled_reports, *_):
        for report_index, report in enumerate(labeled_reports):
            for sent_index, sentence in enumerate(report):
                labeled_reports[report_index][sent_index] = sentence.lower()

        return labeled_reports


class StopWordRemover(TransformerMixin):
    def transform(self, labeled_reports, *_):
        result = []
        swords = set(stopwords.words('english'))
        for report in labeled_reports:
            new_sentences = []
            for sentence in report:
                words = word_tokenize(sentence)
                print('words', words)
                filtered_sentence = [w for w in words if w not in swords]
                print('filtered', filtered_sentence)
                new_sentences.append(" ".join(filtered_sentence))

            result.append(new_sentences)

        return result


class SemanticMapper(TransformerMixin):
    def __init__(self, replacements, regex=False):
        self.replacements = replacements
        self.regex = regex

    def transform_regex(self, labeled_report):
        result = []
        for report in labeled_report:
            new_sentences = []
            for sentence in report:
                for r in self.replacements:
                    sentence = re.sub(r[0], r[1], sentence)
                new_sentences.append(sentence)
            result.append(new_sentences)
        return result

    def transform(self, labeled_report, *_):
        if self.regex:
            return self.transform_regex(labeled_report)

        result = []
        for report in labeled_report:
            new_sentences = []
            for sentence in report:
                sentence = " " + sentence + " "
                for r in self.replacements:
                    sentence = sentence.replace(r[0], r[1])
                sentence = sentence.replace("  ", " ")
                if len(sentence) > 1 and sentence[0] == " ":
                    sentence = sentence[1:]
                if len(sentence) > 1 and sentence[-1] == " ":
                    sentence = sentence[:-1]
                new_sentences.append(sentence)
            result.append(new_sentences)
        return result
