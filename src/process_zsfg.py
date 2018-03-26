import argparse
import pickle
import itertools
import csv
from sklearn.pipeline import Pipeline, make_pipeline
from preprocessing import SectionExtractor, SentenceTokenizer, ReportLabeler, ExtraneousSentenceRemover, get_reports_from_zsfg_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('-i','--in_path', nargs='+', required=True)
    parser.add_argument('-o','--out_path', nargs=1, required=True)

    args = parser.parse_args()

    data = [get_reports_from_zsfg_csv(ip) for ip in args.in_path]
    merged_data = list(set(list(itertools.chain.from_iterable(data))))
    pipeline = make_pipeline(SectionExtractor(sections=["impression", "clinical_history"]), SentenceTokenizer(), ExtraneousSentenceRemover(), ReportLabeler(), None)
    preprocessed = pipeline.transform(merged_data)
    print("Writing " + str(len(preprocessed)) + " preprocessed reports")
    pickle.dump(preprocessed, open(args.out_path[0], "wb"))
