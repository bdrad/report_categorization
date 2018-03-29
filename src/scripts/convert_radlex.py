import argparse
import csv
import pickle
import re
from unidecode import unidecode
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    args = parser.parse_args()

    entries = []
    with open(args.in_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for entry in reader:
            e = {}
            e["term"] = entry["Preferred Label"]
            e["syns"] = entry["Synonyms"].split("|")
            entries.append(e)

    replacements = []
    ignore_vals = "()äößü"
    for e in entries:
        name = e["term"]
        to_ret = " " + name.lower().replace(" ", "_") + " "
        if name.count(" ") < 5:
            if name.lower() not in stop_words:
                replacements.append((" " + name.lower() + " ", to_ret))
                for synonym in e["syns"]:
                    if synonym.replace(" ", "") != "" and synonym.count(" ") < 5:
                        contains_char = True in [iv in synonym for iv in ignore_vals]
                        if (not contains_char) and (synonym.lower() not in stop_words):
                            replacements.append((" " + synonym.lower() + " ", to_ret))


    to_replaces = [r[0].lstrip().rstrip() for r in replacements]
    dupes = set([r for r in to_replaces if to_replaces.count(r) > 1 and r != ""])
    replacements = [r for r in replacements if r[0] not in dupes]
    replacements = replacements + [(" " + d.lower() + " ", " " + d.lower().replace(" ", "_") + " ") for d in dupes]
    print(replacements[:200])
    print(replacements[-200:])
    print(str(len(dupes)) + " duplicates")
    print("Writing " + str(len(replacements)) + " RadLex replacements")
    with open(args.out_path, 'wb') as out_file:
        pickle.dump(replacements, out_file)
