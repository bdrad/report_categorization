from preprocessing import get_reports_from_csv
from random import shuffle
import argparse
import pickle
import csv
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('-i','--in_path', nargs='+', required=True)
    parser.add_argument('-o','--out_path', nargs=1, required=True)

    args = parser.parse_args()

    data = [get_reports_from_csv(ip) for ip in args.in_path]
    merged_data = list(set(list(itertools.chain.from_iterable(data))))
    shuffle(merged_data)

    split_point = int(0.5 * len(merged_data))

    train = merged_data[:split_point]
    print("Training length: " + str(len(train)))
    test = merged_data[split_point:]
    print("Testing length: " + str(len(test)))

    with open(args.out_path[0] + "TRAIN", "w") as csvfile:
        field_names = ["Report Text"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for t in train:
            writer.writerow({"Report Text" : t})

    with open(args.out_path[0] + "TEST", "w") as csvfile:
        field_names = ["Report Text"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for t in test:
            writer.writerow({"Report Text" : t})
