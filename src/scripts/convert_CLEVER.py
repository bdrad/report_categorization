import argparse
import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    args = parser.parse_args()

    replacements = []
    with open(args.in_path, 'r') as in_file:
        for line in in_file:
            elements = [e.replace("\n", "") for e in line.split("|")]
            if len(elements) == 3:
                replacements.append((" " + elements[1] + " ", " " + elements[2]+ " "))
    pickle.dump(replacements, open(args.out_path, "wb"))
