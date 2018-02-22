import argparse
import csv
import pickle
from unidecode import unidecode

def is_descendent_of(rid, ancestor_rid, parent_map):
    while rid in parent_map:
        if rid == ancestor_rid:
            return True
        rid = parent_map[rid]
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    args = parser.parse_args()

    entity_names = {}
    parent_map = {}
    with open(args.in_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for entry in reader:
            entity_names[entry[1]] = entry[0]
            parent_map[entry[1]] = entry[2]
    parents = set(parent_map.values())

    non_parents = set(entity_names.keys()).difference(parents)
    second_level = [parent_map[rid] for rid in non_parents if rid in parent_map]
    third_level = [parent_map[rid] for rid in second_level if rid in parent_map]

    to_replace = non_parents.union(second_level).union(third_level)

    replacements = []
    for rid in to_replace:
        try:
            child_name = entity_names[rid]
            if len(child_name.split(" ")) < 4 and is_descendent_of(rid, RID13176, parent_map):
                parent_rid = parent_map[rid]
                try:
                    parent_name = entity_names[parent_rid]
                    replacements.append((" " + child_name.lower() + " ", " " + parent_name.lower().replace(" ", "_") + " "))
                except:
                    print("Couldn't find parent " + parent_rid)
        except:
            print("Couldn't find " + rid)

    print(replacements)
    print("Writing " + str(len(replacements)) + " RadLex replacements")
    with open(args.out_path, 'wb') as out_file:
        pickle.dump(replacements, out_file)
