from gensim.models import Word2Vec

def train_word2vec(corpus_path, out_path):
    sentences = []
    print("Loading files...")
    with open(corpus_path, 'r') as infile:
        for line in infile:
            sentences.append(line.split(" "))

    print("Training model...")
    model = Word2Vec(sentences, min_count=2, size=200, hs=0, negative=15, iter=15)
    print("Trained!")
    print(model)

    model.save(out_path)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('-c','--corpus_path', nargs=1, required=True)
    parser.add_argument('-o','--out_path', nargs=1, required=True)
    parser.add_argument('-m','--model', nargs='?', default="word2vec", type=str)

    args = parser.parse_args()

    if args.model == "word2vec":
        train_word2vec(args.corpus_path[0], args.out_path[0])
    else:
        print("Unsupported model!")
