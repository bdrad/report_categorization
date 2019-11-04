import fasttext as ft
import nltk
# download gutenberg
import ssl

"""
Thoughts:
- fasttext is really fast so it can take in large amount of data to train. 
- we do not specify the vocab file. it will genereate for us
- potentially need to create another tranformer to change people's names in 
  reports to special person tags? could be helpful in extracting information

Obstacles:
- at what point do we want to use fasttext/to start train embedding?
- we might need to do a custom pre-processing for corpra so that we can convert words into tokens we care about
- chicken and egg problem? 
    - we might want to use advanced matching to break up words with pre/suffix
    - the advanced matching require a good dictionary
    - dictionary is generated by going running fasttext
    - potential solution ------- we pre-process the input and run fasttext
      and generate a good vocab file. using the file, we can develope a matching
      algorithm in the transformers phase and run the actual training
- what kind of corpra can we use to train and utilize radlax? 
    - radiology textbook in txt format? 
    - biodata? 
    - radilogy reports and papers?

"""


def download_text():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    print(nltk.download('gutenberg'))
    from nltk.corpus import gutenberg

    emma_corpra = gutenberg.words('austen-emma.txt')

    with open('test_corpora.txt', 'w+') as f:
        f.write(' '.join(emma_corpra))


def train_model():
    model = ft.train_unsupervised(
        'test_corpora.txt', model='skipgram', dim=512)
    model.save_model("test_model.bin")
    return model