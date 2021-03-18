import json

import spacy
import re

from torchtext.legacy import data


class Tokenize(object):
    
    def __init__(self, lang):
        self.nlp = spacy.load(lang, disable=['ner', 'parser', 'tagger'])
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return " ".join([tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "])

def tokenize(lang, df):

    tk=Tokenize(lang)
    train_examples = [tk.tokenizer(t) for t in df]

    with open('.data/train.json', 'w+') as f:
        for example in train_examples:
            json.dump(example, f)
            f.write('\n')

    with open('.data/test.json', 'w+') as f:
        for example in test_examples:
            json.dump(example, f)
            f.write('\n')