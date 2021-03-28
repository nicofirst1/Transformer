import re

import spacy


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
