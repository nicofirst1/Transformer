import re

import torch
from nltk.corpus import wordnet
from torch.autograd import Variable

from Beam import beam_search
from arch.Models import get_model
from core import move_to, init
from data_gen.Process import load_fields
from data_gen.Tokenize import Tokenize


def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]

    return 0


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt, SRC, TRG):
    model.eval()
    indexed = []
    sentence = Tokenize(opt.src_lang).tokenizer(sentence)
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 :
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))

    sentence = move_to(sentence, opt.device)

    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        trns = translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize()
        translated.append(trns)

    return (' '.join(translated))


def main():
    opts = init()


    assert opts.k > 0
    assert opts.max_len > 10

    SRC, TRG = load_fields(opts.output_dir)
    model = get_model(opts, len(SRC.vocab), len(TRG.vocab))

    while True:
        opts.text = input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opts.text == "q":
            break
        if opts.text == 'f':
            fpath = input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
            try:
                opts.text = ' '.join(open(fpath, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        phrase = translate(opts, model, SRC, TRG)
        print('> ' + phrase + '\n')


if __name__ == '__main__':
    main()
