import re

import torch
from nltk.corpus import wordnet
from torch.autograd import Variable
from torchtext.data import get_tokenizer

from Beam import beam_search
from arch.Models import get_model
from core import move_to, init
from core.parsers import init_parser
from core.util import console
from data_gen.Process import load_vocab


def get_synonym(word, src):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if src[l.name()] != 0:
                return src[l.name()]

    return 0


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opts, src, trg):
    model.eval()
    indexed = []
    tokenizer = get_tokenizer('spacy', language=opts.src_lang)
    sentence = tokenizer(sentence)
    # sentence = [src[x] for x in sentence]
    for tok in sentence:
        if src[tok] != 0:
            indexed.append(src[tok])
        else:
            indexed.append(get_synonym(tok, src))
    sentence = Variable(torch.LongTensor([indexed]))

    sentence = move_to(sentence, opts.device)

    sentence = beam_search(sentence, model, src, trg, opts)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


def translate(opt, model, src, trg):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        trns = translate_sentence(sentence + '.', model, opt, src, trg).capitalize()
        translated.append(trns)

    return (' '.join(translated))


def main():
    parser = init_parser()
    opts = init(parser)

    console.log(sorted(vars(opts).items()))

    assert opts.k > 0
    assert opts.max_len > 10

    src, trg = load_vocab(opts, None)
    model = get_model(opts, len(src), len(trg), weight_path=opts.output_dir)

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
        phrase = translate(opts, model, src, trg)
        print('> ' + phrase + '\n')


if __name__ == '__main__':
    main()
