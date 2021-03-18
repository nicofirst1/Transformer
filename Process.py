import os

import dill as pickle
import pandas as pd
from torchtext.legacy import data

from Batch import MyIterator, batch_size_fn
from Tokenize import Tokenize


def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()


def create_fields(opt):
    if not os.path.exists("translate_transformer_temp.csv"):

        spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it_core_news_sm', 'nl']
        if opt.src_lang not in spacy_langs:
            print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)
        if opt.trg_lang not in spacy_langs:
            print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)

        print("loading spacy tokenizers...")

        raw_data = {'src': [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
        df = pd.DataFrame(raw_data, columns=["src", "trg"])

        mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
        df = df.loc[mask]

        t_src = Tokenize(opt.src_lang)
        t_trg = Tokenize(opt.trg_lang)

        df['src'] = df['src'].apply(lambda x: t_src.tokenizer(x))
        df['trg'] = df['trg'].apply(lambda x: t_trg.tokenizer(x))

        df.to_csv("translate_transformer_temp.csv", index=False)


def create_dataset(opt):
    print("creating dataset and iterator... ")

    def token(x):
        return x.split(" ")

    TRG = data.Field(lower=True, tokenize=token, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=token)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)

    return train_iter, SRC, TRG


def get_len(train):
    for i, b in enumerate(train):
        pass

    return i
