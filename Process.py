import os

import dill as pickle
import numpy as np
import pandas as pd
from torchtext.legacy import data

from Batch import MyIterator, batch_size_fn
from Tokenize import Tokenize

DF_PATH = "data/translate_transformer.csv"
REDUCE_PERC = 0.001
file_perc = str(REDUCE_PERC).replace(".", "")
DF_PATH_REDUCED = f"data/translate_transformer_reduced{file_perc}.csv"


def read_data(opt):
    try:
        src_data = open(opt.src_data).read().strip().split('\n')
    except:
        print("error: '" + opt.src_data + "' file not found")
        quit()

    try:
        trg_data = open(opt.trg_data).read().strip().split('\n')
    except:
        print("error: '" + opt.trg_data + "' file not found")
        quit()

    return src_data, trg_data


def create_fields(opt, src_data, trg_data):
    if not os.path.exists(DF_PATH):

        spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it_core_news_sm', 'nl']
        if opt.src_lang not in spacy_langs:
            print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)
        if opt.trg_lang not in spacy_langs:
            print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)

        print("loading spacy tokenizers...")

        raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}
        df = pd.DataFrame(raw_data, columns=["src", "trg"])

        mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
        df = df.loc[mask]

        t_src = Tokenize(opt.src_lang)
        t_trg = Tokenize(opt.trg_lang)

        df['src'] = df['src'].apply(lambda x: t_src.tokenizer(x))
        df['trg'] = df['trg'].apply(lambda x: t_trg.tokenizer(x))

        df.to_csv(DF_PATH, index=False)
    elif not os.path.exists(DF_PATH_REDUCED):

        print(f"Cannot find {DF_PATH_REDUCED}, creating it...")

        df = pd.read_csv(DF_PATH)
        remove_n = len(df) * (1 - REDUCE_PERC)
        remove_n = int(remove_n)
        drop_indices = np.random.choice(df.index, remove_n, replace=False)
        df_subset = df.drop(drop_indices)
        df_subset.to_csv(DF_PATH_REDUCED, index=False)


def load_fields(opt):
    with open(f"{opt.output_dir}/SRC.pkl", "rb") as file:
        SRC = pickle.load(file)
    with open(f"{opt.output_dir}/TRG.pkl", "rb") as file:
        TRG = pickle.load(file)

    return SRC, TRG


def create_dataset(opt):
    print("creating dataset and iterator... ")

    def token(x):
        return x.split(" ")

    TRG = data.Field(lower=True, tokenize=token, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=token)

    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset(DF_PATH_REDUCED, format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)

        pickle.dump(SRC, open(f'{opt.output_dir}/SRC.pkl', 'wb'))
        pickle.dump(TRG, open(f'{opt.output_dir}/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    return train_iter, SRC, TRG
