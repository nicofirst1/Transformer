import os

import dill as pickle
import numpy as np
import pandas as pd
from torchtext.legacy import data

from core.util import console
from data_gen.Batch import MyIterator, BatchSize
from data_gen.Tokenize import Tokenize

REDUCE_PERC = 0.01
file_perc = str(REDUCE_PERC).replace(".", "")
DF_PATH_REDUCED = f"data/translate_transformer_reduced{file_perc}.csv"


def read_data(src_data_path, trg_data_path):
    try:
        with open(src_data_path, "r") as f:
            src_data = f.read().strip().split('\n')
    except:
        print("error: '" + src_data_path + "' file not found")
        quit()

    try:
        with open(trg_data_path, "r") as f:
            trg_data = f.read().strip().split('\n')
    except:
        print("error: '" + trg_data_path + "' file not found")
        quit()

    return src_data, trg_data


def create_fields(opts, src_data, trg_data):
    if not os.path.exists(opts.df_path):

        spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it_core_news_sm', 'nl']
        if opts.src_lang not in spacy_langs:
            print('invalid src language: ' + opts.src_lang + 'supported languages : ' + spacy_langs)
        if opts.trg_lang not in spacy_langs:
            print('invalid trg language: ' + opts.trg_lang + 'supported languages : ' + spacy_langs)

        print("loading spacy tokenizers...")

        raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}
        df = pd.DataFrame(raw_data, columns=["src", "trg"])

        mask = (df['src'].str.count(' ') < opts.max_strlen) & (df['trg'].str.count(' ') < opts.max_strlen)
        df = df.loc[mask]

        t_src = Tokenize(opts.src_lang)
        t_trg = Tokenize(opts.trg_lang)

        df['src'] = df['src'].apply(lambda x: t_src.tokenizer(x))
        df['trg'] = df['trg'].apply(lambda x: t_trg.tokenizer(x))

        df.to_csv(opts.df_path, index=False)
    elif not os.path.exists(opts.df_path_reduced):

        print(f"Cannot find {opts.df_path_reduced}, creating it...")

        df = pd.read_csv(opts.df_path)
        remove_n = len(df) * (1 - opts.reduce_perc)
        remove_n = int(remove_n)
        drop_indices = np.random.choice(df.index, remove_n, replace=False)
        df_subset = df.drop(drop_indices)
        df_subset.to_csv(opts.df_path_reduced, index=False)
    else:
        console.log(f"Dataset at {opts.df_path_reduced} already created!")


def load_fields(output_dir):
    with open(f"{output_dir}/src.pkl", "rb") as file:
        src = pickle.load(file)
    with open(f"{output_dir}/trg.pkl", "rb") as file:
        trg = pickle.load(file)

    return src, trg


def create_dataset(opts):
    console.log("Loading iterator...")

    def token(x):
        return x.split(" ")

    trg = data.Field(lower=True, tokenize=token, init_token='<sos>', eos_token='<eos>')
    src = data.Field(lower=True, tokenize=token)

    data_fields = [('src', src), ('trg', trg)]
    train = data.TabularDataset(opts.df_path_reduced, format='csv', fields=data_fields)

    batch_size= BatchSize()

    train_iter = MyIterator(train, batch_size=opts.batchsize, device=opts.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size.batch_size_fn, train=True, shuffle=True)

    if opts.load_weights is None:
        src.build_vocab(train)
        trg.build_vocab(train)

        pickle.dump(src, open(f'{opts.output_dir}/src.pkl', 'wb'))
        pickle.dump(trg, open(f'{opts.output_dir}/trg.pkl', 'wb'))

    opts.src_pad = src.vocab.stoi['<pad>']
    opts.trg_pad = trg.vocab.stoi['<pad>']

    return train_iter, src, trg


def data_pipeline(opt):
    src_data, trg_data = read_data(opt.src_data, opt.trg_data)
    create_fields(opt, src_data, trg_data)

    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    train_data, src, trg = create_dataset(opt)

    return train_data, src, trg
