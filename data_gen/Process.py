from collections import Counter

import dill as pickle
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab

from core.util import console


def load_vocab(opts, train_data):
    try:
        src_vocab = pickle.load(open(f'{opts.output_dir}/src.pkl', 'rb'))
        trg_vocab = pickle.load(open(f'{opts.output_dir}/trg.pkl', 'rb'))
        console.log("Vocabulary loaded from file")

    except FileNotFoundError:
        src_tok = get_tokenizer('spacy', language=opts.src_lang)
        trg_tok = get_tokenizer('spacy', language=opts.trg_lang)
        src_counter = Counter()
        trg_counter = Counter()

        for (label, line) in train_data:
            src_counter.update(src_tok(line))
            trg_counter.update(trg_tok(line))

        src_vocab = Vocab(src_counter, min_freq=opts.min_word_freq, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        trg_vocab = Vocab(trg_counter, min_freq=opts.min_word_freq, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

        pickle.dump(src_vocab, open(f'{opts.output_dir}/src.pkl', 'wb'))
        pickle.dump(trg_vocab, open(f'{opts.output_dir}/trg.pkl', 'wb'))
        console.log("Vocabulary created")

    console.log(f"Src vocab len = {len(src_vocab)}\nTrg vocab len = {len(trg_vocab)}")

    return src_vocab, trg_vocab


def preprocess_dataset(opts, dataset, src_vocab, trg_vocab):
    data = list(dataset._iterator)
    BOS_IDX = src_vocab['<bos>']
    EOS_IDX = src_vocab['<eos>']

    src_tok = get_tokenizer('spacy', language=opts.src_lang)
    trg_tok = get_tokenizer('spacy', language=opts.trg_lang)
    for idx in range(len(data)):
        src = data[idx][0]
        trg = data[idx][1]

        src = [src_vocab[x] for x in src_tok(src)]
        trg = [trg_vocab[x] for x in trg_tok(trg)]

        src.insert(0, BOS_IDX)
        trg.insert(0, BOS_IDX)

        src.append(EOS_IDX)
        trg.append(EOS_IDX)

        src = torch.as_tensor(src)
        trg = torch.as_tensor(trg)
        data[idx] = (src, trg)

    dataset._iterator = iter(data)


def batch_generator(opts, src_vocab, trg_vocab):
    src_tok = get_tokenizer('spacy', language=opts.src_lang)
    trg_tok = get_tokenizer('spacy', language=opts.trg_lang)
    BOS_IDX = src_vocab['<bos>']
    EOS_IDX = src_vocab['<eos>']
    PAD_IDX = src_vocab['<pad>']

    def inner(data_batch):
        src_batch = [x[0] for x in data_batch]
        trg_batch = [x[1] for x in data_batch]

        for idx in range(len(src_batch)):
            src = src_batch[idx]
            trg = trg_batch[idx]

            src = [src_vocab[x] for x in src_tok(src)]
            trg = [trg_vocab[x] for x in trg_tok(trg)]

            src.insert(0, BOS_IDX)
            trg.insert(0, BOS_IDX)

            src.append(EOS_IDX)
            trg.append(EOS_IDX)

            src = torch.as_tensor(src)
            trg = torch.as_tensor(trg)

            src_batch[idx] = src
            trg_batch[idx] = trg


        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
        return src_batch, trg_batch

    return inner


def create_dataset(opts):
    with console.status("[bold green]Dataset loading...") as status:
        src_pair = opts.src_lang.split("_")[0]
        trg_pair = opts.trg_lang.split("_")[0]
        train_data = torchtext.datasets.IWSLT2017(root='.data', split='train', language_pair=(src_pair, trg_pair))
        console.log("Dataset loaded.")
        status.status = "[bold green]Creating vocabulary..."
        status.update()
        src_vocab, trg_vocab = load_vocab(opts, train_data)
        status.status = "[bold green]Initializing dataloader.."
        status.update()
        # preprocess_dataset(opts, train_data, src_vocab, trg_vocab)

        train_iter = DataLoader(train_data, batch_size=opts.batch_size, collate_fn=batch_generator(opts, src_vocab,
                                                                                                   trg_vocab))
        console.log("DataLoader initialized")

    return train_iter, src_vocab, trg_vocab
