import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from arch.Embed import  PositionalEncoder
from arch.Layers import EncoderLayer, DecoderLayer
from arch.Sublayers import Norm
from core import move_to
from core.util import console


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)

        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), n_layers)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), n_layers)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


class MultiEncTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, n_layers, heads, dropout, encoder_num):
        super().__init__()
        self.random_state = np.random.RandomState(42)

        self.encoders = [Encoder(src_vocab, d_model, n_layers, heads, dropout) for _ in range(encoder_num)]
        self.decoder = Decoder(trg_vocab, d_model, n_layers, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    @property
    def encoder(self):
        encoder = self.random_state.choice(self.encoders)
        return encoder.cuda()

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


class MultiDecTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, n_layers, heads, dropout, decoder_num):
        super().__init__()
        self.random_state = np.random.RandomState(42)

        self.encoder = Encoder(src_vocab, d_model, n_layers, heads, dropout)
        self.decoders = [Decoder(src_vocab, d_model, n_layers, heads, dropout) for _ in range(decoder_num)]
        self.out = nn.Linear(d_model, trg_vocab)

    @property
    def decoder(self):
        decoder = self.random_state.choice(self.decoders)
        return decoder.cuda()

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def load_weights(path, model) -> int:
    latest_file, latest_time = None, None

    if isinstance(path, str):
        path = Path(path)

    for file in path.glob("*.tar"):
        creation_time = os.stat(file).st_ctime
        if latest_time is None or creation_time > latest_time:
            latest_file, latest_time = file, creation_time

    if latest_file is not None:
        console.log(f"Loading trainer state from {latest_file}")
        checkpoint = torch.load(latest_file)
        model_state = checkpoint.model_state_dict
        model_state = {k.replace("model.", ""): v for k, v in model_state.items()}

        model.load_state_dict(model_state)
        epoch = checkpoint.epoch
        console.log(f"Model trained with {epoch} epochs loaded!")

    else:
        console.log(f"Could not load model from {path}")


def get_model(opts, src_vocab, trg_vocab, weight_path=None):
    assert opts.model_dim % opts.heads == 0
    assert opts.dropout < 1

    if opts.model == "transformer":

        model = Transformer(src_vocab, trg_vocab, opts.model_dim, opts.n_layers, opts.heads, opts.dropout)
    elif opts.model == "multiencoder":
        model = MultiEncTransformer(src_vocab, trg_vocab, opts.model_dim, opts.n_layers, opts.heads, opts.dropout,
                                    opts.encod_num)
    elif opts.model == "multidencoder":
        model = MultiEncTransformer(src_vocab, trg_vocab, opts.model_dim, opts.n_layers, opts.heads, opts.dropout,
                                    opts.decod_num)
    if weight_path is not None:
        try:
            load_weights(weight_path, model)
        except:
            console.log("Could not load weights")

    model = move_to(model, opts.device)
    return model
