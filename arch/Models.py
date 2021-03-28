import copy

import numpy as np
import torch
import torch.nn as nn

from arch.Embed import Embedder, PositionalEncoder
from arch.Layers import EncoderLayer, DecoderLayer
from arch.Sublayers import Norm
from core import move_to


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
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
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, encoder_num):
        super().__init__()
        self.random_state = np.random.RandomState(42)

        self.encoders = [Encoder(src_vocab, d_model, N, heads, dropout) for _ in range(encoder_num)]
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        encoder = self.random_state.choice(self.encoders)
        encoder = encoder.cuda()

        e_outputs = encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


class MultiDecTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, decoder_num):
        super().__init__()
        self.random_state = np.random.RandomState(42)

        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoders = [Decoder(src_vocab, d_model, N, heads, dropout) for _ in range(decoder_num)]
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        decoder = self.random_state.choice(self.decoders)
        decoder = decoder.cuda()
        d_output = decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


def get_model(opt, src_vocab, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.output_dir}/model_weights'))

    model = move_to(model, opt.device)
    return model
