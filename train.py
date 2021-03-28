import argparse

import torch
import torch.nn.functional as F

from arch.Models import get_model
from Optim import CosineWithRestarts
from data_gen.Process import *
from core import Trainer, ProgressBarLogger
from core.games import ClassicGame
from core.util import get_len


def loss_fn(preds, lables, trg_pad):
    loss = F.cross_entropy(preds.view(-1, preds.size(-1)), lables, ignore_index=trg_pad)

    return loss, {}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-validation_freq', type=int, default=1)
    parser.add_argument('-encoder_num', type=int, default=4)
    parser.add_argument('-dencoder_num', type=int, default=4)
    parser.add_argument('-src_data', default='data/europarl-v7.it-en.en')
    parser.add_argument('-trg_data', default='data/europarl-v7.it-en.it')
    parser.add_argument('-src_lang', default='en_core_web_sm')
    parser.add_argument('-trg_lang', default='it_core_news_sm')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=256)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-output_dir', default='output')

    opt = parser.parse_args()
    print(opt)

    opt.device = "cpu" if opt.no_cuda else "cuda"
    if opt.device == "cuda":
        assert torch.cuda.is_available()

    train_data, SRC, TRG = data_pipeline(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    if opt.device == "cuda":
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=get_len(train_data))
    if opt.checkpoint > 0:
        print(
            "model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    game = ClassicGame(opt.src_pad,
                       opt.trg_pad,
                       model,
                       opt.device,
                       loss_fn, )

    trainer = Trainer(game=game,
                      optimizer=optimizer,
                      train_data=train_data,
                      validation_data=None,
                      device=opt.device,
                      callbacks=[
                          ProgressBarLogger(n_epochs=opt.epochs, train_data_len=get_len(train_data))
                      ],
                      opts=opt)

    trainer.train(opt.epochs)


if __name__ == "__main__":
    main()
