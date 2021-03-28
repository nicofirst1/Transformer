import argparse

import torch
import torch.nn.functional as F
import torchtext

from arch.Models import get_model
from Optim import CosineWithRestarts
from core.callbacks import BleuScoreLogger
from data_gen.Process import *
from core import Trainer, ProgressBarLogger
from core.games import ClassicGame
from core.util import get_len, init


def loss_fn(preds, lables, trg_pad):
    loss = F.cross_entropy(preds, lables, ignore_index=trg_pad)
    return loss, {}


def main():
    opts=init()

    print(opts)

    opts.device = "cpu" if opts.no_cuda else "cuda"
    if opts.device == "cuda":
        assert torch.cuda.is_available()

    opts.device=torch.device(opts.device)

    train_data, SRC, TRG = data_pipeline(opts)
    model = get_model(opts, len(SRC.vocab), len(TRG.vocab))


    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)
    if opts.SGDR == True:
        opts.sched = CosineWithRestarts(opts.optimizer, T_max=get_len(train_data))
    if opts.checkpoint > 0:
        print(
            "model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opts.checkpoint))

    game = ClassicGame(opts.src_pad,
                       opts.trg_pad,
                       model,
                       opts.device,
                       loss_fn, )

    trainer = Trainer(game=game,
                      optimizer=optimizer,
                      train_data=train_data,
                      validation_data=None,
                      device=opts.device,
                      callbacks=[
                          ProgressBarLogger(n_epochs=opts.epochs, train_data_len=get_len(train_data)),
                          BleuScoreLogger(),
                      ],
                      opts=opts)

    trainer.train(opts.epochs)


if __name__ == "__main__":
    main()
