import argparse

import torch
import torch.nn.functional as F
import torchtext

from arch.Models import get_model
from Optim import CosineWithRestarts
from data_gen.Process import *
from core import Trainer, ProgressBarLogger
from core.games import ClassicGame
from core.util import get_len, init


def loss_fn(preds, lables, trg_pad):
    loss = F.cross_entropy(preds, lables, ignore_index=trg_pad)
    return loss, {}


def main():
    parser=init()

    opt = parser.parse_args()
    print(opt)

    opt.device = "cpu" if opt.no_cuda else "cuda"
    if opt.device == "cuda":
        assert torch.cuda.is_available()

    opt.device=torch.device(opt.device)

    train_data, SRC, TRG = data_pipeline(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))


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
