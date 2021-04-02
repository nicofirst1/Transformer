import torch
import torch.nn.functional as F

from arch.Models import get_model
from core import Trainer, ProgressBarLogger
from core.callbacks import CheckpointSaver, CustomMetrics
from core.games import ClassicGame
from core.util import init
from data_gen.Process import *


def loss_fn(preds, lables, trg_pad):
    loss = F.cross_entropy(preds, lables, ignore_index=trg_pad)
    return loss, {}


def main():
    opts = init()

    console.log(sorted(vars(opts).items()))

    train_data, src, trg = data_pipeline(opts)
    model = get_model(opts, len(src.vocab), len(trg.vocab))

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)

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
                          CustomMetrics(),
                          ProgressBarLogger(n_epochs=opts.epochs, train_data_len=len(train_data)),
                          CheckpointSaver(checkpoint_path=opts.output_dir, checkpoint_freq=opts.checkpoint_freq,
                                          prefix="model_weights", max_checkpoints=3),
                      ],
                      opts=opts)

    if opts.load_weights:
        trainer.load_from_latest(opts.output_dir)


    trainer.train(opts.epochs)



if __name__ == "__main__":
    main()
