import torch.nn as nn

from data_gen.Batch import create_masks
from .interaction import LoggingStrategy


class ClassicGame(nn.Module):
    """
    A single-symbol Sender/Receiver game implemented with Reinforce.
    """

    def __init__(
            self,
            src_pad,
            trg_pad,
            model,
            device,
            loss,
    ):
        super(ClassicGame, self).__init__()

        self.src_pad = src_pad
        self.trg_pad = trg_pad
        self.model = model
        self.device = device
        self.loss_fn = loss

        self.train_logging_strategy = LoggingStrategy().minimal()

        self.test_logging_strategy = LoggingStrategy().minimal()

    def forward(self, input):
        src = input.src.transpose(0, 1)
        trg = input.trg.transpose(0, 1)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, self.device, self.src_pad, self.trg_pad)
        preds = self.model(src, trg_input, src_mask, trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        loss, aux_info = self.loss_fn(preds.view(-1, preds.size(-1)), ys, self.trg_pad)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            input_=src,
            labels=ys,
            aux=aux_info,
        )

        return loss, interaction
