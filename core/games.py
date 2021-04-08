import inspect

import torch
import torch.nn as nn

from data_gen.Batch import create_masks
from memory_profiler import MemTracker
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

        self.train_logging_strategy = LoggingStrategy()

        self.test_logging_strategy = LoggingStrategy()

        frame = inspect.currentframe()  # define a frame to track
        self.gpu_tracker = MemTracker(frame)  # define a GPU tracker

    def forward(self, input):
        # self.gpu_tracker.track(1)
        src = input[0].transpose(0, 1)
        trg = input[1].transpose(0, 1)
        trg_input = trg[:, :-1]

        ys = trg_input.contiguous().view(-1)

        src_mask, trg_mask = create_masks(src, trg_input, self.device, self.src_pad, self.trg_pad)

        preds = self.model(src, trg_input, src_mask, trg_mask)
        total_loss = 0

        loss, aux_info = self.loss_fn(preds.view(-1, preds.size(-1)), ys, self.trg_pad)
        total_loss += loss

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        preds = torch.argmax(torch.softmax(preds, dim=-1), dim=-1)

        interaction = logging_strategy.filtered_interaction(
            source=src,
            labels=trg,
            preds=preds,
            aux=aux_info,
        )

        return total_loss, interaction


class ModelingGame(nn.Module):
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
        super(ModelingGame, self).__init__()

        self.src_pad = src_pad
        self.trg_pad = trg_pad
        self.model = model
        self.device = device
        self.loss_fn = loss

        self.guessing_step = 0.0001
        self.guessing_weight = 0
        self.train_logging_strategy = LoggingStrategy()

        self.test_logging_strategy = LoggingStrategy()

        frame = inspect.currentframe()  # define a frame to track
        self.gpu_tracker = MemTracker(frame)  # define a GPU tracker

    def forward(self, input):
        # self.gpu_tracker.track(1)
        src = input[0].transpose(0, 1)
        trg = input[1].transpose(0, 1)
        trg_input = trg[:, :-1]

        ys = trg_input.contiguous().view(-1)

        src_mask, trg_mask = create_masks(src, trg_input, self.device, self.src_pad, self.trg_pad)

        preds, sender_guess = self.model(src, trg_input, src_mask, trg_mask)

        loss = nn.functional.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=self.trg_pad)

        actual_preds = torch.argmax(torch.softmax(preds, dim=-1), dim=-1).float()
        guessing_loss = nn.functional.mse_loss(actual_preds, sender_guess)
        scaled_guessing_loss = guessing_loss * (loss / guessing_loss) * self.guessing_weight

        aux_info = {}

        total_loss = loss + scaled_guessing_loss

        aux_info['total_loss'] = total_loss
        aux_info['loss'] = loss
        aux_info['guessing_loss'] = guessing_loss
        aux_info['scaled_guessing_loss'] = scaled_guessing_loss

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        if self.guessing_weight < 1:
            self.guessing_weight += self.guessing_step

        interaction = logging_strategy.filtered_interaction(
            source=src,
            labels=trg,
            preds=actual_preds,
            aux=aux_info,
        )

        return total_loss, interaction
