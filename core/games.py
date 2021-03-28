# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from Batch import create_masks
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
            train_logging_strategy: LoggingStrategy = None,
            test_logging_strategy: LoggingStrategy = None,
    ):
        """
        :param sender: Sender agent. On forward, returns a tuple of (message, log-prob of the message, entropy).
        :param receiver: Receiver agent. On forward, accepts a message and the dedicated receiver input. Returns
            a tuple of (output, log-probs, entropy).
        :param loss: The loss function that accepts:
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs the end-to-end loss. Can be non-differentiable; if it is differentiable, this will be leveraged
        :param sender_entropy_coeff: The entropy regularization coefficient for Sender
        :param receiver_entropy_coeff: The entropy regularizatino coefficient for Receiver
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks
        """
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
