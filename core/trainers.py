# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import inspect
import os
from pathlib import Path
from typing import List, Optional

from memory_profiler import MemTracker

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch
from torch.utils.data import DataLoader

from .callbacks import (
    Callback,
    Checkpoint,
    ConsoleLogger,
)
from .interaction import Interaction
from .util import move_to

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    pass


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
            self,
            game: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            opts,
            optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None,
            grad_norm: float = None,
            aggregate_interaction_logs: bool = True,

    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = opts
        self.validation_freq = common_opts.validation_freq
        self.device = device

        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.aggregate_interaction_logs = aggregate_interaction_logs

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer.state = move_to(self.optimizer.state, self.device)

        frame = inspect.currentframe()  # define a frame to track
        #self.gpu_tracker = MemTracker(frame)  # define a GPU tracker

    def eval(self):
        mean_loss = 0.0
        interactions = []
        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss, interaction = self.game(batch)

                interaction = interaction.to("cpu")
                mean_loss += optimized_loss

                for callback in self.callbacks:
                    callback.on_batch_end(
                        interaction, optimized_loss, n_batches, is_training=False
                    )

                interactions.append(interaction)
                n_batches += 1

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

        return mean_loss.item(), full_interaction

    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.train()

        self.optimizer.zero_grad()

        for batch_id, batch in enumerate(self.train_data):
            #self.gpu_tracker.track(1)

            batch = move_to(batch, self.device)
            #self.gpu_tracker.track(2)

            context = nullcontext()
            with context:
                optimized_loss, interaction = self.game(batch)
            #self.gpu_tracker.track(3)

            optimized_loss.backward()
            #self.gpu_tracker.track(4)

            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.game.parameters(), self.grad_norm
                )

            self.optimizer.step()
            #self.gpu_tracker.track(5)

            self.optimizer.zero_grad()
            #self.gpu_tracker.track(6)

            n_batches += 1
            mean_loss += optimized_loss.detach()

            interaction = interaction.to("cpu")
            #self.gpu_tracker.track(7)

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)
            #self.gpu_tracker.track(8)

            interactions.append(interaction)
            torch.cuda.empty_cache()
            #self.gpu_tracker.track(9)

        if self.optimizer_scheduler:
            self.optimizer_scheduler.step()

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

        return mean_loss.item(), full_interaction

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch + 1)  # noqa: E226

            train_loss, train_interaction = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(
                    train_loss, train_interaction, epoch + 1
                )  # noqa: E226

            validation_loss = validation_interaction = None
            if (
                    self.validation_data is not None
                    and self.validation_freq > 0
                    and (epoch + 1) % self.validation_freq == 0
            ):  # noqa: E226, E501
                for callback in self.callbacks:
                    callback.on_test_begin(epoch + 1)  # noqa: E226
                validation_loss, validation_interaction = self.eval()

                for callback in self.callbacks:
                    callback.on_test_end(
                        validation_loss, validation_interaction, epoch + 1
                    )  # noqa: E226

            if self.should_stop:
                for callback in self.callbacks:
                    callback.on_early_stopping(
                        train_loss,
                        train_interaction,
                        epoch + 1,  # noqa: E226
                        validation_loss,
                        validation_interaction,
                    )
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.optimizer_scheduler_state_dict:
            self.optimizer_scheduler.load_state_dict(checkpoint.optimizer_scheduler_state_dict)
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        if isinstance(path, str):
            path = Path(path)

        for file in path.glob("*.tar"):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)
