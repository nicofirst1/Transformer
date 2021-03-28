# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import sys
from collections import defaultdict
from typing import Any, List, Optional

import numpy as np
import torch

common_opts = None
optimizer = None
summary_writer = None


def get_len(train):
    for i, b in enumerate(train):
        pass

    return i


def _populate_cl_params(arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    arg_parser.add_argument(
        "--random_seed", type=int, default=None, help="Set random seed"
    )
    # core params
    arg_parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Where the checkpoints are stored",
    )

    arg_parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=0,
        help="How often the checkpoints are saved",
    )
    arg_parser.add_argument(
        "--validation_freq",
        type=int,
        default=1,
        help="The validation would be run every `validation_freq` epochs",
    )
    arg_parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to train (default: 10)",
    )
    arg_parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="If the parameter is set, model, core, and optimizer states are loaded from the "
             "checkpoint (default: None)",
    )
    # cuda setup
    arg_parser.add_argument(
        "--no_cuda", default=False, help="disable cuda", action="store_true"
    )
    # dataset
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for training (default: 32)",
    )

    # optimizer
    arg_parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to use [adam, sgd, adagrad] (default: adam)",
    )
    arg_parser.add_argument(
        "--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2)"
    )
    arg_parser.add_argument(
        "--update_freq",
        type=int,
        default=1,
        help="Learnable weights are updated every update_freq batches (default: 1)",
    )

    # Channel parameters
    arg_parser.add_argument(
        "--vocab_size",
        type=int,
        default=10,
        help="Number of symbols (terms) in the vocabulary (default: 10)",
    )
    arg_parser.add_argument(
        "--max_len", type=int, default=1, help="Max length of the sequence (default: 1)"
    )

    # Setting up tensorboard
    arg_parser.add_argument(
        "--tensorboard", default=False, help="enable tensorboard", action="store_true"
    )
    arg_parser.add_argument(
        "--tensorboard_dir", type=str, default="runs/", help="Path for tensorboard log"
    )

    return arg_parser


def _get_params(
        arg_parser: argparse.ArgumentParser, params: List[str]
) -> argparse.Namespace:
    args = arg_parser.parse_args(params)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # just to avoid confusion and be consistent
    args.no_cuda = not args.cuda
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if args.fp16 and torch.__version__ < "1.6.0":
        print("--fp16 is only supported with pytorch >= 1.6.0, please update!")
        args.fp16 = False

    return args


def init(
        arg_parser: Optional[argparse.ArgumentParser] = None,
        params: Optional[List[str]] = None,
) -> argparse.Namespace:
    """
    Should be called before any code using egg; initializes the common components, such as
    seeding logic etc.

    :param arg_parser: An instance of argparse.ArgumentParser that is pre-populated if game-specific arguments.
        `init` would add the commonly used arguments and parse the CL parameters. This allows us to easily obtain
        commonly used parameters and have a full list of parameters obtained by a `--help` argument.
    :param params: An optional list of parameters to be parsed against pre-defined frequently used parameters.
    If set to None (default), command line parameters from sys.argv[1:] are used; setting to an empty list forces
    to use default parameters.
    """
    global common_opts
    global optimizer
    global summary_writer

    if arg_parser is None:
        arg_parser = argparse.ArgumentParser()
    arg_parser = _populate_cl_params(arg_parser)

    if params is None:
        params = sys.argv[1:]
    common_opts = _get_params(arg_parser, params)

    if common_opts.random_seed is None:
        common_opts.random_seed = random.randint(0, 2 ** 31)
    elif common_opts.distributed_context:
        common_opts.random_seed += common_opts.distributed_context.rank

    _set_seed(common_opts.random_seed)

    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
    }
    if common_opts.optimizer in optimizers:
        optimizer = optimizers[common_opts.optimizer]
    else:
        raise NotImplementedError(f"Unknown optimizer name {common_opts.optimizer}!")

    if summary_writer is None and common_opts.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=common_opts.tensorboard_dir)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Cannot load tensorboard module; makes sure you installed everything required"
            )

    if common_opts.update_freq <= 0:
        raise RuntimeError("update_freq should be an integer, >= 1.")

    return common_opts


def close() -> None:
    """
    Should be called at the end of the program - however, not required unless Tensorboard is used
    """
    global summary_writer
    if summary_writer:
        summary_writer.close()


def get_opts() -> argparse.Namespace:
    """
    :return: command line options
    """
    global common_opts
    return common_opts


def get_summary_writer() -> "torch.utils.SummaryWriter":
    """
    :return: Returns an initialized instance of torch.util.SummaryWriter
    """
    global summary_writer
    return summary_writer


def _set_seed(seed) -> None:
    """
    Seeds the RNG in python.random, torch {cpu/cuda}, numpy.
    :param seed: Random seed to be used


    >>> _set_seed(10)
    >>> random.randint(0, 100), torch.randint(0, 100, (1,)).item(), np.random.randint(0, 100)
    (73, 37, 9)
    >>> _set_seed(10)
    >>> random.randint(0, 100), torch.randint(0, 100, (1,)).item(), np.random.randint(0, 100)
    (73, 37, 9)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to(x: Any, device: torch.device) -> Any:
    """
    Simple utility function that moves a tensor or a dict/list/tuple of (dict/list/tuples of ...) tensors
        to a specified device, recursively.
    :param x: tensor, list, tuple, or dict with values that are lists, tuples or dicts with values of ...
    :param device: device to be moved to
    :return: Same as input, but with all tensors placed on device. Non-tensors are not affected.
             For dicts, the changes are done in-place!
    """
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, list) or isinstance(x, tuple):
        return [move_to(i, device) for i in x]
    if isinstance(x, dict) or isinstance(x, defaultdict):
        for k, v in x.items():
            x[k] = move_to(v, device)
        return x
    return x
