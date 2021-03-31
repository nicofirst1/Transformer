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
from rich.console import Console

common_opts = None
optimizer = None
summary_writer = None

console=Console()


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
        default=1,
        help="How often the checkpoints are saved",
    )
    arg_parser.add_argument(
        "--validation_freq",
        type=int,
        default=1,
        help="The validation would be run every `validation_freq` epochs",
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


    # optimizer
    arg_parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to use [adam, sgd, adagrad] (default: adam)",
    )

    arg_parser.add_argument(
        "--update_freq",
        type=int,
        default=1,
        help="Learnable weights are updated every update_freq batches (default: 1)",
    )


    # Setting up tensorboard
    arg_parser.add_argument(
        "--tensorboard", default=False, help="enable tensorboard", action="store_true"
    )
    arg_parser.add_argument(
        "--tensorboard_dir", type=str, default="runs/", help="Path for tensorboard log"
    )

    return arg_parser


def _populate_custom_params(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('-validation_freq', type=int, default=1)
    parser.add_argument('-encoder_num', type=int, default=4)
    parser.add_argument('-dencoder_num', type=int, default=4)
    parser.add_argument('-src_data', default='data/europarl-v7.it-en.en')
    parser.add_argument('-trg_data', default='data/europarl-v7.it-en.it')
    parser.add_argument('-src_lang', default='en_core_web_sm')
    parser.add_argument('-trg_lang', default='it_core_news_sm')
    parser.add_argument('-SGDR', action='store_true')

    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=2048)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-output_dir', default='output')

    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)

    return parser


def _get_params(
        arg_parser: argparse.ArgumentParser, params: List[str]
) -> argparse.Namespace:
    args = arg_parser.parse_args(params)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # just to avoid confusion and be consistent
    args.no_cuda = not args.cuda
    args.device = torch.device("cuda" if args.cuda else "cpu")

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
    global summary_writer

    if arg_parser is None:
        arg_parser = argparse.ArgumentParser()
    arg_parser = _populate_cl_params(arg_parser)
    arg_parser = _populate_custom_params(arg_parser)

    if params is None:
        params = sys.argv[1:]
    common_opts = _get_params(arg_parser, params)

    if common_opts.random_seed is None:
        common_opts.random_seed = random.randint(0, 2 ** 31)
    elif common_opts.distributed_context:
        common_opts.random_seed += common_opts.distributed_context.rank

    _set_seed(common_opts.random_seed)

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
