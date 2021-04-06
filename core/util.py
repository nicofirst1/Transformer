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

console = Console()


def get_len(train):
    for i, b in enumerate(train):
        pass

    return i


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
        arg_parser: argparse.ArgumentParser,
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

    if params is None:
        params = sys.argv[1:]
    common_opts = _get_params(arg_parser, params)

    if common_opts.random_seed is None:
        common_opts.random_seed = random.randint(0, 2 ** 31)
    elif common_opts.distributed_context:
        common_opts.random_seed += common_opts.distributed_context.rank

    _set_seed(common_opts.random_seed)

    return common_opts


def get_opts() -> argparse.Namespace:
    """
    :return: command line options
    """
    global common_opts
    return common_opts


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
