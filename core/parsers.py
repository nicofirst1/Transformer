import argparse

from arch import AVAIABLE_MODELS


def _populate_cl_params(arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    arg_parser.add_argument(
        "--random_seed", type=int, default=None, help="Set random seed"
    )

    # cuda setup
    arg_parser.add_argument(
        "--no_cuda", default=False, help="disable cuda", action="store_true"
    )

    arg_parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1,
        help="How often the checkpoints are saved",
    )
    arg_parser.add_argument('-validation_freq', type=int, default=1)

    arg_parser.add_argument('-epochs', type=int, default=200)

    arg_parser.add_argument('-checkpoint', type=int, default=0)
    arg_parser.add_argument('-output_dir', default='output')

    arg_parser.add_argument('-k', type=int, default=3)
    arg_parser.add_argument('-max_len', type=int, default=80)

    return arg_parser


def data_params(parser):
    parser.add_argument('-data_perc', default=0.003,type=float)
    parser.add_argument('-src_lang', default='en_core_web_sm')
    parser.add_argument('-trg_lang', default='it_core_news_sm')
    parser.add_argument('-min_word_freq', type=int, default=5,
                        help="The minimum number of times a word should appear in the corpora to be included"
                             " in the vocabulary. Increase if you get a cuda OOM error")
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-create_valset', action='store_true')

    return parser


def model_params(parser):
    parser.add_argument('-encod_num', type=int, default=4)
    parser.add_argument('-dencod_num', type=int, default=4)
    parser.add_argument('-model', type=str, choices=AVAIABLE_MODELS.keys(),
                        default="modeling")
    parser.add_argument('-model_dim', type=int, default=128)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-load_weights', default=True)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-dropout', type=int, default=0.1)

    return parser


def init_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser = _populate_cl_params(arg_parser)
    arg_parser = data_params(arg_parser)
    arg_parser = model_params(arg_parser)

    return arg_parser
