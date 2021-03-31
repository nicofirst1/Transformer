import math

import numpy as np
import torch
from torch.autograd import Variable
from torchtext.legacy import data

from core import move_to


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask


def create_masks(src, trg, device, src_pad, trg_pad):
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg = move_to(trg, device)

        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        np_mask = move_to(np_mask, device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

class MyIterator(data.Iterator):
    multiplier = 100

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * self.multiplier):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

    def __len__(self):
        return math.ceil(len(self.data()) / self.batch_size * self.multiplier)


global max_src_in_batch, max_tgt_in_batch


class BatchSize:
    def __init__(self):
        self.max_src_in_batch = 0
        self.max_tgt_in_batch = 0

    def batch_size_fn(self, new, count, sofar):
        "Keep augmenting batch and calculate total number of tokens + padding."

        self.max_src_in_batch = max(self.max_src_in_batch, len(new.src))
        self.max_tgt_in_batch = max(self.max_tgt_in_batch, len(new.trg) + 2)
        src_elements = count * self.max_src_in_batch
        tgt_elements = count * self.max_tgt_in_batch
        return max(src_elements, tgt_elements)
