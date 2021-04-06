import numpy as np
import torch
from torch.autograd import Variable

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
