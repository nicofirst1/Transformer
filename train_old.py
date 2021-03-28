import argparse
import time

import torch
import torch.nn.functional as F
import torchtext

from Batch import create_masks
from Models import get_model
from Optim import CosineWithRestarts
from Process import *
from utils import get_len



def train_model(model, opt, train_data):
    print("training model...")
    model.train()
    train_len = get_len(train_data)

    start = time.time()

    if opt.checkpoint > 0:
        cptime = time.time()

    for epoch in range(opt.epochs):

        total_loss = 0
        print("   %dm: epoch %d [%s]  %d%%  loss = %s" % \
              ((time.time() - start) // 60, epoch + 1, "".join(' ' * 20), 0, '...'), end='\r')

        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        for i, batch in enumerate(train_data):

            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt.device, opt.src_pad, opt.trg_pad)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()

            total_loss += loss.item()

            if (i + 1) % opt.printevery == 0:
                p = int(100 * (i + 1) / train_len)
                avg_loss = total_loss / opt.printevery
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, avg_loss), end='\r')

                total_loss = 0

            if opt.checkpoint > 0 and ((time.time() - cptime) // 60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()

        print("%d s: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
              ((time.time() - start), epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
               avg_loss, epoch + 1, avg_loss))

        print("saving weights to " + opt.output_dir + "/...")
        torch.save(model.state_dict(), f'{opt.output_dir}/model_weights')
        print("weights saved ! ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder_num', type=int, default=4)
    parser.add_argument('-dencoder_num', type=int, default=4)
    parser.add_argument('-src_data', default='data/europarl-v7.it-en.en')
    parser.add_argument('-trg_data', default='data/europarl-v7.it-en.it')
    parser.add_argument('-src_lang', default='en_core_web_sm')
    parser.add_argument('-trg_lang', default='it_core_news_sm')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=256)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-output_dir', default='output')

    opt = parser.parse_args()
    print(opt)

    opt.device = "cpu" if opt.no_cuda else "cuda"
    if opt.device == "cuda":
        assert torch.cuda.is_available()

    src_data, trg_data = read_data(opt)
    create_fields(opt, src_data, trg_data)

    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    train_data, SRC, TRG = create_dataset(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    if opt.device == "cuda":
        model.cuda()

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=get_len(train_data))
    if opt.checkpoint > 0:
        print(
            "model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    print("saving field pickles to " + opt.output_dir + "/...")
    pickle.dump(SRC, open(f'{opt.output_dir}/SRC.pkl', 'wb'))
    pickle.dump(TRG, open(f'{opt.output_dir}/TRG.pkl', 'wb'))
    print("field pickles saved ! ")

    train_model(model, opt, train_data)


if __name__ == "__main__":
    main()
