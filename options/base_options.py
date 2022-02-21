import argparse
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./dataset/CelebA')
        self.parser.add_argument('--savePath', type=str, default='runs')
        self.parser.add_argument('--data_shuffle', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt

if __name__ == '__main__':
    options = BaseOptions()
    options.parse()