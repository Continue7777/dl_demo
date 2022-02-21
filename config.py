import torch.nn as nn


class ConfigBase:
    def __init__(self):
        self.dataroot = "/home/dataset/simswap_filter250/"

        # sys
        self.gpu_ids = [1]
        self.nThreads = 8
        self.name = "exp_test"

        # data
        self.data_same = False
        self.diff_per = 0.0

        # train
        self.batchSize = 4
        self.is_continue = False
        self.continue_exp = None

        # archeticture

        # optimize
        self.lr = 0.00005
        self.beta1 = 0.0

        # loss

        # debug
        self.print_freq = 200
        self.eval_fake_freq = 2000
        self.save_model_freq = 20000


        # name
        self.describe = ""
        self.set_name()

    def set_name(self):
        if type(self).__name__ == "ConfigBase":
            self.name = "exp_test"
        else:
            self.name_id = type(self).__name__.split("_")[-1]
            self.name = "G0" + self.name_id + "_"
            self.name += self.describe


class Exp1(ConfigBase):
    def __init__(self):
        super().__init__()

        # change some params

        self.describe = ""
        self.message = ""

        self.set_name()