import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import base
import time
import numpy as np
from PIL import Image
import sys
import random

# ---------torch-----------
import torch
from torch.autograd import Variable
from torchvision import transforms

# ---------model-----------
from models.model import MineModel

# ---------data-----------
from data.dataset import create_dataloader

# ---------utils-----------
import utils.utils as util
from utils.visualizer import Visualizer

# ---------config-----------
from options.train_options import TrainOptions
from config import *

def save_model_structure(model, opt):
    file_name = os.path.join(opt.savePath, opt.name, 'model.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ G -------------\n')
        opt_file.write(str(model.G))
        opt_file.write('--------------G End ----------------\n')

        opt_file.write('------------ D -------------\n')
        opt_file.write(str(model.D))
        opt_file.write('--------------D End ----------------\n')

def save_opt(opt):
    opt_dir = os.path.join(opt.savePath, opt.name)
    # if not os.path.exists(opt.savePath):
    #     os.mkdir(opt.savePath)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)

    file_name = os.path.join(opt.savePath, opt.name, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(vars(opt).items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

def fill_opts(opt):
    config = getattr(sys.modules[__name__], opt.config_name)()

    config_dic = vars(config)
    opt_dic = vars(opt)
    for k in config_dic:
        #         if k in config_dic:
        opt.__setattr__(k, config_dic[k])
    save_opt(opt)

def save_code(opt):
    os.system("cp models/*.py runs/" + opt.name)
    os.system("cp train.py runs/" + opt.name)
    os.system("cp config.py runs/" + opt.name)


# train config
opt = TrainOptions()
opt.initialize()
# opt = opt.parser.parse_known_args(args=[])[0]
opt = opt.parser.parse_args()
fill_opts(opt)
save_code(opt)

# load data
data_loader = create_dataloader(opt)
dataset_size = len(data_loader.dataset)
print('#training images = %d' % dataset_size)


# load model
model = MineModel(opt)

# model.D.apply(weights_init)
# model.G.apply(weights_init)

save_model_structure(model, opt)
# model.load_network(model.G, 'G', '0_120000',"exp_DwithBN+linear_LossNs+wDr1(10)")
# model.load_network(model.D, 'D1', '0_120000',"exp_DwithBN+linear_LossNs+wDr1(10)")


# load visualizer
visualizer = Visualizer(opt)

optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

total_epoch = 0
total_steps = 0
iter_start_time = time.time()

for cur_epoch in range(opt.epoch_num):
    for cur_itr, data in enumerate(data_loader):

        # itr data
        # xxx.cuda() = data

        # model forward
        # loss_dict = model(data)

        # calculate final loss scalar
        # loss = torch.mean(loss_dict['loss_D_real'])


        ############### Backward Pass ####################
        # optimizer_G.zero_grad()
        # loss_G.backward()
        # optimizer_G.step()
        #
        # optimizer_D.zero_grad()
        # loss_D.backward()
        # optimizer_D.step()

        ############## Display results and errors ##########
        if total_steps % opt.print_freq == 0:
            for name, parms in model.G.named_parameters():
                if ("up" in str(name).lower() or "rgb" in str(name).lower()) and "weight" in str(
                        name) and "adain" not in str(name):
                    visualizer.plot_weight(name, parms, total_steps)
                    if parms.grad is not None:
                        visualizer.plot_gradient(name, parms.grad, total_steps)

        ### print out errors
        if total_steps % opt.print_freq == 0:
            #             errors = {k: float(v.data) if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {k: float(torch.mean(v)) for k, v in loss_dict.items()}
            t = time.time() - iter_start_time
            visualizer.print_current_errors(cur_epoch, cur_itr, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            iter_start_time = time.time()

        ### display output images
        with torch.no_grad():
            pass
            # xxx = model.forward

        ### save latest model
        if total_steps % opt.save_model_freq == 0:
            # print('saving the latest model (epoch %d, total_steps %d)' % (cur_epoch, total_steps))
            model.save(str(cur_epoch) + "_" + str(cur_itr))
            model.save("latest")

        total_steps += 1

