from models.networks import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import random


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        #         torch.nn.init.normal_(m.weight, mean=0., std=1.)
        #         nn.init.xavier_normal_(.data)
        torch.nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1 and classname.find('Equal') == -1:
        #         torch.nn.init.normal_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.constant_(m.bias, 0.0)


class MineModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        if self.opt.is_continue:
            # self.load_network(self.G, 'G', 'latest', self.opt.name)
            # self.load_network(self.D, 'D', 'latest', self.opt.name)
            print("load model successful")
        if self.opt.continue_exp is not None:
            # self.load_network(self.G, 'G', 'latest', self.opt.continue_exp)
            # self.load_network(self.D, 'D', 'latest', self.opt.continue_exp)
            print("load model successful", self.opt.continue_exp)

        if self.opt.is_continue == False and self.opt.continue_exp is None:
            # self.D.apply(weights_init)
            # self.G.apply(weights_init)
            print("init model!")

        # if load pretrain
        # if opt.load_pretrain is not None and len(opt.load_pretrain) > 0:
        #     self.load_network(self.G, 'G', opt.which_epoch)

        if len(self.opt.gpu_ids) > 0:
            self.G.cuda()
            self.D.cuda()
            self.idNet.cuda()

        # define Loss
        # self.criterionGAN = GANLoss(opt.gan_mode, opt=self.opt)
        # self.criterionRec = nn.L1Loss(reduction='none')
        # self.criterionFeat = nn.L1Loss()

        # define optimizer
        # params = list(self.G.parameters())
        # self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))

        # optimizer D
        params = list(self.D.parameters())
        self.optimizer_D = torch.optim.Adam(params, lr=opt.lr * opt.ttur_mult, betas=(opt.beta1, opt.beta2))

    def forward(self,data):

        # get loss

        # sum loss
        return {"loss_x": None}

    def cosin_metric(self, x1, x2):
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def save(self, which_epoch):
        self.save_network(self.G, 'G', which_epoch)
        self.save_network(self.D, 'D', which_epoch)

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.savePath, self.opt.name, "ckpt/", save_filename)
        if not os.path.exists(os.path.join(self.opt.savePath, self.opt.name, "ckpt")):
            os.mkdir(os.path.join(self.opt.savePath, self.opt.name, "ckpt"))
        torch.save(network.cpu().state_dict(), save_path)
        if len(self.opt.gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def load_network(self, network, network_label, epoch_label, name):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join("runs", name, 'ckpt', save_filename)
        network.load_state_dict(torch.load(save_path))