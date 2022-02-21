import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from torchvision import transforms
from PIL import Image


def create_dataloader(opt):
    dataset = MineDataSet(opt.dataroot, opt.batchSize, opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=opt.data_shuffle,
        num_workers=int(opt.nThreads))
    return dataloader


class MineDataSet(Dataset):
    def __init__(self, opt):
        super(MineDataSet, self).__init__()

        self.opt = opt

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
       return None

    def __len__(self):
        length = 0
        return length