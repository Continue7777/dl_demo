import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math


##############################################################################
# normal layer for idNet
##############################################################################



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm_type == "layer":
        norm_layer = nn.LayerNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


##############################################################################
# base modules
##############################################################################
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1.0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul
        self.weight_gain = lr_mul / np.sqrt(in_dim)

    #         self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.weight_gain, bias=self.bias * self.lr_mul)


class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, channels, latent_size=512):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class ApplyStyleSpade(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, in_channels, channels):
        super().__init__()
        #         self.linear = nn.Linear(latent_size, channels * 2)
        self.mlp_gamma = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

    def forward(self, x, fm):
        #         style = self.linear(latent.reshape(-1, fm.size(1)* fm.size(2)* fm.size(3)))  # style => [batch_size,c,h,2]
        #         shape = [-1, 2, fm.size(1), fm.size(2), fm.size(3)]
        #         style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #         x = x * (style[:, 0] + 1.) + style[:, 1]
        #         print(self.mlp_gamma.weight.shape,fm.shape,x.shape)
        gamma = self.mlp_gamma(fm)
        beta = self.mlp_beta(fm)

        out = x * (1 + gamma) + beta
        return out


class ApplyStyleEnc(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, in_channels, channels):
        super().__init__()
        #         self.linear = nn.Linear(latent_size, channels * 2)
        self.mlp = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

    #         self.mlp_beta = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

    def forward(self, x, fm):
        out = self.mlp(fm)
        return out


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H,
                      W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


class StyleVectorizer(nn.Module):
    def __init__(self, emb_num=512, is_equalLinear=False, depth=3):
        super(StyleVectorizer, self).__init__()

        layers = []
        for i in range(depth):
            if is_equalLinear:
                layers.extend([EqualLinear(emb_num, emb_num, 0.1), nn.LeakyReLU(0.2)])
            else:
                layers.extend([nn.Linear(emb_num, emb_num), nn.LeakyReLU(0.2)])
        self.mapping_net = nn.Sequential(*layers)

    def forward(self, input):
        id_mapping = self.mapping_net(input)
        return id_mapping

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'ns':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            if len(self.opt.gpu_ids) > 0:
                return self.real_label_tensor.expand_as(input).cuda()
            else:
                return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)

            if len(self.opt.gpu_ids) > 0:
                return self.fake_label_tensor.expand_as(input).cuda()
            else:
                return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        if len(self.opt.gpu_ids) > 0:
            return self.zero_tensor.expand_as(input).cuda()
        else:
            return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'ns':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)