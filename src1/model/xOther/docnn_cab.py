import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio

from torch.autograd import Variable
from model import common


def make_model(args, parent = False, stride = 2):
    #args.n_resblocks = n_resblocks 
    #args.n_feats = n_feats
    args.stride = stride

    return BSR(args)

class BSR(nn.Module):
    def __init__(self, args, conv=common.default_conv, BBlock = common.BBlock, CALayer = common.CALayer):
        super(BSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        stride = args.stride 
        kernel_size = 3
        n_colors = args.n_colors
        self.scale_idx = 0

        act = nn.ReLU(True)

        # Transform 
        self.FDoC1 = common.FDoCReLU(stride = stride, in_channels= args.n_colors)
        self.FDoC2 = common.FDoCReLU(stride = stride, in_channels= 160 )
        self.FDoC3 = common.FDoCReLU(stride = stride, in_channels= n_feats * 4)
        self.IDoC1 = common.IDoCReLU(stride = stride, in_channels= n_feats * 16)
        self.IDoC2 = common.IDoCReLU(stride = stride, in_channels= 640)
        self.IDoC3 = common.IDoCReLU(stride = stride, in_channels= 4 * args.n_colors)

        # Attention 
        self.FCA2 = CALayer(channel = 160, reduction=16)
        self.FCA3 = CALayer(channel = n_feats * 4, reduction=16)
        self.ICA1 = CALayer(channel = n_feats * 16, reduction=16)
        self.ICA2 = CALayer(channel = 640, reduction=16)
        self.ICA3 = CALayer(channel = 4 * n_colors, reduction=2)

        n = 3
        m_head = [BBlock(conv, 4 * args.n_colors, 160, kernel_size, act=act)]
        d_l1 = []
        for _ in range(n):
            d_l1.append(BBlock(conv, 160, 160, kernel_size, act=act))

        d_l2 = [BBlock(conv, 640, n_feats * 4, kernel_size, act=act)]
        for _ in range(n):
            d_l2.append(BBlock(conv, n_feats * 4, n_feats * 4, kernel_size, act=act))

        pro_l3 = [BBlock(conv, n_feats * 16, n_feats * 4, kernel_size, act=act)]
        for _ in range(n*2):
            pro_l3.append(BBlock(conv, n_feats * 4, n_feats * 4, kernel_size, act=act))
        pro_l3.append(BBlock(conv, n_feats * 4, n_feats * 16, kernel_size, act=act))

        i_l2 = []
        for _ in range(n):
            i_l2.append(BBlock(conv, n_feats * 4, n_feats * 4, kernel_size, act=act))
        i_l2.append(BBlock(conv, n_feats * 4,640, kernel_size, act=act))

        i_l1 = []
        for _ in range(n):
            i_l1.append((BBlock(conv,160, 160, kernel_size, act=act)))

        m_tail = [conv(160, 4 * args.n_colors, 3)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x1 = self.FCA2(self.d_l1(self.head(self.FDoC1(x))))
        x2 = self.FCA3(self.d_l2(self.FDoC2(x1)))
        # x3 = self.d_l2(self.DWT(x2))
        x_ = self.IDoC1(self.ICA1(self.pro_l3(self.FDoC3(x2)))) + x2
        x_ = self.IDoC2(self.ICA2(self.i_l2(x_))) + x1

        # x = self.i_l0(x) + x0
        x = self.IDoC3(self.ICA3(self.tail(self.i_l1(x_)))) + x
        # x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx