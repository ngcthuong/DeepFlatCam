from model import common
import torch
import torch.nn as nn
import scipy.io as sio

def make_model(args, parent=False):
    return BSR(args)

class BSR(nn.Module):
    def __init__(self, args, conv=common.default_conv, BBlock = common.BBlock):
        super(BSR, self).__init__()
        #n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        n_colors = args.n_colors
        self.scale_idx = 0
        act = nn.ReLU(True)
        self.is_fcSim = args.is_fcSim 

        # Sampling layer 
        if args.is_fcSim:
            self.fc_sim = common.FlatCamSampSim(args.batch_size)
            self.add_noise = common.AddNoise(nSig = args.sigma)

        self.init_recon = common.FlatCamSimInitFix()

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        n = args.n_resblocks 

        m_head = [BBlock(conv, 4 * n_colors, 160, 3, act=act)]
        d_l1 = []
        for _ in range(n):
            d_l1.append(BBlock(conv, 160, 160, 3, act=act))

        d_l2 = [BBlock(conv, 640, n_feats * 4, 3, act=act)]
        for _ in range(n):
            d_l2.append(BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))

        pro_l3 = [BBlock(conv, n_feats * 16, n_feats * 4, 3, act=act)]
        for _ in range(n*2):
            pro_l3.append(BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        pro_l3.append(BBlock(conv, n_feats * 4, n_feats * 16, 3, act=act))

        i_l2 = []
        for _ in range(n):
            i_l2.append(BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        i_l2.append(BBlock(conv, n_feats * 4,640, 3, act=act))

        i_l1 = []
        for _ in range(n):
            i_l1.append((BBlock(conv,160, 160, 3, act=act)))

        m_tail = [conv(160, 4 * n_colors, 3)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if self.is_fcSim:
            x = self.add_noise(self.fc_sim(x))
        x = self.init_recon(x)

        x1 = self.d_l1(self.head(self.DWT(x)))
        x2 = self.d_l2(self.DWT(x1))
        # x3 = self.d_l2(self.DWT(x2))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1

        # x = self.i_l0(x) + x0
        x = self.IWT(self.tail(self.i_l1(x_))) + x
        # x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
