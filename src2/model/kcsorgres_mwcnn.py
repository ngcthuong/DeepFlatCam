from model import common
import torch
import torch.nn as nn
import scipy.io as sio

def make_model(args, parent=False):
    return BSR(args)

class BSR(nn.Module):
    def __init__(self, args, conv=common.default_conv, BBlock = common.BBlock):
        super(BSR, self).__init__()
        n_colors = args.n_colors
        self.scale_idx = 0
        #self.in_channels = args.in_channels
        self.is_fcSim = args.is_fcSim
        self.toRGB = common.ApplyBayer2RGB(normalize= False)
        n_feats = args.n_feats
        # Sampling layer
        if args.is_fcSim:
            self.fc_sim = common.FlatCamSampSim(args.batch_size)
            self.add_noise = common.AddNoise(nSig = args.sigma)

        self.toBayer = common.ApplyRaw2Bayer()

        # Initial Reconstruction
        self.init_recon1 = common.KronConv(in_channels=1, out_channels=1, mid_channels=1, act = args.is_act)
        self.init_recon2 = common.KronConv(in_channels=1, out_channels=1, mid_channels=1, act = args.is_act)
        self.init_recon3 = common.KronConv(in_channels=1, out_channels=1, mid_channels=1, act = args.is_act)
        self.init_recon4 = common.KronConv(in_channels=1, out_channels=1, mid_channels=1, act = args.is_act)
        self.conv = nn.Conv2d(4, 3, kernel_size = 1, stride= 1, padding= 0, bias= False)
        self.init_simp = common.FlatCamSimInverse()
        # Enhance reconstruction
        act = nn.ReLU(True)

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        n = 3
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
            x_sim = self.fc_sim(x)
            x_noise = self.add_noise(x_sim)

        x = self.toBayer(x_noise)

        x_ = self.init_recon1(torch.unsqueeze(x[:, 0, :, :], 1))
        x_ = torch.cat((x_, self.init_recon2(torch.unsqueeze(x[:, 1, :, :], 1))), 1)
        x_ = torch.cat((x_, self.init_recon3(torch.unsqueeze(x[:, 2, :, :], 1))), 1)
        x_ = torch.cat((x_, self.init_recon4(torch.unsqueeze(x[:, 3, :, :], 1))), 1)

        x_init = self.toRGB(x_ + self.init_simp(x))

        # Ehance reconstruction
        # Enhance reconstruction
        x1 = self.d_l1(self.head(self.DWT(x_init)))
        x2 = self.d_l2(self.DWT(x1))
        # x3 = self.d_l2(self.DWT(x2))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1

        # x = self.i_l0(x) + x0
        x = self.IWT(self.tail(self.i_l1(x_))) + x_init
        # x = self.add_mean(x)

        return x, x_noise, x_init

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
