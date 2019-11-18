from model import common
import torch
import torch.nn as nn
import scipy.io as sio

def make_model(args, parent=False):
    return BSR(args)

class BSR(nn.Module):
    def __init__(self, args, conv=common.default_conv, BBlock = common.BBlock):
        super(BSR, self).__init__()
        self.out_channels = 4
        self.scale_idx = 0
        self.is_fcSim = args.is_fcSim
        self.toRGB = common.ApplyBayer2RGB(normalize = False)
        self.n_colors = args.n_colors


        # Initial Reconstruction
        self.init_recon = common.KronConv(in_channels = 4, out_channels=self.out_channels,\
                                          mid_channels=args.mid_channels, act = args.is_act)
        self.init_simp = common.FlatCamSimInverse()
        self.conv = nn.Conv2d(4, 3, kernel_size = 1, stride= 1, padding= 0, bias= False)


    def forward(self, x):
      
        x_ = self.init_recon(x)

        #if not self.out_channels == 3:               # Finally convert to rgb
        x = self.toRGB(x_ + self.init_simp(x))

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
