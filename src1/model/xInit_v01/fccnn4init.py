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

        self.init_recon = common.FlatCamSimInitConv4()
        

    def forward(self, x):
        if self.is_fcSim:
            x = self.add_noise(self.fc_sim(x))
        x = self.init_recon(x)
       
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
