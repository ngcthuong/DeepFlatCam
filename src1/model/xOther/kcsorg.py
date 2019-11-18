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
        self.toRGB = common.ApplyBayer2RGB(normalize = False) 

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


    def forward(self, x):
        if self.is_fcSim:
            x_sim = self.fc_sim(x)
            x_noise = self.add_noise(x_sim)    

        x = self.toBayer(x_noise)     
        
        x_ = self.init_recon1(torch.unsqueeze(x[:, 0, :, :], 1))
        x_ = torch.cat((x_, self.init_recon2(torch.unsqueeze(x[:, 1, :, :], 1))), 1)
        x_ = torch.cat((x_, self.init_recon3(torch.unsqueeze(x[:, 2, :, :], 1))), 1)                
        x_ = torch.cat((x_, self.init_recon4(torch.unsqueeze(x[:, 3, :, :], 1))), 1)

        x_ = self.toRGB(x_)

        return x_, x_sim, x_noise

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
