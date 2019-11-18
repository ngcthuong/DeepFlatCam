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

        self.init_recon1 = common.FlatCamSimInitConv1()
        self.init_recon2 = common.FlatCamSimInitConv2()
        self.init_recon3 = common.FlatCamSimInitConv3()
        self.init_recon4 = common.FlatCamSimInitFix()
        
        self.shuffle = ChannelShuffle(groups = 3)
        self.conv = nn.Conv2d(in_channels = 4 * 3 , out_channels = 3, kernel_size = 1, padding=0, stride=1, groups = 3)

    def forward(self, x):
        if self.is_fcSim:
            x_sim = self.fc_sim(x)
            x_noise = self.add_noise(x_sim)
        #x = self.init_recon(x_noise)
        
        #return x, x_sim, x_noise
        #if self.is_fcSim:
        #    x = self.add_noise(self.fc_sim(x))
        x = torch.cat((self.init_recon1(x_noise), self.init_recon2(x_noise), \
                           self.init_recon3(x_noise), self.init_recon4(x_noise)), 1)       
        return self.conv(self.shuffle(x)), x_sim, x_noise

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x