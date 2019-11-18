import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as kr
import scipy.io 
import numpy as np
import kornia as kr 


mat = scipy.io.loadmat('data/flatcam_calibdata.mat')
cSize = np.squeeze(mat['cSize'][:, :]).astype(int)

Plr  = torch.from_numpy(np.squeeze(mat['P1r'][:,:]).astype(float)).float()
Plgb = torch.from_numpy(np.squeeze(mat['P1gb'][:,:]).astype(float)).float()
Plgr = torch.from_numpy(np.squeeze(mat['P1gr'][:,:]).astype(float)).float()
Plb  = torch.from_numpy(np.squeeze(mat['P1b'][:,:]).astype(float)).float()
Qlr  = torch.from_numpy(np.squeeze(mat['Q1r'][:,:]).astype(float)).float()
Qlgb = torch.from_numpy(np.squeeze(mat['Q1gb'][:,:]).astype(float)).float()
Qlgr = torch.from_numpy(np.squeeze(mat['Q1gr'][:,:]).astype(float)).float()
Qlb  = torch.from_numpy(np.squeeze(mat['Q1b'][:,:]).astype(float)).float()

batch_size = 32 
Plr = torch.unsqueeze(Plr, 0).expand((batch_size, Plr.size()[0], Plr.size()[1]))
Plgb = torch.unsqueeze(Plgb, 0).expand((batch_size, Plgb.size()[0], Plgb.size()[1]))
Plgr = torch.unsqueeze(Plgr, 0).expand((batch_size, Plgr.size()[0], Plgr.size()[1]))
Plb = torch.unsqueeze(Plb, 0).expand((batch_size, Plb.size()[0], Plb.size()[1]))
Qlr = torch.unsqueeze(Qlr, 0).expand((batch_size, Qlr.size()[0], Qlr.size()[1]))
Qlgb = torch.unsqueeze(Qlgb, 0).expand((batch_size, Qlgb.size()[0], Qlgb.size()[1]))
Qlgr = torch.unsqueeze(Qlgr, 0).expand((batch_size, Qlgr.size()[0], Qlgr.size()[1]))
Qlb = torch.unsqueeze(Qlb, 0).expand((batch_size, Qlb.size()[0], Qlb.size()[1]))


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def apply_noise(img, nSig = 10):
    r''' This function add simulated noise '''
    if nSig == 0:
        return img 
    for i in range(0, img.size()[1]):
        Y = img[i, :, :, :]
        tempY = Y - torch.min(torch.zeros(1, device = torch.device("cuda")), torch.min(Y.view(-1)))
        normY = torch.dist(tempY, torch.zeros(1, device = torch.device("cuda")), p = 2) 
        noise = torch.randn(Y.size(), device = torch.device('cuda'))
        noise = torch.sqrt((normY / nSig) ** 2 / (Y.numel() * torch.var(noise.view(-1)) )) * noise        
        img[i, :, :, :] = Y + noise

    return img 

class AddNoise(nn.Module):
    r'''Add noise for simulated measurement'''
    def __init__(self, nSig = 10):
        super(AddNoise, self).__init__()
        self.nSig = nSig

    def forward(self, x):
        return apply_noise(x, self.nSig)

class ApplyRaw2Bayer(nn.Module):
    r''' Convert Raw data to Bayer pattern'''
    def __init__(self):
        super(ApplyRaw2Bayer, self).__init__()

    def forward(self, x):
        return Raw2Bayer(x)

def Raw2Bayer(x, crop_size = cSize, is_rotate = False):
    r''' Convert FlatCam raw data to Bayer'''
    
    # Step 1. Convert the Image & rotate 
    c, b, h, w = x.size()
    
    y = torch.zeros((c, 4, int(h/2), int(w/2)), device = torch.device('cuda'))

    if is_rotate:                       # ---> THIS MODES DOESNOT WORK YET!!! (2019.07.14)
        scale = torch.ones(1)
        angle = torch.ones(1) * 0.05 * 360              # 0.05 is angle collected from data measurements 
        center = torch.ones(1, 2)
        center[..., 0] = int(h / 4)  # x
        center[..., 1] = int(w / 4)  # y
        M = kr.get_rotation_matrix2d(center, angle, scale).cuda()
        _, _, h, w = y.size()
        
        y[:, 0, :, : ] = kr.warp_affine(x[:, :, 1::2, 1::2], M, dsize = (h, w))
        y[:, 1, :, : ] = kr.warp_affine(x[:, :, 0::2, 1::2], M, dsize = (h, w))
        y[:, 2, :, : ] = kr.warp_affine(x[:, :, 1::2, 0::2], M, dsize = (h, w))
        y[:, 3, :, : ] = kr.warp_affine(x[:, :, 0::2, 0::2], M, dsize = (h, w))

    else:
        y[:, 0, :, : ] = x[:, 0, 1::2, 1::2]
        y[:, 1, :, : ] = x[:, 0, 0::2, 1::2]
        y[:, 2, :, : ] = x[:, 0, 1::2, 0::2]
        y[:, 3, :, : ] = x[:, 0, 0::2, 0::2]

    # Step 3. Crop the image 
    start_row = int((y.size()[2] - crop_size[0]) / 2) 
    end_row = start_row + crop_size[0]
    start_col = int((y.size()[3] - crop_size[1])/2) 
    end_col = start_col + crop_size[1] 
    return y[:,:, start_row:end_row, start_col:end_col]

def Bayer2RGB(x, normalize = True):
    b, _, h, w = x.size()
    x_rgb = torch.zeros((b, 3, h, w)).cuda()    
    x_rgb[:, 0, :, :] = x[:, 0, :, :]
    x_rgb[:, 1, :, :] = 0.5 * (x[:, 1, :, :]  + x[:, 2, :, :])
    x_rgb[:, 2, :, :] = x[:, 3, :, :]

    if normalize:
        x_rgb = (x_rgb - torch.min(x_rgb.view(-1))) / (torch.max(x_rgb.view(-1)) - torch.min(x_rgb.view(-1)) )
    
    return x_rgb 

class ApplyBayer2RGB(nn.Module):
    def __init__(self, normalize = True ):
        super(ApplyBayer2RGB, self).__init__()
        self.normalize = normalize
    
    def forward(self, x):
        return Bayer2RGB(x, self.normalize)
    

def flatcamSamp(x):
    y = torch.zeros((x.size()[0], 1, Plr.size()[1] * 2, 2 * Qlr.size()[1])).to(torch.device("cuda"))       
    y[:, 0, 1::2, 1::2] = torch.bmm(torch.bmm(Plr[0:x.size()[0], :, :].cuda(),  x[:, 0, :, :]), Qlr[0:x.size()[0], :, :].cuda().permute([0, 2, 1]))
    y[:, 0, 0::2, 1::2] = torch.bmm(torch.bmm(Plgb[0:x.size()[0], :, :].cuda(), x[:, 1, :, :]), Qlgb[0:x.size()[0], :, :].cuda().permute([0, 2, 1]))
    y[:, 0, 1::2, 0::2] = torch.bmm(torch.bmm(Plgr[0:x.size()[0], :, :].cuda(), x[:, 1, :, :]), Qlgr[0:x.size()[0], :, :].cuda().permute([0, 2, 1]))
    y[:, 0, 0::2, 0::2] = torch.bmm(torch.bmm(Plb[0:x.size()[0], :, :].cuda() , x[:, 2, :, :]), Qlb[0:x.size()[0], :, :].cuda().permute([0, 2, 1]))

    return y 

class FlatCamSampSim(nn.Module):
    r''' Simulated Flatcam measurement '''
    def __init__(self, batSize):
        super(FlatCamSampSim, self).__init__()
        if batSize > batch_size:
            raise Exception('batch_size should not exceed {}. Please change the corresponding batch_size values in common.py file'.format(batch_size))      

    def forward(self, x):        
        return flatcamSamp(x)

class FlatCamSimInverse(nn.Module):
    r''' Initial Reconstruction for Simulated'''
    def __init__(self):
        super(FlatCamSimInverse, self).__init__()           
        
    def forward(self, x):
        # Step 0 Convert from raw data to bayer 
        #x = Raw2Bayer(x)
         
        # Step 2: Simple Inverse 
        y = torch.zeros((x.size()[0], 4, Plr.size()[2], Qlr.size()[2])).to(torch.device("cuda"))
        y[:, 0, :, :] = torch.bmm(torch.bmm(Plr[0:x.size()[0], :, :].cuda().permute([0, 2, 1]),  x[:, 0, :, :]), Qlr[0:x.size()[0], :, :].cuda())        
        y[:, 1, :, :] = torch.bmm(torch.bmm(Plgb[0:x.size()[0], :, :].cuda().permute([0, 2, 1]), x[:, 1, :, :]), Qlgb[0:x.size()[0], :, :].cuda())
        y[:, 2, :, :] = torch.bmm(torch.bmm(Plgr[0:x.size()[0], :, :].cuda().permute([0, 2, 1]), x[:, 1, :, :]), Qlgr[0:x.size()[0], :, :].cuda())
        y[:, 3, :, :] = torch.bmm(torch.bmm(Plb[0:x.size()[0], :, :].cuda().permute([0, 2, 1]),  x[:, 2, :, :]), Qlb[0:x.size()[0], :, :].cuda())

        # Step 3: Convert to bayer pattern 
        #y = F.relu(y)               # Remove negative value  --> maybe not necessary 
        #y = Bayer2RGB(y)            # convert to RGB 

        return y 

class KronConv(nn.Module):
    r'''Kronecker Convolution, generate out_channel output, from in_channels input'''
    def __init__(self, in_channels = 4, out_channels = 4, mid_channels = 4, out_size = 256, height = 500, width = 620, act = True):
        super(KronConv, self).__init__()
        self.out_size = out_size
        self.mid_channels = mid_channels
        self.out_channels = out_channels         
        self.act = act 

        self.conv_h = nn.Conv2d(in_channels, out_size * mid_channels, kernel_size=[1, width], padding = 0, stride = 1, bias=False)
        self.conv_v = nn.Conv2d(mid_channels, out_size * out_channels, kernel_size=[height, 1], padding = 0, stride = 1, bias=False)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        b, c, h, w = x.size()         
        y = self.conv_h(x)                                 # (b, 256*c, h, 1)   --> b, 256*c, 1, h ---> b, 256, c, h --> b, c, h, 256 
        
        if self.act:
            y = self.prelu1(y)
        
        y = y.permute(0, 1, 3, 2).view(b, self.mid_channels, self.out_size, h)        
        y = self.conv_v(y.permute(0, 1, 3, 2))             # (b, 256*c, 1, 256) ---> b, c, 256, 256       

        if self.act :
            y  = self.prelu2(y)

        return y.view(b, self.out_channels, self.out_size, self.out_size)



def matrix_init():
    a = torch.linspace(1, 15, steps=15) - 8
    a = a.float()

    mat_1 = torch.mul(a, a)
    mat_1, mat_3 = torch.meshgrid([mat_1, mat_1])
    a = a.view(15, 1)
    mat_2 = torch.mul(a, a.t())
    mat_1 = mat_1.contiguous().view(1, 1, 225, 1, 1)
    mat_2 = mat_2.contiguous().view(1, 1, 225, 1, 1)
    mat_3 = mat_3.contiguous().view(1, 1, 225, 1, 1)

    return torch.cat((mat_1, mat_2, mat_3), 1).cuda()

class FDoCReLU(nn.Module):
    def __init__(self, stride = 1, in_channels = 64):
        super(FDoCReLU, self).__init__()
        self.requires_grad = True 
        self.stride = stride 
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = stride, padding=1, bias=False, groups = in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 5, stride = stride, padding=2, bias=False, groups = in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size = 7, stride = stride, padding=3, bias=False, groups = in_channels)
         
        if stride == 2:
            self.pool1 = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out0 = x
        if self.stride == 2:
            out0 = self.pool1(x)
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))

        res1 = out0 - out1
        res2 = out1 - out2 
        res3 = out2 - out3
        res4 = out3 

        return torch.cat((res1, res2, res3, res4), 1)

 
class IDoCReLU(nn.Module) :   
    ### Need to update with stride > 1 

    def __init__(self, stride = 1, in_channels = 64)    :
        super(IDoCReLU, self).__init__()
        self.stride = stride 
        out_channels = int(in_channels // 4)

        if stride > 1:
            self.convt = nn.Upsample(scale_factor=2, mode='bicubic' , align_corners= True )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding=0, groups= out_channels)

    def forward(self, x):
        out = x 
        if self.stride > 1:
            out = F.relu(self.convt(x))
        out = self.conv1(out)

        return out

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    del x   

    x_LL = x01[:, :, :, 0::2] + x02[:, :, :, 0::2] + x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
    x_HL = -x01[:, :, :, 0::2] - x02[:, :, :, 0::2] + x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
    x_LH = -x01[:, :, :, 0::2] + x02[:, :, :, 0::2] - x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
    x_HH = x01[:, :, :, 0::2] - x02[:, :, :, 0::2] - x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
    del x01, x02 

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    
    x = x /2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x[:, 0:out_channel, :, :] - x[:, out_channel:out_channel * 2, :, :] \
                        - x[:, out_channel * 2:out_channel * 3, :, :] + x[:, out_channel * 3:out_channel * 4, :, :]
    h[:, :, 1::2, 0::2] = x[:, 0:out_channel, :, :] - x[:, out_channel:out_channel * 2, :, :] \
                        + x[:, out_channel * 2:out_channel * 3, :, :] - x[:, out_channel * 3:out_channel * 4, :, :]
    h[:, :, 0::2, 1::2] = x[:, 0:out_channel, :, :] + x[:, out_channel:out_channel * 2, :, :] \
                        - x[:, out_channel * 2:out_channel * 3, :, :] - x[:, out_channel * 3:out_channel * 4, :, :]
    h[:, :, 1::2, 1::2] = x[:, 0:out_channel, :, :] + x[:, out_channel:out_channel * 2, :, :] \
                        + x[:, out_channel * 2:out_channel * 3, :, :] + x[:, out_channel * 3:out_channel * 4, :, :]

    return h



class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class BBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x
        
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y