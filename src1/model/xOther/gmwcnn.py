from model import common
import torch
import torch.nn as nn
import scipy.io as sio
import math 

def make_model(args, parent=False):
    return BSR(args)

class BSR(nn.Module):
    def __init__(self, args, conv=common.default_conv, BBlock = common.BBlock):
        super(BSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        n_colors = args.n_colors
        self.scale_idx = 0
        self.DWT = common.DWT()
        self.IWT = common.IWT()
        act = nn.ReLU(True)
        n = 3
        self.scale = args.scale 

        # =============== Graudually up-scaling ===================

        u2_ = []        
        u2_.append(BBlock(conv, n_colors, 256, 3, act=act))
        for _ in range(n):
            u2_.append(BBlock(conv, 256, 256, 3, act=act))     
        u2_.append(BBlock(conv, 256, 640, 3, act=act))
        self.u2 = nn.Sequential(*u2_)
       
        u1_ = [BBlock(conv, 160, 160, 3, act=act)]        
        for _ in range(n):
            u1_.append(BBlock(conv, 160, 160, 3, act=act)) 
        self.u1 = nn.Sequential(*u1_)       

        u0_ = []    
        u0_.append(BBlock(conv, 160, n_colors * 4, 3, act=act)) 
        self.u0 = nn.Sequential(*u0_)       

        # =============== Main MWCNN =========================      
        
        m_head = [BBlock(conv, 4 * n_colors, 160, 3, act=act)]

        d_l1 = [BBlock(conv, 160, 160, 3, act=act)]         # always happend
        for _ in range(n):
            d_l1.append(BBlock(conv, 160, 160, 3, act=act))

        d_l2 = [BBlock(conv, 640, n_feats * 4, 3, act=act)]
        #if n_scale > 1:                                         # for scale 4 or 8 
        for _ in range(n):
            d_l2.append(BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))

        pro_l3 = [BBlock(conv, n_feats * 16, n_feats * 4, 3, act=act)]        
        #for _ in range(n*2):
        #    pro_l3.append(BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        pro_l3.append(BBlock(conv, n_feats * 4, n_feats * 16, 3, act=act))

        i_l2 = []
        for _ in range(n):
            i_l2.append(BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        i_l2.append(BBlock(conv, n_feats * 4,640, 3, act=act))

        i_l1 = []
        for _ in range(n):
            i_l1.append((BBlock(conv,160, 160, 3, act=act)))

        m_tail = [conv(160, 4, 3)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        u2 = self.u2(x)
        u1 = self.u1(self.IWT(u2))
        x  = self.IWT(self.u0(u1))

        x1 = self.d_l1(self.head(self.DWT(x)) + u1)
        x2 = self.d_l2(self.DWT(x1)  + u2 )


        # x3 = self.d_l2(self.DWT(x2))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2 
        x_ = self.IWT(self.i_l2(x_)) + x1 

        # x = self.i_l0(x) + x0
        x_ = self.IWT(self.tail(self.i_l1(x_))) + x
        # x = self.add_mean(x)

        return x, x_

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
