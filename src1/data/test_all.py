import torch
import utility
import data
import model
import loss
from option import args
import glob
import os 
import imageio
import visvis as vv 
import numpy as np 
from model import common 
import cv2 
import scipy.io as scio 

def main():
    # load image 
    ext = ('.png')
    dir_hr = 'D:/xImageDataset/benchmark/Set5/HR'
    #names_hr = sorted(
    #    glob.glob(os.path.join(dir_hr + test_set + '/HR', '*' + ext[0]))
    #)

    files = [f for f in glob.glob(dir_hr + "**/*.png", recursive=True)]

    for f_hr in files:

        # Reading images
        hr =cv2.imread(f_hr)        
        h, w, _ = np.shape(hr)
        cut_size = min(h, w)
        hr = hr[0:cut_size, 0:cut_size, :]

        hr = cv2.resize(hr, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('HR '  + str(np.shape(hr)[0]), hr)
        cv2.waitKey(10) 
        
        # Simulate flatcam measuremetn 
        img = torch.from_numpy((cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)).float().cuda()
        fc_meas = common.flatcamSamp(torch.unsqueeze(img / 255, 0))
        fc_meas_n = common.apply_noise(fc_meas, nSig= 10)       

        # Simulated reconstruction        
        rec_sim = common.flatcamRecSimple(fc_meas_n)
        x_bayer, x_norm, x_, x_nonneg, rec_org = common.flatcamRecOrg(fc_meas_n)        

        print('Simulated measurement')
        
        file_name = os.path.basename(f_hr)
        file_name = file_name[:-4]
        scio.savemat(file_name + '.mat', {'img' : torch.squeeze(img.permute(1, 2, 0)).cpu().numpy(),
                                    'fc_meas' : torch.squeeze(fc_meas).cpu().numpy(),
                                    'fc_meas_n' : torch.squeeze(fc_meas_n).cpu().numpy(),
                                    'x_bayer' : torch.squeeze(x_bayer).cpu().numpy(),
                                    'x_norm' : torch.squeeze(x_norm).cpu().numpy(),
                                    'x_' : torch.squeeze(x_norm).cpu().numpy(),
                                    'x_nonneg' : torch.squeeze(x_nonneg).cpu().numpy(),
                                    'rec_org' : torch.squeeze(rec_org).permute(1, 2, 0).cpu().numpy(),
                                    'rec_sim' : torch.squeeze(rec_sim).permute(1, 2, 0).cpu().numpy()})
        


if __name__ == '__main__':
    main()
