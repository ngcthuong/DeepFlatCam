import os
import math
from decimal import Decimal
from scipy.misc import imresize
import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
import numpy as np 
import scipy.io
import torch.nn.functional as F 
from model import common
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('tkagg')
# import matlab.engine 
import scipy.io as scio 


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        self.is_upsample = False 
        up_model_list = ('mwcnn', 'vdsr', 'docnn', 'mwcnn_caa', 'mwcnn_cab', \
                         'mwcnn_caab', 'docnn_cab')
        for model in up_model_list:
            if self.args.model == model :
                self.is_upsample = True 
                break 

        self.is_pad = False 
        up_model_list = ('mwcnn', 'docnn', 'mwcnn_caa', 'mwcnn_cab', \
                         'mwcnn_caab', 'docnn_cab')
        for model in up_model_list:
            if self.args.model == model :
                self.is_pad = True 
                break 
        
        # self.eng = matlab.engine.start_matlab()

    def train(self):
        self.loss.step()        
        epoch = self.optimizer.get_last_epoch() + 1
        if self.args.resume > 0:
            epoch = self.args.resume + 1

        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):

            #if batch > 10:
            #    continue

            _, hr = self.prepare(lr, hr)

            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()

            img = utility.quantize(hr , self.args.rgb_range) 
            if self.args.is_fcSim:
                img = common.flatcamSamp(img / self.args.rgb_range)
                img = common.apply_noise(img, self.args.sigma)
                img = common.Raw2Bayer(img)
                img = common.make_separable(img)
                
                #scio.savemat( 'train_test_sig' + str(self.args.sigma) + '_' + str(batch) +'.mat', 
                #                { 'hr' : torch.squeeze(hr).permute(0, 2, 3, 1).detach().cpu().numpy(),
                #                  'hr2' : torch.squeeze(hr2).permute(0, 2, 3, 1).detach().cpu().numpy(),  
                #                  'sim_fc' : torch.squeeze(sim_fc).detach().cpu().numpy(),
                #                  'sim_fc_noise' : torch.squeeze(sim_fc_noise).detach().cpu().numpy(),
                #                  'sim_fc_bayer' : torch.squeeze(sim_fc_bayer).detach().cpu().numpy(),
                #                  'sim_fc_bayerNorm' : torch.squeeze(sim_fc_bayerNorm).detach().cpu().numpy(),})
            
            sr = self.model(img, idx_scale)
            loss = self.loss(sr, hr)

            if self.args.model == 'kcsres_mwcnn2' :
                loss = loss + self.loss(sr_init, hr)


            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        save_folder = 'Results_DL/' + self.args.save + '/' + self.args.data_test[0] + '/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        timer_test = utility.timer()
        # if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    _, hr = self.prepare(lr, hr)    

                    # Prepare data for test_only 
                
                    _, _, h, w = hr.size() 
                    idx = min(h, w)
                    hr = hr[:, :, 0:idx, 0:idx]         # squazsied

                    img = utility.quantize(hr , self.args.rgb_range) 
                    if self.args.is_fcSim:
                        img = common.flatcamSamp(img / self.args.rgb_range)
                        img = common.apply_noise(img, self.args.sigma)
                        img = common.Raw2Bayer(img)
                        img = common.make_separable(img)
                        #img = sim_fc_bayerNorm

                    sr = self.model(img, idx_scale)
                    sr = utility.quantize(sr , self.args.rgb_range)


                    if self.args.test_only:                        
                        plt.imsave(save_folder + filename[0] + '.png',
                        torch.squeeze(sr).permute(1, 2, 0).detach().cpu().numpy()  /self.args.rgb_range ) 
                        plt.imsave(save_folder + '__Org_' + filename[0] + '.png',
                        torch.squeeze(hr).permute(1, 2, 0).detach().cpu().numpy()  /self.args.rgb_range ) 
                        
                    save_list = [sr]
                    #print('\n')
                    #print(hr.size())
                    #print(sr.size())
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                        #print(cur_psnr, init_psnr)

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        # if self.args.save_results:             self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs


