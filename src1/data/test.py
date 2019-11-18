import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    # Test dataset 
    args.model = 'fccnn1mw'
    args.data_test = 'Set5'
    args.data_test = args.data_test.split('+')
    args.resume = -1
    args.n_resblocks = 3
    args.test_only = True
    args.save_results = True 
    args.save_gt = True
    args.save = args.model + '_b' + str(args.n_resblocks) + 'f' + str(args.n_feats) + \
            's' + str(args.batch_size) +'sig' + str(args.sigma)

    # Load model 
    args.pretrain = '../experiment/' + args.save + '/model/model_best.pt'
    _model = model.Model(args, checkpoint)
    
    loader = data.Data(args)
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, _model, _loss, checkpoint)
    t.test() 



if __name__ == '__main__':
    main()
