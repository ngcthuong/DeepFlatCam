import torch
import os 
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
os.environ['CUDA_VISIBLE_DEVICES']='0'

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main(): 
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            
            #args.model_init='kcsres'
            #args.test_only = True 
            #args.resume = -2 
            #args.batch_size = 16 
            #args.is_fcSim = False 

            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
