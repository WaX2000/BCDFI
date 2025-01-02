import torch
from config.cfg import BaseConfig
import training 
import os
from utils import data_load
import numpy as np
from time import time
import random
os.environ["mapreduce_input_fileinputformat_split_maxsize"] = "64" 

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main(args):
    runseed = args.seed
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    seed_everything(runseed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found!")
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found!")
    
    dataset = getattr(data_load, 'mydataset')
    trainer = getattr(training, args.trainer.lower())(args)

    test_data = dataset(
        dataset='test',
        args=args,
    )
    test_load = torch.utils.data.DataLoader(test_data, 
                                            batch_size=args.batch_size, 
                                            shuffle=False, 
                                            num_workers=args.num_workers)
    print('================ {:^30s} ================'.format('Test set loaded'))
    trainer.test(test_load,
                 save_file=os.path.join(args.save_dir, "result.csv"),
                 model_path=os.path.join(args.model_dir,"model.pth"))

if __name__ == '__main__':
    
    args = BaseConfig()
    args = args.initialize()
    print('================ {:^30s} ================'.format('Config loaded'))
    main(args)
    


    

