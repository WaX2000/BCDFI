import os
import torch
import models
from tensorboardX import SummaryWriter
from config.cfg import arg2str
from torchmetrics import Accuracy, AUROC, MeanSquaredError
# from torch.cuda.amp import autocast, GradScaler
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt


 

class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")
        
class DefaultTrainer(object):
    def __init__(self, args,ncont=33,ncate=12):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        self.warmup_steps = args.warmup_steps
        self.cluster=args.cluster
        self.dim=args.dim
        self.depth=args.depth
        self.att_dropout=args.att_dropout
        self.ff_dropout=args.ff_dropout
        if self.args.lr_adjust==0:
            self.args.lr_adjust="fix"
        elif self.args.lr_adjust==1:
            self.args.lr_adjust = 'poly'
 
        # self.model = getattr(models, args.model_name.lower())(args)
        self.model = models.amformer(dim = self.dim,
                                depth = self.depth,
                                heads = 8,
                                attn_dropout = self.att_dropout,
                                ff_dropout = self.ff_dropout,
                                use_cls_token = False,# TRue

                                groups = [136,136,136],
                                sum_num_per_group = [32, 32, 32],
                                prod_num_per_group = [8,8,8],

                                cluster = self.cluster,
                                target_mode = 'mix',
                                token_descent = False, #True,
                                use_prod = True,
                                num_special_tokens = 2,
                                num_unique_categories = 10000,
                                out = 1,
                                num_cont = ncont,
                                num_cate = ncate,
                                use_sigmoid = True,)

        self.flag = 0
        self.model.cuda()
        # self.scaler = GradScaler()
        self.metrics = {
            'pcc':['high', 0],
            'r2':['high', 0],
            'loss':['low', 1000],
            'mse':['low', 1000],
        }
        self.start = 0
        self.wrong = None
        params = []
        params_for_pretrain = []
        for keys, param_value in self.model.named_parameters():
            params += [{'params': [param_value], 'lr': self.lr}]
        self.optim = torch.optim.Adam(params, lr=self.lr,
                                      betas=(0.9, 0.999), eps=1e-08)
        self.grads = []
        self.notprove_epoch=0


    def test(self, test_dataloader,model_path,save_file):
        if os.path.exists(model_path):
            checkpoint=torch.load(model_path)
            self.model.load_state_dict(checkpoint["net_state_dict"])
            print('================ {:^30s} ================'.format('Model parameters loaded'))
        else:
            raise FileNotFoundError(f"Model parameter file not found!")
        
        print('================ {:^30s} ================'.format('Begin Testing'))
        test_iter = iter(test_dataloader)
        epoch_size = len(test_dataloader)
        self.model.eval()
        with torch.no_grad():
            for i in range(epoch_size):
                cate, cont= next(test_iter)
                cate, cont= cate.cuda(), cont.cuda()
                pred= self.model(cate, cont,None).to('cpu')
                if i == 0:
                    total_pred = pred
                    # total_label = label
                else:
                    total_pred = torch.cat([total_pred, pred], dim=0)
                    # total_label = torch.cat([total_label, label], dim=0)
        df = pd.DataFrame(total_pred.tolist(), columns=['BCDFI'])
        df['BCDFI'] = df['BCDFI'].round(4)
        df.to_csv(save_file, index=False)
        print("The result was successfully saved to the path:",os.path.abspath(save_file))



