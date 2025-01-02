import random
from torchvision import transforms
import torch
import torch.utils.data as data_utils
import numpy as np
import os
import copy
import csv
import torch
import pickle
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import h5py
from utils.utils import timer
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer


class MyDataset(data_utils.Dataset):
    
    @timer
    def __init__(self, dataset, args,kfold=0):
        self.mode=dataset
        self.data_list = []
        self.ratio=args.ratio
        epsilon = 1e-9
        mean=pd.read_csv(os.path.join(args.data_dir,"mean.csv"))
        std=pd.read_csv(os.path.join(args.data_dir,"std.csv"))
        if dataset == 'test':
            cate_df=pd.read_csv(os.path.join(args.data_dir,"cate_feat.csv"),index_col=False)
            cont_df=pd.read_csv(os.path.join(args.data_dir,"cont_feat.csv"),index_col=False)
            self.cate = cate_df[cate_col].values
            self.cont = ((cont_df[cont_col] - mean[cont_col].values)/(std[cont_col].values + epsilon)).values
    
    def __getitem__(self, idx):
        cate = copy.deepcopy(self.cate[idx])
        cont = copy.deepcopy(self.cont[idx])
        return torch.LongTensor(cate), torch.FloatTensor(cont)

    def __len__(self):
        return len(self.cont)
    

class make_split():

    def __init__(self):
        self.data_list = []
        # self.transform = transform

        file = '/home/ceyu.cy2/datasets/tabular/Income/income_evaluation.csv'
        now = 0
        with open(file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                self.data_list.append(row)
                # print(row)

        cols = [[] for _ in range(15)]
        for each in self.data_list:
            for i in range(15):
                cols[i].append(each[i])


        cont = []
        cate = []
        mapping = [{} for _ in range(15)]
        for i in range(15):
            if i == 14:
                # label index
                for j, each in enumerate(set(cols[i])):
                    mapping[i][each] = j
            elif i in [0, 2, 4, 10, 11, 12]:
                cont.append(cols[i])
            else:
                lens = len(set(cols[i]))
                # print(lens)
                for j, each in enumerate(set(cols[i])):
                    if each in mapping[i]:
                        raise ValueError
                    else:
                        mapping[i][each] = now + j
                now += lens

        n = 1
        for i in range(len(self.data_list)):
            j = 14
            if mapping[j] != {}:
                self.data_list[i][j] = mapping[j][self.data_list[i][j]]
        


        self.label = [each[14] for each in self.data_list]

        negative = [i for i in range(len(self.label)) if self.label[i] == 0]
        positive = [i for i in range(len(self.label)) if self.label[i] == 1]

        random.shuffle(negative)
        random.shuffle(positive)                  

        train = positive[:int(0.65*len(positive))]
        train += negative[:int(0.65*len(negative))]

        val = positive[int(0.65*len(positive)): int(0.8*len(positive))]
        val += negative[int(0.65*len(negative)): int(0.8*len(negative))]

        test = positive[int(0.8*len(positive)): ]
        test += negative[int(0.8*len(negative)): ]

        root = '/home/ceyu.cy2/datasets/tabular/Income/train_val_test/split4/'
        if not os.path.exists(root):
            os.mkdir(root)
        with open(root + 'train.pkl', 'wb') as f:
            pickle.dump(train, f)

        with open(root + 'val.pkl', 'wb') as f:
            pickle.dump(val, f)

        with open(root + 'test.pkl', 'wb') as f:
            pickle.dump(test, f)


class test_args():
    def __init__(self) -> None:
        self.seed = 0
        self.cate_mask_value = 0
        self.split_sets = 2

if __name__ == '__main__':
    # make_split()
    train_data = MyDataset(
        dataset='train',
        args=test_args()
    )
        

cate_col=['Snoring', 'Able to confide', 'Sleeplessness / insomnia', 'Usual walking pace', 'Current tobacco smoking', 'Past tobacco smoking', 'Alcohol intake frequency', 'Chest pain or discomfort', 'Hearing difficulty/problems', 'Hearing difficulty/problems with background noise', 'Chest pain or discomfort walking normally', 'Sex']
cont_col=['Waist circumference', 'Forced expiratory volume in 1-second', 'BMI', 'Whole body fat mass', 'Hip circumference', 'LA_pct', 'DHA_pct', 'IDL_FC', 'M_HDL_TG_pct', 'L_HDL_CE', 'L_HDL_CE_pct', 'L_HDL_C', 'L_LDL_TG', 'VLDL_C', 'L_VLDL_FC_pct', 'I_inteR', 'III_S', 'aVR_T', 'aVR_inteT', 'aVL_R', 'V6_inteR', 'Ecc_AHA_2', 'RAV_max', 'WT_AHA_10', 'Nitrogen dioxide air pollution; 2010', 'Nitrogen oxides air pollution; 2010', 'Particulate matter air pollution (pm10); 2010', 'Particulate matter air pollution (pm2.5); 2010', 'Age', 'Time spent watching television (TV)', 'Time spend outdoors in summer', 'Age first had sexual intercourse', 'Hand grip strength (left)']