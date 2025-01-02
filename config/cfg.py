import argparse
import yaml
import os
from datetime import datetime
import time

def arg2str(args):
    # args.run_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args_dict = vars(args)
    option_str = 'run_time: ' + datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str


class BaseConfig(object):

    def __init__(self, config = None):
        self.config = config

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--seed', type=int, default=2)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--ratio', type=float, default=0.15)
        self.parser.add_argument('--depth', type=int, default=3)
        self.parser.add_argument('--dim', type=int, default=512)
        self.parser.add_argument('--att_dropout', type=float, default=0.3)
        self.parser.add_argument('--ff_dropout', type=float, default=0.25)
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--lr_adjust', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=256)
        self.parser.add_argument('--Kfold', type=int, default=5)
        
        self.parser.add_argument('--save_dir', type=str, default='./result')
        self.parser.add_argument('--model_dir', type=str, default='./data/model')
        self.parser.add_argument('--data_dir', type=str, default='./data')
        self.parser.add_argument('--trainer', type=str, default='trainer')
        
        self.parser.add_argument('--use_amp', type=bool, default=False)
        self.parser.add_argument('--cluster', type=bool, default=True)
        self.parser.add_argument('--use_amp_in_eval', type=bool, default=False)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--time_delay', type=int, default=0)
        self.parser.add_argument('--stepvalues', type=int, default=4000)
        self.parser.add_argument('--start_iter', type=int, default=0)
        self.parser.add_argument('--max_iter', type=int, default=10000)
        self.parser.add_argument('--val_freq', type=int, default=100)
        # self.parser.add_argument('--display_freq', type=int, default=20)
        # self.parser.add_argument('--validation_after', type=int, default=0)
        # self.parser.add_argument('--save_log', type=str, default='./result/save_log/')
        # self.parser.add_argument('--save_folder', type=str, default='./result/save_model/')
        # self.parser.add_argument('--trainer', type=str, default='trainer')
        
        

    
    def load_base(self, derived_config, config):
        if '__base__' in derived_config:
            for each in derived_config['__base__']:
                with open(each) as f:
                    derived_config_ = yaml.safe_load(f)
                    config = self.load_base(derived_config_, config)
            # config = {**config, **derived_config}
        # else:
        config = {**config, **derived_config}
        return config

    # def load_base(self, derived_config, config):
    #     config = self._load_base(derived_config, config)
    #     if config['exp_param'] not in [None, 'None']:
    #         for each in config['exp_param']:
    #             config['exp_name'] = config['exp_name'] + '-' + each + '=' + config[each]

    #     return config

    def initialize(self, config = None):
        args = self.parser.parse_args()

        # print(self.parser.config)
        # print(self.parser.confag)
        # raise ValueError

        if self.config:
            args.config = self.config

        config = {}
        # with open(args.config) as f:
        #     derived_config = yaml.safe_load(f)
        #     config = self.load_base(derived_config, config)



        if 'exp_param' in config and config['exp_param'] not in [None, 'None']:
            if isinstance(config['exp_param'], str):
                config['exp_name'] = str(config['exp_name']) + '-' + str(config['exp_param']) + '=' + str(config[config['exp_param']])
            else:
                for each in config['exp_param']:
                    config['exp_name'] = str(config['exp_name']) + '-' + str(each) + '=' + str(config[each])

              
        # for key, value in config.items():
        #     setattr(args, key, value)

        # if args.time_delay != 0:
        #     print('================ {:^30s} ================'.format('Delay for {} seconds'.format(args.time_delay)))
        #     time.sleep(args.time_delay)

        return args


# 加载YAML文件
# if args.config:
#     with open(args.config) as f:
#         config = yaml.safe_load(f)
# else:
#     # 如果未指定YAML文件，则使用默认值
#     config = {
#         'arg1': 'default_value1',
#         'arg2': 123,
#         'arg3': 3.14
#     }

# # 将YAML配置加载到args中
# for key, value in config.items():
#     setattr(args, key, value)

# # 输出参数值
# print(args.arg1)
# print(args.arg2)
# print(args.arg3)
