import random
import os
import torch
import argparse
import numpy as np

from load_data import load_dataset, split_dataset, allocate_dataset, Dataset_Config
from attack_model import DCHTA

'''
# Locking random seed 固定随机种子
def seed_setting(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_setting()
'''

parser = argparse.ArgumentParser()
# dataset setting
# 数据集设置用于
parser.add_argument('--dataset', dest='dataset', default='WIKI', choices=['WIKI', 'IAPR', 'FLICKR', 'COCO', 'NUS'], help='name of the dataset')
parser.add_argument('--dataset_path', dest='dataset_path', default='../Datasets/', help='path of the dataset')

# attacked model setting
# 被攻击模型设置，：已经存在的被攻击方法：'DCMH', 'CPAH', 'DADH', 'DJSRH', 'JDSH', 'DGCPN'；被攻击模型的路径
parser.add_argument('--attacked_method', dest='attacked_method', default='DCMH', choices=['DCMH', 'CPAH', 'DADH', 'DJSRH', 'JDSH', 'DGCPN'], help='deep cross-modal hashing methods')
parser.add_argument('--attacked_models_path', dest='attacked_models_path', default='attacked_models/', help='path of attacked models')


# training or test detail
# 训练测试细节
parser.add_argument('--train', dest='train', action='store_true', help='to train or not')
parser.add_argument('--test', dest='test', action='store_true', help='to test or not')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', choices=['0', '1', '2', '3'])
parser.add_argument('--bit', dest='bit', type=int, default=16, choices=[16, 32, 64, 128], help='length of the hashing code')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=24, help='number of images in one batch')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=50, help='number of epoch')
parser.add_argument('--n_epochs_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
# 初试学习率
parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
# 学习率调整策略
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
# 一个训练标志,代表着我们想要在每隔多少个batch之后输出一次信息
parser.add_argument('--print_freq', dest='print_freq', type=int, default=10, help='print the debug information every print_freq iterations')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--sample_dir', dest='sample', default='samples/', help='output image are saved here during training')
# 保存训练输出的图片 路径和文件
parser.add_argument('--output_path', dest='output_path', default='outputs/', help='models are saved here')
parser.add_argument('--output_dir', dest='output_dir', default='output000', help='the name of output')
#
parser.add_argument('--transfer_attack', dest='transfer_attack', action='store_true', help='transfer_attack')
parser.add_argument('--transfer_attacked_method', dest='transfer_attacked_method', default='DCMH', choices=['DCMH', 'CPAH', 'DADH', 'DJSRH', 'JDSH', 'DGCPN'])
parser.add_argument('--transfer_bit', dest='transfer_bit', type=int, default=16, choices=[16, 32, 64, 128], help='length of the hashing code')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 数据集配置
Dcfg = Dataset_Config(args.dataset, args.dataset_path)
# 加载数据集
X, Y, L = load_dataset(Dcfg.data_path)
# 划分数据集为查询（测试），训练，和数据库数据用于检索
X, Y, L = split_dataset(X, Y, L, Dcfg.query_size, Dcfg.training_size, Dcfg.database_size)
# 训练，         数据库，              查询测试
Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L = allocate_dataset(X, Y, L)

# 本文模型
model = DCHTA(args=args, Dcfg=Dcfg)
# ？
# 训练集 数据库集 测试集的文本和标签用于训练
if args.train:
    model.train(Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_T, Te_L)
# 数据库集和测试集用于查询测试
if args.test:
    model.test(Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)
# 数据库和测试集用于迁移攻击
if args.transfer_attack:
    model.transfer_attack(Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)